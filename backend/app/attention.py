from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import torch
import numpy as np
from transformers import BertTokenizer
from .custom_bert_model import CustomBertForMaskedLM

router = APIRouter()


class AttentionRequest(BaseModel):
    text: str

# For prediction probability endpoint
class PredictionProbabilitiesRequest(BaseModel):
    text: str
    known_indices: List[int] = []  # indices of words marked as known
    unknown_indices: List[int] = []  # indices of words marked as unknown
    custom_attention_mask: List[List[float]] = None  # Optional: asymmetric mask (seq_len x seq_len)

class PredictionProbabilitiesResponse(BaseModel):
    tokens: List[str]
    probabilities: List[float]  # probability of the original word at each position

class AttentionResponse(BaseModel):
    tokens: List[str]
    attention: List[List[float]]  # attention[i][j]: how much word j attends to word i
    ffn_activations: List[float] = []  # FFN activation counts for each token
    probabilities: List[float] = []  # Prediction probabilities for each token (Recalled Meaning)

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = CustomBertForMaskedLM.from_pretrained('bert-base-uncased', output_attentions=True)
bert_model.eval()

@router.post('/attention', response_model=AttentionResponse)
def attention_endpoint(req: AttentionRequest):
    # Tokenize input to check length
    inputs = bert_tokenizer(req.text, return_tensors='pt', truncation=False)
    
    # Check if text is too long (BERT max is usually 512 tokens)
    token_length = inputs['input_ids'].shape[1]
    max_length = 512
    
    if token_length > max_length:
        # Truncate the text to fit within BERT's limits
        inputs = bert_tokenizer(req.text, return_tensors='pt', truncation=True, max_length=max_length)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        # outputs.attentions: tuple of (num_layers, batch, num_heads, seq_len, seq_len)
        # We'll sum over layers and heads for a single attention matrix
        attn = torch.stack(outputs.attentions)  # (num_layers, batch, num_heads, seq_len, seq_len)
        attn = attn.sum(dim=0).sum(dim=1)[0]  # (seq_len, seq_len)
        attn = attn.cpu().detach().numpy().tolist()
        
        # Get FFN activation counts
        ffn_counts_dict = bert_model.get_ffn_activation_counts()
        
        # Aggregate FFN activation counts across all layers
        seq_len = inputs['input_ids'].shape[1]
        ffn_activations = [0.0] * seq_len
        
        for layer_idx, counts in ffn_counts_dict.items():
            # counts is (batch_size, seq_len), we want batch 0
            layer_counts = counts[0].cpu().detach().numpy()
            for i in range(min(len(ffn_activations), len(layer_counts))):
                ffn_activations[i] += layer_counts[i]
    
    # Get prediction probabilities for each token (Recalled Meaning)
    tokens_for_probs = bert_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    probabilities = []
    input_ids = inputs['input_ids'][0]
    
    for i, token in enumerate(tokens_for_probs):
        if token in ['[CLS]', '[SEP]']:
            probabilities.append(1.0)
            continue
            
        # Create masked input
        masked_input_ids = input_ids.clone()
        masked_input_ids[i] = bert_tokenizer.mask_token_id
        masked_inputs = {k: v.clone() for k, v in inputs.items()}
        masked_inputs['input_ids'][0] = masked_input_ids
        
        with torch.no_grad():
            outputs_masked = bert_model(**masked_inputs)
            logits = outputs_masked.logits[0, i]
            probs = torch.softmax(logits, dim=-1)
            orig_id = input_ids[i].item()
            prob = probs[orig_id].item()
            probabilities.append(prob)
    
    # Convert tokens to readable words (skip special tokens for visualization)
    tokens = bert_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Optionally, remove [CLS] and [SEP] tokens for visualization
    if tokens[0] == '[CLS]':
        tokens = tokens[1:]
        attn = attn[1:]
        attn = [row[1:] for row in attn]
        ffn_activations = ffn_activations[1:]
        probabilities = probabilities[1:]
    if tokens and tokens[-1] == '[SEP]':
        tokens = tokens[:-1]
        attn = attn[:-1]
        attn = [row[:-1] for row in attn]
        ffn_activations = ffn_activations[:-1]
        probabilities = probabilities[:-1]
    
    return AttentionResponse(tokens=tokens, attention=attn, ffn_activations=ffn_activations, probabilities=probabilities)


# Helper: get probability of original token at each position by masking it
def get_prediction_probabilities(text: str, known_indices: List[int], unknown_indices: List[int], custom_attention_mask=None):
    # Tokenize
    inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = inputs['input_ids'][0]
    tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)
    seq_len = len(tokens)

    # Only create custom mask if we actually have unknown indices or explicit mask
    cam_tensor = None
    if unknown_indices or custom_attention_mask is not None:
        # Create attention mask efficiently
        if custom_attention_mask is None:
            # Build mask from known/unknown indices only when needed
            mask = torch.ones((seq_len, seq_len), dtype=torch.float32)
            
            # Map word indices to token indices
            word_to_token_indices = []
            current = []
            for i, tok in enumerate(tokens):
                if tok.startswith('##'):
                    current.append(i)
                else:
                    if current:
                        word_to_token_indices.append(current)
                    current = [i]
            if current:
                word_to_token_indices.append(current)

            # Set unknown token rows to 0 (they provide no attention)
            for word_idx in unknown_indices:
                if 0 <= word_idx < len(word_to_token_indices):
                    for tidx in word_to_token_indices[word_idx]:
                        if 0 <= tidx < seq_len:
                            mask[tidx, :] = 0.0
        else:
            mask = torch.tensor(custom_attention_mask, dtype=torch.float32)
        
        # Expand mask for all layers and heads only if we need it
        num_layers = bert_model.config.num_hidden_layers
        num_heads = bert_model.config.num_attention_heads
        cam_tensor = mask.unsqueeze(0).unsqueeze(0).expand(
            num_layers, 1, num_heads, seq_len, seq_len
        ).contiguous()

    # Calculate probabilities for each token
    probabilities = []
    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]']:
            probabilities.append(1.0)
            continue
            
        # Create masked input
        masked_input_ids = input_ids.clone()
        masked_input_ids[i] = bert_tokenizer.mask_token_id
        masked_inputs = {k: v.clone() for k, v in inputs.items()}
        masked_inputs['input_ids'][0] = masked_input_ids
        
        with torch.no_grad():
            # Only pass custom mask if we have one
            if cam_tensor is not None:
                outputs = bert_model(**masked_inputs, custom_attention_mask=cam_tensor)
            else:
                outputs = bert_model(**masked_inputs)
                
            logits = outputs.logits[0, i]
            probs = torch.softmax(logits, dim=-1)
            orig_id = input_ids[i].item()
            prob = probs[orig_id].item()
            probabilities.append(prob)

    # Remove special tokens from output
    if tokens[0] == '[CLS]':
        tokens = tokens[1:]
        probabilities = probabilities[1:]
    if tokens and tokens[-1] == '[SEP]':
        tokens = tokens[:-1]
        probabilities = probabilities[:-1]
        
    return tokens, probabilities


@router.post('/prediction-probabilities', response_model=PredictionProbabilitiesResponse)
def prediction_probabilities_endpoint(req: PredictionProbabilitiesRequest):
    tokens, probabilities = get_prediction_probabilities(
        req.text, req.known_indices, req.unknown_indices, req.custom_attention_mask
    )
    return PredictionProbabilitiesResponse(tokens=tokens, probabilities=probabilities)
