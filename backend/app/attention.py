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
    
    # Convert tokens to readable words (skip special tokens for visualization)
    tokens = bert_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Optionally, remove [CLS] and [SEP] tokens for visualization
    if tokens[0] == '[CLS]':
        tokens = tokens[1:]
        attn = attn[1:]
        attn = [row[1:] for row in attn]
    if tokens and tokens[-1] == '[SEP]':
        tokens = tokens[:-1]
        attn = attn[:-1]
        attn = [row[:-1] for row in attn]
    
    return AttentionResponse(tokens=tokens, attention=attn)


# Helper: get probability of original token at each position by masking it
def get_prediction_probabilities(text: str, known_indices: List[int], unknown_indices: List[int], custom_attention_mask=None):
    # Tokenize
    inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = inputs['input_ids'][0]
    tokens = bert_tokenizer.convert_ids_to_tokens(input_ids)
    probabilities = []
    seq_len = len(tokens)

    # If no custom_attention_mask provided, build one from known/unknown indices
    if custom_attention_mask is None:
        # By default, all ones (all tokens attend to all)
        mask = [[1.0 for _ in range(seq_len)] for _ in range(seq_len)]

        # Map word indices to BERT token indices
        # Assume frontend sends word indices (not subword indices)
        # We'll group BERT tokens into words: a new word starts at a token not starting with '##'
        word_to_token_indices = []  # List[List[int]]
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

        print("Word to token mapping:", word_to_token_indices)

        # Flatten all BERT token indices for unknown words
        unknown_token_indices = set()
        for word_idx in unknown_indices:
            if 0 <= word_idx < len(word_to_token_indices):
                for tidx in word_to_token_indices[word_idx]:
                    unknown_token_indices.add(tidx)

        print("Unknown token indices:", unknown_token_indices)

        # For each unknown BERT token, set its row to 0 (provides no attention), but keep its column as is (can receive attention)
        for idx in unknown_token_indices:
            if 0 <= idx < seq_len:
                for j in range(seq_len):
                    mask[idx][j] = 0.0
        custom_attention_mask = mask
        
        # Print a sample of the mask to verify it's being constructed correctly
        if unknown_token_indices:
            print("Sample mask rows for unknown tokens:")
            for idx in list(unknown_token_indices)[:3]:  # Show first 3
                print(f"  Row {idx}: {mask[idx][:5]}...")  # Show first 5 values

    # Broadcast mask to all heads and all layers: (num_layers, batch, num_heads, seq_len, seq_len)
    num_layers = getattr(bert_model.config, 'num_hidden_layers', 12)
    num_heads = getattr(bert_model.config, 'num_attention_heads', 12)
    cam_tensor = torch.tensor(custom_attention_mask, dtype=torch.float32)
    cam_tensor = cam_tensor.unsqueeze(0).unsqueeze(0)  # (1,1,seq_len,seq_len)
    cam_tensor = cam_tensor.expand(num_layers, 1, num_heads, cam_tensor.size(-2), cam_tensor.size(-1)).contiguous()  # (num_layers, 1, num_heads, seq_len, seq_len)
    
    print(f"Custom attention mask tensor shape: {cam_tensor.shape}")
    print(f"Zero entries in mask: {torch.sum(cam_tensor == 0.0).item()}")
    print(f"Total entries in mask: {cam_tensor.numel()}")
    if torch.sum(cam_tensor == 0.0) > 0:
        print("Mask has zero entries - attention masking should be active")

    # For each token (skip special tokens)
    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]']:
            probabilities.append(1.0)
            continue
        masked_input_ids = input_ids.clone()
        masked_input_ids[i] = bert_tokenizer.mask_token_id
        masked_inputs = {k: v.clone() for k, v in inputs.items()}
        masked_inputs['input_ids'][0] = masked_input_ids
        with torch.no_grad():
            outputs = bert_model(
                **masked_inputs,
                custom_attention_mask=cam_tensor
            )
            logits = outputs.logits[0, i]
            probs = torch.softmax(logits, dim=-1)
            orig_id = input_ids[i].item()
            prob = probs[orig_id].item()
            probabilities.append(prob)
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
