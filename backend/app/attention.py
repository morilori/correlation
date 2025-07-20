from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import torch
from transformers import BertTokenizer, BertModel

router = APIRouter()

class AttentionRequest(BaseModel):
    text: str

class AttentionResponse(BaseModel):
    tokens: List[str]
    attention: List[List[float]]  # attention[i][j]: how much word j attends to word i

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
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
        attn = attn.cpu().numpy().tolist()
    
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
