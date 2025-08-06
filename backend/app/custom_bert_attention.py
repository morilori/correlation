import torch
from transformers.models.bert.modeling_bert import BertSelfAttention

class CustomBertSelfAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.layer_index = 0  # Will be set by the model patcher
        self._custom_mask = None
    
    def set_custom_mask(self, mask):
        """Set the custom attention mask for this layer"""
        self._custom_mask = mask
        print(f"[CustomBertSelfAttention Layer {getattr(self, 'layer_index', 'unknown')}] Mask set with shape: {mask.shape if mask is not None else None}")
    
    def clear_custom_mask(self):
        """Clear the stored custom mask"""
        self._custom_mask = None
    
    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, output_attentions=False, custom_attention_mask=None):
        # Check both parameter and stored mask
        has_param_mask = custom_attention_mask is not None
        has_stored_mask = hasattr(self, '_custom_mask') and self._custom_mask is not None
        layer_idx = getattr(self, 'layer_index', 'unknown')
        
        # Only debug first 3 layers to reduce output
        if layer_idx in [0, 1, 2]:
            # Debug: Print when custom mask is received
            if has_param_mask:
                print(f"Layer {layer_idx} received custom attention mask parameter with shape: {custom_attention_mask.shape}")
                print(f"Layer {layer_idx} parameter mask has {torch.sum(custom_attention_mask == 0.0).item()} zero entries")
            elif has_stored_mask:
                print(f"Layer {layer_idx} using stored custom mask with shape: {self._custom_mask.shape}")
                custom_attention_mask = self._custom_mask
            
            # Rest of the forward method...
            print(f"[CustomBertSelfAttention Layer {layer_idx}] Called. Parameter mask: {has_param_mask}, Stored mask: {has_stored_mask}")
        
        # Use the passed mask first, otherwise use the stored one
        mask_to_use = custom_attention_mask if custom_attention_mask is not None else (self._custom_mask if hasattr(self, '_custom_mask') else None)
        
        # Standard BERT attention computation
        mixed_query_layer = self.query(hidden_states)
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / (self.attention_head_size ** 0.5)

        # Inject custom mask (asymmetric, float, 0=block, 1=allow, or any value)
        if mask_to_use is not None:
            print(f"[CustomBertSelfAttention Layer {getattr(self, 'layer_index', 'unknown')}] Applying custom mask. Mask shape:", mask_to_use.shape)
            # Select the correct layer and broadcast to match attention_scores shape
            if mask_to_use.dim() == 5:  # (num_layers, batch, num_heads, seq_len, seq_len)
                layer_mask = mask_to_use[self.layer_index]  # (batch, num_heads, seq_len, seq_len)
            elif mask_to_use.dim() == 4:  # (batch, num_heads, seq_len, seq_len)
                layer_mask = mask_to_use
            elif mask_to_use.dim() == 2:  # (seq_len, seq_len)
                # Expand to match all heads and batch
                layer_mask = mask_to_use.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
                layer_mask = layer_mask.expand(attention_scores.size(0), attention_scores.size(1), -1, -1)
            else:
                layer_mask = mask_to_use
            
            # Apply the mask by multiplication
            attention_scores = attention_scores * layer_mask
            print(f"[CustomBertSelfAttention Layer {self.layer_index}] Mask applied successfully")

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        if past_key_value is not None:
            outputs = outputs + (past_key_value,)
        return outputs
