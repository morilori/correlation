import torch
from transformers import BertForMaskedLM, BertModel
from transformers.models.bert.modeling_bert import BertEncoder, BertLayer, BertAttention
from .custom_bert_attention import CustomBertSelfAttention
import types

# Helper to recursively replace all BertSelfAttention layers with CustomBertSelfAttention
def replace_bert_self_attention(module, layer_idx=0):
    layer_counter = [layer_idx]  # Use list to make it mutable in nested function
    
    def _replace_recursive(mod, counter):
        for name, child in mod.named_children():
            if child.__class__.__name__ in ['BertSelfAttention', 'BertSdpaSelfAttention']:
                # Replace with custom - use the module's config instead of child.config
                custom = CustomBertSelfAttention(module.config)
                # Copy weights if possible
                try:
                    custom.load_state_dict(child.state_dict())
                except Exception as e:
                    print(f"Warning: Could not copy weights from {child.__class__.__name__}: {e}")
                # Set layer index for mask slicing
                custom.layer_index = counter[0]
                setattr(mod, name, custom)
                counter[0] += 1
            else:
                _replace_recursive(child, counter)
    
    _replace_recursive(module, layer_counter)

class CustomBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        replace_bert_self_attention(self)

    def forward(self, *args, custom_attention_mask=None, **kwargs):
        # Only modify encoder behavior if we have a custom mask
        if custom_attention_mask is not None:
            # Store original forward method
            original_forward = self.bert.encoder.forward
            
            def custom_encoder_forward(hidden_states, attention_mask=None, head_mask=None, 
                                     encoder_hidden_states=None, encoder_attention_mask=None, 
                                     past_key_values=None, use_cache=None, output_attentions=False, 
                                     output_hidden_states=False, return_dict=True):
                
                # Set the custom mask on all attention layers
                for layer in self.bert.encoder.layer:
                    if hasattr(layer.attention.self, 'set_custom_mask'):
                        layer.attention.self.set_custom_mask(custom_attention_mask)
                
                return original_forward(hidden_states, attention_mask, head_mask, 
                                      encoder_hidden_states, encoder_attention_mask, 
                                      past_key_values, use_cache, output_attentions, 
                                      output_hidden_states, return_dict)
            
            # Temporarily replace the encoder's forward method
            self.bert.encoder.forward = custom_encoder_forward
            
            try:
                result = super().forward(*args, **kwargs)
            finally:
                # Restore original forward method and clear masks
                self.bert.encoder.forward = original_forward
                for layer in self.bert.encoder.layer:
                    if hasattr(layer.attention.self, 'clear_custom_mask'):
                        layer.attention.self.clear_custom_mask()
        else:
            # No custom mask, use normal forward pass
            result = super().forward(*args, **kwargs)
            
        return result
