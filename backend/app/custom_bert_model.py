import torch
from transformers import BertForMaskedLM, BertModel
from transformers.models.bert.modeling_bert import BertEncoder, BertLayer, BertAttention
from .custom_bert_attention import CustomBertSelfAttention
from .custom_bert_ffn import CustomBertIntermediate, CustomBertOutput
import types

# Helper to recursively replace all BertSelfAttention layers with CustomBertSelfAttention
# and FFN layers with CustomBertIntermediate
def replace_bert_layers(module, layer_idx=0):
    attention_counter = [layer_idx]  # Use list to make it mutable in nested function
    
    def _replace_recursive(mod, counter):
        for name, child in mod.named_children():
            if child.__class__.__name__ in ['BertSelfAttention', 'BertSdpaSelfAttention']:
                # Replace attention with custom
                custom = CustomBertSelfAttention(module.config)
                try:
                    custom.load_state_dict(child.state_dict())
                except Exception as e:
                    print(f"Warning: Could not copy attention weights: {e}")
                custom.layer_index = counter[0]
                setattr(mod, name, custom)
                counter[0] += 1
            elif child.__class__.__name__ == 'BertIntermediate':
                # Replace FFN intermediate with custom
                custom_intermediate = CustomBertIntermediate(module.config)
                try:
                    custom_intermediate.load_state_dict(child.state_dict())
                except Exception as e:
                    print(f"Warning: Could not copy FFN intermediate weights: {e}")
                # Set layer index based on current attention counter - 1 (since FFN comes after attention in same layer)
                custom_intermediate.layer_index = counter[0] - 1 if counter[0] > 0 else 0
                setattr(mod, name, custom_intermediate)
            elif child.__class__.__name__ == 'BertOutput':
                # Replace FFN output with custom (though it doesn't need special handling)
                custom_output = CustomBertOutput(module.config)
                try:
                    custom_output.load_state_dict(child.state_dict())
                except Exception as e:
                    print(f"Warning: Could not copy FFN output weights: {e}")
                setattr(mod, name, custom_output)
            else:
                _replace_recursive(child, counter)
    
    _replace_recursive(module, attention_counter)

class CustomBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        replace_bert_layers(self)
        self._ffn_activation_counts = {}  # Store FFN activation counts

    def forward(self, *args, custom_attention_mask=None, **kwargs):
        # Clear previous activation counts
        self._ffn_activation_counts.clear()
        
        # Set up FFN activation counting
        for layer in self.bert.encoder.layer:
            if hasattr(layer.intermediate, 'set_activation_counter'):
                layer.intermediate.set_activation_counter(self._ffn_activation_counts)
        
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
        
        # Clean up FFN activation counting
        for layer in self.bert.encoder.layer:
            if hasattr(layer.intermediate, 'clear_activation_counter'):
                layer.intermediate.clear_activation_counter()
                
        return result
    
    def get_ffn_activation_counts(self):
        """Get the FFN activation counts from the last forward pass"""
        return self._ffn_activation_counts.copy()
