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
            if 'attention' in name.lower() or 'self' in name.lower():
                print(f"Checking module: {name} -> {child.__class__.__name__}")
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
                print(f"Replaced {child.__class__.__name__} at layer {counter[0]}")
                counter[0] += 1
            else:
                _replace_recursive(child, counter)
    
    _replace_recursive(module, layer_counter)

class CustomBertForMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        print("CustomBertForMaskedLM: Initializing and replacing attention layers...")
        replace_bert_self_attention(self)
        print("CustomBertForMaskedLM: Attention layer replacement complete")
        # Store the custom mask for propagation
        self._custom_attention_mask = None

    def forward(self, *args, custom_attention_mask=None, **kwargs):
        # Store the custom mask so it can be accessed by attention layers
        self._custom_attention_mask = custom_attention_mask
        print(f"CustomBertForMaskedLM.forward called with custom_attention_mask: {custom_attention_mask is not None}")
        if custom_attention_mask is not None:
            print(f"  Mask shape: {custom_attention_mask.shape}")
        
        # Monkey patch the encoder to pass the mask
        original_forward = self.bert.encoder.forward
        
        def custom_encoder_forward(hidden_states, attention_mask=None, head_mask=None, 
                                 encoder_hidden_states=None, encoder_attention_mask=None, 
                                 past_key_values=None, use_cache=None, output_attentions=False, 
                                 output_hidden_states=False, return_dict=True):
            
            # Pass the custom mask to each layer through the forward call
            # We'll modify the forward call to pass custom_attention_mask to each layer
            if custom_attention_mask is not None:
                print(f"CustomBertForMaskedLM: Setting mask on all {len(self.bert.encoder.layer)} layers")
                # Temporarily store the custom mask in each attention layer
                for i, layer in enumerate(self.bert.encoder.layer):
                    if hasattr(layer.attention.self, 'set_custom_mask'):
                        layer.attention.self.set_custom_mask(custom_attention_mask)
                        if i < 3:  # Only print for first 3 layers
                            print(f"  - Set mask on layer {i}")
            else:
                print("CustomBertForMaskedLM: No custom mask to set")
            
            return original_forward(hidden_states, attention_mask, head_mask, 
                                  encoder_hidden_states, encoder_attention_mask, 
                                  past_key_values, use_cache, output_attentions, 
                                  output_hidden_states, return_dict)
        
        # Temporarily replace the encoder's forward method
        self.bert.encoder.forward = custom_encoder_forward
        
        try:
            result = super().forward(*args, **kwargs)
        finally:
            # Restore original forward method
            self.bert.encoder.forward = original_forward
            
        return result
