import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput

class CustomBertIntermediate(BertIntermediate):
    def __init__(self, config):
        super().__init__(config)
        self.layer_index = 0  # Will be set by the model patcher
        self._activation_counts = None
    
    def forward(self, hidden_states):
        # Standard FFN computation
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        
        # Count activations for each token position
        if self._activation_counts is not None:
            # Count neurons with positive activations (assuming GELU/ReLU-like activation)
            # For GELU, we'll count activations > small threshold since GELU can be negative
            threshold = 0.01
            activations = (hidden_states > threshold).float()  # (batch_size, seq_len, intermediate_size)
            activation_counts = activations.sum(dim=-1)  # (batch_size, seq_len)
            
            # Store activation counts for this layer
            batch_size, seq_len = activation_counts.shape
            if self.layer_index not in self._activation_counts:
                self._activation_counts[self.layer_index] = activation_counts.clone()
            else:
                # Accumulate if multiple passes (shouldn't happen in normal forward)
                self._activation_counts[self.layer_index] += activation_counts
        
        return hidden_states
    
    def set_activation_counter(self, counter_dict):
        """Set shared dictionary to store activation counts"""
        self._activation_counts = counter_dict
    
    def clear_activation_counter(self):
        """Clear activation counting"""
        self._activation_counts = None


class CustomBertOutput(BertOutput):
    def __init__(self, config):
        super().__init__(config)
        # This layer doesn't need special handling, just pass through
    
    def forward(self, hidden_states, input_tensor):
        # Standard output computation
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
