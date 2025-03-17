import torch
import torch.nn as nn
from typing import Optional
from .attention import MultiHeadAttention
from .feedforward import FeedForwardLayer, DropoutLayer
from .normalization import LayerNormalization
from .residual import ResidualConnection


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForwardLayer(d_model, d_ff)
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        self.dropout1 = DropoutLayer(dropout)
        self.dropout2 = DropoutLayer(dropout)
        self.dropout3 = DropoutLayer(dropout)
        self.residual1 = ResidualConnection()
        self.residual2 = ResidualConnection()
        self.residual3 = ResidualConnection()
    
    def forward(self, 
                x: torch.Tensor, 
                memory: torch.Tensor, 
                src_mask: Optional[torch.Tensor] = None, 
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        x1 = self.norm1(x)
        x = self.residual1(x, lambda x: self.dropout1(self.self_attn(x1, x1, x1, tgt_mask)))
        
        # Cross-attention
        x2 = self.norm2(x)
        x = self.residual2(x, lambda x: self.dropout2(self.cross_attn(x2, memory, memory, src_mask)))
        
        # Feed-forward
        x3 = self.norm3(x)
        x = self.residual3(x, lambda x: self.dropout3(self.ff(x3)))
        
        return x