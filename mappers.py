import torch
import torch.nn as nn
from torch.nn import functional as F


class ProjectionHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.LayerNorm(d)
        )
    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1)



class QueryAggregator(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.query_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # [1, 1, D]
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, multimodal_sequence):
        B = multimodal_sequence.size(0)
        query = self.query_token.expand(B, -1, -1)  # [B, 1, D]
        attn_output, _ = self.attn(query, multimodal_sequence, multimodal_sequence)  # [B, 1, D]
        # output = attn_output.squeeze(1)
        output = self.norm(attn_output.squeeze(1))
        return output

    
class Linear_proj(nn.Module):
    def __init__(self, input_hidden_size, hidden_size, mlp_depth=2):
        super(Linear_proj, self).__init__()
        modules = [nn.Linear(input_hidden_size, hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        self.projector = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)