import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import NUM_CLASSES, DEVICE

class GatingNetwork(nn.Module):
    def __init__(self, input_shape, num_learners, hidden_dim=256, dropout=0.1):
        """
        MLP-based Gating Network to select top-k weak learners.
        
        Args:
            input_shape: (packet_num, num_features)
            num_learners: Number of weak learners (n experts)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super(GatingNetwork, self).__init__()
        
        flatten_dim = input_shape[0] * input_shape[1]
        
        self.net = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_learners)
        )
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        elif x.dim() == 4:
            x = x.reshape(x.size(0), -1)
        
        return self.net(x)


class NoisyTopKGating(nn.Module):
    """Noisy Top-K Gating (GShard style) - adds noise for exploration"""
    def __init__(self, input_shape, num_learners, hidden_dim=256, noise_std=1.0, dropout=0.1):
        super(NoisyTopKGating, self).__init__()
        
        flatten_dim = input_shape[0] * input_shape[1]
        self.noise_std = noise_std
        
        self.w_gate = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_learners)
        )
        
        self.w_noise = nn.Sequential(
            nn.Linear(flatten_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_learners)
        )
    
    def forward(self, x, training=True):
        if x.dim() == 3:
            x = x.reshape(x.size(0), -1)
        elif x.dim() == 4:
            x = x.reshape(x.size(0), -1)
        
        logits = self.w_gate(x)
        
        if training:
            noise_scale = F.softplus(self.w_noise(x)) + 1e-2
            noise = torch.randn_like(logits) * noise_scale * self.noise_std
            logits = logits + noise
        
        return logits
