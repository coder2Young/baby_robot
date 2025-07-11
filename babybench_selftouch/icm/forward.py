# babybench_icm/icm/forward.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ForwardModel(nn.Module):
    """
    ICM正向模型：(z_t, a_t) -> z_{t+1}_pred
    """
    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        h = F.relu(self.fc1(x))
        return self.fc2(h)

    def compute_loss(self, z_pred, z_next):
        return F.mse_loss(z_pred, z_next, reduction='mean')
