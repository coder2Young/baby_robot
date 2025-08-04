# babybench_selftouch/icm/forward.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import sigma_vae_loss

class ForwardModel(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        delta_z_pred = self.fc3(h) # Predict change in latent state

        return z + delta_z_pred

    def compute_loss(self, z_pred, z_next):
        return sigma_vae_loss(z_pred, z_next)