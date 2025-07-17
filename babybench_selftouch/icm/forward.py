# babybench_selftouch/icm/forward.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# --- NEW: Import the custom loss function ---
from .losses import sigma_vae_loss

class ForwardModel(nn.Module):
    """
    ICM Forward Model: (z_t, a_t) -> z_{t+1}_pred
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
        # --- MODIFIED: Use sigma_vae_loss instead of F.mse_loss ---
        return sigma_vae_loss(z_pred, z_next)