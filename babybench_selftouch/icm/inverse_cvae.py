# babybench_selftouch/icm/inverse_cvae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .losses import sigma_vae_loss

class InverseCVAE(nn.Module):
    """
    Conditional VAE Inverse Model, models state pair -> action distribution
    """
    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.enc_fc1 = nn.Linear(latent_dim * 2 + action_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, hidden_dim)
        self.enc_logvar = nn.Linear(hidden_dim, hidden_dim)
        self.dec_fc1 = nn.Linear(hidden_dim + latent_dim * 2, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, action_dim)

    def encode(self, z_t, z_tp1, a):
        x = torch.cat([z_t, z_tp1, a], dim=-1)
        h = F.relu(self.enc_fc1(x))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, z_t, z_tp1):
        x = torch.cat([z, z_t, z_tp1], dim=-1)
        h = F.relu(self.dec_fc1(x))
        return self.dec_fc2(h)

    def forward(self, z_t, z_tp1, a):
        mu, logvar = self.encode(z_t, z_tp1, a)
        z = self.reparameterize(mu, logvar)
        a_pred = self.decode(z, z_t, z_tp1)
        return a_pred, mu, logvar, z

    def compute_loss(self, a_pred, a_true, mu, logvar, beta=1.0):
        recon_loss = sigma_vae_loss(a_pred, a_true)
        
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / a_true.shape[0]
        total_loss = recon_loss + beta * kl_loss
        return total_loss, recon_loss, kl_loss