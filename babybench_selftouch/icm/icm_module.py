# babybench_selftouch/icm/icm_module.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from babybench_selftouch.icm.vae import VAE
from babybench_selftouch.icm.forward import ForwardModel
from babybench_selftouch.icm.inverse_cvae import InverseCVAE

class ICMModule(nn.Module):
    """
    ICM assembly, managing VAE, forward, and inverse models.
    It is now a proper nn.Module to support saving and loading state_dicts,
    and implements a weighted loss scheme inspired by the original ICM paper.
    """
    def __init__(self, obs_dim, action_dim, latent_dim=8, hidden_dim=256, lr=1e-4, device='cpu'):
        super().__init__()
        
        self.device = device
        self.vae = VAE(obs_dim, latent_dim, hidden_dim).to(device)
        self.forward_model = ForwardModel(latent_dim, action_dim, hidden_dim).to(device)
        self.inverse_cvae = InverseCVAE(latent_dim, action_dim, hidden_dim).to(device)
        
        all_params = list(self.vae.parameters()) + \
                     list(self.forward_model.parameters()) + \
                     list(self.inverse_cvae.parameters())
        self.optimizer = optim.Adam(all_params, lr=lr)
        
        self.forward_loss_ema = 1.0
        self.inverse_loss_ema = 1.0
        self.ema_alpha = 0.99

    def encode_state(self, obs):
        with torch.no_grad():
            mu, logvar = self.vae.encode(obs)
            z = self.vae.reparameterize(mu, logvar)
        return z

    def reconstruct_state(self, obs):
        with torch.no_grad():
            recon_x, _, _, _ = self.vae(obs)
        return recon_x

    def compute_forward_loss(self, obs, action, next_obs, update_ema=True):
        self.eval()
        with torch.no_grad():
            mu, logvar = self.vae.encode(obs)
            z = self.vae.reparameterize(mu, logvar)
            mu_next, logvar_next = self.vae.encode(next_obs)
            z_next = self.vae.reparameterize(mu_next, logvar_next)
            z_pred = self.forward_model(z, action)
            loss = self.forward_model.compute_loss(z_pred, z_next)

        if update_ema:
            self.forward_loss_ema = self.ema_alpha * self.forward_loss_ema + (1 - self.ema_alpha) * loss.item()
        
        norm_loss = loss.item() / (self.forward_loss_ema + 1e-8)
        norm_loss = min(max(norm_loss, 0.0), 1.0)
        return norm_loss, loss

    def compute_inverse_loss(self, obs, next_obs, action, update_ema=True):
        self.eval()
        with torch.no_grad():
            mu, logvar = self.vae.encode(obs)
            z = self.vae.reparameterize(mu, logvar)
            mu_next, logvar_next = self.vae.encode(next_obs)
            z_next = self.vae.reparameterize(mu_next, logvar_next)
            a_pred, mu_z, logvar_z, _ = self.inverse_cvae(z, z_next, action)
            loss, recon_loss, kl_loss = self.inverse_cvae.compute_loss(a_pred, action, mu_z, logvar_z)

        if update_ema:
            self.inverse_loss_ema = self.ema_alpha * self.inverse_loss_ema + (1 - self.ema_alpha) * loss.item()
        
        norm_loss = loss.item() / (self.inverse_loss_ema + 1e-8)
        norm_loss = min(max(norm_loss, 0.0), 1.0)
        return norm_loss, loss

    def train_on_batch(self, obs_batch, action_batch, next_obs_batch, n_epochs=4, batch_size=256):
        self.train()
        
        dataset_size = obs_batch.shape[0]
        final_epoch_losses = {}
        
        # Define weights for the loss components, based on ICM paper and best practices
        beta_icm = 0.2
        w_vae = 1.0
        w_icm = 1.0
        
        for epoch in range(n_epochs):
            epoch_losses = {
                'vae_recon_loss': [], 'vae_kl_loss': [], 'forward_loss': [],
                'inverse_recon_loss': [], 'inverse_kl_loss': []
            }
            permutation = torch.randperm(dataset_size).to(self.device)
            
            for i in range(0, dataset_size, batch_size):
                indices = permutation[i : i + batch_size]
                obs, actions, next_obs = obs_batch[indices], action_batch[indices], next_obs_batch[indices]

                # --- Loss Calculation ---
                recon_x, mu, logvar, z = self.vae(obs)
                vae_total_loss, vae_recon_loss, vae_kl_loss = self.vae.compute_loss(obs, recon_x, mu, logvar)

                with torch.no_grad():
                    mu_next, logvar_next = self.vae.encode(next_obs)
                    z_next = self.vae.reparameterize(mu_next, logvar_next)
                
                z_pred = self.forward_model(z, actions)
                forward_loss = self.forward_model.compute_loss(z_pred, z_next)

                a_pred, mu_z, logvar_z, _ = self.inverse_cvae(z, z_next, actions)
                inverse_total_loss, inverse_recon_loss, inverse_kl_loss = self.inverse_cvae.compute_loss(a_pred, actions, mu_z, logvar_z)

                # --- CORRECTED: Apply weighted loss scheme based on ICM Paper ---
                icm_dynamics_loss = (1 - beta_icm) * inverse_total_loss + beta_icm * forward_loss
                
                # The final loss combines the VAE's task with the ICM's task
                total_loss = (w_vae * vae_total_loss) + (w_icm * icm_dynamics_loss)

                # --- Backpropagation ---
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Append individual losses for logging
                epoch_losses['vae_recon_loss'].append(vae_recon_loss.item())
                epoch_losses['vae_kl_loss'].append(vae_kl_loss.item())
                epoch_losses['forward_loss'].append(forward_loss.item())
                epoch_losses['inverse_recon_loss'].append(inverse_recon_loss.item())
                epoch_losses['inverse_kl_loss'].append(inverse_kl_loss.item())

            final_epoch_losses = {key: np.mean(val) for key, val in epoch_losses.items()}
        
        return final_epoch_losses