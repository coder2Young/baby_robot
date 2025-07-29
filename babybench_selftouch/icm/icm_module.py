# babybench_selftouch/icm/icm_module.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from babybench_selftouch.icm.vae import VAE
from babybench_selftouch.icm.forward import ForwardModel
# --- MODIFIED: Import the new ProbabilisticInverseModel ---
from babybench_selftouch.icm.probabilistic_inverse import ProbabilisticInverseModel


class ICMModule(nn.Module):
    """
    ICM assembly, managing VAE, forward, and inverse models.
    --- MODIFIED ---
    This version uses a stable ProbabilisticInverseModel instead of InverseCVAE.
    """
    def __init__(self, 
                 proprio_obs_dim, 
                 touch_obs_dim, 
                 action_dim, 
                 proprio_latent_dim=6, 
                 touch_latent_dim=2, 
                 hidden_dim=256, 
                 lr=1e-4, 
                 device='none',
                 vae_beta: float = 1.0):
        super().__init__()
        
        # If no device is specified, use the cuda, then mps, then cpu
        if device == 'none':
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        self.vae_beta = vae_beta
        self.proprio_latent_dim = proprio_latent_dim
        self.touch_latent_dim = touch_latent_dim
        combined_latent_dim = proprio_latent_dim + touch_latent_dim

        self.proprio_vae = VAE(proprio_obs_dim, proprio_latent_dim, hidden_dim).to(device)
        self.touch_vae = VAE(touch_obs_dim, touch_latent_dim, hidden_dim).to(device)
        self.forward_model = ForwardModel(combined_latent_dim, action_dim, hidden_dim).to(device)
        
        # --- MODIFIED: Instantiate the new ProbabilisticInverseModel ---
        self.inverse_model = ProbabilisticInverseModel(combined_latent_dim, action_dim, hidden_dim).to(device)
        
        # --- MODIFIED: Optimizer now includes parameters from the new inverse model ---
        all_params = list(self.proprio_vae.parameters()) + \
                     list(self.touch_vae.parameters()) + \
                     list(self.forward_model.parameters()) + \
                     list(self.inverse_model.parameters()) # Updated from inverse_cvae
        self.optimizer = optim.Adam(all_params, lr=lr)
        
        self.forward_loss_ema = 1.0
        # self.inverse_loss_ema is no longer needed in the same way, but can be kept for consistency
        self.ema_alpha = 0.99

    def _encode_and_combine(self, proprio_obs, touch_obs):
        proprio_recon, p_mu, p_logvar, z_proprio = self.proprio_vae(proprio_obs)
        touch_recon, t_mu, t_logvar, z_touch = self.touch_vae(touch_obs)
        z_combined = torch.cat([z_proprio, z_touch], dim=-1)
        return (z_combined, 
                (proprio_recon, p_mu, p_logvar), 
                (touch_recon, t_mu, t_logvar))

    def encode_states_for_inference(self, proprio_obs, touch_obs):
        self.eval()
        with torch.no_grad():
            p_mu, p_logvar = self.proprio_vae.encode(proprio_obs)
            z_proprio = self.proprio_vae.reparameterize(p_mu, p_logvar)
            t_mu, t_logvar = self.touch_vae.encode(touch_obs)
            z_touch = self.touch_vae.reparameterize(t_mu, t_logvar)
        return torch.cat([z_proprio, z_touch], dim=-1)

    def compute_forward_loss(self, proprio_obs, touch_obs, action, next_proprio_obs, next_touch_obs, update_ema=True):
        self.eval()
        with torch.no_grad():
            z_combined = self.encode_states_for_inference(proprio_obs, touch_obs)
            z_next_combined = self.encode_states_for_inference(next_proprio_obs, next_touch_obs)
            z_pred = self.forward_model(z_combined, action)
            loss = self.forward_model.compute_loss(z_pred, z_next_combined)
        if update_ema:
            self.forward_loss_ema = self.ema_alpha * self.forward_loss_ema + (1 - self.ema_alpha) * loss.item()
        norm_loss = max(0.0, loss.item() / (self.forward_loss_ema))
        return norm_loss, loss

    def train_on_batch(self, 
                       proprio_batch, 
                       touch_batch, 
                       action_batch, 
                       next_proprio_batch, 
                       next_touch_batch, 
                       n_epochs=4, 
                       batch_size=256):
        self.train()
        dataset_size = proprio_batch.shape[0]
        final_epoch_losses = {}
        
        beta_icm = 1.0 # Disable inverse model now
        w_proprio_vae = 1.0
        w_touch_vae = 5.0
        w_icm = 1.0
        
        for epoch in range(n_epochs):
            # --- MODIFIED: Updated keys for epoch_losses dictionary ---
            epoch_losses = {
                'proprio_vae_recon_loss': [], 'proprio_vae_kl_loss': [], 
                'touch_vae_recon_loss': [], 'touch_vae_kl_loss': [],
                'forward_loss': [],
                #'inverse_nll_loss': [] # Changed from recon/kl to a single NLL loss
            }
            permutation = torch.randperm(dataset_size).to(self.device)
            
            for i in range(0, dataset_size, batch_size):
                indices = permutation[i : i + batch_size]
                proprio, touch, actions = proprio_batch[indices], touch_batch[indices], action_batch[indices]
                next_proprio, next_touch = next_proprio_batch[indices], next_touch_batch[indices]

                # 1. & 2. VAE encoding and loss calculation (unchanged)
                (z_combined, 
                 (proprio_recon, p_mu, p_logvar), 
                 (touch_recon, t_mu, t_logvar)) = self._encode_and_combine(proprio, touch)
                proprio_vae_total_loss, p_recon_loss, p_kl_loss = self.proprio_vae.compute_loss(proprio, proprio_recon, p_mu, p_logvar, beta=self.vae_beta)
                touch_vae_total_loss, t_recon_loss, t_kl_loss = self.touch_vae.compute_loss(touch, touch_recon, t_mu, t_logvar, beta=self.vae_beta)

                # 3. Encode next state for dynamics models
                with torch.no_grad():
                    z_next_combined = self.encode_states_for_inference(next_proprio, next_touch)
                
                # 4. Calculate Forward Model loss (unchanged)
                z_pred = self.forward_model(z_combined, actions)
                forward_loss = self.forward_model.compute_loss(z_pred, z_next_combined.detach())

                # --- MODIFIED: 5. Calculate Inverse Model loss using the new model ---
                # The new model only takes the state transition as input. Detach z_next to stabilize.
                # mu_a, log_sigma_a = self.inverse_model(z_combined, z_next_combined.detach())
                # The new loss is a simple Negative Log-Likelihood
                # inverse_nll_loss = self.inverse_model.compute_loss(mu_a, log_sigma_a, actions)

                # 6. Combine all losses with weights
                # --- MODIFIED: Use the new inverse_nll_loss ---
                #icm_dynamics_loss = (1 - beta_icm) * inverse_nll_loss + beta_icm * forward_loss
                icm_dynamics_loss = forward_loss * beta_icm

                total_loss = (w_proprio_vae * proprio_vae_total_loss) + \
                             (w_touch_vae * touch_vae_total_loss) + \
                             (w_icm * icm_dynamics_loss)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                # Append losses for logging
                epoch_losses['proprio_vae_recon_loss'].append(p_recon_loss.item())
                epoch_losses['proprio_vae_kl_loss'].append(p_kl_loss.item())
                epoch_losses['touch_vae_recon_loss'].append(t_recon_loss.item())
                epoch_losses['touch_vae_kl_loss'].append(t_kl_loss.item())
                epoch_losses['forward_loss'].append(forward_loss.item())
                # --- MODIFIED: Log the new inverse_nll_loss ---
                #epoch_losses['inverse_nll_loss'].append(inverse_nll_loss.item())

            final_epoch_losses = {key: np.mean(val) for key, val in epoch_losses.items()}
        
        return final_epoch_losses