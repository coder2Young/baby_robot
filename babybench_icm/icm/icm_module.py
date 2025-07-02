# babybench_icm/icm/icm_module.py

import torch
from babybench_icm.icm.vae import VAE
from babybench_icm.icm.forward import ForwardModel
from babybench_icm.icm.inverse_cvae import InverseCVAE

class ICMModule:
    """
    ICM总成，管理VAE、正向、逆向模型及loss归一化
    """
    def __init__(self, obs_dim, action_dim, latent_dim=8, hidden_dim=256, device='cpu'):
        self.device = device
        self.vae = VAE(obs_dim, latent_dim, hidden_dim).to(device)
        self.forward_model = ForwardModel(latent_dim, action_dim, hidden_dim).to(device)
        self.inverse_cvae = InverseCVAE(latent_dim, action_dim, hidden_dim).to(device)
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
    
