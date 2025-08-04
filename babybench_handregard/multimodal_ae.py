# babybench_handregard/multimodal_ae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VisualAutoEncoder(nn.Module):
    def __init__(self, img_shape=(64,64), z_dim=32):
        super().__init__()
        C, H, W = 3, img_shape[0], img_shape[1]
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(C, 32, 4, 2, 1),   # (B,32,32,32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # (B,64,16,16)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # (B,128,8,8)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*8*8, z_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 128*8*8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # (B,64,16,16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # (B,32,32,32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, C, 4, 2, 1),    # (B,3,64,64)
            nn.Sigmoid()
        )

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

    def encode_and_recon(self, img):
        # img: (B, 3, 64, 64) and float32 in [0,1]
        if img.ndim == 3:
            img = img.unsqueeze(0)
        z = self.encode(img)
        recon = self.decode(z)
        loss = F.mse_loss(recon, img)
        return z, loss.item(), recon.detach()

class ProprioAutoEncoder(nn.Module):
    def __init__(self, in_dim=466, z_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, z_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, in_dim)
        )

    def encode(self, x):
        return self.encoder(x)
    def decode(self, z):
        return self.decoder(z)
    def encode_and_recon(self, obs):
        # obs: (B, 466)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        z = self.encode(obs)
        recon = self.decode(z)
        loss = F.mse_loss(recon, obs)
        return z, loss.item(), recon.detach()

class MultimodalAutoEncoder(nn.Module):
    def __init__(self, z_v_dim=32, z_p_dim=32, z_m_dim=32):
        super().__init__()
        in_dim = z_v_dim + z_p_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, z_m_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_m_dim, 128),
            nn.ReLU(),
            nn.Linear(128, in_dim)
        )
        self.z_v_dim = z_v_dim
        self.z_p_dim = z_p_dim

    def encode(self, z_v, z_p):
        z_joint = torch.cat([z_v, z_p], dim=-1)
        return self.encoder(z_joint)
    def decode(self, z_m):
        return self.decoder(z_m)
    def encode_and_recon(self, z_v, z_p):
        z_joint = torch.cat([z_v, z_p], dim=-1)
        z_m = self.encoder(z_joint)
        recon = self.decoder(z_m)
        loss = F.mse_loss(recon, z_joint)
        return z_m, loss.item(), recon.detach()

class MultimodalAEManager:
    def __init__(self, img_shape=(64,64), proprio_dim=466, z_v_dim=32, z_p_dim=32, z_m_dim=32, device='cpu'):
        self.visual_ae = VisualAutoEncoder(img_shape, z_v_dim).to(device)
        self.proprio_ae = ProprioAutoEncoder(proprio_dim, z_p_dim).to(device)
        self.multi_ae = MultimodalAutoEncoder(z_v_dim, z_p_dim, z_m_dim).to(device)
        self.device = device

    def encode_and_recon(self, img_left, obs):
        # img_left: np.ndarray (64, 64, 3) uint8 or float32
        # obs: np.ndarray (466,)
        img = img_left
        if img.dtype == torch.uint8 or img.max() > 1.1:
            img = img.astype('float32') / 255.0
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()
        if img.ndim == 3:
            img = img.permute(2,0,1)  # (3,64,64)
        img = img.unsqueeze(0).to(self.device)
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        obs = obs.unsqueeze(0) if obs.ndim == 1 else obs

        # 1. Single-modal AE
        z_v, L_v, _ = self.visual_ae.encode_and_recon(img)
        z_p, L_p, _ = self.proprio_ae.encode_and_recon(obs)
        # 2. Multimodal AE
        z_m, L_m, _ = self.multi_ae.encode_and_recon(z_v, z_p)
        return {
            'z_v': z_v, 'z_p': z_p, 'z_m': z_m,
            'L_v': L_v, 'L_p': L_p, 'L_m': L_m
        }

