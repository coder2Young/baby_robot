# babybench_handregard/forward_model/forward_model.py

import torch
import torch.nn as nn
import numpy as np

class CVAEForwardModel(nn.Module):
    """
    条件VAE正模型，输入observation(466)+action，预测下一帧eye_left图像(64x64x3)
    """
    def __init__(self, obs_dim, action_dim, img_shape=(64,64,3), latent_dim=32, hidden_dim=256):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        # 条件编码
        self.cond_encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # 图像编码器
        C, H, W = img_shape[2], img_shape[0], img_shape[1]
        self.img_encoder = nn.Sequential(
            nn.Conv2d(C, 32, 4, 2, 1),   # (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), # (128, 8, 8)
            nn.ReLU(),
            nn.Flatten()
        )
        img_enc_outdim = 128 * 8 * 8

        # 合并条件与图像，输出mu,sigma
        self.fc_mu = nn.Linear(hidden_dim + img_enc_outdim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim + img_enc_outdim, latent_dim)

        # 解码器，先MLP到大向量再reshape成特征图
        self.fc_decode = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 128*8*8),
            nn.ReLU()
        )
        self.img_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), # (64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # (32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, C, 4, 2, 1),   # (3, 64, 64)
            nn.Sigmoid()  # [0,1]输出
        )

    def encode(self, obs, action, img):
        # obs: (B, 466), action: (B, A), img: (B, 64,64,3)
        cond = self.cond_encoder(torch.cat([obs, action], dim=-1))     # (B, H)
        img = img.permute(0, 3, 1, 2)  # NHWC→NCHW
        img_feat = self.img_encoder(img)  # (B, feat)
        feat = torch.cat([cond, img_feat], dim=-1)
        mu = self.fc_mu(feat)
        logvar = self.fc_logvar(feat)
        return mu, logvar, cond

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        # z: (B, latent_dim), cond: (B, hidden_dim)
        inp = torch.cat([z, cond], dim=-1)
        x = self.fc_decode(inp)        # (B, 8192)
        x = x.view(-1, 128, 8, 8)     # (B, 128, 8, 8)
        img = self.img_decoder(x)      # (B, 3, 64, 64)
        img = img.permute(0, 2, 3, 1) # NCHW→NHWC
        return img

    def forward(self, obs, action, img):
        # 用于训练：输入t时刻obs/action和t+1图像
        mu, logvar, cond = self.encode(obs, action, img)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar

    def generate(self, obs, action):
        # 用于推理：输入obs/action，采样z=0向量或随机采样
        device = obs.device
        cond = self.cond_encoder(torch.cat([obs, action], dim=-1))
        z = torch.zeros((obs.shape[0], self.latent_dim), device=device)
        img = self.decode(z, cond)
        return img

##############################
# 外部API
##############################

# ------- 外部接口：双分支管理 -------
class ForwardModel:
    """
    - 分别训练/预测左右眼
    - 支持 fit, predict, save, load
    """
    def __init__(self, obs_dim, action_dim, img_shape=(64,64,3), device='cpu'):
        self.left_model = CVAEForwardModel(obs_dim, action_dim, img_shape)
        self.right_model = CVAEForwardModel(obs_dim, action_dim, img_shape)
        self.device = device
        self.left_model.to(device)
        self.right_model.to(device)
        self.img_shape = img_shape

    def fit(self, obs_arr, action_arr, left_img_arr, right_img_arr, epochs=10, batch_size=128, lr=1e-3):
        """
        分别训练两只眼
        obs_arr: (N,466)
        action_arr: (N,A)
        left_img_arr, right_img_arr: (N,64,64,3)
        """
        self._fit_one(self.left_model, obs_arr, action_arr, left_img_arr, epochs, batch_size, lr, "left")
        self._fit_one(self.right_model, obs_arr, action_arr, right_img_arr, epochs, batch_size, lr, "right")

    def _fit_one(self, model, obs_arr, action_arr, img_arr, epochs, batch_size, lr, tag="left"):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        N = obs_arr.shape[0]
        for epoch in range(epochs):
            idxs = np.arange(N)
            np.random.shuffle(idxs)
            epoch_loss = 0
            for start in range(0, N, batch_size):
                end = min(N, start+batch_size)
                idx = idxs[start:end]
                obs = torch.tensor(obs_arr[idx], device=self.device, dtype=torch.float32)
                action = torch.tensor(action_arr[idx], device=self.device, dtype=torch.float32)
                imgs = torch.tensor(img_arr[idx].copy(), device=self.device, dtype=torch.float32)  # <--- .copy()加在这里
                if imgs.max() > 1.1:  # 若输入为uint8
                    imgs = imgs / 255.0
                recon, mu, logvar = model(obs, action, imgs)
                BCE = torch.mean((recon - imgs) ** 2)
                KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = BCE + 1e-4 * KLD
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * (end-start)
            epoch_loss /= N
            print(f"[CVAE-{tag}] epoch {epoch+1}/{epochs}, loss={epoch_loss:.4f}")

    def predict(self, obs, action, left_img, right_img):
        """
        obs: (466,), action: (A,), left_img/right_img: (64,64,3), 都是np.ndarray
        return: left_pred, right_pred, 均为(64,64,3) uint8
        """
        self.left_model.eval()
        self.right_model.eval()
        with torch.no_grad():
            # 单batch预测，全部 .copy()，保证不含负stride
            obs_t = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
            action_t = torch.tensor(action[None], dtype=torch.float32, device=self.device)
            left_img_t = torch.tensor(left_img[None].copy(), dtype=torch.float32, device=self.device)
            right_img_t = torch.tensor(right_img[None].copy(), dtype=torch.float32, device=self.device)
            if left_img_t.max() > 1.1:
                left_img_t = left_img_t / 255.0
            if right_img_t.max() > 1.1:
                right_img_t = right_img_t / 255.0
            # 使用encode-重参数-解码的方式
            mu, logvar, cond = self.left_model.encode(obs_t, action_t, left_img_t)
            z = torch.zeros_like(mu)
            left_pred = self.left_model.decode(z, cond)[0].cpu().numpy()
            mu, logvar, cond = self.right_model.encode(obs_t, action_t, right_img_t)
            z = torch.zeros_like(mu)
            right_pred = self.right_model.decode(z, cond)[0].cpu().numpy()
            left_pred = np.clip(left_pred*255.0, 0, 255).astype(np.uint8)
            right_pred = np.clip(right_pred*255.0, 0, 255).astype(np.uint8)
            return left_pred, right_pred

    def save(self, path_prefix):
        torch.save(self.left_model.state_dict(), path_prefix+"_left.pth")
        torch.save(self.right_model.state_dict(), path_prefix+"_right.pth")

    def load(self, path_prefix):
        self.left_model.load_state_dict(torch.load(path_prefix+"_left.pth", map_location=self.device))
        self.right_model.load_state_dict(torch.load(path_prefix+"_right.pth", map_location=self.device))
        self.left_model.to(self.device)
        self.right_model.to(self.device)
