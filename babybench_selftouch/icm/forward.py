# babybench_selftouch/icm/forward.py

import torch
import torch.nn as nn
import torch.nn.functional as F
# --- NEW: Import the custom loss function ---
from .losses import sigma_vae_loss

# forward.py 的残差连接版本
class ForwardModel(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=512): # 可以适当加深
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # 新增一层
        self.fc3 = nn.Linear(hidden_dim, latent_dim) # 输出层

    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        delta_z_pred = self.fc3(h) # 网络现在输出的是预测的变化量

        # 【关键】将预测的变化量与当前状态相加，得到下一状态的预测
        return z + delta_z_pred

    # compute_loss 函数无需任何改动
    def compute_loss(self, z_pred, z_next):
        return sigma_vae_loss(z_pred, z_next)