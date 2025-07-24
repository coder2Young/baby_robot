# babybench_selftouch/icm/losses.py

import torch
import torch.nn.functional as F

def sigma_vae_loss(prediction, target, log_offset=10.0, min_mse=1e-6):
    """
    Calculates a more stable version of the σ-VAE reconstruction loss.
    """
    mse = F.mse_loss(prediction, target, reduction='mean')

    # --- 【关键修正】 ---
    # 在取对数之前，将MSE的值限制在一个安全的最小范围内。
    # 这是防止梯度爆炸的核心步骤 (d(log(x))/dx = 1/x)。
    safe_mse = torch.clamp(mse, min=min_mse)

    # 使用限制后的safe_mse进行计算
    loss = torch.log(safe_mse) + log_offset # 移除了不再需要的epsilon

    return loss