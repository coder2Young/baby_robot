# babybench_selftouch/icm/losses.py

import torch
import torch.nn.functional as F

def sigma_vae_loss(prediction, target, log_offset=10.0):
    """
    Calculates a more stable version of the σ-VAE reconstruction loss.
    
    The loss is based on log(MSE), but a constant offset is added.
    This prevents the loss from becoming a large negative value when MSE is small,
    which is a major source of numerical instability in complex training loops.
    """
    mse = F.mse_loss(prediction, target, reduction='mean')
    
    # Add a small epsilon for stability when MSE is near zero.
    # Add a larger offset to keep the final loss value in a stable, non-negative range.
    loss = torch.log(mse + 1e-8) + log_offset
    
    return loss