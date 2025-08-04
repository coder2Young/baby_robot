# babybench_selftouch/icm/losses.py

import torch
import torch.nn.functional as F

def sigma_vae_loss(prediction, target, log_offset=10.0, min_mse=1e-6):
    """
    Calculates a more stable version of the sigma-VAE reconstruction loss.
    """
    mse = F.mse_loss(prediction, target, reduction='mean')

    # Clamp the MSE to a minimum value to avoid log(0) or log(negative)
    safe_mse = torch.clamp(mse, min=min_mse)

    # Use the clamped safe_mse for the calculation
    loss = torch.log(safe_mse) + log_offset

    return loss