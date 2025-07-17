# babybench_selftouch/icm/losses.py

import torch
import torch.nn.functional as F

def sigma_vae_loss(prediction, target):
    """
    Calculates the reconstruction loss based on the σ-VAE objective.
    This involves analytically finding the optimal decoder variance, which
    results in a loss proportional to the log of the Mean Squared Error.
    Ref: "Simple and effective VAE training with calibrated decoders" (Rybkin et al., 2021)
    
    :param prediction: The output of the decoder network.
    :param target: The ground truth data.
    :return: The scalar loss value.
    """
    # Calculate MSE, which is the optimal variance sigma*^2
    mse = F.mse_loss(prediction, target, reduction='mean')
    
    # The loss to be minimized is proportional to the log of this variance.
    # Add a small epsilon for numerical stability if MSE is close to zero.
    loss = torch.log(mse + 1e-8)
    
    return loss