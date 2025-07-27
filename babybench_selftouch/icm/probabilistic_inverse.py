# babybench_selftouch/icm/probabilistic_inverse.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ProbabilisticInverseModel(nn.Module):
    """
    A probabilistic inverse model that outputs a distribution over actions.
    It models p(a | z_t, z_{t+1}) as a Gaussian distribution.
    """
    def __init__(self, latent_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Two heads for the mean and log_std of the action distribution
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_log_sigma = nn.Linear(hidden_dim, action_dim)

    def forward(self, z_t, z_tp1):
        # Input is only the state transition (the condition)
        x = torch.cat([z_t, z_tp1], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        
        mu = self.fc_mu(h)
        # Clamp the log_sigma to prevent extreme values (very large or small variance)
        log_sigma = torch.clamp(self.fc_log_sigma(h), min=-10, max=2)
        
        return mu, log_sigma

    def compute_loss(self, mu, log_sigma, a_true):
        """
        Calculates the Negative Log-Likelihood (NLL) of the true actions
        under the predicted Gaussian distribution.
        """
        # Ensure sigma is positive
        sigma = torch.exp(log_sigma)
        
        # Create a Normal distribution object
        dist = torch.distributions.Normal(mu, sigma)
        
        # Calculate the log probability of the true actions
        log_prob = dist.log_prob(a_true)
        
        # The NLL loss is the negative of the sum of log probabilities, averaged over the batch
        # We sum over the action dimension and then take the mean over the batch dimension
        nll_loss = -log_prob.sum(dim=-1).mean()
        
        return nll_loss

    def sample(self, z_t, z_tp1):
        """
        For inference: sample an action from the predicted distribution.
        """
        mu, log_sigma = self.forward(z_t, z_tp1)
        sigma = torch.exp(log_sigma)
        
        # Sample from the distribution
        return mu + sigma * torch.randn_like(mu)