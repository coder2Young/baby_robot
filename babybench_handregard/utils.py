# babybench_handregard/utils.py

import numpy as np
from scipy.ndimage import convolve

def torchify(x):
    import torch
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float().unsqueeze(0)
    elif isinstance(x, (list, tuple)):
        x = torch.tensor(x).float().unsqueeze(0)
    elif hasattr(x, 'ndim') and x.ndim == 1:
        x = x.unsqueeze(0)
    return x

def flatten_obs(obs):
    # 可以按需拼接观测（如proprio+observation）
    if 'proprio' in obs and 'observation' in obs:
        return np.concatenate([obs['proprio'], obs['observation']]).astype(np.float32)
    elif 'proprio' in obs:
        return obs['proprio'].astype(np.float32)
    else:
        return obs['observation'].astype(np.float32)

def to_grayscale(x):
   return 0.2989*x[:,:,0] + 0.5870*x[:,:,1] + 0.1140*x[:,:,2]

def simple_saliency(rgb_img):
    gray_img = to_grayscale(rgb_img)
    # Define a simple Laplacian kernel
    laplacian_kernel = np.array([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]])
    # Apply the kernel using convolution
    edges = convolve(gray_img, laplacian_kernel, mode='reflect')
    # Compute energy as sum of squared edge intensities (normalized)
    energy = np.sqrt(np.sum(edges**2)) / (gray_img.shape[0] * gray_img.shape[1])
    return energy
