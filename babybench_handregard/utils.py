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

def to_grayscale(img):
    # RGB转灰度
    if img.ndim == 3 and img.shape[2] == 3:
        return np.dot(img[...,:3], [0.299, 0.587, 0.114]).astype(np.float32)
    else:
        return img.astype(np.float32)

def simple_saliency(rgb_img):
    """
    简单拉普拉斯边缘能量，作为saliency奖励
    """
    gray_img = to_grayscale(rgb_img)
    laplacian_kernel = np.array([[0, -1, 0],
                                 [-1, 4, -1],
                                 [0, -1, 0]])
    edges = convolve(gray_img, laplacian_kernel, mode='reflect')
    energy = np.sqrt(np.sum(edges ** 2)) / (gray_img.shape[0] * gray_img.shape[1])
    return energy
