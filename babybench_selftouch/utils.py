# babybench_icm/utils.py

import babybench.utils as bb_utils
import torch
from babybench_selftouch.icm.icm_module import ICMModule
from babybench_selftouch.rewards import SoftmaxTouchReward
import numpy as np

def torchify(x, device='cpu'):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float().unsqueeze(0)
    elif isinstance(x, torch.Tensor) and x.ndim == 1:
        x = x.unsqueeze(0)
    return x.to(device)

def flatten_obs(obs):
    return np.concatenate([obs['observation'], obs['touch']]).astype(np.float32)