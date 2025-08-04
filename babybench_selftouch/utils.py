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

def get_body_subtree(model, root_id):
    """Get the subtree of bodies rooted at root_id."""
    subtree = {root_id}
    for body_id in range(model.nbody):
        current_id = body_id
        while current_id != -1:
            # Body's parent ID
            parent_id = model.body_parentid[current_id]
            if parent_id in subtree:
                subtree.add(body_id)
                break
            # If the parent is not the root, continue to search up the hierarchy
            current_id = parent_id
            if current_id == 0:  # Reached worldbody, stop
                break
    return subtree