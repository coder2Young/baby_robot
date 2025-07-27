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
    """获取一个根节点下的所有子身体的ID集合。"""
    subtree = {root_id}
    for body_id in range(model.nbody):
        current_id = body_id
        while current_id != -1:
            # body_parentid 数组记录了每个body的父body的id
            parent_id = model.body_parentid[current_id]
            if parent_id in subtree:
                subtree.add(body_id)
                break
            # 如果父节点不是root，则继续向上查找父节点的父节点
            current_id = parent_id
            if current_id == 0: # 到达worldbody，停止
                break
    return subtree