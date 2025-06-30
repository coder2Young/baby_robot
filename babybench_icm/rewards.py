# babybench_icm/rewards.py

import numpy as np

class SoftmaxTouchReward:
    """
    动态softmax奖励分配管理器
    - 跟踪每个部位触摸次数，分配归一化奖励
    - 鼓励优先触摸冷门部位
    """
    def __init__(self, num_parts, tau=2.0, total_reward=2.0, part_names=None):
        self.num_parts = num_parts
        self.tau = tau
        self.total_reward = total_reward
        self.touch_counts = np.zeros(num_parts, dtype=np.int32)
        self.part_names = part_names if part_names is not None else [f"part_{i}" for i in range(num_parts)]

    def reset(self):
        self.touch_counts = np.zeros(self.num_parts, dtype=np.int32)

    def update(self, part_idx):
        if isinstance(part_idx, (list, tuple, np.ndarray)):
            for idx in part_idx:
                self.touch_counts[idx] += 1
        else:
            self.touch_counts[part_idx] += 1

    def compute_rewards(self):
        exp_neg_counts = np.exp(-self.touch_counts / self.tau)
        probs = exp_neg_counts / np.sum(exp_neg_counts)
        rewards = self.total_reward * probs
        return rewards

    def get_touch_distribution(self):
        return self.touch_counts.copy()
