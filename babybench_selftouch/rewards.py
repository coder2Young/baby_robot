# babybench_selftouch/rewards.py

import numpy as np

class SoftmaxTouchReward:
    """
    Manages dynamic softmax reward allocation.
    - Tracks touch counts for each part to assign normalized rewards.
    - Encourages prioritizing less-touched parts.
    - NOW supports part-specific decay rates (tau).
    """
    def __init__(self, num_parts, tau=10.0, total_reward=2.0, part_names=None):
        self.num_parts = num_parts
        self.total_reward = total_reward
        self.touch_counts = np.zeros(num_parts, dtype=np.int32)
        self.part_names = part_names if part_names is not None else [f"part_{i}" for i in range(num_parts)]

        # --- MODIFIED: Allow tau to be a vector for part-specific decay ---
        # If tau is a single number, create a vector with that value for all parts.
        # If tau is already a vector/list, use it directly.
        if np.isscalar(tau):
            self.tau = np.full(self.num_parts, tau, dtype=np.float32)
        else:
            self.tau = np.asarray(tau, dtype=np.float32)
            assert self.tau.shape[0] == self.num_parts, "Tau vector must have the same size as num_parts"

    def reset(self):
        self.touch_counts = np.zeros(self.num_parts, dtype=np.int32)

    def update(self, part_idx):
        # This works for single index or a list/array of indices
        self.touch_counts[part_idx] += 1

    def compute_rewards(self):
        # The calculation remains the same. NumPy will automatically perform
        # element-wise division if self.tau is a vector.
        exp_neg_counts = np.exp(-self.touch_counts / self.tau)
        probs = exp_neg_counts / np.sum(exp_neg_counts)
        rewards = self.total_reward * probs
        
        return rewards

    def get_touch_distribution(self):
        return self.touch_counts.copy()