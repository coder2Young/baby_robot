# babybench_selftouch/rewards.py

import numpy as np

class SoftmaxTouchReward:
    """
    Manages dynamic softmax reward allocation for a given set of body parts.
    - Tracks touch counts for each part to assign normalized rewards.
    - Encourages prioritizing less-touched parts.
    - Supports part-specific decay rates (tau).
    - Note: This class is now agnostic to specific body parts like hands.
      The logic for hand-specific bonuses has been moved to the wrapper.
    """
    def __init__(self, num_parts, tau=10.0, total_reward=2.0):
        # --- MODIFIED: Store initial tau for reconfiguration ---
        self._init_tau = tau 
        self.total_reward = total_reward
        self.reconfigure(num_parts) # Use reconfigure to do the main setup

    def reset(self):
        """Resets the touch counts for all parts."""
        self.touch_counts.fill(0)

    def update(self, part_idx):
        """Increments the touch count for the given part(s)."""
        self.touch_counts[part_idx] += 1

    def compute_rewards(self):
        """
        Computes the potential reward for all parts based on touch history.
        """
        exp_neg_counts = np.exp(-self.touch_counts / self.tau)
        denominator = np.sum(exp_neg_counts)
        probs = exp_neg_counts / (denominator + 1e-8) 
        rewards = self.total_reward * (probs * self.num_parts)
        return rewards

    def get_touch_distribution(self):
        """Returns a copy of the current touch counts."""
        return self.touch_counts.copy()
        
    # --- 【新增方法】 ---
    def reconfigure(self, num_parts):
        """
        Re-initializes the module with a new number of parts.
        This allows the wrapper to configure it correctly at runtime.
        """
        self.num_parts = num_parts
        self.touch_counts = np.zeros(self.num_parts, dtype=np.int32)

        # Re-initialize tau array with the correct new size
        if np.isscalar(self._init_tau):
            self.tau = np.full(self.num_parts, self._init_tau, dtype=np.float32)
        else:
            self.tau = np.asarray(self._init_tau, dtype=np.float32)
            assert self.tau.shape[0] == self.num_parts, "Tau vector must have the same size as num_parts"
    # -------------------