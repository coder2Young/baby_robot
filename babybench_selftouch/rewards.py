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
        self.num_parts = num_parts
        self.total_reward = total_reward
        self.touch_counts = np.zeros(num_parts, dtype=np.int32)

        # Tau can be a single value for all parts or a vector for part-specific rates.
        if np.isscalar(tau):
            self.tau = np.full(self.num_parts, tau, dtype=np.float32)
        else:
            self.tau = np.asarray(tau, dtype=np.float32)
            assert self.tau.shape[0] == self.num_parts, "Tau vector must have the same size as num_parts"

    def reset(self):
        """Resets the touch counts for all parts."""
        self.touch_counts = np.zeros(self.num_parts, dtype=np.int32)

    def update(self, part_idx):
        """Increments the touch count for the given part(s)."""
        # This works for a single index or a list/array of indices
        self.touch_counts[part_idx] += 1

    def compute_rewards(self):
        """
        Computes the potential reward for all parts based on touch history.
        This no longer includes any special hand bonus logic.
        """
        exp_neg_counts = np.exp(-self.touch_counts / self.tau)
        
        # Add epsilon for numerical stability to prevent division by zero
        denominator = np.sum(exp_neg_counts)
        probs = exp_neg_counts / (denominator + 1e-8) 
        
        rewards = self.total_reward * (probs * self.num_parts)
        
        return rewards

    def get_touch_distribution(self):
        """Returns a copy of the current touch counts."""
        return self.touch_counts.copy()