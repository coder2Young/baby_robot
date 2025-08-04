# babybench_handregard/rewards.py

import numpy as np
from utils import simple_saliency
import cv2
class HandSaliencyReward:
    """
    Hand saliency reward based on simple saliency detection.
    - Uses a simple Laplacian filter to compute saliency.
    - Encourages high saliency in the hand region.
    - Can be extended to use both eyes or a single eye.
    - Scales the reward to match the expected range.
    - Note: This class is agnostic to specific body parts like hands.
    """
    def __init__(self, mode="laplacian", use_both_eyes=True, reward_scale=1.0):
        """
        Initialize the HandSaliencyReward.
        """
        self.mode = mode
        self.use_both_eyes = use_both_eyes
        self.reward_scale = reward_scale

    def reset(self):
        pass

    def compute_reward(self, obs):
        """
        Compute the saliency reward based on the observation.
        """
        if self.mode == "laplacian":
            sal_left = simple_saliency(obs['eye_left'])
            if self.use_both_eyes:
                sal_right = simple_saliency(obs['eye_right'])
                reward = (sal_left + sal_right) / 2.0
                #reward = min(sal_left, sal_right)
            else:
                reward = sal_left
            return self.reward_scale * reward

        elif self.mode == "mask":
            if "hand_mask_left" in obs:
                mask_left = obs["hand_mask_left"]
                mask_reward_left = np.mean(mask_left)
            else:
                mask_reward_left = 0.0
            if self.use_both_eyes and "hand_mask_right" in obs:
                mask_right = obs["hand_mask_right"]
                mask_reward_right = np.mean(mask_right)
                reward = (mask_reward_left + mask_reward_right) / 2.0
            else:
                reward = mask_reward_left
            return self.reward_scale * reward

        else:
            raise NotImplementedError(f"Unknown saliency mode: {self.mode}")

class HandSkinColorReward:
    """
    Detect skin color pixels in the field of view as a reward.
    """
    def __init__(self, use_both_eyes=True, reward_scale=1.0, color_lower=None, color_upper=None):
        """
        Initialize the HandSkinColorReward.
        - use_both_eyes: If True, requires both eyes to see skin
        """
        self.use_both_eyes = use_both_eyes
        self.reward_scale = reward_scale
        self.color_lower = np.array([20, 40, 80]) if color_lower is None else color_lower
        self.color_upper = np.array([40, 255, 255]) if color_upper is None else color_upper

    def reset(self):
        pass

    def _skin_mask(self, rgb_img):

        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)
        skin_ratio = np.mean(mask > 0)
        return skin_ratio

    def compute_reward(self, obs):
        skin_left = self._skin_mask(obs['eye_left'])
        if self.use_both_eyes:
            skin_right = self._skin_mask(obs['eye_right'])
            reward = min(skin_left, skin_right)
        else:
            reward = skin_left
        return self.reward_scale * reward
