# babybench_handregard/rewards.py

import numpy as np
from utils import simple_saliency
import cv2
class HandSaliencyReward:
    """
    Hand regard用saliency奖励管理器
    - 支持多种显著性reward函数
    - 可配置单眼/双眼reward
    - 默认采用Laplacian边缘能量
    """
    def __init__(self, mode="laplacian", use_both_eyes=True, reward_scale=1.0):
        """
        mode: 可选'sobel', 'laplacian', 'mask'等（方便后续拓展）
        use_both_eyes: 是否对左右眼都计算reward
        reward_scale: 用于统一reward尺度（可根据MSE reward对齐scale）
        """
        self.mode = mode
        self.use_both_eyes = use_both_eyes
        self.reward_scale = reward_scale

    def reset(self):
        pass

    def compute_reward(self, obs):
        """
        obs: dict, 包含'eye_left', 'eye_right'
        return: scalar, reward数值
        """
        if self.mode == "laplacian":
            sal_left = simple_saliency(obs['eye_left'])
            if self.use_both_eyes:
                sal_right = simple_saliency(obs['eye_right'])
                # 平均左右眼的saliency
                reward = (sal_left + sal_right) / 2.0
                # 直接使用左右眼的最小值作为reward，鼓励两只眼睛都高
                #reward = min(sal_left, sal_right)
            else:
                reward = sal_left
            return self.reward_scale * reward

        # 拓展：mask-based reward（占位范例）
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
    检测视野中黄色（肤色）像素比例作为奖励
    """
    def __init__(self, use_both_eyes=True, reward_scale=1.0, color_lower=None, color_upper=None):
        """
        color_lower, color_upper: np.array, HSV空间下的lower/upper阈值，默认适合浅黄色
        """
        self.use_both_eyes = use_both_eyes
        self.reward_scale = reward_scale
        # 适合仿真浅黄色皮肤，你可根据你的仿真/真实肤色调整
        self.color_lower = np.array([20, 40, 80]) if color_lower is None else color_lower
        self.color_upper = np.array([40, 255, 255]) if color_upper is None else color_upper

    def reset(self):
        pass

    def _skin_mask(self, rgb_img):

        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, self.color_lower, self.color_upper)
        # 比例归一化
        skin_ratio = np.mean(mask > 0)
        return skin_ratio

    def compute_reward(self, obs):
        skin_left = self._skin_mask(obs['eye_left'])
        if self.use_both_eyes:
            skin_right = self._skin_mask(obs['eye_right'])
            reward = min(skin_left, skin_right)  # 只有两只眼都看到reward才高
        else:
            reward = skin_left
        return self.reward_scale * reward
