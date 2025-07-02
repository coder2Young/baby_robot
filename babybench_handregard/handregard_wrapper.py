# babybench_handregard/handregard_wrapper.py

import gymnasium as gym
import numpy as np

from forward_model.forward_model import ForwardModel
from rewards import HandSaliencyReward
from utils import torchify, flatten_obs

class HandRegardRewardWrapper(gym.Wrapper):
    """
    Hand regard专用奖励包装器：
    - 内在奖励1：视觉正模型负MSE（分别对左右眼，平均后作为reward）
    - 内在奖励2：手部saliency奖励（鼓励agent将手保持在视野内）
    """
    def __init__(self, env, forward_model, hand_reward_mod, lambda_pred=-1.0, lambda_sal=10.0):
        """
        lambda_pred: 负MSE的系数（默认-1.0）
        lambda_sal:  saliency奖励的系数（建议MSE的100倍）
        """
        super().__init__(env)
        self.fwd_model = forward_model
        self.hand_reward_mod = hand_reward_mod
        self.lambda_pred = lambda_pred
        self.lambda_sal = lambda_sal
        self.last_obs = None
        self.last_action = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_obs = None
        self.last_action = None
        self.hand_reward_mod.reset()
        return obs

    def step(self, action):
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        # 视觉正模型reward（负的MSE损失，分别对左右眼）
        if self.last_obs is not None and self.last_action is not None:
            pred_left, pred_right = self.fwd_model.predict(
                flatten_obs(self.last_obs),
                np.array(self.last_action, dtype=np.float32),
                self.last_obs["eye_left"],
                self.last_obs["eye_right"]
            )
            gt_left = obs["eye_left"]
            gt_right = obs["eye_right"]

            mse_left = np.mean((pred_left.astype(np.float32) - gt_left.astype(np.float32)) ** 2)
            mse_right = np.mean((pred_right.astype(np.float32) - gt_right.astype(np.float32)) ** 2)
            pred_reward = 0.5 * (mse_left + mse_right)  # 平均左右眼的MSE
            #pred_reward = -pred_reward  # 负MSE作为奖励，鼓励模型预测准确；正的则会鼓励模型探索
        else:
            pred_reward = 0.0

        # Saliency手部奖励
        saliency_reward = self.hand_reward_mod.compute_reward(obs)

        # # 打印奖励
        # print(f"Predicted Reward: {pred_reward:.4f}, Saliency Reward: {saliency_reward:.4f}")
        # # 打印赋权后的奖励
        # print(f"Weighted Predicted Reward: {self.lambda_pred * pred_reward:.4f}, Weighted Saliency Reward: {self.lambda_sal * saliency_reward:.4f}")
        
        # 总奖励
        total_reward = self.lambda_pred * pred_reward + self.lambda_sal * saliency_reward

        self.last_obs = obs
        self.last_action = action
        return obs, total_reward, terminated, truncated, info
