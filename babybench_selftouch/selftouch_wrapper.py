# babybench_selftouch/selftouch_wrapper.py

import gymnasium as gym
import numpy as np
import torch

# =============================================================================
#  Design Philosophy Change Note:
#
#  This file has been refactored. The original `ICMRewardWrapper` was responsible
#  for calculating BOTH touch rewards and ICM curiosity rewards.
#
#  To create a cleaner, more modular design, this logic has been split:
#
#  1.  This Wrapper (`TouchRewardWrapper`): Is now ONLY responsible for logic
#      directly related to the environment's state, i.e., calculating rewards
#      based on the `obs['touch']` values.
#
#  2.  A new Callback (`icm_callback.py`): Is now responsible for ALL logic
#      related to the ICM module. It calculates the ICM reward at each step
#      and orchestrates the batch training of the ICM model at the end of
#      each rollout.
#
#  This separation of concerns makes the system more robust and easier to modify.
# =============================================================================


class TouchRewardWrapper(gym.Wrapper):
    """
    一个只负责计算和添加触摸相关奖励的装饰器。
    ICM奖励的逻辑被移至一个专门的Callback中。
    """
    def __init__(self, env, reward_module, lambda_touch=100.0, lambda_hand_touch=0.5, touch_threshold=1e-6):
        super().__init__(env)
        self.reward_mod = reward_module
        self.lambda_touch = lambda_touch
        self.lambda_hand_touch = lambda_hand_touch
        self.touch_threshold = touch_threshold

    def reset(self, **kwargs):
        """Resets the environment and the touch reward counter."""
        self.reward_mod.reset()
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Performs a step in the environment and calculates ONLY the touch-based rewards.
        """
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        
        # --- 触摸奖励计算 (保留这部分) ---
        touched_array = obs['touch']
        touched_parts = np.where(touched_array > self.touch_threshold)[0]
        
        # 更新触摸计数器
        self.reward_mod.update(touched_parts)
        
        # 基于更新后的计数器，计算当前步的奖励
        touch_rewards_per_part = self.reward_mod.compute_rewards()
        touch_reward = np.sum(touch_rewards_per_part[touched_parts]) if len(touched_parts) > 0 else 0.0
        
        # --- 手部触摸奖励计算 (保留这部分) ---
        hand_touch_reward = 0.0
        # 手部触摸给额外奖励, 左手的part位置为19-21,右手为13-15
        # Note: A more robust way might be to get hand part indices from the env config
        hand_parts = {13, 14, 15, 19, 20, 21}
        if any(part in touched_parts for part in hand_parts):
            hand_touch_reward = 1.0

        # --- 计算总触摸奖励 ---
        # 这个wrapper现在只负责返回所有与触摸相关的奖励。
        # extrinsic_reward (通常为0) + 我们计算的触摸奖励。
        total_touch_reward = extrinsic_reward + self.lambda_touch * touch_reward + self.lambda_hand_touch * hand_touch_reward
        
        # --- 关于ICM奖励的重要说明 ---
        # 请注意，此函数返回的 'total_touch_reward' *不包含* ICM好奇心奖励。
        # 好奇心奖励将在我们的自定义 `ICMCallback` 的 `_on_step` 方法中被计算出来，
        # 并直接添加到SB3的奖励缓冲区中。
        # 这样做，PPO代理在进行策略更新时，看到的是一个已经包含了所有分量的最终奖励。
        
        return obs, total_touch_reward, terminated, truncated, info

# --- 辅助函数 ---
# 这些函数现在将被Callback使用，所以我们把它们留在这里作为一个公共的工具模块。

def torchify(x, device='cpu'):
    """Converts numpy arrays or lists to torch tensors."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float().unsqueeze(0)
    elif isinstance(x, (list, tuple)):
        x = torch.tensor(x).float().unsqueeze(0)
    elif isinstance(x, torch.Tensor) and x.ndim == 1:
        x = x.unsqueeze(0)
    return x.to(device)

def flatten_obs(obs):
    """
    Flattens the observation dictionary from the environment into a single vector.
    """
    # 拼接observation + touch
    return np.concatenate([obs['observation'], obs['touch']]).astype(np.float32)