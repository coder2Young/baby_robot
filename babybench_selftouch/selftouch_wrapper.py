# babybench_icm/icm_wrapper.py

import gymnasium as gym
import numpy as np
import torch

class ICMRewardWrapper(gym.Wrapper):
    """
    自定义奖励包装器，适配BabyBench obs['touch'] API。
    """
    def __init__(self, env, icm_module, reward_module, lambda_icm=0.5, lambda_touch=0.5, lambda_hand_touch=0.5,touch_threshold=1e-6):
        super().__init__(env)
        self.icm = icm_module
        self.reward_mod = reward_module
        self.lambda_icm = lambda_icm
        self.lambda_touch = lambda_touch
        self.lambda_hand_touch = lambda_hand_touch
        self.touch_threshold = touch_threshold
        self.last_obs = None
        self.last_action = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_obs = None
        self.last_action = None
        self.reward_mod.reset()
        return obs

    def step(self, action):
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        if self.last_obs is not None and self.last_action is not None:
            flat_obs = flatten_obs(self.last_obs)
            flat_next_obs = flatten_obs(obs)
            flat_action = np.array(self.last_action, dtype=np.float32)
            norm_fwd_loss, _ = self.icm.compute_forward_loss(
                torchify(flat_obs), torchify(flat_action), torchify(flat_next_obs)
            )
            norm_inv_loss, _ = self.icm.compute_inverse_loss(
                torchify(flat_obs), torchify(flat_next_obs), torchify(flat_action)
            )
            icm_reward = (norm_fwd_loss + norm_inv_loss) / 2.0
        else:
            icm_reward = 0.0
        # 触摸奖励
        touched_array = obs['touch']
        touched_parts = np.where(touched_array > self.touch_threshold)[0]
        self.reward_mod.update(touched_parts)
        touch_rewards = self.reward_mod.compute_rewards()
        touch_reward = np.sum(touch_rewards[touched_parts]) if len(touched_parts) > 0 else 0.0
        
        hand_touch_reward = 0.0
        # 手部触摸给额外奖励, 左手的part位置为19-21,右手为13-15
        if 19 in touched_parts or 20 in touched_parts or 21 in touched_parts:
            hand_touch_reward += 1.0
        if 13 in touched_parts or 14 in touched_parts or 15 in touched_parts:
            hand_touch_reward += 1.0

        total_reward = self.lambda_icm * icm_reward + self.lambda_touch * touch_reward + self.lambda_hand_touch * hand_touch_reward
        
        # 打印 用lambda加权后 的reward分量
        print(f"ICM Reward: {self.lambda_icm * icm_reward:.4f}, "
              f"Touch Reward: {self.lambda_touch * touch_reward:.4f}, "
                f"Hand Touch Reward: {self.lambda_hand_touch * hand_touch_reward:.4f}, "
              f"Total Reward: {total_reward:.4f}")
        

        
        self.last_obs = obs
        self.last_action = action
        return obs, total_reward, terminated, truncated, info

def torchify(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float().unsqueeze(0)
    elif isinstance(x, (list, tuple)):
        x = torch.tensor(x).float().unsqueeze(0)
    elif isinstance(x, torch.Tensor) and x.ndim == 1:
        x = x.unsqueeze(0)
    return x

def flatten_obs(obs):
    # 拼接observation + touch（如有需要也可加vestibular）
    return np.concatenate([obs['observation'], obs['touch']]).astype(np.float32)
