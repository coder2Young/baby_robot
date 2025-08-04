# babybench_handregard/handregard_wrapper.py

import gymnasium as gym
import numpy as np

from rewards import HandSaliencyReward, HandSkinColorReward

class HandRegardRewardWrapper(gym.Wrapper):
    """
    Wrapper that calculates hand regard rewards based on saliency and
    """
    def __init__(self, env, multimodal_ae, hand_saliency_mod, hand_skin_mod,
                 lambda_v=1.0, lambda_p=1.0, lambda_m=1.0,
                 lambda_sal=10.0, lambda_skin=1.0,
                 motion_bonus_scale=1.0, decay_steps=10000):
        super().__init__(env)
        self.ae = multimodal_ae  # MultimodalAEManager
        self.hand_saliency_mod = hand_saliency_mod
        self.hand_skin_mod = hand_skin_mod
        self.lambda_v = lambda_v
        self.lambda_p = lambda_p
        self.lambda_m = lambda_m
        self.lambda_sal = lambda_sal
        self.lambda_skin = lambda_skin
        self.motion_bonus_scale = motion_bonus_scale
        self.decay_steps = decay_steps
        self.last_obs = None
        self.last_action = None
        self.L_v_last = None
        self.L_p_last = None
        self.L_m_last = None
        self.timestep = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_obs = None
        self.last_action = None
        self.L_v_last = None
        self.L_p_last = None
        self.L_m_last = None
        self.timestep = 0
        self.hand_saliency_mod.reset()
        self.hand_skin_mod.reset()
        return obs

    def step(self, action):
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        # Multimodal AE encoding and reconstruction
        ae_out = self.ae.encode_and_recon(obs['eye_left'], obs['observation'])
        L_v = ae_out['L_v']
        L_p = ae_out['L_p']
        L_m = ae_out['L_m']

        eps = 1e-8
        if self.L_v_last is not None:
            reward_v = (self.L_v_last - L_v) / (self.L_v_last + eps)
            reward_p = (self.L_p_last - L_p) / (self.L_p_last + eps)
            reward_m = (self.L_m_last - L_m) / (self.L_m_last + eps)
        else:
            reward_v = reward_p = reward_m = 0.0

        saliency_reward = self.hand_saliency_mod.compute_reward(obs)
        skin_reward = self.hand_skin_mod.compute_reward(obs)

        if self.last_action is not None:
            action_diff = np.linalg.norm(np.array(action) - np.array(self.last_action))
            decay_factor = max(0.0, 1.0 - self.timestep / self.decay_steps)
            motion_bonus = self.motion_bonus_scale * action_diff * decay_factor
        else:
            motion_bonus = 0.0

        total_reward = (self.lambda_v * reward_v +
                        self.lambda_p * reward_p +
                        self.lambda_m * reward_m +
                        self.lambda_sal * saliency_reward +
                        self.lambda_skin * skin_reward +
                        motion_bonus)

        self.L_v_last = L_v
        self.L_p_last = L_p
        self.L_m_last = L_m
        self.last_obs = obs
        self.last_action = action
        self.timestep += 1

        return obs, total_reward, terminated, truncated, info
