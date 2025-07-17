# babybench_selftouch/selftouch_wrapper.py

import gymnasium as gym
import numpy as np
import torch

class TouchRewardWrapper(gym.Wrapper):
    """
    A wrapper that calculates touch-based rewards using a sophisticated
    "Reward Window with Cooldown" mechanism for general touches, and a
    separate "Event-Based Bonus with Cooldown" for special hand touches.
    """
    def __init__(self, env, reward_module, reward_window_duration=25, cooldown_period=50, hand_bonus_cooldown_period=20, lambda_touch=100.0, lambda_hand_touch=0.5, touch_threshold=1e-6):
        super().__init__(env)
        self.reward_mod = reward_module
        self.lambda_touch = lambda_touch
        self.lambda_hand_touch = lambda_hand_touch
        self.touch_threshold = touch_threshold
        
        # Parameters for the general touch reward system
        self.reward_window_duration = reward_window_duration
        self.cooldown_period = cooldown_period
        
        # --- NEW: Parameters and trackers for the special hand bonus ---
        self.hand_bonus_cooldown_period = hand_bonus_cooldown_period
        self.hand_bonus_cooldown_timer = 0
        self.is_hand_touching_last_step = False
        self.hand_parts = {13, 14, 15, 19, 20, 21}
        
        # State trackers for the general touch reward system
        num_parts = self.env.observation_space['touch'].shape[0]
        self.current_touch_durations = np.zeros(num_parts, dtype=np.int32)
        self.touch_cooldown_timers = np.zeros(num_parts, dtype=np.int32)
        self.parts_in_contact_last_step = set()

    def reset(self, **kwargs):
        """Resets the environment and all stateful trackers."""
        self.reward_mod.reset()
        self.current_touch_durations.fill(0)
        self.touch_cooldown_timers.fill(0)
        self.parts_in_contact_last_step = set()
        # --- NEW: Reset hand bonus trackers ---
        self.hand_bonus_cooldown_timer = 0
        self.is_hand_touching_last_step = False
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Performs a step and calculates rewards using the combined logic.
        """
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        
        # --- General Touch Reward Logic (remains the same) ---
        # ... (Code for Reward Window and Cooldown for all parts) ...
        self.touch_cooldown_timers = np.maximum(0, self.touch_cooldown_timers - 1)
        touched_array = obs['touch']
        parts_in_contact_now = set(np.where(touched_array > self.touch_threshold)[0])
        is_in_contact_mask = np.zeros(self.current_touch_durations.shape[0], dtype=bool)
        if parts_in_contact_now:
            is_in_contact_mask[list(parts_in_contact_now)] = True
        self.current_touch_durations[is_in_contact_mask] += 1
        self.current_touch_durations[~is_in_contact_mask] = 0
        released_parts = self.parts_in_contact_last_step - parts_in_contact_now
        for part_idx in released_parts:
            self.touch_cooldown_timers[part_idx] = self.cooldown_period
        eligible_for_reward = [p for p in parts_in_contact_now if self.touch_cooldown_timers[p] == 0 and 0 < self.current_touch_durations[p] <= self.reward_window_duration]
        touch_reward = 0.0
        if eligible_for_reward:
            potential_rewards = self.reward_mod.compute_rewards()
            touch_reward = np.sum(potential_rewards[eligible_for_reward])
        if parts_in_contact_now:
            self.reward_mod.update(list(parts_in_contact_now))
        self.parts_in_contact_last_step = parts_in_contact_now

        # --- NEW: Event-Based Hand Bonus Logic ---
        self.hand_bonus_cooldown_timer = max(0, self.hand_bonus_cooldown_timer - 1)
        hand_touch_reward = 0.0
        is_hand_touching_now = any(part in parts_in_contact_now for part in self.hand_parts)
        
        # Check for a NEW hand touch event that is NOT on cooldown
        if is_hand_touching_now and not self.is_hand_touching_last_step and self.hand_bonus_cooldown_timer == 0:
            hand_touch_reward = 1.0  # Give the one-time bonus
            self.hand_bonus_cooldown_timer = self.hand_bonus_cooldown_period # Start the cooldown

        # Update the state for the next step
        self.is_hand_touching_last_step = is_hand_touching_now

        # --- Final reward calculation ---
        total_reward = extrinsic_reward + self.lambda_touch * touch_reward + self.lambda_hand_touch * hand_touch_reward
        
        return obs, total_reward, terminated, truncated, info

# --- Helper functions remain unchanged ---

def torchify(x, device='cpu'):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float().unsqueeze(0)
    elif isinstance(x, (list, tuple)):
        x = torch.tensor(x).float().unsqueeze(0)
    elif isinstance(x, torch.Tensor) and x.ndim == 1:
        x = x.unsqueeze(0)
    return x.to(device)

def flatten_obs(obs):
    return np.concatenate([obs['observation'], obs['touch']]).astype(np.float32)