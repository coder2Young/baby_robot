# babybench_selftouch/selftouch_wrapper.py

import gymnasium as gym
import numpy as np
import torch

class TouchRewardWrapper(gym.Wrapper):
    """
    A wrapper that calculates touch-based rewards.
    It now includes a "cooldown" or "refractory period" mechanism to encourage
    discrete touch events rather than continuous contact.
    """
    def __init__(self, env, reward_module, cooldown_period=50, lambda_touch=100.0, lambda_hand_touch=0.5, touch_threshold=1e-6):
        super().__init__(env)
        self.reward_mod = reward_module
        self.lambda_touch = lambda_touch
        self.lambda_hand_touch = lambda_hand_touch
        self.touch_threshold = touch_threshold
        
        # --- NEW: Initialize cooldown attributes ---
        self.cooldown_period = cooldown_period
        num_parts = self.env.observation_space['touch'].shape[0]
        self.touch_cooldown_timers = np.zeros(num_parts, dtype=np.int32)

    def reset(self, **kwargs):
        """Resets the environment, touch reward counter, and cooldown timers."""
        self.reward_mod.reset()
        # --- NEW: Reset cooldown timers at the start of an episode ---
        self.touch_cooldown_timers.fill(0)
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Performs a step and calculates touch rewards based on the "touch onset"
        with a cooldown period.
        """
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)
        
        # --- NEW COOLDOWN-BASED REWARD LOGIC ---

        # 1. Decrement all active cooldown timers by 1.
        self.touch_cooldown_timers = np.maximum(0, self.touch_cooldown_timers - 1)

        # 2. Identify all parts currently in physical contact.
        touched_array = obs['touch']
        parts_in_contact_now = np.where(touched_array > self.touch_threshold)[0]

        # 3. Determine which of these parts are "eligible" for a reward,
        #    i.e., they are both being touched AND their cooldown timer is zero.
        eligible_for_reward = [p for p in parts_in_contact_now if self.touch_cooldown_timers[p] == 0]

        # 4. Calculate reward ONLY for these newly eligible touches.
        touch_reward = 0.0
        if eligible_for_reward:
            # Get the potential rewards for all parts from our diversity module.
            potential_rewards = self.reward_mod.compute_rewards()
            # Sum the rewards only for the parts that are eligible.
            touch_reward = np.sum(potential_rewards[eligible_for_reward])
            
            # 5. For each part that just gave a reward, reset its cooldown timer.
            #    This prevents it from giving rewards for the next `cooldown_period` steps.
            for part_idx in eligible_for_reward:
                self.touch_cooldown_timers[part_idx] = self.cooldown_period

        # IMPORTANT: We update the diversity counter with ALL currently touched parts,
        # not just the rewarded ones. This ensures the system knows which parts are
        # touched frequently, even if they are on cooldown.
        self.reward_mod.update(parts_in_contact_now)
        
        # --- Hand touch reward logic remains the same ---
        hand_touch_reward = 0.0
        hand_parts = {13, 14, 15, 19, 20, 21}
        if any(part in parts_in_contact_now for part in hand_parts):
            hand_touch_reward = 1.0

        # --- Final reward calculation ---
        total_touch_reward = extrinsic_reward + self.lambda_touch * touch_reward + self.lambda_hand_touch * hand_touch_reward
        
        return obs, total_touch_reward, terminated, truncated, info

# --- Helper functions remain unchanged ---

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
    # This might need to be updated if you want to include vestibular data
    return np.concatenate([obs['observation'], obs['touch']]).astype(np.float32)