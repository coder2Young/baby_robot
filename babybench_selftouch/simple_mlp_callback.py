# babybench_selftouch/simple_mlp_callback.py

import os
import torch
import numpy as np
import mujoco
from collections import defaultdict
from stable_baselines3.common.callbacks import BaseCallback
from .utils import torchify, get_body_subtree
from icm.direct_prediction_mlp import DirectPredictionMLP
import sys
sys.path.append(".")
sys.path.append("..")

class SimpleMLPCallback(BaseCallback):
    """
    A callback for training a simple MLP model to predict proprioception and touch observations.
    This callback is similar to ICMCallback but uses a DirectPredictionMLP instead of an ICMModule.
    It is designed to be used in a reinforcement learning training loop with Stable Baselines3.
    """
    def __init__(self, 
                 proprio_dim: int,
                 touch_dim: int,
                 action_dim: int,
                 total_training_steps: int,
                 save_path: str,
                 save_freq: int = 100000,
                 lr: float = 1e-4,
                 lambda_icm_schedule: tuple = (5.0, 50.0),
                 lambda_touch_schedule: tuple = (10.0, 1.0),
                 lambda_hand_touch_schedule: tuple = (20.0, 2.0),
                 dynamic_weight_stop_step: int = 1000000,
                 n_epochs: int = 8, 
                 batch_size: int = 512,
                 device: str = 'cpu',
                 verbose: int = 0):
        super().__init__(verbose)

        self.model_mlp = DirectPredictionMLP(proprio_dim, touch_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model_mlp.parameters(), lr=lr)
        self.device = device

        self.total_training_steps = total_training_steps
        self.lambda_icm_start, self.lambda_icm_end = lambda_icm_schedule
        self.lambda_touch_start, self.lambda_touch_end = lambda_touch_schedule
        self.lambda_hand_touch_start, self.lambda_hand_touch_end = lambda_hand_touch_schedule
        self.dynamic_weight_stop_step = dynamic_weight_stop_step
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.save_path = save_path
        self.save_freq = save_freq

        self.robot_geom_to_log_name = {}
        self.hand_geoms = set()
        self.body_part_geoms = set()

        self.rollout_touch_counts = defaultdict(int)
        self.rollout_touch_durations = defaultdict(int)
        self.rollout_touched_parts_by_hand = set()
        self.rollout_hand_to_hand_count = 0
        self.rollout_hand_to_hand_duration = 0
        
        self.active_hand_body_contacts = set()
        self.active_hand_hand_contacts = set()
        
        self.raw_touch_reward_sum = 0.0
        self.raw_hand_reward_sum = 0.0
        self.raw_icm_reward_sum = 0.0
        self.weighted_touch_reward_sum = 0.0
        self.weighted_hand_reward_sum = 0.0
        self.weighted_icm_reward_sum = 0.0
        
    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print("--- Initializing Behavior Analyzer Mappings (SimpleMLPCallback Version) ---")
        
        env = self.training_env.envs[0].env
        model = env.model
        
        try:
            robot_root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hip")
            if robot_root_id == -1: raise ValueError
            robot_body_ids = get_body_subtree(model, robot_root_id)
        except (ValueError, KeyError):
            robot_body_ids = set()

        hand_body_ids = set()
        for body_id in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name and 'hand' in body_name:
                hand_body_ids.add(body_id)
        
        self.hand_geoms = set()
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] in hand_body_ids:
                self.hand_geoms.add(geom_id)
        
        all_robot_geoms = set()
        for geom_id in range(model.ngeom):
            if model.geom_bodyid[geom_id] in robot_body_ids:
                all_robot_geoms.add(geom_id)
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                self.robot_geom_to_log_name[geom_id] = geom_name or mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id])

        self.body_part_geoms = all_robot_geoms - self.hand_geoms
        
    def _on_rollout_start(self) -> None:
        self.rollout_touch_counts.clear()
        self.rollout_touch_durations.clear()
        self.rollout_touched_parts_by_hand.clear()
        self.rollout_hand_to_hand_count = 0
        self.rollout_hand_to_hand_duration = 0
        self.active_hand_body_contacts.clear()
        self.active_hand_hand_contacts.clear()
        self.raw_touch_reward_sum = 0.0
        self.raw_hand_reward_sum = 0.0
        self.raw_icm_reward_sum = 0.0
        self.weighted_touch_reward_sum = 0.0
        self.weighted_hand_reward_sum = 0.0
        self.weighted_icm_reward_sum = 0.0

    def _on_step(self) -> bool:
        # 1. Calculate dynamic lambda values based on training progress
        progress = min(1.0, self.num_timesteps / self.dynamic_weight_stop_step)
        current_lambda_icm = self.lambda_icm_start + (self.lambda_icm_end - self.lambda_icm_start) * progress
        current_lambda_touch = self.lambda_touch_start + (self.lambda_touch_end - self.lambda_touch_start) * progress
        current_lambda_hand_touch = self.lambda_hand_touch_start + (self.lambda_hand_touch_end - self.lambda_hand_touch_start) * progress

        # 2. Compute curiosity reward from the new MLP model
        self.model_mlp.eval()
        with torch.no_grad():
            last_obs = {k: v[0] for k, v in self.model._last_obs.items()}
            action = self.locals['actions'][0]
            new_obs = {k: v[0] for k, v in self.locals['new_obs'].items()}
            
            p_obs = torchify(last_obs['observation'], self.device)
            t_obs = torchify(last_obs['touch'], self.device)
            action_tensor = torchify(action, self.device)
            next_p_true = torchify(new_obs['observation'], self.device)
            next_t_true = torchify(new_obs['touch'], self.device)

            next_p_pred, next_t_pred = self.model_mlp(p_obs, t_obs, action_tensor)
            curiosity_loss, _, _ = self.model_mlp.compute_loss(next_p_pred, next_t_pred, next_p_true, next_t_true)
        unweighted_icm_reward = curiosity_loss.item()
        
        # 3. Update the locals with the unweighted curiosity reward
        info = self.locals['infos'][0]
        unweighted_touch = info.get('reward_components', {}).get('unweighted_touch', 0.0)
        unweighted_hand = info.get('reward_components', {}).get('unweighted_hand', 0.0)

        # 4. Combine rewards
        weighted_icm = current_lambda_icm * unweighted_icm_reward
        weighted_touch = current_lambda_touch * unweighted_touch
        weighted_hand = current_lambda_hand_touch * unweighted_hand
        self.locals['rewards'][0] = weighted_icm + weighted_touch + weighted_hand

        # 5. Accumulate rewards for logging
        self.raw_icm_reward_sum += unweighted_icm_reward
        self.raw_touch_reward_sum += unweighted_touch
        self.raw_hand_reward_sum += unweighted_hand
        self.weighted_icm_reward_sum += weighted_icm
        self.weighted_touch_reward_sum += weighted_touch
        self.weighted_hand_reward_sum += weighted_hand

        # 6. Update the contact information for the current step
        env = self.training_env.envs[0].env
        contacts = env.data.contact
        current_hand_body_contacts, current_hand_hand_contacts = set(), set()

        for i in range(contacts.geom1.shape[0]):
            geom1, geom2 = contacts.geom1[i], contacts.geom2[i]
            is_g1_hand, is_g2_hand = geom1 in self.hand_geoms, geom2 in self.hand_geoms
            is_g1_body, is_g2_body = geom1 in self.body_part_geoms, geom2 in self.body_part_geoms

            if (is_g1_hand and is_g2_body) or (is_g2_hand and is_g1_body):
                body_id = geom2 if is_g1_hand else geom1
                body_name = self.robot_geom_to_log_name.get(body_id)
                if body_name:
                    contact_pair = tuple(sorted((geom1, geom2)))
                    current_hand_body_contacts.add(contact_pair)
                    self.rollout_touch_durations[body_name] += 1
                    if contact_pair not in self.active_hand_body_contacts:
                        self.rollout_touch_counts[body_name] += 1
                        self.rollout_touched_parts_by_hand.add(body_name)
            elif is_g1_hand and is_g2_hand:
                contact_pair = tuple(sorted((geom1, geom2)))
                current_hand_hand_contacts.add(contact_pair)
                self.rollout_hand_to_hand_duration += 1
                if contact_pair not in self.active_hand_hand_contacts:
                    self.rollout_hand_to_hand_count += 1
        
        self.active_hand_body_contacts = current_hand_body_contacts
        self.active_hand_hand_contacts = current_hand_hand_contacts

        # 7. Model saving logic
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            # PPO model saving
            self.model.save(os.path.join(self.save_path, "ppo_model", f'p_model_{self.num_timesteps}_steps.zip'))
            # MLP model saving
            torch.save(self.model_mlp.state_dict(), os.path.join(self.save_path, "icm_model", f'mlp_model_{self.num_timesteps}_steps.pth'))

        return True

    def _on_rollout_end(self) -> None:
        # 1. Log the accumulated rewards and contact statistics
        self.logger.record('behavior/touch_diversity_by_hand', len(self.rollout_touched_parts_by_hand))
        self.logger.record('behavior/hand_to_hand_freq', self.rollout_hand_to_hand_count)
        self.logger.record('behavior/hand_to_hand_duration', self.rollout_hand_to_hand_duration)
        for part_name, count in self.rollout_touch_counts.items():
            self.logger.record(f'behavior_freq/{part_name}', count)
        for part_name, duration in self.rollout_touch_durations.items():
            self.logger.record(f'behavior_duration/{part_name}', duration)
        
        rollout_steps = self.model.n_steps * self.model.n_envs
        if rollout_steps > 0:
            self.logger.record('reward_raw/mean_touch', self.raw_touch_reward_sum / rollout_steps)
            self.logger.record('reward_raw/mean_hand', self.raw_hand_reward_sum / rollout_steps)
            self.logger.record('reward_raw/mean_icm', self.raw_icm_reward_sum / rollout_steps)
            self.logger.record('reward_weighted/mean_touch', self.weighted_touch_reward_sum / rollout_steps)
            self.logger.record('reward_weighted/mean_hand', self.weighted_hand_reward_sum / rollout_steps)

        # 2. Train the new MLP model
        if self.verbose > 1:
            print(f"\n--- Rollout ended. Training DirectPredictionMLP for {self.n_epochs} epochs... ---")
        
        self.model_mlp.train()
        buffer = self.model.rollout_buffer
        num_samples = buffer.buffer_size * buffer.n_envs
        
        proprio_obs_np = buffer.observations['observation'].reshape(num_samples, -1)
        touch_obs_np = buffer.observations['touch'].reshape(num_samples, -1)
        actions_np = buffer.actions.reshape(num_samples, -1)
        
        next_proprio_obs_np = np.roll(proprio_obs_np, -1, axis=0)
        next_touch_obs_np = np.roll(touch_obs_np, -1, axis=0)
        
        p_obs = torch.from_numpy(proprio_obs_np).float().to(self.device)
        t_obs = torch.from_numpy(touch_obs_np).float().to(self.device)
        actions = torch.from_numpy(actions_np).float().to(self.device)
        next_p_obs = torch.from_numpy(next_proprio_obs_np).float().to(self.device)
        next_t_obs = torch.from_numpy(next_touch_obs_np).float().to(self.device)

        for epoch in range(self.n_epochs):
            permutation = torch.randperm(num_samples)
            for i in range(0, num_samples, self.batch_size):
                indices = permutation[i : i + self.batch_size]
                
                next_p_pred, next_t_pred = self.model_mlp(p_obs[indices], t_obs[indices], actions[indices])
                loss, p_loss, t_loss = self.model_mlp.compute_loss(next_p_pred, next_t_pred, next_p_obs[indices], next_t_obs[indices])
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        self.logger.record('mlp_loss/total', loss.item())
        self.logger.record('mlp_loss/proprio', p_loss.item())
        self.logger.record('mlp_loss/touch', t_loss.item())