# babybench_selftouch/icm_callback.py

import os
import torch
import numpy as np
import mujoco
from collections import defaultdict
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from babybench_selftouch.utils import torchify 
from babybench_selftouch.icm.icm_module import ICMModule

def get_body_subtree(model, root_id):
    """Get all body IDs in the subtree rooted at root_id."""
    subtree = {root_id}
    # body_parentid is a list where each index corresponds to a body ID and the value is its parent body ID
    for body_id in range(model.nbody):
        current_id = body_id
        while current_id != -1:
            parent_id = model.body_parentid[current_id]
            if parent_id in subtree:
                subtree.add(body_id)
                break
            # If the parent is not the root, continue to search up the hierarchy
            current_id = parent_id
            if current_id == 0:
                break
    return subtree

class ICMCallback(BaseCallback):
    """
    A custom callback for ICM integration with advanced features,
    including a memory-safe logger for thesis data.
    """
    def __init__(self, 
                 icm_module: ICMModule,
                 total_training_steps: int,
                 save_path: str, 
                 save_freq: int = 100000, 
                 lambda_icm_schedule: tuple = (5.0, 50.0),
                 lambda_touch_schedule: tuple = (10.0, 1.0),
                 lambda_hand_touch_schedule: tuple = (20.0, 2.0),
                 dynamic_weight_stop_step: int = 1000000,
                 n_epochs: int = 8, 
                 batch_size: int = 512, 
                 verbose: int = 0):
        super().__init__(verbose)
        self.icm = icm_module
        self.total_training_steps = total_training_steps
        
        self.lambda_icm_start, self.lambda_icm_end = lambda_icm_schedule
        self.lambda_touch_start, self.lambda_touch_end = lambda_touch_schedule
        self.lambda_hand_touch_start, self.lambda_hand_touch_end = lambda_hand_touch_schedule

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.save_path = save_path
        self.save_freq = save_freq

        # One-time setup attributes
        self.robot_geom_to_log_name = {}
        self.hand_geoms = set()

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
        self.dynamic_weight_stop_step = dynamic_weight_stop_step

        self.last_save_timestep = 0

    def _on_training_start(self) -> None:
        if self.verbose > 0:
            print("--- Initializing Behavior Analyzer Mappings ---")
        
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
            #if body_name and ('hand' in body_name or 'fingers' in body_name):
            # only hand no fingers
            if body_name and 'hand' in body_name:
                hand_body_ids.add(body_id)
        
        self.hand_geoms = set()
        for geom_id in range(model.ngeom):
            parent_body_id = model.geom_bodyid[geom_id]
            if parent_body_id in hand_body_ids:
                self.hand_geoms.add(geom_id)
        
        self.robot_geom_to_log_name = {}
        all_robot_geoms = set()
        for geom_id in range(model.ngeom):
            parent_body_id = model.geom_bodyid[geom_id]
            if parent_body_id in robot_body_ids:
                all_robot_geoms.add(geom_id)
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                if geom_name:
                    self.robot_geom_to_log_name[geom_id] = geom_name
                else:
                    parent_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_body_id)
                    if parent_body_name:
                        self.robot_geom_to_log_name[geom_id] = parent_body_name
        
        self.body_part_geoms = all_robot_geoms - self.hand_geoms
        
        if self.verbose > 0:
            print(f"Identified {len(self.hand_geoms)} hand geoms (not including fingers).")
            print(f"Identified {len(self.body_part_geoms)} non-hand body geoms.")
            print(f"Mapped {len(self.robot_geom_to_log_name)} robot geoms to logging names.")
        print("---------------------------------------------")

    def _on_rollout_start(self) -> None:
        # Check if the current number of timesteps has surpassed the next save milestone
        if self.num_timesteps >= self.last_save_timestep + self.save_freq:
            self.last_save_timestep += self.save_freq
            save_step_milestone = self.last_save_timestep

            # --- Save PPO policy model ---
            ppo_model_path = os.path.join(self.save_path, "ppo_model",f'p_model_{save_step_milestone}_steps.zip')
            self.model.save(ppo_model_path)

            # --- Save ICM model ---
            icm_model_path = os.path.join(self.save_path, "icm_model", f'icm_model_{save_step_milestone}_steps.pth')
            torch.save(self.icm.state_dict(), icm_model_path)
            
            if self.verbose > 0:
                print(f"\n--- Saving models at timestep milestone {save_step_milestone} ---")
                print(f"Saved PPO model to {ppo_model_path}")
                print(f"Saved ICM model to {icm_model_path}")
                print("-------------------------------------------------")

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
        # 1. Update dynamic weights based on training progress
        progress = min(1.0, self.num_timesteps / self.dynamic_weight_stop_step)
        self.current_lambda_icm = self.lambda_icm_start + (self.lambda_icm_end - self.lambda_icm_start) * progress
        self.current_lambda_touch = self.lambda_touch_start + (self.lambda_touch_end - self.lambda_touch_start) * progress
        self.current_lambda_hand_touch = self.lambda_hand_touch_start + (self.lambda_hand_touch_end - self.lambda_hand_touch_start) * progress
        
        # 2. Get unweighted reward components
        info = self.locals['infos'][0]
        unweighted_touch = 0.0
        unweighted_hand = 0.0
        if 'reward_components' in info:
            rc = info['reward_components']
            unweighted_touch = rc.get('unweighted_touch', 0.0)
            unweighted_hand = rc.get('unweighted_hand', 0.0)

        # 3. Compute ICM forward loss
        last_obs_single = {k: v[0] for k, v in self.model._last_obs.items()}
        action_single = self.locals['actions'][0]
        new_obs_single = {k: v[0] for k, v in self.locals['new_obs'].items()}
        p_obs = torchify(last_obs_single['observation'], self.icm.device)
        t_obs = torchify(last_obs_single['touch'], self.icm.device)
        action_tensor = torchify(action_single, self.icm.device)
        next_p_obs = torchify(new_obs_single['observation'], self.icm.device)
        next_t_obs = torchify(new_obs_single['touch'], self.icm.device)
        
        norm_fwd_loss, fwd_loss = self.icm.compute_forward_loss(
            p_obs, t_obs, action_tensor, next_p_obs, next_t_obs, update_ema=True
        )
        
        # 4. Compute weighted rewards
        weighted_touch = self.current_lambda_touch * unweighted_touch
        weighted_hand = self.current_lambda_hand_touch * unweighted_hand
        weighted_icm = self.current_lambda_icm * fwd_loss.item()
        
        # 5. Update the raw and weighted reward sums
        self.raw_touch_reward_sum += unweighted_touch
        self.raw_hand_reward_sum += unweighted_hand
        self.raw_icm_reward_sum += fwd_loss.item()
        
        self.weighted_touch_reward_sum += weighted_touch
        self.weighted_hand_reward_sum += weighted_hand
        self.weighted_icm_reward_sum += weighted_icm
        
        # 6. Update the locals dictionary with the new rewards
        self.locals['rewards'][0] = weighted_touch + weighted_hand + weighted_icm
        
        # 7. Log the body part touch counts and durations
        env = self.training_env.envs[0].env
        contacts = env.data.contact
        current_hand_body_contacts = set()
        current_hand_hand_contacts = set()

        for i in range(contacts.geom1.shape[0]):
            geom1_id, geom2_id = contacts.geom1[i], contacts.geom2[i]
            
            is_geom1_hand = geom1_id in self.hand_geoms
            is_geom2_hand = geom2_id in self.hand_geoms
            
            # --- Case 1: Hand-body contact ---
            is_geom1_body = geom1_id in self.body_part_geoms
            is_geom2_body = geom2_id in self.body_part_geoms
            
            if (is_geom1_hand and is_geom2_body) or (is_geom2_hand and is_geom1_body):
                hand_id = geom1_id if is_geom1_hand else geom2_id
                body_id = geom2_id if is_geom1_hand else geom1_id
                
                body_name = self.robot_geom_to_log_name.get(body_id)
                if body_name:
                    contact_pair = tuple(sorted((hand_id, body_id)))
                    current_hand_body_contacts.add(contact_pair)
                    
                    self.rollout_touch_durations[body_name] += 1
                    if contact_pair not in self.active_hand_body_contacts:
                        self.rollout_touch_counts[body_name] += 1
                        self.rollout_touched_parts_by_hand.add(body_name)

            # --- Case 2: Hand-hand contact ---
            elif is_geom1_hand and is_geom2_hand:
                contact_pair = tuple(sorted((geom1_id, geom2_id)))
                current_hand_hand_contacts.add(contact_pair)
                
                self.rollout_hand_to_hand_duration += 1
                if contact_pair not in self.active_hand_hand_contacts:
                    self.rollout_hand_to_hand_count += 1

        self.active_hand_body_contacts = current_hand_body_contacts
        self.active_hand_hand_contacts = current_hand_hand_contacts

        return True

    def _on_rollout_end(self) -> None:
        self.logger.record('behavior/touch_diversity_by_hand', len(self.rollout_touched_parts_by_hand))
        self.logger.record('behavior/hand_to_hand_freq', self.rollout_hand_to_hand_count)
        self.logger.record('behavior/hand_to_hand_duration', self.rollout_hand_to_hand_duration)

        for part_name, count in self.rollout_touch_counts.items():
            self.logger.record(f'behavior_freq/{part_name}', count)
        for part_name, duration in self.rollout_touch_durations.items():
            self.logger.record(f'behavior_duration/{part_name}', duration)
        rollout_steps = self.model.n_steps
        if rollout_steps > 0:
            self.logger.record('reward_raw/mean_touch', self.raw_touch_reward_sum / rollout_steps)
            self.logger.record('reward_raw/mean_hand', self.raw_hand_reward_sum / rollout_steps)
            self.logger.record('reward_raw/mean_icm', self.raw_icm_reward_sum / rollout_steps)
            self.logger.record('reward_weighted/mean_touch', self.weighted_touch_reward_sum / rollout_steps)
            self.logger.record('reward_weighted/mean_hand', self.weighted_hand_reward_sum / rollout_steps)
            self.logger.record('reward_weighted/mean_icm', self.weighted_icm_reward_sum / rollout_steps)
        
        if self.verbose > 1:
            print("\n--- Rollout ended. Starting to train ICM model... ---")
        buffer = self.model.rollout_buffer
        
        num_samples = buffer.buffer_size * buffer.n_envs
        proprio_obs = buffer.observations['observation'].reshape(num_samples, -1)
        touch_obs = buffer.observations['touch'].reshape(num_samples, -1)
        actions = buffer.actions.reshape(num_samples, -1)
        next_proprio_obs = np.roll(proprio_obs, -1, axis=0)
        next_touch_obs = np.roll(touch_obs, -1, axis=0)
        proprio_tensor = torch.from_numpy(proprio_obs).float().to(self.icm.device)
        touch_tensor = torch.from_numpy(touch_obs).float().to(self.icm.device)
        action_tensor = torch.from_numpy(actions).float().to(self.icm.device)
        next_proprio_tensor = torch.from_numpy(next_proprio_obs).float().to(self.icm.device)
        next_touch_tensor = torch.from_numpy(next_touch_obs).float().to(self.icm.device)

        final_losses = self.icm.train_on_batch(
            proprio_tensor, touch_tensor, action_tensor, 
            next_proprio_tensor, next_touch_tensor,
            n_epochs=self.n_epochs, batch_size=self.batch_size
        )

        if self.verbose > 1:
            log_str = f"[ICM Training] Timestep: {self.num_timesteps}\n"
            proprio_recon_loss = final_losses.get('proprio_vae_recon_loss', 0.0)
            proprio_kl_loss = final_losses.get('proprio_vae_kl_loss', 0.0)
            touch_recon_loss = final_losses.get('touch_vae_recon_loss', 0.0)
            touch_kl_loss = final_losses.get('touch_vae_kl_loss', 0.0)
            forward_loss = final_losses.get('forward_loss', 0.0)
            log_str += f"  Proprio VAE(R/KL): {proprio_recon_loss:.4f}/{proprio_kl_loss:.4f} | "
            log_str += f"Touch VAE(R/KL): {touch_recon_loss:.4f}/{touch_kl_loss:.4f}\n"
            log_str += f"  Forward Loss: {forward_loss:.4f}"
            print(log_str)
        
        for key, value in final_losses.items():
            self.logger.record(f'icm/{key}', value)