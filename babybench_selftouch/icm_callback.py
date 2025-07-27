# babybench_selftouch/icm_callback.py

import os
import torch
import numpy as np
import mujoco
# --- 【修正】: 导入defaultdict ---
from collections import defaultdict
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from babybench_selftouch.utils import torchify 
from babybench_selftouch.icm.icm_module import ICMModule

def get_body_subtree(model, root_id):
    """获取一个根节点下的所有子身体的ID集合。"""
    subtree = {root_id}
    # MjModel中的body_parentid数组记录了每个body的父body的id
    for body_id in range(model.nbody):
        current_id = body_id
        while current_id != -1: # -1表示没有父节点
            parent_id = model.body_parentid[current_id]
            if parent_id in subtree:
                subtree.add(body_id)
                break
            # 如果父节点不是root，则继续向上查找父节点的父节点
            current_id = parent_id
            if current_id == 0: # 到达worldbody，停止
                break
    return subtree
# ---------------------------------------------

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
                 n_epochs: int = 8, 
                 batch_size: int = 512, 
                 verbose: int = 0):
        super().__init__(verbose)
        self.icm = icm_module
        self.total_training_steps = total_training_steps
        
        # Normalized reward weight logic (This part is correct)
        self.lambda_icm_start, self.lambda_icm_end_unnormalized = lambda_icm_schedule
        self.lambda_touch_start, self.lambda_touch_end_unnormalized = lambda_touch_schedule
        self.lambda_hand_touch_start, self.lambda_hand_touch_end_unnormalized = lambda_hand_touch_schedule
        self.total_weight = self.lambda_icm_start + self.lambda_touch_start + self.lambda_hand_touch_start
        if self.verbose > 0:
            print(f"Normalized reward weight activated. Constant Total Weight = {self.total_weight:.4f}")
        end_weight_sum_unnormalized = self.lambda_icm_end_unnormalized + self.lambda_touch_end_unnormalized + self.lambda_hand_touch_end_unnormalized
        normalization_factor = self.total_weight / end_weight_sum_unnormalized if end_weight_sum_unnormalized > 0 else 0
        self.lambda_icm_end = self.lambda_icm_end_unnormalized * normalization_factor
        self.lambda_touch_end = self.lambda_touch_end_unnormalized * normalization_factor
        self.lambda_hand_touch_end = self.lambda_hand_touch_end_unnormalized * normalization_factor

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.save_path = save_path
        self.save_freq = save_freq

        # One-time setup attributes
        self.robot_geom_to_log_name = {}
        self.hand_geoms = set()

        # --- 【修正】: 使用defaultdict来避免KeyError ---
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
        
        # --- 【修正】: 创建精确的非手部身体geom集合 ---
        self.body_part_geoms = all_robot_geoms - self.hand_geoms
        # -------------------------------------------
        
        if self.verbose > 0:
            # --- 【修正/优化】: 统一打印日志 ---
            print(f"Identified {len(self.hand_geoms)} hand geoms (not including fingers).")
            print(f"Identified {len(self.body_part_geoms)} non-hand body geoms.")
            print(f"Mapped {len(self.robot_geom_to_log_name)} robot geoms to logging names.")
            # ----------------------------------
        print("---------------------------------------------")

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
        # This method's logic was implemented correctly, no changes needed.
        progress = min(1.0, self.num_timesteps / self.total_training_steps)
        self.current_lambda_icm = self.lambda_icm_start + (self.lambda_icm_end - self.lambda_icm_start) * progress
        self.current_lambda_touch = self.lambda_touch_start + (self.lambda_touch_end - self.lambda_touch_start) * progress
        self.current_lambda_hand_touch = self.lambda_hand_touch_start + (self.lambda_hand_touch_end - self.lambda_hand_touch_start) * progress
        self.training_env.set_attr('lambda_touch', self.current_lambda_touch)
        self.training_env.set_attr('lambda_hand_touch', self.current_lambda_hand_touch)

        touch_components_reward = self.locals['rewards'].copy()
        
        info = self.locals['infos'][0]
        if 'reward_components' in info:
            rc = info['reward_components']
            self.raw_touch_reward_sum += rc.get('unweighted_touch', 0.0)
            self.raw_hand_reward_sum += rc.get('unweighted_hand', 0.0)
            self.weighted_touch_reward_sum += rc.get('weighted_touch', 0.0)
            self.weighted_hand_reward_sum += rc.get('weighted_hand', 0.0)

        last_obs_single = {k: v[0] for k, v in self.model._last_obs.items()}
        action_single = self.locals['actions'][0]
        new_obs_single = {k: v[0] for k, v in self.locals['new_obs'].items()}
        p_obs = torchify(last_obs_single['observation'], self.icm.device)
        t_obs = torchify(last_obs_single['touch'], self.icm.device)
        action_tensor = torchify(action_single, self.icm.device)
        next_p_obs = torchify(new_obs_single['observation'], self.icm.device)
        next_t_obs = torchify(new_obs_single['touch'], self.icm.device)
        norm_fwd_loss, _ = self.icm.compute_forward_loss(
            p_obs, t_obs, action_tensor, next_p_obs, next_t_obs, update_ema=True
        )
        self.raw_icm_reward_sum += norm_fwd_loss
        
        weighted_icm_step_reward = self.current_lambda_icm * norm_fwd_loss
        
        self.weighted_icm_reward_sum += weighted_icm_step_reward
        self.locals['rewards'][:] += weighted_icm_step_reward
        
        # --- 【关键修正】: 区分“手碰身体”和“手碰手”的接触 ---
        env = self.training_env.envs[0].env
        contacts = env.data.contact
        current_hand_body_contacts = set()
        current_hand_hand_contacts = set()

        for i in range(contacts.geom1.shape[0]):
            geom1_id, geom2_id = contacts.geom1[i], contacts.geom2[i]
            
            is_geom1_hand = geom1_id in self.hand_geoms
            is_geom2_hand = geom2_id in self.hand_geoms
            
            # --- Case 1: 手碰身体 ---
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

            # --- Case 2: 手碰手 ---
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
        # Logging of behavior and raw rewards is correct
        # --- 【修改】: 增加手碰手的日志记录 ---
        self.logger.record('behavior/touch_diversity_by_hand', len(self.rollout_touched_parts_by_hand))
        self.logger.record('behavior/hand_to_hand_freq', self.rollout_hand_to_hand_count)
        self.logger.record('behavior/hand_to_hand_duration', self.rollout_hand_to_hand_duration)
        # -------------------------------------

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
        
        # ICM MODEL TRAINING & LOGGING
        if self.verbose > 1:
            print("\n--- Rollout ended. Starting to train ICM model... ---")
        buffer = self.model.rollout_buffer
        
        # --- 【修正】: 使用更简洁、健壮的数据准备逻辑 ---
        num_samples = buffer.buffer_size * buffer.n_envs
        # 直接从buffer的字典中获取完整的、正确的数组
        proprio_obs = buffer.observations['observation'].reshape(num_samples, -1)
        touch_obs = buffer.observations['touch'].reshape(num_samples, -1)
        actions = buffer.actions.reshape(num_samples, -1)
        # 使用np.roll来高效地创建next_obs
        next_proprio_obs = np.roll(proprio_obs, -1, axis=0)
        next_touch_obs = np.roll(touch_obs, -1, axis=0)
        # 转换为Tensor
        proprio_tensor = torch.from_numpy(proprio_obs).float().to(self.icm.device)
        touch_tensor = torch.from_numpy(touch_obs).float().to(self.icm.device)
        action_tensor = torch.from_numpy(actions).float().to(self.icm.device)
        next_proprio_tensor = torch.from_numpy(next_proprio_obs).float().to(self.icm.device)
        next_touch_tensor = torch.from_numpy(next_touch_obs).float().to(self.icm.device)
        # ----------------------------------------------------

        final_losses = self.icm.train_on_batch(
            proprio_tensor, touch_tensor, action_tensor, 
            next_proprio_tensor, next_touch_tensor,
            n_epochs=self.n_epochs, batch_size=self.batch_size
        )

        # Logging ICM losses is correct
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