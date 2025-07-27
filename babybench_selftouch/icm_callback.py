# babybench_selftouch/icm_callback.py

import os
import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

# --- MODIFIED: Removed flatten_obs, as it's no longer used ---
from babybench_selftouch.utils import torchify 
from babybench_selftouch.icm.icm_module import ICMModule


class ICMCallback(BaseCallback):
    """
    A custom callback for ICM integration with advanced features.
    --- MODIFIED ---
    This version handles separate data modalities (proprioception, touch)
    for the dual-VAE ICM model.
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
        
        self.lambda_icm_start, self.lambda_icm_end = lambda_icm_schedule
        self.lambda_touch_start, self.lambda_touch_end = lambda_touch_schedule
        self.lambda_hand_touch_start, self.lambda_hand_touch_end = lambda_hand_touch_schedule

        self.lambda_icm = self.lambda_icm_start

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.save_path = save_path
        self.save_freq = save_freq

        self.cumulative_touch_reward = 0.0
        self.cumulative_hand_reward = 0.0
        self.cumulative_icm_reward = 0.0

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.total_training_steps)
        
        self.current_lambda_icm = self.lambda_icm_start + (self.lambda_icm_end - self.lambda_icm_start) * progress
        self.current_lambda_touch = self.lambda_touch_start + (self.lambda_touch_end - self.lambda_touch_start) * progress
        self.current_lambda_hand_touch = self.lambda_hand_touch_start + (self.lambda_hand_touch_end - self.lambda_hand_touch_start) * progress

        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            if self.verbose > 0:
                print(f"\n--- Saving models at step {self.num_timesteps} ---")
            ppo_path = os.path.join(self.save_path, "ppo_model", f"{self.num_timesteps}_steps")
            icm_path = os.path.join(self.save_path, "icm_model", f"{self.num_timesteps}_steps")
            os.makedirs(ppo_path, exist_ok=True)
            os.makedirs(icm_path, exist_ok=True)
            self.model.save(os.path.join(ppo_path, "model.zip"))
            torch.save(self.icm.state_dict(), os.path.join(icm_path, "icm_model.pth"))
            if self.verbose > 0:
                print(f"PPO model saved to {ppo_path}")
                print(f"ICM model saved to {icm_path}\n")

        touch_components_reward = self.locals['rewards'].copy()

        current_pos = self.model.rollout_buffer.pos
        last_obs_dict = {
            key: obs_array[current_pos] 
            for key, obs_array in self.model.rollout_buffer.observations.items()
        }
        actions = self.locals['actions']
        new_obs_dict = self.locals['new_obs']
        icm_rewards = []

        for i in range(self.training_env.num_envs):
            last_obs_single = {k: v[i] for k, v in last_obs_dict.items()}
            action_single = actions[i]
            new_obs_single = {k: v[i] for k, v in new_obs_dict.items()}
            
            # --- MODIFIED: Prepare separate tensors for each modality ---
            p_obs = torchify(last_obs_single['observation'], self.icm.device)
            t_obs = torchify(last_obs_single['touch'], self.icm.device)
            action_tensor = torchify(action_single, self.icm.device)
            next_p_obs = torchify(new_obs_single['observation'], self.icm.device)
            next_t_obs = torchify(new_obs_single['touch'], self.icm.device)

            # --- MODIFIED: Call the new compute_forward_loss with separate modalities ---
            norm_fwd_loss, _ = self.icm.compute_forward_loss(
                p_obs, t_obs, action_tensor, next_p_obs, next_t_obs, update_ema=True
            )
            icm_rewards.append(norm_fwd_loss)
            
        weighted_icm_step_reward = self.current_lambda_icm * np.array(icm_rewards)
        un_smoothed_total_reward = touch_components_reward + weighted_icm_step_reward
        self.locals['rewards'][:] = un_smoothed_total_reward

        info = self.locals['infos'][0]
        if 'reward_components' in info:
            rc = info['reward_components']
            self.cumulative_touch_reward += rc['weighted_touch']
            self.cumulative_hand_reward += rc['weighted_hand']
        
        self.cumulative_icm_reward += weighted_icm_step_reward[0]

        return True

    def _on_rollout_start(self) -> None:
        self.cumulative_touch_reward = 0.0
        self.cumulative_hand_reward = 0.0
        self.cumulative_icm_reward = 0.0

    def _on_rollout_end(self) -> None:
        if self.verbose > 0:
            rollout_steps = self.model.n_steps
            if rollout_steps > 0:
                mean_touch = self.cumulative_touch_reward / rollout_steps
                mean_hand = self.cumulative_hand_reward / rollout_steps
                mean_icm = self.cumulative_icm_reward / rollout_steps
                mean_total_pre_smoothing = mean_touch + mean_hand + mean_icm

                log_str = "\n--- Rollout Mean Reward Breakdown (Pre-Smoothing) ---\n"
                log_str += f"Total: {mean_total_pre_smoothing:.4f} = "
                log_str += f"Touch({mean_touch:.4f}) + "
                log_str += f"Hand({mean_hand:.4f}) + "
                log_str += f"ICM({mean_icm:.4f})"
                print(log_str)

                self.logger.record('rollout/mean_pre_smoothing_total', mean_total_pre_smoothing)
                self.logger.record('rollout/mean_pre_smoothing_touch', mean_touch)
                self.logger.record('rollout/mean_pre_smoothing_hand', mean_hand)
                self.logger.record('rollout/mean_pre_smoothing_icm', mean_icm)

        if self.verbose > 1:
            print("\n--- Rollout ended. Starting to train ICM model... ---")
        
        buffer = self.model.rollout_buffer
        
        # --- MODIFIED: Prepare separate lists and tensors for each modality ---
        proprio_obs_list = []
        touch_obs_list = []
        next_proprio_obs_list = []
        next_touch_obs_list = []

        # Iterate through the buffer to collect transition data
        for env_idx in range(buffer.n_envs):
            for step in range(buffer.buffer_size - 1):
                obs_dict = {k: v[step, env_idx] for k, v in buffer.observations.items()}
                next_obs_dict = {k: v[step + 1, env_idx] for k, v in buffer.observations.items()}
                
                proprio_obs_list.append(obs_dict['observation'])
                touch_obs_list.append(obs_dict['touch'])
                next_proprio_obs_list.append(next_obs_dict['observation'])
                next_touch_obs_list.append(next_obs_dict['touch'])

        # Prepare action tensor (this part remains the same)
        actions_t_sliced = buffer.actions[:-1].reshape(-1, buffer.action_space.shape[0])
        action_tensor = torch.from_numpy(actions_t_sliced).float().to(self.icm.device)

        # Convert lists to tensors
        proprio_tensor = torch.from_numpy(np.array(proprio_obs_list)).float().to(self.icm.device)
        touch_tensor = torch.from_numpy(np.array(touch_obs_list)).float().to(self.icm.device)
        next_proprio_tensor = torch.from_numpy(np.array(next_proprio_obs_list)).float().to(self.icm.device)
        next_touch_tensor = torch.from_numpy(np.array(next_touch_obs_list)).float().to(self.icm.device)

        # --- MODIFIED: Call the new train_on_batch with separate tensors ---
        final_losses = self.icm.train_on_batch(
            proprio_tensor,
            touch_tensor,
            action_tensor, 
            next_proprio_tensor,
            next_touch_tensor,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size
        )

        # --- MODIFIED: Update logging to show separate VAE losses ---
        if self.verbose > 1:
            # Create a more detailed log string
            log_str = f"[ICM Training] Timestep: {self.num_timesteps}\n"
            log_str += f"  Proprio VAE(R/KL): {final_losses['proprio_vae_recon_loss']:.4f}/{final_losses['proprio_vae_kl_loss']:.4f} | "
            log_str += f"Touch VAE(R/KL): {final_losses['touch_vae_recon_loss']:.4f}/{final_losses['touch_vae_kl_loss']:.4f}\n"
            log_str += f"  Forward Loss: {final_losses['forward_loss']:.4f} | "
            log_str += f"Inverse(R/KL): {final_losses['inverse_recon_loss']:.4f}/{final_losses['inverse_kl_loss']:.4f}"
            print(log_str)
        
        # Log all new loss components to TensorBoard
        for key, value in final_losses.items():
            self.logger.record(f'icm/{key}', value)