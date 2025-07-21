# babybench_selftouch/icm_callback.py

import os
import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

# Import necessary components
from babybench_selftouch.utils import flatten_obs, torchify 
from babybench_selftouch.icm.icm_module import ICMModule


class ICMCallback(BaseCallback):
    """
    A custom callback for ICM integration with advanced features:
    1. Calculates ICM curiosity reward with a dynamically annealed weight.
    2. Triggers batch training of the ICM model at the end of each rollout.
    3. Periodically saves model checkpoints.
    4. Dynamically adjusts separate touch reward weights for hands and body.
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
        
        # --- MODIFIED: Store all three schedule parameters ---
        self.lambda_icm_start, self.lambda_icm_end = lambda_icm_schedule
        self.lambda_touch_start, self.lambda_touch_end = lambda_touch_schedule
        self.lambda_hand_touch_start, self.lambda_hand_touch_end = lambda_hand_touch_schedule

        # Initialize current lambda_icm with the starting value
        self.lambda_icm = self.lambda_icm_start

        # Other parameters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.save_path = save_path
        self.save_freq = save_freq

        self.cumulative_touch_reward = 0.0
        self.cumulative_hand_reward = 0.0
        self.cumulative_icm_reward = 0.0

    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment. It is the central
        hub for calculating final rewards, updating weights, and saving models.
        """
        # --- 1. Update Dynamic Lambda Weights ---
        progress = min(1.0, self.num_timesteps / self.total_training_steps)
        
        # Linearly interpolate all three weights based on the training progress
        self.current_lambda_icm = self.lambda_icm_start + (self.lambda_icm_end - self.lambda_icm_start) * progress
        self.current_lambda_touch = self.lambda_touch_start + (self.lambda_touch_end - self.lambda_touch_start) * progress
        self.current_lambda_hand_touch = self.lambda_hand_touch_start + (self.lambda_hand_touch_end - self.lambda_hand_touch_start) * progress

        # --- 2. Periodically Save Models ---
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

        # --- 3. Final Reward Calculation and Smoothing ---

        # 3a. Get the un-smoothed, touch-related rewards returned by the wrapper
        # At this point, self.locals['rewards'] contains the sum of weighted touch and hand rewards
        touch_components_reward = self.locals['rewards'].copy()

        # 3b. Calculate the ICM curiosity reward for the current step
        # This part requires fetching observations and actions from the buffer/locals
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
            
            flat_obs = torchify(flatten_obs(last_obs_single), self.icm.device)
            flat_action = torchify(action_single, self.icm.device)
            flat_next_obs = torchify(flatten_obs(new_obs_single), self.icm.device)
            
            norm_fwd_loss, _ = self.icm.compute_forward_loss(flat_obs, flat_action, flat_next_obs, update_ema=True)
            icm_rewards.append(norm_fwd_loss)
            
        weighted_icm_step_reward = self.current_lambda_icm * np.array(icm_rewards)

        # 3c. Sum all components to get the final un-smoothed total reward
        un_smoothed_total_reward = touch_components_reward + weighted_icm_step_reward

        # 3d. Apply logarithmic scaling to the final total reward to smooth it
        smoothed_reward = np.sign(un_smoothed_total_reward) * np.log1p(np.abs(un_smoothed_total_reward))

        # 3e. Overwrite the rewards in `locals` with the smoothed version for the PPO update
        self.locals['rewards'][:] = smoothed_reward

        # --- 4. Accumulate Pre-Smoothing Rewards for Logging ---
        info = self.locals['infos'][0]
        if 'reward_components' in info:
            rc = info['reward_components']
            # Accumulate the original, un-smoothed weighted values for accurate analysis
            self.cumulative_touch_reward += rc['weighted_touch']
            self.cumulative_hand_reward += rc['weighted_hand']
        
        # Accumulate the original, un-smoothed weighted ICM reward
        self.cumulative_icm_reward += weighted_icm_step_reward[0]

        return True

    def _on_rollout_start(self) -> None:
        """
        This method is called at the beginning of a new rollout.
        We use it to reset our reward accumulators.
        """
        self.cumulative_touch_reward = 0.0
        self.cumulative_hand_reward = 0.0
        self.cumulative_icm_reward = 0.0

    def _on_rollout_end(self) -> None:
        """
        This method is called at the end of each rollout.
        Its primary role is to train the ICM model and log the training progress.
        """
        # --- 任务一：报告与记录 Rollout 的奖励情况 ---
        if self.verbose > 0:
            # The number of steps in the rollout is n_steps
            rollout_steps = self.model.n_steps
            
            # Avoid division by zero if rollout is empty for some reason
            if rollout_steps > 0:
                # Calculate the MEAN of the pre-smoothed rewards accumulated during the rollout.
                # The self.cumulative_*_reward variables were updated in every _on_step call.
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

                # Log to TensorBoard for analysis
                self.logger.record('rollout/mean_pre_smoothing_total', mean_total_pre_smoothing)
                self.logger.record('rollout/mean_pre_smoothing_touch', mean_touch)
                self.logger.record('rollout/mean_pre_smoothing_hand', mean_hand)
                self.logger.record('rollout/mean_pre_smoothing_icm', mean_icm)

        # --- 任务二：训练 ICM 模型 ---
        if self.verbose > 1:
            print("\n--- Rollout ended. Starting to train ICM model... ---")
        
        buffer = self.model.rollout_buffer
        
        # Extract all necessary data from the buffer for ICM batch training
        obs_t_list = [
             {k: v[step, env_idx] for k, v in buffer.observations.items()}
            for env_idx in range(buffer.n_envs)
            for step in range(buffer.buffer_size - 1)
        ]
        next_obs_t_list = [
             {k: v[step + 1, env_idx] for k, v in buffer.observations.items()}
            for env_idx in range(buffer.n_envs)
            for step in range(buffer.buffer_size - 1)
        ]
        actions_t_sliced = buffer.actions[:-1].reshape(-1, buffer.action_space.shape[0])

        # Convert data to numpy arrays and then to torch tensors
        flat_obs_batch = np.array([flatten_obs(obs) for obs in obs_t_list])
        flat_next_obs_batch = np.array([flatten_obs(obs) for obs in next_obs_t_list])

        obs_tensor = torch.from_numpy(flat_obs_batch).float().to(self.icm.device)
        action_tensor = torch.from_numpy(actions_t_sliced).float().to(self.icm.device)
        next_obs_tensor = torch.from_numpy(flat_next_obs_batch).float().to(self.icm.device)
        
        # Perform the actual training
        final_losses = self.icm.train_on_batch(
            obs_tensor, 
            action_tensor, 
            next_obs_tensor,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size
        )

        # Log the losses from the ICM training
        if self.verbose > 1:
            log_str = f"[ICM Training] Timestep: {self.num_timesteps} | "
            log_str += f"VAE(R/KL): {final_losses['vae_recon_loss']:.4f}/{final_losses['vae_kl_loss']:.4f} | "
            log_str += f"Fwd: {final_losses['forward_loss']:.4f} | "
            log_str += f"Inv(R/KL): {final_losses['inverse_recon_loss']:.4f}/{final_losses['inverse_kl_loss']:.4f}"
            print(log_str)
        
        for key, value in final_losses.items():
            self.logger.record(f'icm/{key}', value)