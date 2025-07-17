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
        This method is called after each step in the environment.
        """
        # --- MODIFIED: DYNAMIC WEIGHTING LOGIC FOR ALL THREE LAMBDAS ---
        progress = min(1.0, self.num_timesteps / self.total_training_steps)
        
        # Linearly interpolate the weights based on the progress
        current_lambda_icm = self.lambda_icm_start + (self.lambda_icm_end - self.lambda_icm_start) * progress
        current_lambda_touch = self.lambda_touch_start + (self.lambda_touch_end - self.lambda_touch_start) * progress
        current_lambda_hand_touch = self.lambda_hand_touch_start + (self.lambda_hand_touch_end - self.lambda_hand_touch_start) * progress

        # Update the weights for the current step's reward calculation
        self.lambda_icm = current_lambda_icm
        self.training_env.set_attr('lambda_touch', current_lambda_touch)
        self.training_env.set_attr('lambda_hand_touch', current_lambda_hand_touch)
        
        # --- The rest of the logic remains the same ---
        
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
            icm_reward = norm_fwd_loss
            icm_rewards.append(icm_reward)
            
        weighted_icm_step_reward = self.lambda_icm * np.array(icm_rewards)
        self.locals['rewards'] += weighted_icm_step_reward
        info = self.locals['infos'][0] 
        if 'reward_components' in info:
            rc = info['reward_components']
            self.cumulative_touch_reward += rc['weighted_touch']
            self.cumulative_hand_reward += rc['weighted_hand']

        # Accumulate the ICM reward we just calculated
        # Note: assuming n_envs=1, so we take the first element
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
        if self.verbose > 0:
            # The number of steps in the rollout is n_steps
            rollout_steps = self.model.n_steps
            
            # Avoid division by zero if rollout is empty for some reason
            if rollout_steps > 0:
                mean_touch = self.cumulative_touch_reward / rollout_steps
                mean_hand = self.cumulative_hand_reward / rollout_steps
                mean_icm = self.cumulative_icm_reward / rollout_steps
                mean_total = mean_touch + mean_hand + mean_icm

                log_str = "\n--- Rollout Mean Reward Breakdown ---\n"
                log_str += f"Total: {mean_total:.4f} = "
                log_str += f"Touch({mean_touch:.4f}) + "
                log_str += f"Hand({mean_hand:.4f}) + "
                log_str += f"ICM({mean_icm:.4f})"

                # Log to TensorBoard
                self.logger.record('rollout/mean_total_reward', mean_total)
                self.logger.record('rollout/mean_touch_reward', mean_touch)
                self.logger.record('rollout/mean_hand_reward', mean_hand)
                self.logger.record('rollout/mean_icm_reward', mean_icm)

        if self.verbose > 1:
            print("\n--- Rollout ended. Starting to train ICM model... ---")
        
        buffer = self.model.rollout_buffer
        
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

        flat_obs_batch = np.array([flatten_obs(obs) for obs in obs_t_list])
        flat_next_obs_batch = np.array([flatten_obs(obs) for obs in next_obs_t_list])

        obs_tensor = torch.from_numpy(flat_obs_batch).float().to(self.icm.device)
        action_tensor = torch.from_numpy(actions_t_sliced).float().to(self.icm.device)
        next_obs_tensor = torch.from_numpy(flat_next_obs_batch).float().to(self.icm.device)
        
        final_losses = self.icm.train_on_batch(
            obs_tensor, 
            action_tensor, 
            next_obs_tensor,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size
        )

        if self.verbose > 1:
            log_str = f"[ICM Training] Timestep: {self.num_timesteps} | "
            log_str += f"VAE(R/KL): {final_losses['vae_recon_loss']:.4f}/{final_losses['vae_kl_loss']:.4f} | "
            log_str += f"Fwd: {final_losses['forward_loss']:.4f} | "
            log_str += f"Inv(R/KL): {final_losses['inverse_recon_loss']:.4f}/{final_losses['inverse_kl_loss']:.4f}"
            print(log_str)
        
        for key, value in final_losses.items():
            self.logger.record(f'icm/{key}', value)