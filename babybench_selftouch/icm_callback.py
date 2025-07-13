# babybench_selftouch/icm_callback.py

import os
import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from babybench_selftouch.selftouch_wrapper import flatten_obs, torchify 
from babybench_selftouch.icm.icm_module import ICMModule


class ICMCallback(BaseCallback):
    """
    A custom callback for ICM integration.
    1. Calculates ICM curiosity reward at each step.
    2. Triggers batch training of the ICM model at the end of each rollout.
    3. Periodically saves model checkpoints.
    """
    def __init__(self, icm_module: ICMModule, save_path: str, save_freq: int = 100000, lambda_icm: float = 0.5, n_epochs: int = 4, batch_size: int = 256, verbose: int = 0):
        super().__init__(verbose)
        self.icm = icm_module
        self.lambda_icm = lambda_icm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment.
        """
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

            # --- CORRECTED: Reward is now ONLY from the forward model loss ---
            # We call compute_forward_loss to get the normalized prediction error.
            # We no longer need to call compute_inverse_loss here.
            norm_fwd_loss, _ = self.icm.compute_forward_loss(flat_obs, flat_action, flat_next_obs, update_ema=True)
            icm_reward = norm_fwd_loss
            icm_rewards.append(icm_reward)

        self.locals['rewards'] += self.lambda_icm * np.array(icm_rewards)
        
        return True

    def _on_rollout_end(self) -> None:
        """
        This method is called at the end of each rollout.
        Its primary role is to train the ICM model and log the training progress.
        """
        if self.verbose > 0:
            print("\n--- Rollout ended. Starting to train ICM model... ---")
        
        buffer = self.model.rollout_buffer
        
        # Slicing to get N-1 valid transitions
        obs_t_list = [
             {k: v[step, env_idx] for k, v in buffer.observations.items()}
            for env_idx in range(buffer.n_envs)
            for step in range(buffer.buffer_size -1)
        ]
        next_obs_t_list = [
             {k: v[step + 1, env_idx] for k, v in buffer.observations.items()}
            for env_idx in range(buffer.n_envs)
            for step in range(buffer.buffer_size -1)
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

        if self.verbose > 0:
            log_str = f"[ICM Training] Timestep: {self.num_timesteps} | "
            log_str += f"VAE(R/KL): {final_losses['vae_recon_loss']:.4f}/{final_losses['vae_kl_loss']:.4f} | "
            log_str += f"Fwd: {final_losses['forward_loss']:.4f} | "
            log_str += f"Inv(R/KL): {final_losses['inverse_recon_loss']:.4f}/{final_losses['inverse_kl_loss']:.4f}"
            print(log_str)
        
        for key, value in final_losses.items():
            self.logger.record(f'icm/{key}', value)