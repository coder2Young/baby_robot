# babybench_selftouch/icm_callback.py

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
    """
    def __init__(self, icm_module: ICMModule, lambda_icm: float = 0.5, n_epochs: int = 4, batch_size: int = 256, verbose: int = 0):
        super().__init__(verbose)
        self.icm = icm_module
        self.lambda_icm = lambda_icm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        # The self.last_obs attribute is no longer needed.

    # The _on_rollout_start method is no longer needed.

    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment.
        Its primary role is to calculate the curiosity reward and add it to the reward buffer.
        """
        # --- NEW: Get s_t directly from the rollout buffer ---
        # The observation that was used to take the current action is stored
        # in the buffer at the current position.
        current_pos = self.model.rollout_buffer.pos
        last_obs_dict = {
            key: obs_array[current_pos] 
            for key, obs_array in self.model.rollout_buffer.observations.items()
        }
        
        # Get the action and resulting next state from the callback's locals
        actions = self.locals['actions']
        new_obs_dict = self.locals['new_obs']

        # The logic is the same, but now we get s_t from a reliable source.
        # This handles VecEnv correctly since SB3 stores obs in a (n_envs, *obs_shape) array at each buffer step.
        icm_rewards = []
        for i in range(self.training_env.num_envs):
            last_obs_single = {k: v[i] for k, v in last_obs_dict.items()}
            action_single = actions[i]
            new_obs_single = {k: v[i] for k, v in new_obs_dict.items()}
            
            flat_obs = torchify(flatten_obs(last_obs_single), self.icm.device)
            flat_action = torchify(action_single, self.icm.device)
            flat_next_obs = torchify(flatten_obs(new_obs_single), self.icm.device)

            norm_fwd_loss, _ = self.icm.compute_forward_loss(flat_obs, flat_action, flat_next_obs, update_ema=False)
            norm_inv_loss, _ = self.icm.compute_inverse_loss(flat_obs, flat_next_obs, flat_action, update_ema=False)
            
            icm_reward = (norm_fwd_loss + norm_inv_loss) / 2.0
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
        
        batch_size_total = buffer.buffer_size * buffer.n_envs
        
        obs_list = [
            {k: v[step, env_idx] for k, v in buffer.observations.items()}
            for env_idx in range(buffer.n_envs)
            for step in range(buffer.buffer_size)
        ]
        
        next_obs_list = [
            {k: v[step + 1, env_idx] for k, v in buffer.observations.items()}
            for env_idx in range(buffer.n_envs)
            for step in range(buffer.buffer_size - 1)
        ]
        # We need to correctly handle the last `next_obs`
        for env_idx in range(buffer.n_envs):
            last_obs_of_rollout = {k: v[-1, env_idx] for k, v in buffer.observations.items()}
            next_obs_list.append(last_obs_of_rollout) # Placeholder, ideally get from `infos` if available

        # Correct action reshaping
        actions_reshaped = np.array([buffer.actions[step, env_idx] 
                                     for env_idx in range(buffer.n_envs) 
                                     for step in range(buffer.buffer_size)])

        flat_obs_batch = np.array([flatten_obs(obs) for obs in obs_list])
        # The length of next_obs_list will be different, so we slice obs and actions
        flat_next_obs_batch = np.array([flatten_obs(obs) for obs in next_obs_list])

        obs_tensor = torch.from_numpy(flat_obs_batch).float().to(self.icm.device)
        action_tensor = torch.from_numpy(actions_reshaped).float().to(self.icm.device)
        next_obs_tensor = torch.from_numpy(flat_next_obs_batch).float().to(self.icm.device)
        
        # --- MODIFIED PART ---
        # Call the ICM module's training method, which now returns the losses
        final_losses = self.icm.train_on_batch(
            obs_tensor, 
            action_tensor, 
            next_obs_tensor,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size
        )

        # --- NEW PART: Logging ---
        # Log to console
        if self.verbose > 0:
            log_str = f"[ICM Training] Timestep: {self.num_timesteps} | "
            log_str += f"VAE(R/KL): {final_losses['vae_recon_loss']:.4f}/{final_losses['vae_kl_loss']:.4f} | "
            log_str += f"Fwd: {final_losses['forward_loss']:.4f} | "
            log_str += f"Inv(R/KL): {final_losses['inverse_recon_loss']:.4f}/{final_losses['inverse_kl_loss']:.4f}"
            print(log_str)
        
        # Log to TensorBoard (or other loggers)
        for key, value in final_losses.items():
            self.logger.record(f'icm/{key}', value)