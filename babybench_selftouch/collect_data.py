import os
import argparse
import yaml
import numpy as np
import h5py
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer

import sys
sys.path.append(".")
sys.path.append("..")
import babybench.utils as bb_utils
from babybench_selftouch.rewards import SoftmaxTouchReward
from babybench_selftouch.selftouch_wrapper import TouchRewardWrapper


class SaveTransitionsCallback(BaseCallback):
    def __init__(self, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.h5_file = None

    def _on_step(self) -> bool:
        """
        This method is required by the BaseCallback abstract class.
        We don't need to do anything here, as our logic is in _on_rollout_end.
        Returning True ensures that training continues.
        """
        return True

    def _on_training_start(self) -> None:
        """Initializes the HDF5 file for data collection."""
        # From the training environment, we can get the observation space and action space dimensions
        obs_space = self.training_env.observation_space
        action_dim = self.model.action_space.shape[0]
        proprio_dim = obs_space['observation'].shape[0]
        touch_dim = obs_space['touch'].shape[0]
        
        self.h5_file = h5py.File(self.save_path, 'w')
        self.h5_file.create_dataset('proprio_obs', shape=(0, proprio_dim), maxshape=(None, proprio_dim), dtype='f4')
        self.h5_file.create_dataset('touch_obs', shape=(0, touch_dim), maxshape=(None, touch_dim), dtype='f4')
        self.h5_file.create_dataset('actions', shape=(0, action_dim), maxshape=(None, action_dim), dtype='f4')
        self.h5_file.create_dataset('next_proprio_obs', shape=(0, proprio_dim), maxshape=(None, proprio_dim), dtype='f4')
        self.h5_file.create_dataset('next_touch_obs', shape=(0, touch_dim), maxshape=(None, touch_dim), dtype='f4')
        
        if self.verbose > 0:
            print(f"HDF5 file created at {self.save_path}. Ready for data collection.")

    def _on_rollout_end(self) -> bool:
        """ Collects data from the rollout buffer and saves it to the HDF5 file."""
        buffer = self.model.rollout_buffer

        # Extract data
        num_samples = buffer.buffer_size * buffer.n_envs
        proprio_obs = buffer.observations['observation'].reshape(num_samples, -1)
        touch_obs = buffer.observations['touch'].reshape(num_samples, -1)
        actions = buffer.actions.reshape(num_samples, -1)
        
        next_proprio_obs = np.roll(proprio_obs, -1, axis=0)
        next_touch_obs = np.roll(touch_obs, -1, axis=0)

        # Remove the last invalid s_t+1
        proprio_obs = proprio_obs[:-1]
        touch_obs = touch_obs[:-1]
        actions = actions[:-1]
        next_proprio_obs = next_proprio_obs[:-1]
        next_touch_obs = next_touch_obs[:-1]

        # Incrementally write to the HDF5 file
        current_size = self.h5_file['actions'].shape[0]
        chunk_size = len(actions)
        
        for key, data_arr in zip(['proprio_obs', 'touch_obs', 'actions', 'next_proprio_obs', 'next_touch_obs'],
                                 [proprio_obs, touch_obs, actions, next_proprio_obs, next_touch_obs]):
            self.h5_file[key].resize(current_size + chunk_size, axis=0)
            self.h5_file[key][-chunk_size:] = data_arr

        if self.verbose > 0:
            print(f"  -> Saved {chunk_size} transitions. Total saved: {self.h5_file['actions'].shape[0]}")
            
        return True

    def _on_training_end(self) -> None:
        """Closes the HDF5 file at the end of training."""
        if self.h5_file:
            self.h5_file.close()
            print(f"\nData collection finished. HDF5 file closed at {self.save_path}.")


def main():
    parser = argparse.ArgumentParser(description="RL-Driven Data Collection Script (Final Version)")
    parser.add_argument('--config', default='babybench_selftouch/config_selftouch.yml', help='Path to the environment config file')
    parser.add_argument('--steps', default=200000, type=int, help='Total number of steps to run the agent')
    parser.add_argument('--out', default='data/rl_driven_data.h5', help='Path to save the output HDF5 file')
    parser.add_argument('--ent_coef', type=float, default=1.0, help='Entropy coefficient for PPO to encourage exploration')
    args = parser.parse_args()

    # 1. Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    env = bb_utils.make_env(config)
    
    num_total_parts = env.observation_space['touch'].shape[0]
    hand_parts_indices = {13, 14, 15, 19, 20, 21}
    body_parts_indices_list = sorted(list(set(range(num_total_parts)) - hand_parts_indices))
    num_body_parts = len(body_parts_indices_list)
    body_idx_map = {full_idx: reduced_idx for reduced_idx, full_idx in enumerate(body_parts_indices_list)}
    reward_mod = SoftmaxTouchReward(num_parts=num_body_parts, tau=10.0, total_reward=1)
    wrapped_env = TouchRewardWrapper(
        env,
        reward_module=reward_mod,
        body_idx_map=body_idx_map,
        general_reward_window=60,
        general_cooldown_period=600,
        hand_reward_value=1,
        hand_reward_window=120,
        hand_cooldown_period=30,
        hand_overhold_threshold=300,
        hand_overhold_penalty=1,
        lambda_touch=0.1,
        lambda_hand_touch=10.0,
    )

    # 2. Initialize the PPO model
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_callback = SaveTransitionsCallback(save_path=args.out, verbose=1)

    model = PPO(
        "MultiInputPolicy", 
        wrapped_env, 
        verbose=1,
        n_steps=2048,
        ent_coef=args.ent_coef,
        learning_rate=1e-3
    )

    # 3. Run the training and data collection process
    model.learn(total_timesteps=args.steps, callback=save_callback)
    
    env.close()

if __name__ == '__main__':
    main()