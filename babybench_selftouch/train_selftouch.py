# babybench_selftouch/train_selftouch.py

import os
import argparse
import yaml
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env # Good practice to be explicit

import sys
sys.path.append(".")
sys.path.append("..")
import babybench.utils as bb_utils

from babybench_selftouch.icm.icm_module import ICMModule
from babybench_selftouch.rewards import SoftmaxTouchReward
from babybench_selftouch.selftouch_wrapper import TouchRewardWrapper 
from babybench_selftouch.icm_callback import ICMCallback

def main():
    # ---- 1. 解析参数 (无变化) ----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='babybench_selftouch/config_selftouch.yml', type=str)
    parser.add_argument('--train_for', default=500000, type=int)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ---- 2. 初始化环境 (结构优化) ----
    # We use a helper function to correctly pass arguments when creating the vectorized environment
    def make_custom_env():
        env = bb_utils.make_env(config)
        
        # Get necessary dimensions from the environment's observation space
        num_parts = env.observation_space['touch'].shape[0]

        # Create the reward module with part-specific tau
        tau_hands = 10.0
        tau_others = 1.3
        tau_vector = np.full(num_parts, tau_others, dtype=np.float32)
        hand_parts_indices = [13, 14, 15, 19, 20, 21]
        tau_vector[hand_parts_indices] = tau_hands
        reward_mod = SoftmaxTouchReward(num_parts=num_parts, tau=tau_vector, total_reward=2.0)
        
        # Wrap the environment
        # The lambda_touch here is the *initial* value. The callback will change it during training.
        wrapped_env = TouchRewardWrapper(
            env, 
            reward_mod,
            reward_window_duration=50,
            cooldown_period=100, # Using the refined cooldown mechanism
            lambda_touch=10.0,   # Start with a strong touch incentive
            lambda_hand_touch=50.0
        )
        return wrapped_env

    # Create the vectorized environment
    vec_env = make_vec_env(make_custom_env, n_envs=1)
    
    # Get dimensions from the now-wrapped env's spaces
    obs_space = vec_env.observation_space
    obs_dim = len(obs_space['observation'].low) + len(obs_space['touch'].low)
    action_dim = vec_env.action_space.shape[0]

    

    # ---- 3. 构建ICM和Callback ----
    icm = ICMModule(obs_dim=obs_dim, action_dim=action_dim, latent_dim=64, hidden_dim=512, lr=1e-3)
    
    # --- MODIFIED: Pass schedule parameters to the callback ---
    icm_callback = ICMCallback(
        icm_module=icm,
        total_training_steps=args.train_for,
        save_path=config['save_dir'],
        save_freq=20000,
        lambda_icm_schedule=(0.5, 50.0),    # Schedule for ICM reward: (start, end)
        lambda_touch_schedule=(50.0, 5.0), # Schedule for Touch reward: (start, end)
        n_epochs=8,
        batch_size=512,
        verbose=1
    )
    
    # ---- 4. RL算法集成 ----
    # Using the tuned PPO hyperparameters we discussed for better stability
    model = PPO("MultiInputPolicy", vec_env, verbose=1, ent_coef=0.05, vf_coef=1.0, gae_lambda=0.9)

    model.learn(total_timesteps=args.train_for, callback=icm_callback)
    
    # ---- 5. 保存最终模型 (无变化) ----
    print("\n--- Training finished. Saving final models... ---")
    final_step_count = args.train_for
    
    final_ppo_path = os.path.join(config['save_dir'], "ppo_model", f"{final_step_count}_steps_final")
    final_icm_path = os.path.join(config['save_dir'], "icm_model", f"{final_step_count}_steps_final")
    
    os.makedirs(final_ppo_path, exist_ok=True)
    os.makedirs(final_icm_path, exist_ok=True)
    
    model.save(os.path.join(final_ppo_path, "model.zip"))
    torch.save(icm.state_dict(), os.path.join(final_icm_path, "icm_model.pth"))

    print(f"Final PPO model saved to {final_ppo_path}")
    print(f"Final ICM model saved to {final_icm_path}")


if __name__ == '__main__':
    main()