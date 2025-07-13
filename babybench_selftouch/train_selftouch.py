# babybench_selftouch/train_selftouch.py

import os
import argparse
import yaml
import torch

from stable_baselines3 import PPO
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
    parser.add_argument('--train_for', default=300000, type=int)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ---- 2. 初始化环境 (无变化) ----
    env = bb_utils.make_env(config)
    obs, _ = env.reset()
    obs0 = obs
    obs_dim = len(obs0['observation']) + len(obs0['touch'])
    action_dim = env.action_space.shape[0]
    num_parts = len(obs0['touch'])

    # ---- 3. 构建ICM、奖励管理器、wrapper和callback ----
    icm = ICMModule(obs_dim=obs_dim, action_dim=action_dim, latent_dim=64, hidden_dim=512, lr=1e-4)
    reward_mod = SoftmaxTouchReward(num_parts=num_parts, tau=10.0, total_reward=2.0)
    
    # --- MODIFIED: Pass new arguments to the wrapper ---
    wrapped_env = TouchRewardWrapper(
        env, 
        reward_mod, 
        #num_parts=num_parts,      # Pass the number of touchable parts
        cooldown_period=50,       # Set the refractory period to 50 steps
        lambda_touch=100, 
        lambda_hand_touch=0.5
    )
    
    # Callback instantiation remains the same
    icm_callback = ICMCallback(
        icm_module=icm, 
        save_path=config['save_dir'],
        save_freq=50000,
        lambda_icm=1.0,
        n_epochs=4,
        batch_size=256,
        verbose=0
    )
    
    # ---- 4. RL算法集成 (无变化) ----
    model = PPO("MultiInputPolicy", wrapped_env, verbose=1, ent_coef=0.01)

    # learn() call remains the same
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