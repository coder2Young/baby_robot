# babybench_selftouch/train_selftouch.py

import os
import argparse
import yaml
import torch
import numpy as np
from stable_baselines3 import PPO

# --- 项目模块导入 ---
import sys
sys.path.append(".")
sys.path.append("..")
import babybench.utils as bb_utils
from babybench_selftouch.icm.icm_module import ICMModule
from babybench_selftouch.rewards import SoftmaxTouchReward
from babybench_selftouch.selftouch_wrapper import TouchRewardWrapper
from babybench_selftouch.icm_callback import ICMCallback
from babybench_selftouch.utils import flatten_obs # 假设您已采纳重构建议

def main():
    """
    Main training script for self-touch with ICM, structured like official examples.
    """
    # === 1. 配置与参数解析 ===
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='babybench_selftouch/config_selftouch.yml', type=str)
    parser.add_argument('--train_for', default=500000, type=int)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # === 2. 创建并包裹环境 (简化流程) ===
    # 直接创建基础环境，不再使用额外的函数包裹
    env = bb_utils.make_env(config)
    
    # --- 准备多样性奖励模块 (SoftmaxTouchReward) ---
    num_total_parts = env.observation_space['touch'].shape[0]
    hand_parts_indices = {13, 14, 15, 19, 20, 21}
    
    # 计算非手部身体部位的索引和数量
    body_parts_indices_list = sorted(list(set(range(num_total_parts)) - hand_parts_indices))
    num_body_parts = len(body_parts_indices_list)
    body_idx_map = {full_idx: reduced_idx for reduced_idx, full_idx in enumerate(body_parts_indices_list)}

    # 初始化多样性奖励模块，它只管理非手部
    reward_mod = SoftmaxTouchReward(num_parts=num_body_parts, tau=1.3, total_reward=2.0)
    
    # --- 将所有功能包裹到最终的环境中 ---
    wrapped_env = TouchRewardWrapper(
        env, 
        reward_module=reward_mod,
        body_idx_map=body_idx_map,
        general_reward_window=80,
        general_cooldown_period=100,
        hand_reward_window=200,
        hand_cooldown_period=40,
        lambda_touch=10.0,
        lambda_hand_touch=20.0
    )

    # === 3. 计算真实维度并初始化ICM和Callback ===
    # 通过一次reset获取真实观测样本，确保维度准确无误
    obs, _ = wrapped_env.reset()
    actual_obs_dim = flatten_obs(obs).shape[0]
    action_dim = wrapped_env.action_space.shape[0]

    # 初始化ICM模块
    icm = ICMModule(obs_dim=actual_obs_dim, action_dim=action_dim, latent_dim=64, hidden_dim=512, lr=1e-3)

    # 初始化Callback，传入所有调度参数
    icm_callback = ICMCallback(
        icm_module=icm,
        total_training_steps=args.train_for,
        save_path=config['save_dir'],
        save_freq=50000,
        lambda_icm_schedule=(5.0, 50.0),
        lambda_touch_schedule=(10.0, 1.0),
        lambda_hand_touch_schedule=(20.0, 10.0),
        n_epochs=8,
        batch_size=512,
        verbose=1
    )

    # === 4. 初始化PPO模型 ===
    # 直接将单一的、完整包裹的环境传递给PPO
    model = PPO(
        "MultiInputPolicy", 
        wrapped_env, 
        verbose=1,
        ent_coef=0.05,
        vf_coef=1.0,
        gae_lambda=0.95
    )

    # === 5. 开始训练 ===
    model.learn(total_timesteps=args.train_for, callback=icm_callback)
    
    # === 6. 保存最终模型 ===
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