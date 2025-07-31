# babybench_selftouch/train_selftouch.py

import os
import argparse
import yaml
import torch
import numpy as np
from stable_baselines3 import PPO
import mujoco

import sys
sys.path.append(".")
sys.path.append("..")
import babybench.utils as bb_utils
from babybench_selftouch.icm.icm_module import ICMModule
from babybench_selftouch.rewards import SoftmaxTouchReward
from babybench_selftouch.selftouch_wrapper import TouchRewardWrapper
from babybench_selftouch.icm_callback import ICMCallback
# --- MODIFIED: flatten_obs is no longer needed ---
# from babybench_selftouch.utils import flatten_obs

LAMBDA_ICM_SCHEDULE = (0.005, 1.1) # Best is 0.005, 0.1
LAMBDA_TOUCH_SCHEDULE = (10.0, 1.0)
LAMBDA_HAND_TOUCH_SCHEDULE = (100.0, 10.0)
DYNAMIC_WEIGHT_STOP_STEP = 1000000  # New parameter for dynamic weight adjustment

def main():
    """
    Main training script for self-touch with ICM, structured like official examples.
    """
    # === 1. 配置与参数解析 ===
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='babybench_selftouch/config_selftouch.yml', type=str)
    parser.add_argument('--train_for', default=4000000, type=int)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # === 2. 创建并包裹环境 (简化流程) ===
    env = bb_utils.make_env(config)

    # --- 【关键修正】: 训练脚本只负责找出“手部”的Body ID ---
    print("--- Automatically identifying hand BODY IDs ---")
    hand_body_ids = set()
    for body_id in range(env.model.nbody):
        body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        #if body_name and ('hand' in body_name or 'fingers' in body_name):
        if body_name and 'hand' in body_name:
            hand_body_ids.add(body_id)
    print(f"Identified hand body IDs: {hand_body_ids}")
    print("---------------------------------------------")

    reward_mod = SoftmaxTouchReward(num_parts=1, tau=15.0, total_reward=1)
    
    wrapped_env = TouchRewardWrapper(
        env, 
        reward_module=reward_mod,
        general_reward_window=40,
        general_cooldown_period=200,
        hand_reward_value=1,
        hand_reward_window=60,
        hand_cooldown_period=30,
        hand_overhold_threshold=300,
        hand_overhold_penalty=1,
        hand_body_ids=hand_body_ids 
    )
    
    # === 3. 计算真实维度并初始化ICM和Callback ===
    obs, _ = wrapped_env.reset()

    # --- MODIFIED: Get separate dimensions for each modality ---
    proprio_obs_dim = obs['observation'].shape[0]
    touch_obs_dim = obs['touch'].shape[0]
    action_dim = wrapped_env.action_space.shape[0]

    # --- MODIFIED: Instantiate ICMModule with the new signature ---
    icm = ICMModule(
        proprio_obs_dim=proprio_obs_dim,
        touch_obs_dim=touch_obs_dim,
        action_dim=action_dim,
        proprio_latent_dim=64,  # New hyperparameter
        touch_latent_dim=24,    # New hyperparameter
        hidden_dim=512,
        lr=1e-3,
        vae_beta=0.01,
    )

    # 初始化Callback (Callback's initialization remains unchanged)
    icm_callback = ICMCallback(
        icm_module=icm,
        total_training_steps=args.train_for,
        save_path=config['save_dir'],
        save_freq=20000,
        lambda_icm_schedule=LAMBDA_ICM_SCHEDULE,
        lambda_touch_schedule=LAMBDA_TOUCH_SCHEDULE,
        lambda_hand_touch_schedule=LAMBDA_HAND_TOUCH_SCHEDULE,
        dynamic_weight_stop_step=DYNAMIC_WEIGHT_STOP_STEP,  # New parameter for dynamic weight adjustment
        n_epochs=2,
        batch_size=512,
        verbose=2
    )

    tensorboard_log_path = os.path.join(config['save_dir'], "logs")
    # === 4. 初始化PPO模型 ===
    model = PPO(
        "MultiInputPolicy", 
        wrapped_env, 
        verbose=1,
        ent_coef=0.05,
        n_steps=1024,
        learning_rate=3e-4,
        tensorboard_log=tensorboard_log_path 
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