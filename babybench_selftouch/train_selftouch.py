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
# from babybench_selftouch.utils import flatten_obs

LAMBDA_ICM_SCHEDULE = (0.005, 0.1)
LAMBDA_TOUCH_SCHEDULE = (10.0, 2.5)
LAMBDA_HAND_TOUCH_SCHEDULE = (80.0, 8.0)
DYNAMIC_WEIGHT_STOP_STEP = 1000000
SEED = 42

def main():
    """
    Main training script for self-touch with ICM, structured like official examples.
    """
    # 1. Argument parsing and configuration loading
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='babybench_selftouch/config_selftouch.yml', type=str)
    parser.add_argument('--train_for', default=2000000, type=int) # Baseline is 4M steps
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 2. Create and wrap environment
    env = bb_utils.make_env(config)

    hand_body_ids = set()
    for body_id in range(env.model.nbody):
        body_name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        #if body_name and ('hand' in body_name or 'fingers' in body_name):
        if body_name and 'hand' in body_name:
            hand_body_ids.add(body_id)

    reward_mod = SoftmaxTouchReward(num_parts=1, tau=20.0, total_reward=1)
    
    wrapped_env = TouchRewardWrapper(
        env, 
        reward_module=reward_mod,
        general_reward_window=80,
        general_cooldown_period=200,
        hand_reward_value=1,
        hand_reward_window=60,
        hand_cooldown_period=30,
        hand_overhold_threshold=300,
        hand_overhold_penalty=1,
        hand_body_ids=hand_body_ids 
    )
    
    # 3. Reset the environment to get initial observations
    obs, _ = wrapped_env.reset()
    proprio_obs_dim = obs['observation'].shape[0]
    touch_obs_dim = obs['touch'].shape[0]
    action_dim = wrapped_env.action_space.shape[0]

    # 4. Initialize ICMModule
    icm = ICMModule(
        proprio_obs_dim=proprio_obs_dim,
        touch_obs_dim=touch_obs_dim,
        action_dim=action_dim,
        proprio_latent_dim=64,
        touch_latent_dim=24,
        hidden_dim=512,
        lr=3e-4,
        vae_beta=0.01,
    )

    # 5. ICMCallback setup
    icm_callback = ICMCallback(
        icm_module=icm,
        total_training_steps=args.train_for,
        save_path=config['save_dir'],
        save_freq=4096,
        lambda_icm_schedule=LAMBDA_ICM_SCHEDULE,
        lambda_touch_schedule=LAMBDA_TOUCH_SCHEDULE,
        lambda_hand_touch_schedule=LAMBDA_HAND_TOUCH_SCHEDULE,
        dynamic_weight_stop_step=DYNAMIC_WEIGHT_STOP_STEP,  # New parameter for dynamic weight adjustment
        n_epochs=2,
        batch_size=256,
        verbose=2
    )

    # 6. Initialize PPO model
    tensorboard_log_path = os.path.join(config['save_dir'], "logs")
    model = PPO(
        "MultiInputPolicy", 
        wrapped_env, 
        verbose=1,
        ent_coef=3e-5,
        n_steps=1024 * 4,
        learning_rate=1e-4,
        tensorboard_log=tensorboard_log_path,
        seed=SEED
    )
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print("seed used during training:", model.seed)

    # 7. Train the model with ICM
    model.learn(total_timesteps=args.train_for, callback=icm_callback)

if __name__ == '__main__':
    main()