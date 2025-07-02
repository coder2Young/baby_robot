# babybench_icm/train_icm.py

import os
import argparse
import yaml

from stable_baselines3 import PPO, SAC

import sys
sys.path.append(".")
sys.path.append("..")
import babybench.utils as bb_utils

from icm.icm_module import ICMModule
from rewards import SoftmaxTouchReward
from icm_wrapper import ICMRewardWrapper

def main():
    # ---- 1. 解析参数 ----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='babybench_icm/config_icm.yml', type=str)
    parser.add_argument('--train_for', default=10000, type=int)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    # ---- 2. 初始化环境 ----
    env = bb_utils.make_env(config)
    obs = env.reset()
    # BabyBench环境通常是vectorized env，这里用obs[0]作为第一个环境的观测
    obs0 = obs[0]

    obs_dim = len(obs0['observation']) + len(obs0['touch'])  # 观测维度 = 观测 + 触摸部位
    action_dim = env.action_space.shape[0]
    num_parts = len(obs0['touch'])

    # ---- 3. 构建ICM、奖励管理器、wrapper ----
    icm = ICMModule(obs_dim=obs_dim, action_dim=action_dim, latent_dim=8, hidden_dim=256)
    # 修改tau，鼓励触摸部位的多样性
    reward_mod = SoftmaxTouchReward(num_parts=num_parts, tau=10.0, total_reward=2.0)
    wrapped_env = ICMRewardWrapper(env, icm, reward_mod, lambda_icm=0.5, lambda_touch=0.5)
    wrapped_env.reset()

    # ---- 4. RL算法集成，直接用PPO训练 ----
    model = PPO("MultiInputPolicy", wrapped_env, verbose=1, ent_coef=0.05)

    model.learn(total_timesteps=args.train_for)
    
    # ---- 5. 保存模型 ----
    os.makedirs(config['save_dir'], exist_ok=True)
    model.save(os.path.join(config['save_dir'], "model"))

if __name__ == '__main__':
    main()
