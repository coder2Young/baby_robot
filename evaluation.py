import numpy as np
import os
import gymnasium as gym
import time
import argparse
import mujoco
import yaml

import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
import mimoEnv.utils as env_utils
import babybench.utils as bb_utils
import babybench.eval as bb_eval

from stable_baselines3 import PPO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='examples/config_handregard.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--render', default=True, type=bool,
                        help='Renders a video for each episode during the evaluation.')
    parser.add_argument('--duration', default=10000, type=int,
                        help='Total timesteps per evaluation episode')
    parser.add_argument('--episodes', default=10, type=int,
                        help='Number of evaluation episode')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    env = bb_utils.make_env(config, training=False)
    obs, _ = env.reset()

    evaluation = bb_eval.EVALS[config['behavior']](
        env=env,
        duration=args.duration,
        render=args.render,
        save_dir=config['save_dir'],
    )

    # Preview evaluation of training log
    evaluation.eval_logs()

    # ====== 加载训练好的模型 ======
    model_path = os.path.join(config['save_dir'], "model")  # 注意模型名
    model = PPO.load(model_path, env=env)

    for ep_idx in range(args.episodes):
        print(f'Running evaluation episode {ep_idx+1}/{args.episodes}')

        # Reset environment and evaluation
        obs, _ = env.reset()
        evaluation.reset()

        for t_idx in range(args.duration):

            # Select action
            action, _ = model.predict(obs, deterministic=True)

            # Perform step in simulation
            obs, _, _, _, info = env.step(action)

            # Perform evaluations of step
            evaluation.eval_step(info)
            
        evaluation.end(episode=ep_idx)

if __name__ == '__main__':
    main()
