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

DEFAULT_STEPS = 50000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='babybench_selftouch/config_selftouch.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--render', default=True, type=bool,
                        help='Renders a video for each episode during the evaluation.')
    parser.add_argument('--duration', default=1000, type=int,
                        help='Total timesteps per evaluation episode')
    parser.add_argument('--episodes', default=1, type=int,
                        help='Number of evaluation episodes')
    parser.add_argument('--trained_steps', default=DEFAULT_STEPS, type=int,
                        help='The training timestep of the model checkpoint to evaluate (e.g., 100000)')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # --- Path Construction (remains the same) ---
    base_save_dir = config['save_dir']
    model_path = os.path.join(base_save_dir, "ppo_model", f"{args.trained_steps}_steps", "model.zip")
    eval_output_path = os.path.join(base_save_dir, "evaluation", f"{args.trained_steps}_steps")

    # --- NEW: Explicitly create the output directories for the evaluation ---
    # This call fixes the FileNotFoundError.
    bb_utils.make_save_dirs(eval_output_path)
    
    print(f"=================================================")
    print(f"Starting evaluation for model trained for {args.trained_steps} steps.")
    print(f"Loading PPO model from: {model_path}")
    print(f"Saving new evaluation results to: {eval_output_path}")
    print(f"Reading original training logs from: {base_save_dir}")
    print(f"=================================================")

    # Create the environment for evaluation
    env = bb_utils.make_env(config, training=False)
    obs, _ = env.reset()

    # --- MODIFIED: Pass both paths to the Eval object ---
    evaluation = bb_eval.EVALS[config['behavior']](
        env=env,
        duration=args.duration,
        render=args.render,
        save_dir=eval_output_path,             # Where to SAVE new eval files
        training_log_dir=base_save_dir        # Where to READ old training logs from
    )

    # This will now correctly find the training.pkl file.
    evaluation.eval_logs() 

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure you have a trained model checkpoint at that location.")
        return

    model = PPO.load(model_path, env=env)

    # The evaluation loop remains the same
    for ep_idx in range(args.episodes):
        print(f'Running evaluation episode {ep_idx+1}/{args.episodes}')
        obs, _ = env.reset()
        evaluation.reset()

        for t_idx in range(args.duration):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _, info = env.step(action)
            evaluation.eval_step(info)
            
        # This will now correctly save logs and videos to the new directories.
        evaluation.end(episode=ep_idx)

    print(f"Evaluation for model at {args.trained_steps} steps complete.")

if __name__ == '__main__':
    main()