# babybench_handregard/train_handregard.py

import os
import argparse
import yaml

from stable_baselines3 import PPO

import sys
sys.path.append(".")
sys.path.append("..")
import babybench.utils as bb_utils

from multimodal_ae import MultimodalAEManager
from rewards import HandSaliencyReward, HandSkinColorReward
from handregard_wrapper import HandRegardRewardWrapper

import babybench.eval as bb_eval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='babybench_handregard/config_handregard.yml', type=str)
    parser.add_argument('--train_for', default=10000, type=int)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    env = bb_utils.make_env(config, training=True)
    obs = env.reset()
    obs0 = obs[0]

    img_shape = obs0['eye_left'].shape    # (64,64,3)
    proprio_dim = len(obs0['observation'])
    multimodal_ae = MultimodalAEManager(img_shape=img_shape[:2], proprio_dim=proprio_dim, device='cuda' if hasattr(obs0['eye_left'], 'cuda') else 'cpu')

    hand_saliency_reward_mod = HandSaliencyReward(use_both_eyes=True)
    hand_skin_color_reward_mod = HandSkinColorReward()

    wrapped_env = HandRegardRewardWrapper(
        env,
        multimodal_ae,
        hand_saliency_reward_mod,
        hand_skin_color_reward_mod,
        lambda_v=1.0, lambda_p=0.2, lambda_m=1.0,
        lambda_sal=20.0, lambda_skin=10.0,
        motion_bonus_scale=1.0, decay_steps=10000
    )
    wrapped_env.reset()
    model = PPO("MultiInputPolicy", wrapped_env, verbose=1, ent_coef=0.01, learning_rate=3e-4)
    model.learn(total_timesteps=args.train_for)

    os.makedirs(config['save_dir'], exist_ok=True)
    model.save(os.path.join(config['save_dir'], "model"))

if __name__ == '__main__':
    main()
