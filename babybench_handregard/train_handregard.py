# babybench_handregard/train_handregard.py

import os
import argparse
import yaml

from stable_baselines3 import PPO

import sys
sys.path.append(".")
sys.path.append("..")
import babybench.utils as bb_utils

from forward_model.forward_model import ForwardModel
from rewards import HandSaliencyReward
from handregard_wrapper import HandRegardRewardWrapper

import babybench.eval as bb_eval


def main():
    # ---- 1. 解析参数 ----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='babybench_handregard/config_handregard.yml', type=str)
    parser.add_argument('--train_for', default=100000, type=int)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    # ---- 2. 初始化环境 ----
    env = bb_utils.make_env(config)
    obs = env.reset()
    # BabyBench环境通常是vectorized env，这里用obs[0]作为第一个环境的观测
    obs0 = obs[0]

    # 打印结构（debug用）
    # print("Observation structure:", obs0.keys())
    # print("obs[observation]", "type", type(obs0['observation']),"shape", obs0['observation'].shape)
    # print("obs[eye_left]", "type", type(obs0['eye_left']),obs0['eye_left'].shape, "dtype", obs0['eye_left'].dtype)
    # print("obs[eye_right]", "type", type(obs0['eye_right']),obs0['eye_right'].shape, "dtype", obs0['eye_right'].dtype)

    obs_dim = len(obs0['observation'])    # 只用'observation'的长度
    action_dim = env.action_space.shape[0]
    img_shape = obs0['eye_left'].shape    # (64,64,3)，左右眼shape一致

    # ---- 3. 构建视觉正模型与saliency奖励 ----
    forward_model = ForwardModel(obs_dim=obs_dim, action_dim=action_dim, img_shape=img_shape)
    hand_reward_mod = HandSaliencyReward(use_both_eyes=True)
    # 注意：lambda_pred和lambda_sal是权重系数，调节预测误差和saliency奖励的影响
    # Predicted Reward: 4359.6851, Saliency Reward: 0.1498
    # Weighted Predicted Reward: -4359.6851, Weighted Saliency Reward: 14975.5957 for lambda_pred=-1.0, lambda_sal=1e5
    wrapped_env = HandRegardRewardWrapper(env, forward_model, hand_reward_mod, lambda_pred=-1e-4, lambda_sal=1e2)
    wrapped_env.reset()

    # ---- 4. RL算法集成 ----
    model = PPO("MultiInputPolicy", wrapped_env, verbose=1, ent_coef=0.05)
    model.learn(total_timesteps=args.train_for)

    # ---- 5. 保存模型 ----
    os.makedirs(config['save_dir'], exist_ok=True)
    model.save(os.path.join(config['save_dir'], "model"))

if __name__ == '__main__':
    main()
