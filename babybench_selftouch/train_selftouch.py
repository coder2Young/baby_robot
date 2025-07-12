# babybench_selftouch/train_selftouch.py

import os
import argparse
import yaml

from stable_baselines3 import PPO
import sys
sys.path.append(".")
sys.path.append("..")
import babybench.utils as bb_utils

from babybench_selftouch.icm.icm_module import ICMModule
from babybench_selftouch.rewards import SoftmaxTouchReward

# --- 修改/新增的导入 ---
# 1. 导入修改后的、更简单的TouchRewardWrapper
from babybench_selftouch.selftouch_wrapper import TouchRewardWrapper 
# 2. 导入我们新建的ICMCallback
from babybench_selftouch.icm_callback import ICMCallback

def main():
    # ---- 1. 解析参数 (无变化) ----
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='babybench_selftouch/config_selftouch.yml', type=str)
    parser.add_argument('--train_for', default=1000000, type=int) # Increased timesteps for meaningful training
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # ---- 2. 初始化环境 (无变化) ----
    env = bb_utils.make_env(config)
    obs = env.reset()
    obs0 = obs[0]
    obs_dim = len(obs0['observation']) + len(obs0['touch'])
    action_dim = env.action_space.shape[0]
    num_parts = len(obs0['touch'])

    # ---- 3. 构建ICM、奖励管理器、wrapper和callback (核心修改部分) ----
    # 将Latent Dim设为64，Hidden Dim设为256
    icm = ICMModule(obs_dim=obs_dim, action_dim=action_dim, latent_dim=64, hidden_dim=256, lr=3e-4)
    reward_mod = SoftmaxTouchReward(num_parts=num_parts, tau=10.0, total_reward=2.0)

    # --- 修改部分 ---
    # 使用新的、简化的TouchRewardWrapper，它不再需要ICM模块
    wrapped_env = TouchRewardWrapper(env, reward_mod, lambda_touch=100, lambda_hand_touch=0.5)
    
    # --- 新增部分 ---
    # 实例化我们的新Callback，并将ICM模块传递给它
    # 这里的超参数可以根据需要进行调整
    icm_callback = ICMCallback(
        icm_module=icm, 
        lambda_icm=0.5, 
        n_epochs=8,         # 每个rollout结束后，训练ICM 8个epoch
        batch_size=256      # 每个epoch内，使用256的mini-batch大小
    )
    
    # wrapped_env.reset() is called inside PPO, so we don't need to call it here.

    # ---- 4. RL算法集成 (核心修改部分) ----
    # PPO的实例化保持不变
    model = PPO("MultiInputPolicy", wrapped_env, verbose=1, ent_coef=0.01)

    # --- 修改部分 ---
    # 在learn方法中传入我们自定义的callback
    model.learn(total_timesteps=args.train_for, callback=icm_callback)
    
    # ---- 5. 保存模型 (无变化) ----
    os.makedirs(config['save_dir'], exist_ok=True)
    model.save(os.path.join(config['save_dir'], "ppo_icm_model"))

if __name__ == '__main__':
    main()