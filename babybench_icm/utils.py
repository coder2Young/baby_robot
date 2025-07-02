# babybench_icm/utils.py

import babybench.utils as bb_utils
from babybench_icm.icm.icm_module import ICMModule
from babybench_icm.rewards import SoftmaxTouchReward
from babybench_icm.icm_wrapper import ICMRewardWrapper
import numpy as np

def make_icm_env(config, icm_kwargs=None, reward_kwargs=None, wrapper_kwargs=None, training=True):
    """
    初始化BabyBench官方环境，并包裹ICM+奖励Wrapper，统一接口
    :param config: yaml加载的dict配置
    :param icm_kwargs: dict, 传递给ICMModule的参数（可选，自动推断obs_dim、action_dim）
    :param reward_kwargs: dict, 传递给SoftmaxTouchReward的参数
    :param wrapper_kwargs: dict, 传递给ICMRewardWrapper的参数
    :param training: 是否训练模式（官方make_env参数）
    :return: 完整包装的env，直接可被PPO/DDPG等RL库和评测代码使用
    """
    # 1. 创建原始环境（支持vectorized、dict obs）
    env = bb_utils.make_env(config, training=training)
    obs = env.reset()
    # 兼容单环境、多环境（vector env）输出
    obs0 = obs[0] if isinstance(obs, (list, tuple, np.ndarray)) else obs

    # 2. 自动推断输入维度
    obs_dim = len(obs0['observation']) + len(obs0['touch'])
    action_dim = env.action_space.shape[0]
    num_parts = len(obs0['touch'])

    # 3. 初始化ICM、奖励管理器（可自定义参数）
    icm_params = dict(obs_dim=obs_dim, action_dim=action_dim, latent_dim=8, hidden_dim=256)
    reward_params = dict(num_parts=num_parts, tau=2.0, total_reward=2.0)
    wrapper_params = dict(lambda_icm=0.5, lambda_touch=0.5)

    if icm_kwargs:
        icm_params.update(icm_kwargs)
    if reward_kwargs:
        reward_params.update(reward_kwargs)
    if wrapper_kwargs:
        wrapper_params.update(wrapper_kwargs)

    icm = ICMModule(**icm_params)
    reward_mod = SoftmaxTouchReward(**reward_params)
    wrapped_env = ICMRewardWrapper(env, icm, reward_mod, **wrapper_params)
    wrapped_env.reset()
    return wrapped_env

