# -*- coding: utf-8 -*-
"""
Publication-Quality Time-Series Plot Generator (V1)

This script reads various metrics from a TensorBoard log directory,
processes the data, and generates two multi-panel, publication-quality
figures visualizing the evolution of agent behavior and internal model learning.

Each plot includes both the raw data and an EMA-smoothed trend line.

Author: Gemini
Date: 2025-07-30
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator
import scienceplots # 使用一个流行的学术绘图风格库
import math

# --- 1. 配置参数 (CONFIGURATION PARAMETERS) ---

# TODO: 请将此路径修改为您实际的TensorBoard日志目录
LOG_DIR = './results/self_touch/logs/PPO_7' 

# EMA 平滑系数 (alpha). 0.99的平滑度对应alpha为0.01
EMA_ALPHA = 0.01

# 为各部位的详细图表创建一个单独的输出文件夹
OUTPUT_DIR_PARTS = './timeseries_per_part/'

# 输出图像文件名
OUTPUT_KEY_METRICS_FIGURE = 'timeseries_key_metrics.png'
OUTPUT_LOSS_FIGURE = 'timeseries_loss_evolution.png'

# --- 2. 核心功能函数 (CORE FUNCTIONS) ---
# 为各部位的详细图表创建一个单独的输出文件夹
OUTPUT_DIR_PARTS = './timeseries_per_part/'

# 输出图像文件名
OUTPUT_KEY_METRICS_FIGURE = 'timeseries_key_metrics.png'
OUTPUT_LOSS_FIGURE = 'timeseries_loss_evolution.png'
OUTPUT_REWARD_FIGURE = 'timeseries_reward_evolution.png' # 【新功能】: 新增奖励图文件名

# --- 2. 核心功能函数 (CORE FUNCTIONS) ---

def parse_all_scalars(log_dir):
    """
    从TensorBoard日志目录中解析出所有标量数据。
    包含数据清洗步骤，处理重复的时间步。
    """
    print(f"正在从 '{log_dir}' 读取所有TensorBoard标量数据...")
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"错误: 日志目录 '{log_dir}' 不存在。请检查路径。")
    
    ea = event_accumulator.EventAccumulator(log_dir,
        size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()

    scalar_data = defaultdict(pd.Series)
    tags = ea.Tags()['scalars']
    for tag in tags:
        events = ea.Scalars(tag)
        steps = [event.step for event in events]
        values = [event.value for event in events]
        
        s = pd.Series(values, index=steps)
        
        if s.index.has_duplicates:
            s = s.groupby(s.index).mean()
        
        scalar_data[tag] = s.sort_index()

    print(f"成功解析并清洗了 {len(scalar_data)} 个标量系列。")
    return scalar_data

def plot_key_metrics(scalar_data, alpha):
    """
    绘制关键宏观指标的演化图：多样性、总频率、总时长。
    """
    print("正在生成关键指标演化图...")
    plt.style.use(['science', 'notebook', 'grid'])
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, dpi=200)
    fig.suptitle('Evolution of Key Exploratory Metrics', fontsize=16, fontweight='bold')

    # Panel (a): Touch Diversity
    ax = axes[0]
    diversity_raw = scalar_data.get('behavior/touch_diversity_by_hand', pd.Series(dtype=np.float64))
    if not diversity_raw.empty:
        diversity_smooth = diversity_raw.ewm(alpha=alpha).mean()
        ax.plot(diversity_raw.index, diversity_raw.values, color='lightblue', alpha=0.4, label='Raw')
        ax.plot(diversity_smooth.index, diversity_smooth.values, color='dodgerblue', label=f'EMA (α={alpha})')
    ax.set_ylabel('Unique Body Parts Touched')
    ax.set_title('Panel (a): Touch Diversity')
    ax.legend()
    ax.set_ylim(bottom=0)

    # 采用更稳健的求和方法
    def robust_sum_series(series_list):
        if not series_list:
            return pd.Series(dtype=np.float64)
        long_series = pd.concat(series_list)
        total_series = long_series.groupby(long_series.index).sum()
        return total_series.sort_index()

    # Panel (b): Total Touch Frequency
    ax = axes[1]
    freq_series_list = [scalar_data[tag] for tag in scalar_data if tag.startswith('behavior_freq/')]
    total_freq_raw = robust_sum_series(freq_series_list)
    
    if not total_freq_raw.empty:
        total_freq_smooth = total_freq_raw.ewm(alpha=alpha).mean()
        ax.plot(total_freq_raw.index, total_freq_raw.values, color='lightcoral', alpha=0.4, label='Raw')
        ax.plot(total_freq_smooth.index, total_freq_smooth.values, color='crimson', label=f'EMA (α={alpha})')
    ax.set_ylabel('Total Touch Frequency')
    ax.set_title('Panel (b): Total Touch Frequency')
    ax.legend()
    ax.set_ylim(bottom=0)

    # Panel (c): Total Touch Duration
    ax = axes[2]
    duration_series_list = [scalar_data[tag] for tag in scalar_data if tag.startswith('behavior_duration/')]
    total_duration_raw = robust_sum_series(duration_series_list)

    if not total_duration_raw.empty:
        total_duration_smooth = total_duration_raw.ewm(alpha=alpha).mean()
        ax.plot(total_duration_raw.index, total_duration_raw.values, color='lightgreen', alpha=0.4, label='Raw')
        ax.plot(total_duration_smooth.index, total_duration_smooth.values, color='forestgreen', label=f'EMA (α={alpha})')
    ax.set_ylabel('Total Touch Duration (steps)')
    ax.set_title('Panel (c): Total Touch Duration')
    ax.legend()
    ax.set_ylim(bottom=0)

    axes[-1].set_xlabel('Training Steps')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_KEY_METRICS_FIGURE, bbox_inches='tight')
    print(f"关键指标演化图已成功保存至 '{OUTPUT_KEY_METRICS_FIGURE}'")
    plt.close(fig)

def plot_detailed_behavior_per_part(scalar_data, alpha, output_dir):
    """
    为每个身体部位创建独立的、包含频率和时长对比的图表。
    """
    print(f"正在为每个身体部位生成详细的行为演化图...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    freq_parts = {tag.replace('behavior_freq/', '') for tag in scalar_data if tag.startswith('behavior_freq/')}
    duration_parts = {tag.replace('behavior_duration/', '') for tag in scalar_data if tag.startswith('behavior_duration/')}
    all_parts = sorted(list(freq_parts.union(duration_parts)))

    for part_name in all_parts:
        plt.style.use(['science', 'notebook', 'grid'])
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, dpi=150)
        fig.suptitle(f'Behavioral Evolution for: {part_name}', fontsize=14, fontweight='bold')

        # Panel (a): Frequency
        ax = axes[0]
        freq_tag = f'behavior_freq/{part_name}'
        freq_raw = scalar_data.get(freq_tag, pd.Series(dtype=np.float64))
        if not freq_raw.empty:
            freq_smooth = freq_raw.ewm(alpha=alpha).mean()
            ax.plot(freq_raw.index, freq_raw.values, color='lightcoral', alpha=0.4, label='Raw')
            ax.plot(freq_smooth.index, freq_smooth.values, color='crimson', label=f'EMA (α={alpha})')
        ax.set_ylabel('Frequency')
        ax.set_title('Panel (a): Touch Frequency')
        ax.legend()
        ax.set_ylim(bottom=0)

        # Panel (b): Duration
        ax = axes[1]
        duration_tag = f'behavior_duration/{part_name}'
        duration_raw = scalar_data.get(duration_tag, pd.Series(dtype=np.float64))
        if not duration_raw.empty:
            duration_smooth = duration_raw.ewm(alpha=alpha).mean()
            ax.plot(duration_raw.index, duration_raw.values, color='lightgreen', alpha=0.4, label='Raw')
            ax.plot(duration_smooth.index, duration_smooth.values, color='forestgreen', label=f'EMA (α={alpha})')
        ax.set_ylabel('Duration (steps)')
        ax.set_title('Panel (b): Touch Duration')
        ax.legend()
        ax.set_ylim(bottom=0)

        axes[-1].set_xlabel('Training Steps')
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        output_filename = os.path.join(output_dir, f'behavior_{part_name}.png')
        fig.savefig(output_filename, bbox_inches='tight')
        plt.close(fig)
    
    print(f"已在 '{output_dir}' 目录中为 {len(all_parts)} 个身体部位生成了详细图表。")


def plot_loss_curves(scalar_data, alpha):
    """绘制模型损失的演化图。"""
    print("正在生成模型损失演化图...")
    plt.style.use(['science', 'notebook', 'grid'])
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=200)
    fig.suptitle('Evolution of Internal Model Losses', fontsize=16, fontweight='bold')

    # Panel (a): VAE Reconstruction Losses
    ax = axes[0]
    prop_loss_raw = scalar_data.get('icm/proprio_vae_recon_loss', pd.Series(dtype=np.float64))
    touch_loss_raw = scalar_data.get('icm/touch_vae_recon_loss', pd.Series(dtype=np.float64))
    if not prop_loss_raw.empty:
        prop_loss_smooth = prop_loss_raw.ewm(alpha=alpha).mean()
        ax.plot(prop_loss_raw.index, prop_loss_raw.values, color='mediumpurple', alpha=0.3)
        ax.plot(prop_loss_smooth.index, prop_loss_smooth.values, color='indigo', label='Proprioception VAE Loss (EMA)')
    if not touch_loss_raw.empty:
        touch_loss_smooth = touch_loss_raw.ewm(alpha=alpha).mean()
        ax.plot(touch_loss_raw.index, touch_loss_raw.values, color='sandybrown', alpha=0.3)
        ax.plot(touch_loss_smooth.index, touch_loss_smooth.values, color='saddlebrown', label='Touch VAE Loss (EMA)')
    ax.set_ylabel('Reconstruction Loss')
    ax.set_title('Panel (a): VAE Reconstruction Losses')
    ax.legend()

    # Panel (b): Forward Model Prediction Loss
    ax = axes[1]
    fwd_loss_raw = scalar_data.get('icm/forward_loss', pd.Series(dtype=np.float64))
    if not fwd_loss_raw.empty:
        fwd_loss_smooth = fwd_loss_raw.ewm(alpha=alpha).mean()
        ax.plot(fwd_loss_raw.index, fwd_loss_raw.values, color='lightskyblue', alpha=0.3, label='Raw')
        ax.plot(fwd_loss_smooth.index, fwd_loss_smooth.values, color='deepskyblue', label=f'EMA (α={alpha})')
    ax.set_ylabel('Prediction Loss')
    ax.set_title('Panel (b): Forward Model Prediction Loss (Curiosity Source)')
    ax.legend()

    axes[-1].set_xlabel('Training Steps')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_LOSS_FIGURE, bbox_inches='tight')
    print(f"模型损失演化图已成功保存至 '{OUTPUT_LOSS_FIGURE}'")
    plt.close(fig)

def plot_reward_evolution(scalar_data, alpha):
    """
    【新功能】: 绘制加权奖励分量的演化图。
    """
    print("正在生成加权奖励演化图...")
    plt.style.use(['science', 'notebook', 'grid'])
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
    fig.suptitle('Evolution of Weighted Reward Components', fontsize=16, fontweight='bold')

    # ICM Reward
    icm_reward_raw = scalar_data.get('reward_weighted/mean_icm', pd.Series(dtype=np.float64))
    if not icm_reward_raw.empty:
        icm_reward_smooth = icm_reward_raw.ewm(alpha=alpha).mean()
        ax.plot(icm_reward_raw.index, icm_reward_raw.values, color='cyan', alpha=0.3)
        ax.plot(icm_reward_smooth.index, icm_reward_smooth.values, color='blue', label='Weighted ICM Reward (EMA)')
        
    # Hand Touch Reward
    hand_reward_raw = scalar_data.get('reward_weighted/mean_hand', pd.Series(dtype=np.float64))
    if not hand_reward_raw.empty:
        hand_reward_smooth = hand_reward_raw.ewm(alpha=alpha).mean()
        ax.plot(hand_reward_raw.index, hand_reward_raw.values, color='magenta', alpha=0.3)
        ax.plot(hand_reward_smooth.index, hand_reward_smooth.values, color='purple', label='Weighted Hand Touch Reward (EMA)')

    ax.set_ylabel('Mean Reward per Step')
    ax.set_title('Crossover of Learning Drivers: Touch-Guidance vs. Curiosity')
    ax.set_xlabel('Training Steps')
    ax.legend()
    ax.set_ylim(bottom=0)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_REWARD_FIGURE, bbox_inches='tight')
    print(f"加权奖励演化图已成功保存至 '{OUTPUT_REWARD_FIGURE}'")
    plt.close(fig)

# --- 3. 主程序入口 (MAIN EXECUTION BLOCK) ---

if __name__ == '__main__':
    try:
        scalar_data = parse_all_scalars(LOG_DIR)
        
        if scalar_data:
            # 绘制所有图表
            plot_key_metrics(scalar_data, EMA_ALPHA)
            plot_detailed_behavior_per_part(scalar_data, EMA_ALPHA, OUTPUT_DIR_PARTS)
            plot_loss_curves(scalar_data, EMA_ALPHA)
            plot_reward_evolution(scalar_data, EMA_ALPHA) # 【新功能】: 调用新的绘图函数
            print("\n所有图表生成完毕！")
        else:
            print("未找到可处理的数据，脚本执行完毕。")

    except ImportError:
        print("\n错误: scienceplots库未安装。")
        print("请运行: pip install scienceplots")
    except FileNotFoundError as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")
        import traceback
        traceback.print_exc()
