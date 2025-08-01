# -*- coding: utf-8 -*-
"""
Publication-Quality Time-Series Plot Generator (V2)

This script reads various metrics from a TensorBoard log directory,
processes the data, and generates several publication-quality figures:
  1. Multi-panel “key metrics” figure (diversity, total frequency, total duration)
  2. Detailed per-part time-series figures
  3. Multi-panel “internal model losses” figure
  4. Reward evolution figure
  5. 【新增】Three‐panel composite figure for head, ub3, and right_eye behaviors

Author: Gemini
Date: 2025-07-30 (updated 2025-08-01)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator
import scienceplots
import math

# --- 1. 配置参数 (CONFIGURATION PARAMETERS) ---
LOG_DIR = './results/self_touch/logs/PPO_7'
EMA_ALPHA = 0.01

OUTPUT_DIR_PARTS           = './timeseries_per_part/'
OUTPUT_KEY_METRICS_FIGURE = 'timeseries_key_metrics.png'
OUTPUT_LOSS_FIGURE        = 'timeseries_loss_evolution.png'
OUTPUT_REWARD_FIGURE      = 'timeseries_reward_evolution.png'
OUTPUT_BEHAVIOR_PANEL     = 'timeseries_behavior_panel.png'

# --- 2. 核心功能函数 (CORE FUNCTIONS) ---

def parse_all_scalars(log_dir):
    """
    从 TensorBoard 日志目录中解析出所有标量数据，
    并对重复的时间步做平均处理后返回一个 dict of pandas.Series。
    """
    print(f"正在从 '{log_dir}' 读取所有 TensorBoard 标量数据...")
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"日志目录 '{log_dir}' 不存在。")
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    scalar_data = {}
    for tag in ea.Tags().get('scalars', []):
        events = ea.Scalars(tag)
        steps  = [e.step  for e in events]
        values = [e.value for e in events]
        s = pd.Series(values, index=steps)
        if s.index.has_duplicates:
            s = s.groupby(level=0).mean()
        scalar_data[tag] = s.sort_index()
    print(f"成功解析了 {len(scalar_data)} 个标量系列。")
    return scalar_data

def plot_key_metrics(scalar_data, alpha):
    """绘制多面板关键指标演化图：多样性、总频率、总时长。"""
    print("生成关键指标演化图...")
    plt.style.use(['science', 'notebook', 'grid'])
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True, dpi=200)
    fig.suptitle('Evolution of Key Exploratory Metrics', fontsize=16, fontweight='bold')

    # Panel (a): Touch Diversity
    ax = axes[0]
    raw = scalar_data.get('behavior/touch_diversity_by_hand', pd.Series(dtype=float))
    if not raw.empty:
        smooth = raw.ewm(alpha=alpha).mean()
        ax.plot(raw.index, raw.values, alpha=0.4, label='Raw', color='lightblue')
        ax.plot(smooth.index, smooth.values, label=f'EMA (α={alpha})', color='dodgerblue')
    ax.set_title('Panel (a): Touch Diversity')
    ax.set_ylabel('Unique Parts Touched')
    ax.legend()
    ax.set_ylim(bottom=0)

    # Panel (b): Total Touch Frequency
    ax = axes[1]
    freq_tags = [t for t in scalar_data if t.startswith('behavior_freq/')]
    freq_total = pd.concat([scalar_data[t] for t in freq_tags]).groupby(level=0).sum() if freq_tags else pd.Series(dtype=float)
    if not freq_total.empty:
        smooth = freq_total.ewm(alpha=alpha).mean()
        ax.plot(freq_total.index, freq_total.values, alpha=0.4, label='Raw', color='lightcoral')
        ax.plot(smooth.index, smooth.values, label=f'EMA (α={alpha})', color='crimson')
    ax.set_title('Panel (b): Total Touch Frequency')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.set_ylim(bottom=0)

    # Panel (c): Total Touch Duration
    ax = axes[2]
    dur_tags = [t for t in scalar_data if t.startswith('behavior_duration/')]
    dur_total = pd.concat([scalar_data[t] for t in dur_tags]).groupby(level=0).sum() if dur_tags else pd.Series(dtype=float)
    if not dur_total.empty:
        smooth = dur_total.ewm(alpha=alpha).mean()
        ax.plot(dur_total.index, dur_total.values, alpha=0.4, label='Raw', color='lightgreen')
        ax.plot(smooth.index, smooth.values, label=f'EMA (α={alpha})', color='forestgreen')
    ax.set_title('Panel (c): Total Touch Duration')
    ax.set_ylabel('Duration (steps)')
    ax.legend()
    ax.set_ylim(bottom=0)

    axes[-1].set_xlabel('Training Steps')
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(OUTPUT_KEY_METRICS_FIGURE, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {OUTPUT_KEY_METRICS_FIGURE}")

def plot_detailed_behavior_per_part(scalar_data, alpha, output_dir):
    """为每个部位绘制频率与时长的详细子图。"""
    print("生成每个部位的详细行为图...")
    os.makedirs(output_dir, exist_ok=True)
    parts = set(t.split('/',1)[1] for t in scalar_data if t.startswith('behavior_freq/') or t.startswith('behavior_duration/'))
    for part in sorted(parts):
        plt.style.use(['science', 'notebook', 'grid'])
        fig, axes = plt.subplots(2,1, figsize=(8,6), sharex=True, dpi=150)
        fig.suptitle(f'Behavior Evolution: {part}', fontsize=14, fontweight='bold')

        # 频率
        ax = axes[0]
        freq = scalar_data.get(f'behavior_freq/{part}', pd.Series(dtype=float))
        if not freq.empty:
            smooth = freq.ewm(alpha=alpha).mean()
            ax.plot(freq.index, freq.values, alpha=0.4, label='Raw', color='lightcoral')
            ax.plot(smooth.index, smooth.values, label=f'EMA (α={alpha})', color='crimson')
        ax.set_title('Panel (a): Touch Frequency')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.set_ylim(bottom=0)

        # 时长
        ax = axes[1]
        dur = scalar_data.get(f'behavior_duration/{part}', pd.Series(dtype=float))
        if not dur.empty:
            smooth = dur.ewm(alpha=alpha).mean()
            ax.plot(dur.index, dur.values, alpha=0.4, label='Raw', color='lightgreen')
            ax.plot(smooth.index, smooth.values, label=f'EMA (α={alpha})', color='forestgreen')
        ax.set_title('Panel (b): Touch Duration')
        ax.set_ylabel('Duration (steps)')
        ax.legend()
        ax.set_ylim(bottom=0)

        axes[-1].set_xlabel('Training Steps')
        fig.tight_layout(rect=[0,0,1,0.95])
        out_fn = os.path.join(output_dir, f'behavior_{part}.png')
        fig.savefig(out_fn, bbox_inches='tight')
        plt.close(fig)
    print(f"详细图表已保存至目录: {output_dir}")

def plot_loss_curves(scalar_data, alpha):
    """绘制内部模型损失演化图。"""
    print("生成模型损失演化图...")
    plt.style.use(['science', 'notebook', 'grid'])
    fig, axes = plt.subplots(2,1, figsize=(10,8), sharex=True, dpi=200)
    fig.suptitle('Evolution of Internal Model Losses', fontsize=16, fontweight='bold')

    # VAE 重建 Loss
    ax = axes[0]
    for tag, label, col in [
        ('icm/proprio_vae_recon_loss', 'Proprio VAE Loss', 'indigo'),
        ('icm/touch_vae_recon_loss',    'Touch VAE Loss',    'saddlebrown')
    ]:
        raw = scalar_data.get(tag, pd.Series(dtype=float))
        if not raw.empty:
            smooth = raw.ewm(alpha=alpha).mean()
            ax.plot(raw.index, raw.values, alpha=0.3)
            ax.plot(smooth.index, smooth.values, label=label+' (EMA)', color=col)
    ax.set_title('Panel (a): VAE Reconstruction Losses')
    ax.set_ylabel('Loss')
    ax.legend()

    # Forward Prediction Loss
    ax = axes[1]
    raw = scalar_data.get('icm/forward_loss', pd.Series(dtype=float))
    if not raw.empty:
        smooth = raw.ewm(alpha=alpha).mean()
        ax.plot(raw.index, raw.values, alpha=0.3)
        ax.plot(smooth.index, smooth.values, label='Forward Loss (EMA)', color='deepskyblue')
    ax.set_title('Panel (b): Forward Model Prediction Loss')
    ax.set_ylabel('Loss')
    ax.legend()

    axes[-1].set_xlabel('Training Steps')
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(OUTPUT_LOSS_FIGURE, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {OUTPUT_LOSS_FIGURE}")

def plot_reward_evolution(scalar_data, alpha):
    """绘制加权奖励演化图。"""
    print("生成奖励演化图...")
    plt.style.use(['science', 'notebook', 'grid'])
    fig, ax = plt.subplots(1,1, figsize=(10,6), dpi=200)
    fig.suptitle('Evolution of Weighted Reward Components', fontsize=16, fontweight='bold')

    for tag, label in [
        ('reward_weighted/mean_icm',  'Weighted ICM Reward'),
        ('reward_weighted/mean_hand', 'Weighted Hand Touch Reward')
    ]:
        raw = scalar_data.get(tag, pd.Series(dtype=float))
        if not raw.empty:
            smooth = raw.ewm(alpha=alpha).mean()
            ax.plot(raw.index, raw.values, alpha=0.3)
            ax.plot(smooth.index, smooth.values, label=label+' (EMA)')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Reward')
    ax.legend()
    ax.set_ylim(bottom=0)

    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(OUTPUT_REWARD_FIGURE, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {OUTPUT_REWARD_FIGURE}")

def plot_behavior_panel(scalar_data, alpha, output_path):
    """
    绘制组合图：Panel (a) head, (b) ub3, (c) right_eye
    (a) Stable High-Frequency Behavior
    (b) Learned and Maintained Behavior
    (c) Habituated Behavior
    """
    print("生成三面板行为对比图...")
    plt.style.use(['science', 'notebook', 'grid'])
    fig, axes = plt.subplots(3,1, figsize=(8,12), sharex=True, dpi=200)
    fig.suptitle('Behavioral Learning and Habituation', fontsize=16, fontweight='bold')

    panels = [
        ('behavior_freq/head',      'Panel (a): Stable High-Frequency Behavior'),
        ('behavior_freq/ub3',       'Panel (b): Learned and Maintained Behavior'),
        ('behavior_freq/right_eye', 'Panel (c): Habituated Behavior')
    ]
    colors = ['teal', 'green', 'purple']

    for ax, (tag, title), col in zip(axes, panels, colors):
        raw = scalar_data.get(tag, pd.Series(dtype=float))
        if not raw.empty:
            smooth = raw.ewm(alpha=alpha).mean()
            ax.plot(raw.index, raw.values, alpha=0.3, label='Raw', color=col)
            ax.plot(smooth.index, smooth.values, label='EMA Trend', color=col, linewidth=2)
        ax.set_title(title)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel('Training Steps')
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f"已保存: {output_path}")

# --- 3. 主程序入口 (MAIN EXECUTION BLOCK) ---
if __name__ == '__main__':
    try:
        data = parse_all_scalars(LOG_DIR)
        if not data:
            print("未找到标量数据，退出。")
            exit(1)

        # 生成各类图表
        plot_key_metrics(data, EMA_ALPHA)
        plot_detailed_behavior_per_part(data, EMA_ALPHA, OUTPUT_DIR_PARTS)
        plot_loss_curves(data, EMA_ALPHA)
        plot_reward_evolution(data, EMA_ALPHA)
        # 新增：三面板组合图
        plot_behavior_panel(data, EMA_ALPHA, OUTPUT_BEHAVIOR_PANEL)

        print("\n所有图表已生成完毕！")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback; traceback.print_exc()
