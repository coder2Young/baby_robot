# -*- coding: utf-8 -*-
"""
Unified, Publication-Quality Figure Generator for BabyRobot Project (V3)

This version incorporates detailed user feedback for improved plotting logic,
figure composition, and data handling, tailored for the final paper.

Key Improvements:
- Y-axis auto-scaling based on smoothed data to improve readability.
- Robust time-series summation to fix empty plot bugs.
- Figure composition for direct multi-panel comparisons.
- Comprehensive ablation comparisons and simplified file organization.

Author: Gemini
Date: 2025-08-04
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator
import scienceplots
from typing import Dict, List, Tuple

# =============================================================================
# --- 1. 全局配置 (GLOBAL CONFIGURATION) ---
# =============================================================================

# TODO: 请务必修改此字典，将实验名称映射到其正确的TensorBoard日志路径
# 这是唯一需要您手动修改的地方
EXPERIMENT_LOG_DIRS = {
    'Baseline': 'results/self_touch/logs/PPO_10',
    'Ablation_NoCuriosity': 'results/self_touch/logs/PPO_11',
    'Ablation_NoTouch': 'results/self_touch/logs/PPO_13',
    'Ablation_NoDynamicWeights': 'results/self_touch/logs/PPO_14',
    'Ablation_SimpleTouch': 'results/self_touch/logs/PPO_17',
    'Ablation_SimpleMLP': 'results/self_touch/logs/PPO_24',
}

# --- 绘图参数 ---
BASE_OUTPUT_DIR = 'results/plot'
EMA_ALPHA = 0.01
STAGES = {
    'Early (0-1.5M)': (0, 1500000),
    'Middle (1.5-3M)': (1500000, 3000000),
    'Late (3-4M)': (3000000, 4000000)
}
GROUPING_THRESHOLD = 0.01

# =============================================================================
# --- 2. 数据加载与处理模块 (DATA LOADING & PROCESSING) ---
# =============================================================================

def parse_tensorboard_log(log_dir: str) -> Dict[str, pd.Series]:
    """健壮地从单个TensorBoard日志目录中解析出所有标量数据。"""
    print(f"--> Parsing data from: {log_dir}")
    if not os.path.isdir(log_dir):
        print(f"  [Warning] Directory not found: {log_dir}. Skipping.")
        return {}
    
    try:
        ea = event_accumulator.EventAccumulator(log_dir, size_guidance={event_accumulator.SCALARS: 0})
        ea.Reload()
        tags = ea.Tags()['scalars']
        
        scalar_data = defaultdict(pd.Series)
        for tag in tags:
            events = ea.Scalars(tag)
            steps = [event.step for event in events]
            values = [event.value for event in events]
            
            s = pd.Series(values, index=steps)
            if s.index.has_duplicates:
                s = s.groupby(s.index).mean()
            
            scalar_data[tag] = s.sort_index()
        
        print(f"  --> Successfully parsed {len(scalar_data)} scalar tags.")
        return dict(scalar_data)
    except Exception as e:
        print(f"  [Error] Failed to parse {log_dir}: {e}")
        return {}

def load_all_experiment_data(exp_log_dirs: Dict[str, str]) -> Dict[str, Dict[str, pd.Series]]:
    """加载所有实验的数据。"""
    print("\n--- Starting Data Loading Process ---")
    all_data = {}
    for exp_name, log_dir in exp_log_dirs.items():
        all_data[exp_name] = parse_tensorboard_log(log_dir)
    print("--- Data Loading Complete ---\n")
    return all_data

def robust_series_sum(series_list: List[pd.Series]) -> pd.Series:
    """
    【修复】健壮地求和多个Pandas Series，处理不对齐的索引。
    """
    if not series_list:
        return pd.Series(dtype=np.float64)
    # 移除空的Series
    series_list = [s for s in series_list if s is not None and not s.empty]
    if not series_list:
        return pd.Series(dtype=np.float64)
    
    combined_df = pd.concat(series_list, axis=1)
    return combined_df.sum(axis=1).sort_index()

# =============================================================================
# --- 3. 绘图核心函数 (PLOTTING CORE FUNCTIONS) ---
# =============================================================================

def setup_plot_style():
    plt.style.use(['science', 'notebook', 'grid'])
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 200})

def save_plot(fig, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches='tight')
    print(f"  <-- Figure saved to: {output_path}")
    plt.close(fig)

def plot_time_series(data: Dict, plot_configs: List[Dict], output_path: str, title: str):
    """灵活的时序图绘制函数，支持多子图、Y轴智能缩放和双版本输出。"""
    num_plots = len(plot_configs)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
    if num_plots == 1: axes = [axes]
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for i, config in enumerate(plot_configs):
        ax = axes[i]
        
        # --- 【关键修正】: 为每个子图独立计算Y轴范围 ---
        all_smooth_series_in_ax = []
        # ----------------------------------------------

        for line_conf in config['lines']:
            series = data.get(line_conf['tag'])
            if series is not None and not series.empty:
                series_smooth = series.ewm(alpha=EMA_ALPHA).mean()
                all_smooth_series_in_ax.append(series_smooth) # 收集平滑曲线
                
                # 绘制带原始数据的图 (设置半透明)
                ax.plot(series.index, series.values, color=line_conf.get('raw_color', line_conf['color']), alpha=0.25)
                # 绘制平滑曲线 (不设置透明度，默认为1.0)
                ax.plot(series_smooth.index, series_smooth.values, color=line_conf['color'], label=line_conf['label'])
        
        # --- 【关键修正】: 实现Y轴智能缩放 ---
        if all_smooth_series_in_ax:
            combined_smooth_data = pd.concat(all_smooth_series_in_ax)
            min_val = combined_smooth_data.min()
            max_val = combined_smooth_data.max()
            data_range = max_val - min_val
            
            # 设置Y轴范围，上下各增加10%的padding
            ax.set_ylim(
                bottom=max(0, min_val - data_range * 0.1), 
                top=max_val + data_range * 0.1
            )
        # --- [修正结束] ---
        
        ax.set_ylabel(config['ylabel'])
        ax.set_title(config['title'], fontsize=12)
        ax.legend()

    axes[-1].set_xlabel('Training Steps')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 保存包含原始数据的版本
    save_plot(fig, output_path.replace('.png', '_raw_and_smooth.png'))
    
    # 移除原始数据线，生成仅平滑曲线的版本
    for ax in axes:
        for line in ax.lines:
            alpha = line.get_alpha()
            if alpha is not None and alpha < 1.0:
                line.set_visible(False)
    
    save_plot(fig, output_path.replace('.png', '_smooth.png'))

def plot_comparison_time_series(all_data: Dict, metric_tag: str, experiments_to_plot: List[str], output_path: str, title: str, ylabel: str):
    """在同一张图上对比多个实验的同一个指标。"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    max_smooth_val = 0
    all_series_smooth = []

    for exp_name in experiments_to_plot:
        data = all_data.get(exp_name, {})
        series = data.get(metric_tag)
        if series is not None and not series.empty:
            series_smooth = series.ewm(alpha=EMA_ALPHA).mean()
            max_smooth_val = max(max_smooth_val, series_smooth.max())
            ax.plot(series.index, series.values, alpha=0.15)
            all_series_smooth.append(series_smooth)
    
    for series_smooth, exp_name in zip(all_series_smooth, experiments_to_plot):
        line, = ax.plot(series_smooth.index, series_smooth.values, label=exp_name)
        # 确保原始数据和EMA曲线颜色一致
        for line_raw in ax.lines:
            if line_raw.get_label() == '' and np.array_equal(line_raw.get_xdata(), series_smooth.index):
                line_raw.set_color(line.get_color())
                break
                
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Training Steps')
    ax.legend(title='Experiment')
    ax.set_ylim(bottom=0, top=max_smooth_val * 1.15) # 【改进3】
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_plot(fig, output_path.replace('.png', '_raw_and_smooth.png'))
    for line in ax.lines:
        if line.get_alpha() is not None and line.get_alpha() < 1.0: line.set_visible(False)
    save_plot(fig, output_path.replace('.png', '_smooth.png'))

def plot_multi_panel_comparison(all_data: Dict, plot_configs: List[Dict], output_path: str, title: str):
    """【新功能】绘制多面板对比图，每个面板对比不同实验的同一个指标。"""
    num_plots = len(plot_configs)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
    if num_plots == 1: axes = [axes]
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for i, config in enumerate(plot_configs):
        ax = axes[i]
        max_smooth_val = 0
        for exp_name in config['experiments']:
            data = all_data.get(exp_name, {})
            series = data.get(config['tag'])
            if series is not None and not series.empty:
                series_smooth = series.ewm(alpha=EMA_ALPHA).mean()
                max_smooth_val = max(max_smooth_val, series_smooth.max())
                ax.plot(series_smooth.index, series_smooth.values, label=exp_name)
        
        ax.set_ylabel(config['ylabel'])
        ax.set_title(config['title'], fontsize=12)
        ax.legend()
        ax.set_ylim(bottom=0, top=max_smooth_val * 1.15)

    axes[-1].set_xlabel('Training Steps')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot(fig, output_path)

# (plot_stacked_bar_distribution 函数保持不变，这里省略以节省空间)
def plot_stacked_bar_distribution(exp_data: Dict, stages: Dict, output_path: str, title: str):
    stage_aggregation = {stage_name: defaultdict(float) for stage_name in stages}
    all_parts_set = set()
    for tag, series in exp_data.items():
        if tag.startswith('behavior_freq/'):
            part_name = tag.replace('behavior_freq/', '')
            all_parts_set.add(part_name)
            for step, value in series.items():
                for stage_name, (start_step, end_step) in stages.items():
                    if start_step <= step < end_step:
                        stage_aggregation[stage_name][part_name] += value
                        break
    
    data_to_plot = defaultdict(list)
    parts_to_group = defaultdict(list)
    stage_names = list(stages.keys())
    
    all_parts_with_data = sorted([p for p in all_parts_set if any(st.get(p,0)>0 for st in stage_aggregation.values())])

    for stage_name in stage_names:
        part_counts = stage_aggregation[stage_name]
        total_touches = sum(part_counts.values())
        if total_touches > 0:
            for part in all_parts_with_data:
                proportion = part_counts.get(part, 0) / total_touches
                if proportion < GROUPING_THRESHOLD:
                    parts_to_group[stage_name].append(proportion)
                else:
                    data_to_plot[part].append(proportion)
        else: 
             for part in all_parts_with_data:
                 if part in data_to_plot: data_to_plot[part].append(0)

    for part, values in data_to_plot.items():
        while len(values) < len(stage_names): data_to_plot[part].append(0)
            
    others_proportions = [sum(parts_to_group[stage]) for stage in stage_names]
    if any(p > 0 for p in others_proportions):
        data_to_plot['Others'] = others_proportions

    fig, ax = plt.subplots(figsize=(10, 7))
    final_parts = sorted(data_to_plot.keys(), key=lambda p: (p == 'Others', -sum(data_to_plot.get(p, [0]))))
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(final_parts)))
    
    bottoms = np.zeros(len(stage_names))
    for part_name, color in zip(final_parts, colors):
        values = np.array(data_to_plot[part_name])
        ax.bar(stage_names, values, label=part_name, bottom=bottoms, color=color)
        bottoms += values
        
    ax.set_ylabel('Proportion of All Touches')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title='Body Parts', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    save_plot(fig, output_path)

# =============================================================================
# --- 5. 主执行逻辑 (MAIN EXECUTION LOGIC) ---
# =============================================================================

def main():
    setup_plot_style()
    all_data = load_all_experiment_data(EXPERIMENT_LOG_DIRS)
    print("\n--- Generating All Figures Based on a New Plan ---")

    # --- 图组一：基准模型 (Baseline Model) ---
    exp_name = 'Baseline'
    exp_data = all_data.get(exp_name)
    if exp_data:
        exp_dir = os.path.join(BASE_OUTPUT_DIR, exp_name)
        print(f"\n[1] Generating plots for {exp_name}...")
        # 1.1: 触摸分布
        plot_stacked_bar_distribution(exp_data, STAGES, os.path.join(exp_dir, 'fig1_1_distribution_evolution.png'), f'Touch Distribution Evolution ({exp_name})')
        # 1.2: 核心部位趋势
        plot_time_series(exp_data, [{'title': 'Panel (a): Touch Frequency', 'ylabel': 'Frequency', 'lines': [{'tag': f'behavior_freq/{p}', 'label': p, 'color': c} for p, c in zip(['head', 'ub3', 'cb', 'lb'], ['red', 'orange', 'green', 'blue'])]},
                                    {'title': 'Panel (b): Touch Duration', 'ylabel': 'Duration', 'lines': [{'tag': f'behavior_duration/{p}', 'label': p, 'color': c} for p, c in zip(['head', 'ub3', 'cb', 'lb'], ['red', 'orange', 'green', 'blue'])]}],
                           os.path.join(exp_dir, 'fig1_2_rostro_caudal_trend.png'), f'Exploration Trend ({exp_name})')
        # 1.3: 内部模型loss
        plot_time_series(exp_data, [{'title': 'Panel (a): VAE Reconstruction Loss', 'ylabel': 'Loss', 'lines': [{'tag': 'icm/proprio_vae_recon_loss', 'label': 'Proprio VAE', 'color': 'purple'}, {'tag': 'icm/touch_vae_recon_loss', 'label': 'Touch VAE', 'color': 'brown'}]},
                                    {'title': 'Panel (b): Forward Model Prediction Loss', 'ylabel': 'Loss', 'lines': [{'tag': 'icm/forward_loss', 'label': 'Forward Model', 'color': 'deepskyblue'}]}],
                           os.path.join(exp_dir, 'fig1_3_internal_model_loss.png'), f'Internal Model Learning ({exp_name})')
        # 1.4: 驱动力交叉
        plot_time_series(exp_data, [{'title': 'Crossover of Learning Drivers', 'ylabel': 'Mean Weighted Reward', 'lines': [{'tag': 'reward_weighted/mean_hand', 'label': 'Weighted Hand Touch', 'color': 'purple'}, {'tag': 'reward_weighted/mean_icm', 'label': 'Weighted ICM', 'color': 'blue'}]}],
                           os.path.join(exp_dir, 'fig1_4_driver_crossover.png'), f'Evolution of Reward Components ({exp_name})')

    # --- 图组二：奖励机制消融 ---
    print("\n[2] Generating plots for Reward Ablations...")
    # 2.1: 多样性大对比 (包含所有实验)
    plot_comparison_time_series(all_data, 'behavior/touch_diversity_by_hand', list(EXPERIMENT_LOG_DIRS.keys()), os.path.join(BASE_OUTPUT_DIR, 'fig2_1_diversity_full_comparison.png'), 'Impact of All Ablations on Exploration Diversity', 'Touch Diversity')
    
    # 2.2: 纯好奇心
    exp_name = 'Ablation_NoTouch'
    exp_data = all_data.get(exp_name)
    if exp_data:
        exp_dir = os.path.join(BASE_OUTPUT_DIR, exp_name)
        plot_time_series(exp_data, [{'title': 'Panel (a): Touch Frequency', 'ylabel': 'Frequency', 'lines': [{'tag': f'behavior_freq/{p}', 'label': p, 'color': c} for p, c in zip(['cb', 'lb', 'left_fingers', 'right_fingers'], ['g', 'b', 'c', 'm'])] + [{'tag': 'behavior/hand_to_hand_freq', 'label': 'hand_to_hand', 'color': 'k'}]},
                                    {'title': 'Panel (b): Touch Duration', 'ylabel': 'Duration', 'lines': [{'tag': f'behavior_duration/{p}', 'label': p, 'color': c} for p, c in zip(['cb', 'lb', 'left_fingers', 'right_fingers'], ['g', 'b', 'c', 'm'])] + [{'tag': 'behavior/hand_to_hand_duration', 'label': 'hand_to_hand', 'color': 'k'}]}],
                           os.path.join(exp_dir, 'fig2_2_pure_curiosity_behavior.png'), f'Exploration Pattern ({exp_name})')

    # 2.3: 纯触摸
    exp_name = 'Ablation_NoCuriosity'
    exp_data = all_data.get(exp_name)
    if exp_data:
        exp_dir = os.path.join(BASE_OUTPUT_DIR, exp_name)
        total_freq = robust_series_sum([s for t, s in exp_data.items() if t.startswith('behavior_freq/')])
        total_duration = robust_series_sum([s for t, s in exp_data.items() if t.startswith('behavior_duration/')])
        plot_time_series({'total_freq': total_freq, 'total_duration': total_duration}, [{'title': 'Total Touch Frequency', 'ylabel': 'Frequency', 'lines': [{'tag': 'total_freq', 'label': 'Total Freq', 'color': 'crimson'}]},
                                                                                        {'title': 'Total Touch Duration', 'ylabel': 'Duration', 'lines': [{'tag': 'total_duration', 'label': 'Total Duration', 'color': 'forestgreen'}]}],
                           os.path.join(exp_dir, 'fig2_3_pure_touch_behavior.png'), f'Total Touch Behavior ({exp_name})')

    # --- 图组三：课程与表征消融 ---
    print("\n[3] Generating plots for Curriculum and Representation Ablations...")
    # 3.1: 动态权重对比
    plot_multi_panel_comparison(all_data, [{'title': 'Impact on "lb" Exploration', 'ylabel': 'Touch Duration', 'tag': 'behavior_duration/lb', 'experiments': ['Baseline', 'Ablation_NoDynamicWeights']},
                                           {'title': 'Impact on "right_upper_leg" Exploration', 'ylabel': 'Touch Duration', 'tag': 'behavior_duration/right_upper_leg', 'experiments': ['Baseline', 'Ablation_NoDynamicWeights']}],
                                os.path.join(BASE_OUTPUT_DIR, 'Ablation_NoDynamicWeights', 'fig3_1_dynamic_weights_comparison.png'), 'Effectiveness of Dynamic Weight Curriculum')
    # 3.2: 简单触摸对比
    plot_multi_panel_comparison(all_data, [{'title': 'Touch Duration on "head"', 'ylabel': 'Touch Duration', 'tag': 'behavior_duration/head', 'experiments': ['Baseline', 'Ablation_SimpleTouch']},
                                           {'title': 'Touch Duration on "ub3"', 'ylabel': 'Touch Duration', 'tag': 'behavior_duration/ub3', 'experiments': ['Baseline', 'Ablation_SimpleTouch']}],
                                os.path.join(BASE_OUTPUT_DIR, 'Ablation_SimpleTouch', 'fig3_2_simple_touch_comparison.png'), 'Behavioral Trap of Simple Touch Reward')

    # 3.3: VAE vs MLP 对比
    baseline_loss = all_data.get('Baseline', {}).get('icm/forward_loss', pd.Series(dtype=np.float64))
    mlp_loss = all_data.get('Ablation_SimpleMLP', {}).get('mlp_loss/total', pd.Series(dtype=np.float64))
    if not baseline_loss.empty: baseline_loss_norm = (baseline_loss - baseline_loss.min()) / (baseline_loss.max() - baseline_loss.min())
    else: baseline_loss_norm = pd.Series(dtype=np.float64)
    if not mlp_loss.empty: mlp_loss_norm = (mlp_loss - mlp_loss.min()) / (mlp_loss.max() - mlp_loss.min())
    else: mlp_loss_norm = pd.Series(dtype=np.float64)

    plot_time_series({'baseline_loss_norm': baseline_loss_norm, 'mlp_loss_norm': mlp_loss_norm}, [
        {'title': 'Normalized Learning Dynamics', 'ylabel': 'Normalized Prediction Loss', 'lines': [
            {'tag': 'baseline_loss_norm', 'label': 'VAE-based Model', 'color': 'blue'}, 
            {'tag': 'mlp_loss_norm', 'label': 'End-to-End MLP', 'color': 'red'}
        ]}
    ], os.path.join(BASE_OUTPUT_DIR, 'Ablation_SimpleMLP', 'fig3_3_panel_A_loss_dynamics.png'), 'Comparison of Internal Model Learning Dynamics')
    
    fig, ax = plt.subplots(figsize=(6, 5))
    div_baseline = all_data.get('Baseline', {}).get('behavior/touch_diversity_by_hand', pd.Series(dtype=np.float64))
    div_mlp = all_data.get('Ablation_SimpleMLP', {}).get('behavior/touch_diversity_by_hand', pd.Series(dtype=np.float64))
    final_div_baseline = div_baseline.ewm(alpha=EMA_ALPHA).mean().iloc[-1] if not div_baseline.empty else 0
    final_div_mlp = div_mlp.ewm(alpha=EMA_ALPHA).mean().iloc[-1] if not div_mlp.empty else 0
    ax.bar(['VAE-based Model', 'End-to-End MLP'], [final_div_baseline, final_div_mlp], color=['blue', 'red'])
    ax.set_ylabel('Final Touch Diversity')
    ax.set_title('Impact of State Representation on Behavior')
    save_plot(fig, os.path.join(BASE_OUTPUT_DIR, 'Ablation_SimpleMLP', 'fig3_3_panel_B_behavioral_outcome.png'))

    print("\n--- All plotting tasks are complete! ---")

if __name__ == '__main__':
    main()