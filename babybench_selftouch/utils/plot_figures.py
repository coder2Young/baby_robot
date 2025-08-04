# -*- coding: utf-8 -*-
"""
Unified, Publication-Quality Figure Generator for BabyRobot Project (V5 - Final)

This final version incorporates an extensive list of user feedback,
including advanced plot compositions, aesthetic fixes, and completion of
all required figures for the paper.

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
        scalar_data = {}
        for tag in tags:
            s = pd.Series([e.value for e in ea.Scalars(tag)], index=[e.step for e in ea.Scalars(tag)]).sort_index()
            if s.index.has_duplicates:
                s = s.groupby(s.index).mean()
            scalar_data[tag] = s
        
        print(f"  --> Successfully parsed {len(scalar_data)} scalar tags.")
        return scalar_data
    except Exception as e:
        print(f"  [Error] Failed to parse {log_dir}: {e}")
        return {}

def load_all_experiment_data(exp_log_dirs: Dict[str, str]) -> Dict[str, Dict[str, pd.Series]]:
    """加载所有实验的数据。"""
    print("\n--- Starting Data Loading Process ---")
    all_data = {exp_name: parse_tensorboard_log(log_dir) for exp_name, log_dir in exp_log_dirs.items()}
    print("--- Data Loading Complete ---\n")
    return all_data

def robust_series_sum(series_list: List[pd.Series]) -> pd.Series:
    """健壮地求和多个Pandas Series，处理不对齐的索引。"""
    series_list = [s for s in series_list if s is not None and not s.empty]
    if not series_list: return pd.Series(dtype=np.float64)
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
    num_plots = len(plot_configs)
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)
    if num_plots == 1: axes = [axes]
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for i, config in enumerate(plot_configs):
        ax = axes[i]
        all_smooth_series_in_ax = []

        for line_conf in config['lines']:
            series = data.get(line_conf['tag'])
            if series is not None and not series.empty:
                series_smooth = series.ewm(alpha=EMA_ALPHA).mean()
                all_smooth_series_in_ax.append(series_smooth)
                ax.plot(series.index, series.values, color=line_conf.get('raw_color', line_conf['color']), alpha=0.25)
                ax.plot(series_smooth.index, series_smooth.values, color=line_conf['color'], label=line_conf['label'])
        
        if all_smooth_series_in_ax:
            combined_smooth_data = pd.concat(all_smooth_series_in_ax)
            min_val, max_val = combined_smooth_data.min(), combined_smooth_data.max()
            data_range = max_val - min_val if max_val > min_val else 1
            padding = data_range * 0.1
            bottom_limit = min_val - padding
            # 【改进8】如果数据区间远大于0，则不再强制从0开始
            if min_val > (max_val * 0.25) and min_val > 0:
                 ax.set_ylim(bottom=bottom_limit, top=max_val + padding)
            else:
                 ax.set_ylim(bottom=max(0, bottom_limit), top=max_val + padding)
        
        ax.set_ylabel(config['ylabel'])
        ax.set_title(config['title'], fontsize=12)
        ax.legend(loc='best', fontsize='small') # 【改进4】

    axes[-1].set_xlabel('Training Steps')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    save_plot(fig, output_path.replace('.png', '_raw_and_smooth.png'))
    
    for ax in axes:
        for line in ax.lines:
            alpha = line.get_alpha()
            if alpha is not None and alpha < 1.0: line.set_visible(False)
    save_plot(fig, output_path.replace('.png', '_smooth.png'))

def plot_comparison_time_series(all_data: Dict, metric_tag: str, experiments_to_plot: List[str], output_path: str, title: str, ylabel: str):
    """在同一张图上对比多个实验的同一个指标。"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    max_smooth_val = 0
    
    # 绘制平滑曲线和半透明的原始数据背景
    for exp_name in experiments_to_plot:
        data = all_data.get(exp_name, {})
        series = data.get(metric_tag)
        if series is not None and not series.empty:
            series_smooth = series.ewm(alpha=EMA_ALPHA).mean()
            max_smooth_val = max(max_smooth_val, series_smooth.max())
            
            line, = ax.plot(series_smooth.index, series_smooth.values, label=exp_name)
            ax.plot(series.index, series.values, color=line.get_color(), alpha=0.2)
                
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Training Steps')
    
    # --- 【最终修正】 ---
    # 将图例放在左上角，并使用更小的字号
    ax.legend(title='Experiment', loc='upper left', fontsize=9)
    # -------------------
    
    ax.set_ylim(bottom=0, top=max_smooth_val * 1.15)
    
    # 调整布局以适应图表标题
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存双版本图表
    save_plot(fig, output_path.replace('.png', '_raw_and_smooth.png'))
    for line in ax.lines:
        alpha = line.get_alpha()
        if alpha is not None and alpha < 1.0:
            line.set_visible(False)
    save_plot(fig, output_path.replace('.png', '_smooth.png'))

def plot_multi_panel_comparison(all_data: Dict, plot_configs: List[Dict], output_path: str, title: str):
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

    for i, stage_name in enumerate(stage_names):
        part_counts = stage_aggregation[stage_name]
        total_touches = sum(part_counts.values())
        if total_touches > 0:
            for part in all_parts_with_data:
                proportion = part_counts.get(part, 0) / total_touches
                if proportion < GROUPING_THRESHOLD: parts_to_group[i].append(proportion)
                else: data_to_plot[part].append(proportion)
        else: 
             for part in all_parts_with_data:
                 if part in data_to_plot: data_to_plot[part].append(0)

    for part, values in data_to_plot.items():
        while len(values) < len(stage_names): data_to_plot[part].append(0)
            
    others_proportions = [sum(parts_to_group.get(i, [])) for i in range(len(stage_names))]
    if any(p > 0 for p in others_proportions): data_to_plot['Others'] = others_proportions

    fig, ax = plt.subplots(figsize=(10, 7))
    final_parts = sorted(data_to_plot.keys(), key=lambda p: (p == 'Others', -sum(data_to_plot.get(p, [0]))))
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, len(final_parts)))
    
    x_pos = np.arange(len(stage_names))
    bottoms = np.zeros(len(stage_names))
    for part_name, color in zip(final_parts, colors):
        values = np.array(data_to_plot[part_name])
        ax.bar(x_pos, values, label=part_name, bottom=bottoms, color=color, width=0.6)
        bottoms += values
        
    ax.set_ylabel('Proportion of All Touches')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(title='Body Parts', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stage_names, rotation=15, ha="center") # 【改进1】
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    save_plot(fig, output_path)

# =============================================================================
# --- 5. 主执行逻辑 (MAIN EXECUTION LOGIC) ---
# =============================================================================

def main():
    """主函数，加载数据并生成所有图表。"""
    setup_plot_style()
    all_data = load_all_experiment_data(EXPERIMENT_LOG_DIRS)
    print("\n--- Generating All Figures Based on Final Plan (V5) ---")

    # --- 图组一：基准模型 (Baseline Model) ---
    exp_name = 'Baseline'
    exp_data = all_data.get(exp_name)
    if exp_data:
        exp_dir = os.path.join(BASE_OUTPUT_DIR, exp_name)
        print(f"\n[1] Generating plots for {exp_name}...")
        plot_stacked_bar_distribution(exp_data, STAGES, os.path.join(exp_dir, 'fig1_1_distribution_evolution.png'), f'Touch Distribution Evolution ({exp_name})')
        plot_time_series(exp_data, [{'title': 'Panel (a): Touch Frequency', 'ylabel': 'Frequency', 'lines': [{'tag': f'behavior_freq/{p}', 'label': p, 'color': c} for p, c in zip(['head', 'ub3', 'cb', 'lb'], ['red', 'orange', 'green', 'blue'])]},
                                    {'title': 'Panel (b): Touch Duration', 'ylabel': 'Duration', 'lines': [{'tag': f'behavior_duration/{p}', 'label': p, 'color': c} for p, c in zip(['head', 'ub3', 'cb', 'lb'], ['red', 'orange', 'green', 'blue'])]}],
                           os.path.join(exp_dir, 'fig1_2_rostro_caudal_trend.png'), f'Exploration Trend ({exp_name})')
        plot_time_series(exp_data, [{'title': 'Panel (a): VAE Reconstruction Loss', 'ylabel': 'Loss', 'lines': [{'tag': 'icm/proprio_vae_recon_loss', 'label': 'Proprio VAE', 'color': 'purple'}, {'tag': 'icm/touch_vae_recon_loss', 'label': 'Touch VAE', 'color': 'brown'}]},
                                    {'title': 'Panel (b): Forward Model Prediction Loss', 'ylabel': 'Loss', 'lines': [{'tag': 'icm/forward_loss', 'label': 'Forward Model', 'color': 'deepskyblue'}]}],
                           os.path.join(exp_dir, 'fig1_3_internal_model_loss.png'), f'Internal Model Learning ({exp_name})')
        plot_time_series(exp_data, [{'title': 'Crossover of Learning Drivers', 'ylabel': 'Mean Weighted Reward', 'lines': [{'tag': 'reward_weighted/mean_hand', 'label': 'Weighted Hand Touch', 'color': 'purple'}, {'tag': 'reward_weighted/mean_icm', 'label': 'Weighted ICM', 'color': 'blue'}]}],
                           os.path.join(exp_dir, 'fig1_4_driver_crossover.png'), f'Evolution of Reward Components ({exp_name})')

    # --- 图组二：核心对比与奖励机制消融 ---
    print("\n[2] Generating Core Comparison and Reward Ablation Plots...")
    # 2.1: 多样性大对比 【改进1】
    plot_comparison_time_series(all_data, 'behavior/touch_diversity_by_hand', list(EXPERIMENT_LOG_DIRS.keys()), os.path.join(BASE_OUTPUT_DIR, 'fig2_1_diversity_full_comparison.png'), 'Impact of All Ablations on Exploration Diversity', 'Touch Diversity')
    
    # 2.2 “纯触摸”+ Baseline对比 【改进2】
    exp_name_ablation = 'Ablation_NoCuriosity'
    exp_dir_ablation = os.path.join(BASE_OUTPUT_DIR, exp_name_ablation)
    if all_data.get('Baseline') and all_data.get(exp_name_ablation):
        for exp in ['Baseline', exp_name_ablation]:
            all_data[exp]['total_freq'] = robust_series_sum([s for t, s in all_data[exp].items() if t.startswith('behavior_freq/')])
            all_data[exp]['total_duration'] = robust_series_sum([s for t, s in all_data[exp].items() if t.startswith('behavior_duration/')])
        
        plot_multi_panel_comparison(all_data, 
            [{'title': 'Total Touch Frequency Comparison', 'ylabel': 'Frequency', 'tag': 'total_freq', 'experiments': ['Baseline', exp_name_ablation]},
             {'title': 'Total Touch Duration Comparison', 'ylabel': 'Duration', 'tag': 'total_duration', 'experiments': ['Baseline', exp_name_ablation]}],
            os.path.join(exp_dir_ablation, 'fig2_2_pure_touch_vs_baseline.png'), f'Total Touch Behavior ({exp_name_ablation} vs Baseline)')

    # 2.3: “纯好奇心”分析
    exp_name = 'Ablation_NoTouch'
    exp_data = all_data.get(exp_name)
    if exp_data:
        exp_dir = os.path.join(BASE_OUTPUT_DIR, exp_name)
        plot_time_series(exp_data, [{'title': 'Panel (a): Touch Frequency', 'ylabel': 'Frequency', 'lines': [{'tag': f'behavior_freq/{p}', 'label': p, 'color': c} for p, c in zip(['cb', 'lb', 'left_fingers', 'right_fingers'], ['g', 'b', 'c', 'm'])] + [{'tag': 'behavior/hand_to_hand_freq', 'label': 'hand_to_hand', 'color': 'k'}]},
                                    {'title': 'Panel (b): Touch Duration', 'ylabel': 'Duration', 'lines': [{'tag': f'behavior_duration/{p}', 'label': p, 'color': c} for p, c in zip(['cb', 'lb', 'left_fingers', 'right_fingers'], ['g', 'b', 'c', 'm'])] + [{'tag': 'behavior/hand_to_hand_duration', 'label': 'hand_to_hand', 'color': 'k'}]}],
                           os.path.join(exp_dir, 'fig2_3_pure_curiosity_behavior.png'), f'Exploration Pattern ({exp_name})')
        # 【新增图9】
        plot_stacked_bar_distribution(exp_data, STAGES, os.path.join(exp_dir, 'fig2_4_distribution.png'), f'Touch Distribution ({exp_name})')
        plot_time_series(exp_data, [{'title': 'Panel (a): VAE Reconstruction Loss', 'ylabel': 'Loss', 'lines': [{'tag': 'icm/proprio_vae_recon_loss', 'label': 'Proprio VAE', 'color': 'purple'}, {'tag': 'icm/touch_vae_recon_loss', 'label': 'Touch VAE', 'color': 'brown'}]},
                                    {'title': 'Panel (b): Forward Model Prediction Loss', 'ylabel': 'Loss', 'lines': [{'tag': 'icm/forward_loss', 'label': 'Forward Model', 'color': 'deepskyblue'}]}],
                           os.path.join(exp_dir, 'fig2_5_internal_model_loss.png'), f'Internal Model Learning ({exp_name})')

    # --- 图组三：课程与表征消融 ---
    print("\n[3] Generating plots for Curriculum and Representation Ablations...")
    # 3.1: 动态权重对比
    exp_name = 'Ablation_NoDynamicWeights'
    exp_dir = os.path.join(BASE_OUTPUT_DIR, exp_name)
    plot_multi_panel_comparison(all_data, [{'title': 'Impact on "lb" Exploration', 'ylabel': 'Touch Duration', 'tag': 'behavior_duration/lb', 'experiments': ['Baseline', exp_name]},
                                           {'title': 'Impact on "right_upper_leg" Exploration', 'ylabel': 'Touch Duration', 'tag': 'behavior_duration/right_upper_leg', 'experiments': ['Baseline', exp_name]}],
                                os.path.join(exp_dir, 'fig3_1_dynamic_weights_comparison.png'), 'Effectiveness of Dynamic Weight Curriculum')
    # 【新增图3/10】
    exp_data = all_data.get(exp_name)
    if exp_data:
        plot_time_series(exp_data, [{'title': 'Touch Duration on "ub2"', 'ylabel': 'Duration', 'lines': [{'tag': 'behavior_duration/ub2', 'label': 'ub2', 'color': 'red'}]},
                                    {'title': 'Touch Duration on "ub1"', 'ylabel': 'Duration', 'lines': [{'tag': 'behavior_duration/ub1', 'label': 'ub1', 'color': 'green'}]},
                                    {'title': 'Touch Duration on "cb"', 'ylabel': 'Duration', 'lines': [{'tag': 'behavior_duration/cb', 'label': 'cb', 'color': 'blue'}]}],
                           os.path.join(exp_dir, 'fig3_2_minor_parts_trend.png'), f'Minor Part Exploration ({exp_name})')

    # 3.2: 简单触摸对比
    exp_name_ablation = 'Ablation_SimpleTouch'
    exp_dir_ablation = os.path.join(BASE_OUTPUT_DIR, exp_name_ablation)
    plot_multi_panel_comparison(all_data, [{'title': 'Touch Duration on "head"', 'ylabel': 'Touch Duration', 'tag': 'behavior_duration/head', 'experiments': ['Baseline', exp_name_ablation]},
                                           {'title': 'Touch Duration on "ub3"', 'ylabel': 'Touch Duration', 'tag': 'behavior_duration/ub3', 'experiments': ['Baseline', exp_name_ablation]}],
                                os.path.join(exp_dir_ablation, 'fig3_3_simple_touch_comparison.png'), 'Behavioral Trap of Simple Touch Reward')
    # 【新增图6】
    if all_data.get('Baseline') and all_data.get(exp_name_ablation):
        for exp in ['Baseline', exp_name_ablation]:
            if 'total_freq' not in all_data[exp]:
                all_data[exp]['total_freq'] = robust_series_sum([s for t, s in all_data[exp].items() if t.startswith('behavior_freq/')])
                all_data[exp]['total_duration'] = robust_series_sum([s for t, s in all_data[exp].items() if t.startswith('behavior_duration/')])
        plot_multi_panel_comparison(all_data, 
            [{'title': 'Total Touch Frequency Comparison', 'ylabel': 'Frequency', 'tag': 'total_freq', 'experiments': ['Baseline', exp_name_ablation]},
             {'title': 'Total Touch Duration Comparison', 'ylabel': 'Duration', 'tag': 'total_duration', 'experiments': ['Baseline', exp_name_ablation]}],
            os.path.join(exp_dir_ablation, 'fig3_4_total_touch_vs_baseline.png'), f'Total Touch Behavior ({exp_name_ablation} vs Baseline)')

    # 3.3: VAE vs MLP 对比
    exp_name = 'Ablation_SimpleMLP'
    exp_dir = os.path.join(BASE_OUTPUT_DIR, exp_name)
    baseline_loss = all_data.get('Baseline', {}).get('icm/forward_loss', pd.Series(dtype=np.float64))
    mlp_loss = all_data.get(exp_name, {}).get('mlp_loss/total', pd.Series(dtype=np.float64))
    if not baseline_loss.empty: baseline_loss_norm = (baseline_loss - baseline_loss.min()) / (baseline_loss.max() - baseline_loss.min())
    else: baseline_loss_norm = pd.Series(dtype=np.float64)
    if not mlp_loss.empty: mlp_loss_norm = (mlp_loss - mlp_loss.min()) / (mlp_loss.max() - mlp_loss.min())
    else: mlp_loss_norm = pd.Series(dtype=np.float64)

    plot_time_series({'baseline_loss_norm': baseline_loss_norm, 'mlp_loss_norm': mlp_loss_norm}, [
        {'title': 'Normalized Learning Dynamics', 'ylabel': 'Normalized Prediction Loss', 'lines': [
            {'tag': 'baseline_loss_norm', 'label': 'VAE-based Model', 'color': 'blue'}, 
            {'tag': 'mlp_loss_norm', 'label': 'End-to-End MLP', 'color': 'red'}
        ]}
    ], os.path.join(exp_dir, 'fig3_5_panel_A_loss_dynamics.png'), 'Comparison of Internal Model Learning Dynamics')
    
    # 【改进5】
    exp_data = all_data.get(exp_name)
    if exp_data:
        plot_stacked_bar_distribution(exp_data, STAGES, os.path.join(exp_dir, 'fig3_5_panel_B_distribution.png'), f'Touch Distribution ({exp_name})')

    print("\n--- All plotting tasks are complete! ---")

if __name__ == '__main__':
    main()
