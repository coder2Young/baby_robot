# -*- coding: utf-8 -*-
"""
TensorBoard Log to Body Touch Heatmap Generator (V2 - Model-Based)

This script reads touch frequency data from a TensorBoard log directory,
aggregates it into developmental stages, and generates a heatmap visualization
by directly rendering the robot's 3D model geometry from the simulation
environment, colored by touch frequency.

This version replaces the 2D schematic with a high-fidelity point cloud
representation of the robot's body for improved accuracy and visual quality.

Author: Gemini
Date: 2025-07-30
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from tensorboard.backend.event_processing import event_accumulator
import yaml
import mujoco
# Assuming the utils.py and other necessary files from babybench are in the path 
import sys
sys.path.append(".")
sys.path.append("..")
from babybench.utils import make_env
from babybench_selftouch.icm_callback import get_body_subtree

# --- 1. 配置参数 (CONFIGURATION PARAMETERS) ---

# TODO: 请将此路径修改为您实际的TensorBoard日志目录
# 例如: 'path/to/your/experiment/logs'
LOG_DIR = './results/self_touch/logs/PPO_7/' 

# TODO: 请将此路径修改为您实验的配置文件路径
# 这个文件是训练时生成的，通常在实验保存目录的根目录下
CONFIG_PATH = './babybench_selftouch/config_selftouch.yml'

# 定义三个发育阶段 (时间步)
STAGES = {
    'Early Stage (0-150k steps)': (0, 1500000),
    'Middle Stage (450k-550k steps)': (1500000, 3000000),
    'Late Stage (850k-1M steps)': (3000000, 4000000)
}

# 【新功能】: 定义要从热力图颜色归一化中排除的主导部位
# 这可以是一个列表，例如 ['head', 'another_part']
EXCLUDE_PART_FOR_ANALYSIS = 'head'

GROUPING_THRESHOLD = 0.02 # 2%

# 输出图像的文件名
OUTPUT_HEATMAP_FULL = 'touch_heatmap_full_story.png'
OUTPUT_HEATMAP_ZOOMED = 'touch_heatmap_zoomed_in.png'
OUTPUT_BAR_CHART = 'touch_distribution_barchart.png'

# 选择一个色谱
COLORMAP = 'plasma'

# 输出图像的文件名
OUTPUT_HEATMAP_FULL = 'touch_heatmap_full_body.png'
OUTPUT_HEATMAP_ZOOMED = 'touch_heatmap_zoomed_limbs.png'
OUTPUT_BAR_CHART_FULL = 'touch_distribution_full.png'
OUTPUT_BAR_CHART_NON_HEAD = 'touch_distribution_non_head.png' # 新图表的文件名

# 选择一个色谱
COLORMAP = 'viridis' # 换一个色谱以示区别

def parse_tensorboard_logs(log_dir):
    """从TensorBoard日志目录中解析出所有与触摸频率相关的数据。"""
    print(f"正在从 '{log_dir}' 读取TensorBoard日志...")
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"错误: 日志目录 '{log_dir}' 不存在。请检查路径。")
    ea = event_accumulator.EventAccumulator(log_dir, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    touch_data = defaultdict(list)
    freq_tags = [tag for tag in ea.Tags()['scalars'] if tag.startswith('behavior_freq/')]
    if not freq_tags:
        print("警告: 在日志中没有找到任何 'behavior_freq/' 相关的数据。")
        return {}
    for tag in freq_tags:
        part_name = tag.replace('behavior_freq/', '')
        events = ea.Scalars(tag)
        touch_data[part_name] = [(event.step, event.value) for event in events]
    print(f"成功解析到 {len(touch_data)} 个身体部位的触摸频率数据。")
    return dict(touch_data)

def aggregate_data_by_stage(touch_data, stages):
    """将解析出的数据按定义的阶段进行聚合和归一化。"""
    print("正在按发育阶段聚合数据...")
    stage_aggregation = {stage_name: defaultdict(float) for stage_name in stages}
    for part_name, events in touch_data.items():
        for step, value in events:
            for stage_name, (start_step, end_step) in stages.items():
                if start_step <= step < end_step:
                    stage_aggregation[stage_name][part_name] += value
                    break
    normalized_data = {}
    for stage_name, part_counts in stage_aggregation.items():
        total_touches_in_stage = sum(part_counts.values())
        normalized_stage = {}
        if total_touches_in_stage > 0:
            for part_name, count in part_counts.items():
                normalized_stage[part_name] = count / total_touches_in_stage
        normalized_data[stage_name] = normalized_stage
        print(f"  - {stage_name}: 总触摸频率 {total_touches_in_stage:.0f}")
    return normalized_data

def plot_stacked_bar_chart(aggregated_data, colormap_name, output_filename, grouping_threshold):
    """
    【V9 优化】: 绘制100%堆叠柱状图，并合并零碎项。
    """
    print("正在生成整体分布的堆叠柱状图...")
    stage_names = list(aggregated_data.keys())
    x_labels = [name.replace(' (', '\n(') for name in stage_names]

    # --- 数据分组逻辑 ---
    data_to_plot = defaultdict(lambda: [0.0] * len(stage_names))
    parts_to_group = defaultdict(lambda: [0.0] * len(stage_names))
    all_parts_original = sorted(list(set(part for stage_data in aggregated_data.values() for part in stage_data.keys() if stage_data.get(part, 0) > 0)))

    for i, stage in enumerate(stage_names):
        for part in all_parts_original:
            proportion = aggregated_data[stage].get(part, 0)
            if proportion < grouping_threshold:
                parts_to_group['Others'][i] += proportion
            else:
                data_to_plot[part][i] = proportion
    
    if 'Others' in parts_to_group:
        data_to_plot['Others'] = parts_to_group['Others']

    # --- 绘图逻辑 ---
    final_parts = sorted(data_to_plot.keys(), key=lambda p: (p == 'Others', -sum(data_to_plot[p]))) # 按总占比排序
    colors = plt.get_cmap(colormap_name)(np.linspace(0, 1, len(final_parts)))
    color_map = {part: color for part, color in zip(final_parts, colors)}
    if 'Others' in color_map:
        color_map['Others'] = 'grey'

    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
    
    bottoms = np.zeros(len(stage_names))
    for part_name in final_parts:
        values = data_to_plot[part_name]
        ax.bar(x_labels, values, label=part_name, bottom=bottoms, color=color_map.get(part_name))
        bottoms += np.array(values)

    ax.set_ylabel('Proportion of All Touches', fontsize=12)
    ax.set_title('Evolution of Full Body Touch Distribution', fontsize=14, fontweight='bold')
    ax.legend(title='Body Parts', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', labelsize=10, pad=10)

    for bar_container in ax.containers:
        for bar in bar_container:
            height = bar.get_height()
            if height >= grouping_threshold: # 只为大于阈值的块标注
                bar_color = bar.get_facecolor()
                luminance = 0.299*bar_color[0] + 0.587*bar_color[1] + 0.114*bar_color[2]
                text_color = 'white' if luminance < 0.5 else 'black'
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_y() + height / 2., f"{height:.1%}", ha='center', va='center', color=text_color, fontsize=9, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"整体分布柱状图已成功保存至 '{output_filename}'")

def plot_non_head_distribution_chart(aggregated_data, colormap_name, output_filename, exclude_part, grouping_threshold):
    """绘制排除主导项后的图表，并合并零碎项。"""
    print(f"正在生成排除'{exclude_part}'后的精细分布柱状图...")
    stage_names = list(aggregated_data.keys())
    x_labels = [name.replace(' (', '\n(') for name in stage_names]

    # --- 数据重归一化与分组逻辑 ---
    data_to_plot = defaultdict(lambda: [0.0] * len(stage_names))
    parts_to_group = defaultdict(lambda: [0.0] * len(stage_names))
    
    non_head_parts = sorted([p for p in touch_data.keys() if p != exclude_part])

    for i, stage in enumerate(stage_names):
        stage_data = aggregated_data[stage]
        total_non_head_proportion = sum(prop for part, prop in stage_data.items() if part != exclude_part)
        
        if total_non_head_proportion == 0: continue
            
        for part in non_head_parts:
            proportion_in_full = stage_data.get(part, 0)
            proportion_in_non_head = proportion_in_full / total_non_head_proportion if total_non_head_proportion > 0 else 0
            
            if proportion_in_non_head < grouping_threshold:
                parts_to_group['Others'][i] += proportion_in_non_head
            else:
                data_to_plot[part][i] = proportion_in_non_head

    if 'Others' in parts_to_group:
        data_to_plot['Others'] = parts_to_group['Others']

    # --- 绘图逻辑 ---
    final_parts = sorted(data_to_plot.keys(), key=lambda p: (p == 'Others', -sum(data_to_plot[p])))
    colors = plt.get_cmap(colormap_name)(np.linspace(0, 1, len(final_parts)))
    color_map = {part: color for part, color in zip(final_parts, colors)}
    if 'Others' in color_map:
        color_map['Others'] = 'grey'

    fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
    
    bottoms = np.zeros(len(stage_names))
    for part_name in final_parts:
        values = data_to_plot[part_name]
        ax.bar(x_labels, values, label=part_name, bottom=bottoms, color=color_map.get(part_name))
        bottoms += np.array(values)

    ax.set_ylabel('Proportion of Non-Head Touches', fontsize=12)
    ax.set_title(f'Evolution of Touch Distribution (Excluding "{exclude_part}")', fontsize=14, fontweight='bold')
    ax.legend(title='Body Parts', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', labelsize=10, pad=10)

    for bar_container in ax.containers:
        for bar in bar_container:
            height = bar.get_height()
            if height >= grouping_threshold:
                bar_color = bar.get_facecolor()
                luminance = 0.299*bar_color[0] + 0.587*bar_color[1] + 0.114*bar_color[2]
                text_color = 'white' if luminance < 0.5 else 'black'
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_y() + height / 2., f"{height:.1%}", ha='center', va='center', color=text_color, fontsize=9, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"非头部精细分布柱状图已成功保存至 '{output_filename}'")


# --- 3. 主程序入口 (MAIN EXECUTION BLOCK) ---

if __name__ == '__main__':
    try:
        touch_data = parse_tensorboard_logs(LOG_DIR)
        
        if touch_data:
            aggregated_data = aggregate_data_by_stage(touch_data, STAGES)
            
            # 生成两种最终的、出版质量的图表
            plot_stacked_bar_chart(aggregated_data, COLORMAP, OUTPUT_BAR_CHART_FULL, grouping_threshold=GROUPING_THRESHOLD)
            plot_non_head_distribution_chart(aggregated_data, COLORMAP, OUTPUT_BAR_CHART_NON_HEAD, exclude_part=EXCLUDE_PART_FOR_ANALYSIS, grouping_threshold=GROUPING_THRESHOLD)

        else:
            print("未找到可处理的数据，脚本执行完毕。")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"发生未知错误: {e}", )
        import traceback
        traceback.print_exc()
