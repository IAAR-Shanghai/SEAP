# analysis_utils.py

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attributions_heatmap(
    attributions, 
    task_type, 
    aggregation='mean', 
    std_multiplier=1, 
    fmt=".3e",  # 使用科学计数法显示梯度值
    annot_fontsize=10,  # 热力图注释字体大小
    label_fontsize=10,  # 坐标轴标签字体大小
    title_fontsize=12  # 标题字体大小
):
    """
    可视化某任务类型的各层各模块梯度大小的热力图。

    Args:
        attributions (dict): 梯度归因数据，结构为 attributions[task_type][layer_idx][module_name]。
        task_type (str): 要可视化的任务类型。
        aggregation (str): 统计方法，可选 'mean' 或 'sum'。
        std_multiplier (float): 用于过滤极值的标准差倍数。
        fmt (str): 热力图注释的格式，如 ".3f" 或 ".3e"。
        annot_fontsize (int): 热力图注释字体大小。
        label_fontsize (int): 坐标轴标签字体大小。
        title_fontsize (int): 标题字体大小。
    """
    task_attributions = attributions[task_type]
    layers = sorted(task_attributions.keys())
    modules = sorted(next(iter(task_attributions.values())).keys())  # 获取模块名称列表

    # 创建一个二维数组存储梯度大小
    data_matrix = np.zeros((len(layers), len(modules)))

    # 填充数据矩阵
    for i, layer_idx in enumerate(layers):
        for j, module_name in enumerate(modules):
            module_values = task_attributions[layer_idx][module_name]
            if aggregation == 'mean':
                data_matrix[i, j] = np.mean(np.abs(module_values))
            elif aggregation == 'sum':
                data_matrix[i, j] = np.sum(np.abs(module_values))
            else:
                raise ValueError("Unsupported aggregation method. Use 'mean' or 'sum'.")

    # 计算数据的上下限
    mean_value = np.mean(data_matrix)
    std_value = np.std(data_matrix)
    vmin = max(data_matrix.min(), mean_value - std_multiplier * std_value)
    vmax = min(data_matrix.max(), mean_value + std_multiplier * std_value)

    # 创建热力图
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        data_matrix,
        xticklabels=modules,
        yticklabels=layers,
        cmap='coolwarm',  # 以 0 为中间点
        center=0,
        cbar_kws={'label': 'Gradient Magnitude'},
        annot=True, fmt=fmt,  # 使用科学计数法显示
        annot_kws={'fontsize': annot_fontsize},  # 调整注释字体大小
        vmin=vmin, vmax=vmax  # 设置颜色范围
    )
    plt.title(f'Attributions Heatmap for Task: {task_type}', fontsize=title_fontsize)
    plt.xlabel('Modules', fontsize=label_fontsize)
    plt.ylabel('Layers', fontsize=label_fontsize)
    plt.xticks(fontsize=label_fontsize)
    plt.yticks(fontsize=label_fontsize)
    plt.show()


def collect_experiment_results(base_results_root, tasks_to_test, sparsity_levels):
    """
    收集实验结果。

    参数：
        base_results_root (str): 实验结果的根目录。
        tasks_to_test (list): 要测试的任务列表。
        sparsity_levels (list): 稀疏度水平列表。

    返回：
        pd.DataFrame: 包含所有实验结果的数据框。
    """
    experiment_results = []

    for sparsity in sparsity_levels:
        results_root = os.path.join(base_results_root, f'results_{sparsity}')

        for task_type in tasks_to_test:
            task_dir = os.path.join(results_root, task_type)
            if not os.path.exists(task_dir):
                continue

            # 遍历层目录
            for layer_name in os.listdir(task_dir):
                layer_dir = os.path.join(task_dir, layer_name)
                if not os.path.isdir(layer_dir):
                    continue

                layer_idx = int(layer_name.split('_')[-1])  # 提取层索引

                # 遍历实验目录
                for experiment_name in os.listdir(layer_dir):
                    experiment_dir = os.path.join(layer_dir, experiment_name)
                    if not os.path.isdir(experiment_dir):
                        continue

                    # 读取 test_results.json 文件
                    test_results_file = os.path.join(experiment_dir, 'test_results.json')
                    if not os.path.exists(test_results_file):
                        continue

                    with open(test_results_file, 'r', encoding='utf-8') as f:
                        test_results = json.load(f)
                        accuracy = test_results.get('accuracy', None)

                    if accuracy is not None:
                        experiment_results.append({
                            'task_type': task_type,
                            'sparsity': sparsity,
                            'layer_idx': layer_idx,
                            'accuracy': accuracy
                        })

    # 将结果转换为 DataFrame
    df_results = pd.DataFrame(experiment_results)
    return df_results


def plot_experiment_results(df_results, tasks_to_test):
    """
    绘制实验结果。

    参数：
        df_results (pd.DataFrame): 实验结果数据框。
        tasks_to_test (list): 要绘制的任务列表。
    """
    sns.set(style="whitegrid", context="talk")  # 学术化风格和更大的字体
    palette = sns.color_palette("deep")  # 学术经典配色

    for task_type in tasks_to_test:
        plt.figure(figsize=(12, 6))
        df_task = df_results[df_results['task_type'] == task_type]

        sns.lineplot(
            data=df_task,
            x='layer_idx',
            y='accuracy',
            hue='sparsity',
            palette=palette,
            marker='o',
            linewidth=2.5,  # 加粗线条
            markersize=8  # 放大标记点
        )

        plt.title(f'Accuracy vs. Pruned Layer for Task: {task_type}', fontsize=18, weight='bold')
        plt.xlabel('Pruned Layer Index', fontsize=16, weight='bold')
        plt.ylabel('Accuracy (%)', fontsize=16, weight='bold')
        plt.legend(title='Sparsity Level (%)', fontsize=14, title_fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.8)  # 使用虚线网格

        plt.show()
