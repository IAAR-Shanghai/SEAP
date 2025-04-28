# src/visualization.py

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, davies_bouldin_score
from mpl_toolkits.mplot3d import Axes3D  # 用于3D绘图
from typing import Dict, List
import seaborn as sns


def _compute_tsne(hidden_states: np.ndarray, n_components: int, perplexity: float, random_state: int = 42):
    """
    对给定的隐藏状态数据进行 t-SNE 降维。

    参数：
        hidden_states (np.ndarray): 待降维的隐藏状态数据。
        n_components (int): 降维目标维度（2或3）。
        perplexity (float): t-SNE 的 perplexity 参数。
        random_state (int): 随机种子。

    返回：
        np.ndarray: t-SNE 降维后的结果。
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    return tsne.fit_transform(hidden_states)


def _plot_tsne_scatter(ax, tsne_results: np.ndarray, labels: List[str], unique_labels: List[str], cmap, 
                       n_components: int, point_size: float = 40, alpha: float = 0.2, elev: float = 30, azim: float = 45):
    """
    在给定的子图上绘制 t-SNE 降维结果的散点图。

    参数：
        ax: Matplotlib Axes 对象。
        tsne_results (np.ndarray): t-SNE 降维后的数据点坐标。
        labels (List[str]): 数据点对应的标签列表。
        unique_labels (List[str]): 唯一标签列表。
        cmap: 色图。
        n_components (int): 数据维度（2或3）。
        point_size (float): 点的大小。
        alpha (float): 点的透明度。
        elev (float): 3D 视角的仰角。
        azim (float): 3D 视角的方位角。
    """
    for i, label in enumerate(unique_labels):
        idx = [j for j, l in enumerate(labels) if l == label]
        color = cmap(i)
        if n_components == 3:
            ax.scatter(tsne_results[idx, 0], tsne_results[idx, 1], tsne_results[idx, 2],
                       label=label, color=color, s=point_size, alpha=alpha)
            ax.view_init(elev=elev, azim=azim)
        else:
            ax.scatter(tsne_results[idx, 0], tsne_results[idx, 1],
                       label=label, color=color, s=point_size, alpha=alpha)
    ax.set_xticks([])
    ax.set_yticks([])
    if n_components == 3:
        ax.set_zticks([])


def plot_tsne_layers(hidden_states_list, labels, perplexity=50, n_components=2, cols=8, elev=30, azim=45):
    """
    为每一层的隐藏状态绘制 t-SNE 可视化，支持二维和三维绘图。

    参数：
        hidden_states_list (list): 每个任务的隐藏状态列表（列表的每个元素代表一个任务的所有层数据）。
        labels (list): 每个任务对应的标签列表，与 hidden_states_list 对应。
        perplexity (int): t-SNE 的 perplexity 参数。
        n_components (int): 降维到2D或3D，取值为2或3。
        cols (int): 每行显示的子图数量。
        elev (int): 3D图的仰角。
        azim (int): 3D图的方位角。
    """
    num_layers = len(hidden_states_list[0]) - 1
    rows = math.ceil(num_layers / cols)

    fig = plt.figure(figsize=(24, rows * 3))
    axes = []
    for i in range(num_layers):
        if n_components == 3:
            ax = fig.add_subplot(rows, cols, i+1, projection='3d')
        else:
            ax = fig.add_subplot(rows, cols, i+1)
        axes.append(ax)

    unique_labels = list(set(labels))
    cmap = plt.cm.get_cmap("Set2", len(unique_labels))

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]

        # 提取该层所有任务的隐藏状态
        hidden_states_layer = [hidden_states[layer_idx] for hidden_states in hidden_states_list]
        hidden_states_layer = np.array(hidden_states_layer)

        if hidden_states_layer.ndim != 2:
            print(f"Layer {layer_idx} has unexpected shape: {hidden_states_layer.shape}")
            continue

        # t-SNE 降维
        tsne_results = _compute_tsne(hidden_states_layer, n_components, perplexity)

        # 计算聚类评价指标
        silhouette_avg = silhouette_score(tsne_results, labels_encoded)
        db_score = davies_bouldin_score(tsne_results, labels_encoded)

        # 绘制散点图
        _plot_tsne_scatter(ax, tsne_results, labels, unique_labels, cmap, n_components,
                           point_size=40, alpha=0.2, elev=elev, azim=azim)

        # 设置标题
        ax.set_title(f'Layer {layer_idx}\nSilhouette: {silhouette_avg:.3f}, DB: {db_score:.3f}')

    # 删除多余的子图（若有）
    total_subplots = rows * cols
    if total_subplots > num_layers:
        for idx in range(num_layers, total_subplots):
            if idx < len(axes):
                fig.delaxes(axes[idx])

    plt.tight_layout()

    # 创建全局图例（最后一幅子图的图例即可）
    handles, labels_legend = axes[-1].get_legend_handles_labels()
    fig.legend(handles, unique_labels, loc='upper right', bbox_to_anchor=(1.13, 1))

    plt.show()

def plot_selected_layers(
        hidden_states_list,
        labels,
        perplexity=50,
        unique_labels=None,
):
    """
    绘制选定层的 t-SNE 可视化，按任务类别分组色系。
    可通过 unique_labels 控制展示哪些任务。
    """
    if unique_labels is None:
        unique_labels = sorted(list(set(labels)))
    else:
        unique_labels = sorted(unique_labels)

    # === 定义任务分组 ===
    task_groups = {
        "commonsense": ["hellaswag", "winogrande", "piqa", "boolq"],
        "knowledge_qa": ["arc_c", "arc_e", "obqa"],
        "language_model": ["c4", "wikitext2"],
        "code_gen": ["humaneval", "mbpp"],
        "math_reasoning": ["math_qa", "gsm8k"]
    }

    group_colormaps = {
        "commonsense": plt.cm.Blues,
        "knowledge_qa": plt.cm.Purples,
        "language_model": plt.cm.Greens,
        "code_gen": plt.cm.Greys,
        "math_reasoning": plt.cm.Oranges,
    }

    label_to_group = {label: group for group, labels in task_groups.items() for label in labels}
    group_order = list(task_groups.keys())

    # 仅保留在 unique_labels 中的
    group_label_list = [label for group in group_order for label in task_groups[group] if label in unique_labels]

    def get_label_color(label):
        group = label_to_group.get(label, "misc")
        cmap = group_colormaps.get(group, plt.cm.Greys)
        idx_in_group = task_groups[group].index(label)
        n_in_group = len(task_groups[group])
        return cmap((idx_in_group + 1) / (n_in_group + 1))

    label_colors = {label: get_label_color(label) for label in group_label_list}

    point_size = 150
    font_size = 18

    total_layers = len(hidden_states_list[0])
    middle_layers = np.linspace(1, total_layers - 2, 8, dtype=int)
    layers = [0] + list(middle_layers) + [total_layers - 1]

    fig = plt.figure(figsize=(22, 9))
    rows, cols = 2, 5

    # === 过滤样本 ===
    filtered_idxs = [i for i, l in enumerate(labels) if l in unique_labels]
    filtered_hidden_states = [hidden_states_list[i] for i in filtered_idxs]
    filtered_labels = [labels[i] for i in filtered_idxs]

    le = LabelEncoder()
    labels_encoded = le.fit_transform(filtered_labels)

    for idx, layer_idx in enumerate(layers):
        ax = fig.add_subplot(rows, cols, idx + 1)
        ax.set_facecolor('#f9f9f9')

        hidden_states_layer = [h[layer_idx] for h in filtered_hidden_states]
        hidden_states_array = np.array(hidden_states_layer)

        if hidden_states_array.ndim == 2 and len(hidden_states_array) > perplexity:
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            hidden_states_tsne = tsne.fit_transform(hidden_states_array)

            for label in group_label_list:
                idxs = [k for k, l in enumerate(filtered_labels) if l == label]
                if not idxs:
                    continue
                ax.scatter(
                    hidden_states_tsne[idxs, 0], hidden_states_tsne[idxs, 1],
                    label=label,
                    color=label_colors[label],
                    s=point_size,
                    alpha=0.12
                )

            ax.set_title(f'Layer {layer_idx}', fontsize=font_size)
            ax.tick_params(axis='both', which='major', labelsize=font_size - 2)
            ax.grid(True, linestyle='--', linewidth=0.5, color='lightgrey')
        else:
            ax.set_title(f"Layer {layer_idx} (skipped)", fontsize=font_size)
            print(f"⚠️ Layer {layer_idx} skipped due to invalid shape or low sample size.")

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    # 图例
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=label_colors[label], markersize=10, label=label)
        for label in group_label_list if label in filtered_labels
    ]

    fig.legend(
        handles=handles,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=font_size,
        frameon=False
    )

    plt.subplots_adjust(hspace=0.3, wspace=0.3, right=0.85)
    plt.tight_layout()
    plt.show()

