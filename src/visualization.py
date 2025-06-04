"""
Visualization utilities for analyzing model hidden states.

This module provides visualization functions for analyzing and plotting
hidden states from transformer models, including t-SNE visualizations
and task-specific color schemes.

Author: why
Date: 2024
"""

# Standard library imports
import math
from typing import List, Dict, Optional

# Third-party imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Colormap, to_hex
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder

# === Task Groups and Color Schemes ===
task_groups = {
    "code_gen":       ["humaneval", "mbpp"],
    "math_reasoning": ["gsm8k", "mathqa"],
    "knowledge_qa":   ["arc_challenge", "arc_easy", "openbookqa"],
    "comparison":     ["boolq", "race"],
    "commonsense":    ["piqa", "winogrande", "hellaswag"],
    "language_model": ["c4", "wikitext2"],
}

# Display names mapping for plot labels
task_display_name = {
    "gsm8k": "GSM8K",
    "mathqa": "MathQA",
    "arc_challenge": "ARC-C",
    "arc_easy": "ARC-E",
    "openbookqa": "OBQA",
    "hellaswag": "HellaSwag",
    "winogrande": "WinoG.",
    "piqa": "PIQA",
    "boolq": "BoolQ",
    "wikitext2": "WikiText2",
    "c4": "C4",
    "mbpp": "MBPP",
    "humaneval": "HumanEval",
    "race": "RACE"
}

# Color maps for each task group
group_colormaps: Dict[str, Colormap] = {
    "code_gen":       plt.cm.Greys,
    "math_reasoning": plt.cm.PuBu,
    "knowledge_qa":   plt.cm.PuRd,
    "comparison":     plt.cm.YlOrBr,
    "commonsense":    plt.cm.Reds,
    "language_model": plt.cm.YlGn,
}

# Mapping from task labels to groups
label_to_group = {lbl: grp for grp, lst in task_groups.items() for lbl in lst}
group_order = list(task_groups.keys())


def get_label_color(label: str) -> str:
    """Get color for a task label based on its position within its group.
    
    Args:
        label: Task label to get color for
        
    Returns:
        Hex color code string
    """
    grp = label_to_group.get(label, "misc")
    cmap = group_colormaps.get(grp, plt.cm.Greys)
    idx = task_groups[grp].index(label)
    n = len(task_groups[grp])
    return to_hex(cmap((idx + 1) / (n + 1)))


def plot_tsne_layers(
    hidden_states_list: List[np.ndarray],
    labels: List[str],
    perplexity: int = 50,
    n_components: int = 2,
    cols: int = 8,
    elev: int = 30,
    azim: int = 45,
) -> None:
    """Plot t-SNE visualizations for each layer's hidden states.
    
    Creates a grid of t-SNE plots showing the clustering of hidden states
    for each layer, with consistent colors and legends across task groups.
    
    Args:
        hidden_states_list: List of hidden states arrays for each sample
        labels: List of task labels for each sample
        perplexity: Perplexity parameter for t-SNE
        n_components: Number of t-SNE dimensions (2 or 3)
        cols: Number of columns in the plot grid
        elev: Elevation angle for 3D plots
        azim: Azimuth angle for 3D plots
    """
    num_layers = len(hidden_states_list[0]) - 1
    rows = math.ceil(num_layers / cols)
    fig = plt.figure(figsize=(24, rows * 3))
    axes = [fig.add_subplot(rows, cols, i + 1,
            projection="3d" if n_components == 3 else None)
            for i in range(num_layers)]

    # Order legend by task groups
    unique_labels = [lbl for grp in group_order for lbl in task_groups[grp] if lbl in set(labels)]
    label_colors = {lbl: get_label_color(lbl) for lbl in unique_labels}
    le = LabelEncoder().fit(labels)
    labels_enc = le.transform(labels)

    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        layer_states = np.array([h[layer_idx] for h in hidden_states_list])
        if layer_states.ndim != 2:
            print(f"⚠️ Layer {layer_idx} skipped (shape={layer_states.shape})")
            continue

        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        tsne_xy = tsne.fit_transform(layer_states)

        sil = silhouette_score(tsne_xy, labels_enc)
        db = davies_bouldin_score(tsne_xy, labels_enc)

        for lbl in unique_labels:
            idxs = [i for i, l in enumerate(labels) if l == lbl]
            color = label_colors[lbl]
            if n_components == 3:
                ax.scatter(tsne_xy[idxs, 0], tsne_xy[idxs, 1], tsne_xy[idxs, 2],
                          s=30, alpha=0.25, color=color)
                ax.view_init(elev=elev, azim=azim)
            else:
                ax.scatter(tsne_xy[idxs, 0], tsne_xy[idxs, 1],
                          s=30, alpha=0.25, color=color)

        ax.set_title(f'Layer {layer_idx}\nSilhouette={sil:.3f}, DB={db:.3f}')
        ax.set_xticks([]); ax.set_yticks([])
        if n_components == 3:
            ax.set_zticks([])

    # Remove empty subplots
    for i in range(num_layers, rows * cols):
        fig.delaxes(axes[i])

    # Global legend
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=label_colors[l], markersize=9, label=l)
               for l in unique_labels]
    fig.legend(
        handles,
        [task_display_name.get(h.get_label(), h.get_label()) for h in handles],
        loc='upper right',
        bbox_to_anchor=(1.14, 1))
    plt.tight_layout()
    plt.show()


def plot_selected_layers(
    hidden_states_list: List[np.ndarray],
    labels: List[str],
    perplexity: int = 50,
    unique_labels: Optional[List[str]] = None,
) -> None:
    """Plot t-SNE visualizations for selected layers with enhanced legend.
    
    Creates a grid of t-SNE plots for key layers (first, last, and evenly spaced
    middle layers) with a large legend on the right side.
    
    Args:
        hidden_states_list: List of hidden states arrays for each sample
        labels: List of task labels for each sample
        perplexity: Perplexity parameter for t-SNE
        unique_labels: Optional list of task labels to include in visualization
    """
    if unique_labels is None:
        unique_labels = sorted(set(labels))

    group_label_list = [lbl for grp in group_order for lbl in task_groups[grp] if lbl in unique_labels]
    label_colors = {lbl: get_label_color(lbl) for lbl in group_label_list}

    tot_layers = len(hidden_states_list[0])
    middle = np.linspace(1, tot_layers - 2, 8, dtype=int)
    layers_shown = [0] + list(middle) + [tot_layers - 1]  # 10 layers

    fig = plt.figure(figsize=(27, 9))
    gs = gridspec.GridSpec(2, 6, width_ratios=[1, 1, 1, 1, 1, 0.6])
    rows, cols = 2, 6

    # Filter required samples
    idx_keep = [i for i, l in enumerate(labels) if l in unique_labels]
    states_keep = [hidden_states_list[i] for i in idx_keep]
    labels_keep = [labels[i] for i in idx_keep]

    for p, layer_idx in enumerate(layers_shown):
        row, col = divmod(p, 5)
        ax = fig.add_subplot(gs[row, col])
        ax.set_facecolor("#f9f9f9")

        layer_states = np.array([h[layer_idx] for h in states_keep])
        if layer_states.ndim != 2 or len(layer_states) <= perplexity:
            ax.set_title(f"Layer {layer_idx} (skipped)")
            continue

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        xy = tsne.fit_transform(layer_states)

        for lbl in group_label_list:
            pts = [i for i, l in enumerate(labels_keep) if l == lbl]
            if pts:
                ax.scatter(xy[pts, 0], xy[pts, 1],
                          color=label_colors[lbl], s=140, alpha=0.12)

        ax.set_title(f"Layer {layer_idx}", fontsize=24)
        ax.grid(True, linestyle='--', linewidth=0.5, color='lightgrey')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ['top', 'right']: ax.spines[spine].set_visible(False)

    # Legend subplot (bottom right)
    ax_legend = fig.add_subplot(gs[:, 5])
    ax_legend.axis("off")

    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=label_colors[l], markersize=12, label=l)
        for l in group_label_list if l in labels_keep
    ]
    ax_legend.legend(
        handles,
        [task_display_name.get(h.get_label(), h.get_label()) for h in handles],
        loc='center',
        frameon=False,
        fontsize=24,
        labelspacing=0.6,
        borderpad=0.5
    )

    plt.tight_layout()
    plt.show()

