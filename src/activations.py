# activations.py

import os
import random
from collections import defaultdict
from typing import Dict
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class OnlineStats:
    """
    Maintains statistics using an online update strategy:
      - running_mean: Average value for each feature dimension
      - running_var:  Variance approximation for each feature dimension
      - running_l2:   Mean of squared values (L2) for each feature dimension

    The idea is that when receiving a new batch x, we scale the existing statistics and blend them with the new batch 
    in the reverse proportion. This allows us to dynamically estimate the mean, approximate variance, and L2 mean 
    without storing all the sample data.
    """

    def __init__(self, hidden_dim=None):
        self.hidden_dim = hidden_dim

        self.running_mean = None  # [D], running mean of features
        self.running_var  = None  # [D], running variance of features
        self.running_l2   = None  # [D], mean of squared features (L2)
        self.sample_count = 0     # Total number of samples (including batch and sequence dimensions)

    def update(self, x: torch.Tensor):
        """
        Update the statistics using a new batch x.
        Supports input shapes of (batch, length, dim) or (N, dim); internally flattened to (N, D).

        Steps:
          1. Flatten to [N, D]
          2. Clamp to avoid extreme values or overflows
          3. If it's the first update, directly initialize the statistics
          4. If statistics already exist, update them using online merging strategy
        """
        # 1. Flatten => [N, D]
        x = x.view(-1, x.shape[-1])

        # 2. Clamp to avoid extreme values causing floating point overflow
        x = x.to(torch.float32)  # Or x.float()

        # Debug: Check for Inf/NaN
        if torch.isinf(x).any():
            print(f"[OnlineStats] After clamp, x still has Inf, shape={x.shape}")
        if torch.isnan(x).any():
            print(f"[OnlineStats] x has NaN after clamp, shape={x.shape}")

        n_new = x.shape[0]
        
        # 3. If it's the first update, initialize the statistics
        if self.sample_count == 0:
            self.hidden_dim = x.shape[-1]
            self.running_mean = x.mean(dim=0)        # [D]
            self.running_var  = torch.zeros_like(self.running_mean)
            self.running_l2   = (x ** 2).mean(dim=0) # [D]
            self.sample_count = n_new
            return

        # 4. If statistics already exist, update using online merging
        old_mean = self.running_mean.clone()
        total_count = self.sample_count + n_new

        alpha_old = self.sample_count / total_count
        alpha_new = n_new / total_count

        batch_mean = x.mean(dim=0)  # [D]
        self.running_mean = alpha_old * self.running_mean + alpha_new * batch_mean

        # 4.1 Update variance (running_var)
        if total_count > 1:
            self.running_var *= (self.sample_count - 1) / (total_count - 1)

        diff_sum = ((x - self.running_mean) * (x - old_mean)).sum(dim=0)  # [D]

        self.running_var += diff_sum / total_count

        # 4.2 Update L2 mean
        batch_l2 = (x ** 2).mean(dim=0)  # [D]
        self.running_l2 = alpha_old * self.running_l2 + alpha_new * batch_l2

        # 4.3 Update sample count
        self.sample_count = total_count

    def get_stats(self):
        """
        Return the current statistics: mean, variance, and L2.
        """
        if self.sample_count == 0:
            return {
                "mean": None,
                "var":  None,
                "l2":   None
            }
        return {
            "mean": self.running_mean,
            "var":  self.running_var,
            "l2":   self.running_l2
        }

    def reset(self):
        """
        Reset the statistics and clear recorded data.
        """
        self.running_mean = None
        self.running_var  = None
        self.running_l2   = None
        self.sample_count = 0

class ActivationHookManager:
    """
    Manages forward hooks to collect activations.
    """
    def __init__(self):
        self.layer_activations = {}

    def _init_stats_dict(self):
        return {
            'attention_input_states': OnlineStats(),
            'attention_post_aggregation': OnlineStats(),
            'mlp_input_states': OnlineStats(),
            'mlp_intermediate_states': OnlineStats()
        }

    def get_layer_hooks(self, layer_idx, layer):
        """
        Return hook functions for each module to update activation statistics.
        Here we focus on the activation statistics, without passing weights to update.
        """
        def q_proj_hook(module, input, output):
            self.layer_activations[layer_idx]['attention_input_states'].update(
                input[0].detach().cpu()
            )

        def o_proj_hook(module, input, output):
            self.layer_activations[layer_idx]['attention_post_aggregation'].update(
                input[0].detach().cpu()
            )

        def gate_proj_hook(module, input, output):
            self.layer_activations[layer_idx]['mlp_input_states'].update(
                input[0].detach().cpu()
            )

        def down_proj_hook(module, input, output):
            self.layer_activations[layer_idx]['mlp_intermediate_states'].update(
                input[0].detach().cpu()
            )

        return q_proj_hook, o_proj_hook, gate_proj_hook, down_proj_hook

    def register_activation_hooks(self, model):
        """Register hooks for each layer in the model to collect activations."""
        self.layer_activations.clear()
        for i, layer in enumerate(model.model.layers):
            self.layer_activations[i] = self._init_stats_dict()
            q_hook, o_hook, g_hook, d_hook = self.get_layer_hooks(i, layer)
            layer.self_attn.q_proj.register_forward_hook(q_hook)
            layer.self_attn.o_proj.register_forward_hook(o_hook)
            layer.mlp.gate_proj.register_forward_hook(g_hook)
            layer.mlp.down_proj.register_forward_hook(d_hook)

    def clear_activations(self):
        """Clear the currently collected activation statistics."""
        for layer_idx in self.layer_activations:
            for key in self.layer_activations[layer_idx]:
                self.layer_activations[layer_idx][key].reset()


def capture_activations(model, tokenizer, hook_manager, shot_inputs, task_types):
    """
    Collect activation statistics in memory (no saving).

    Returns:
        Dict[str, Dict[int, Dict[str, Dict[str, torch.Tensor]]]]:
        A dictionary of task -> layer_idx -> module_name -> stats dicts (mean/var/l2)
    """
    task_to_prompts = {}
    for prompt, ttype in zip(shot_inputs, task_types):
        task_to_prompts.setdefault(ttype, []).append(prompt)

    model.eval()
    task_activations = {}

    with torch.no_grad():
        for ttype, prompts in task_to_prompts.items():
            hook_manager.clear_activations()
            for prompt in tqdm(prompts, desc=f"Processing {ttype}", unit="prompt"):
                inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
                _ = model(**inputs)
                del inputs
                torch.cuda.empty_cache()

            final_dict = {}
            for layer_idx, keys_dict in hook_manager.layer_activations.items():
                final_dict[layer_idx] = {}
                for key, stats_obj in keys_dict.items():
                    stats = stats_obj.get_stats()
                    final_dict[layer_idx][key] = {
                        "mean": stats["mean"].float(),
                        "var":  stats["var"].float(),
                        "l2":   stats["l2"].float()
                    }

            task_activations[ttype] = final_dict

    return task_activations

def save_activations_dict(activations_dict: Dict[str, Dict], output_root: str = '../activations'):
    """
    Save activations from memory to disk.

    Args:
        activations_dict (dict): A dict returned by `capture_activations`.
        output_root (str): Root folder to store each task's activations.
    """
    os.makedirs(output_root, exist_ok=True)

    for task_type, task_data in activations_dict.items():
        task_dir = os.path.join(output_root, task_type)
        os.makedirs(task_dir, exist_ok=True)
        save_path = os.path.join(task_dir, 'activations.pt')
        torch.save(task_data, save_path)
        print(f"[save_activations_dict] Saved {task_type} to {save_path}")

def load_activations(root_path='../activations'):
    """
    Load all saved activations from the given directory and return them organized by task type.
    """
    task_to_activations = {}
    if not os.path.exists(root_path):
        return task_to_activations

    for ttype in os.listdir(root_path):
        task_dir = os.path.join(root_path, ttype)
        if os.path.isdir(task_dir):
            print('Loading:', ttype)
            activations_file = os.path.join(task_dir, 'activations.pt')
            if os.path.exists(activations_file):
                loaded_acts = torch.load(activations_file)
                task_to_activations[ttype] = loaded_acts
    return task_to_activations

def compute_and_save_weight_l2(model, save_path):
    """
    Compute the L2 norm of weights from specific layers in the model (o_proj and down_proj),
    and save them as a file. The L2 norm is computed across the input channels (dim=0) for 
    both 'o_proj' and 'down_proj' weights.
    """
    weight_l2_info = {}
    for i, layer in enumerate(model.model.layers):
        layer_dict = {}

        # o_proj.weight.shape = [hidden_size, hidden_size] (out_features, in_features)
        # sum(dim=0) -> shape=[in_features=hidden_size]
        o_l2 = (layer.self_attn.o_proj.weight ** 2).sum(dim=0).cpu()

        # down_proj.weight.shape = [hidden_size, intermediate_size]
        # sum(dim=0) -> shape=[in_features=intermediate_size]
        down_l2 = (layer.mlp.down_proj.weight ** 2).sum(dim=0).cpu()

        layer_dict['o_proj'] = o_l2
        layer_dict['down_proj'] = down_l2

        weight_l2_info[i] = layer_dict

    torch.save(weight_l2_info, save_path)
    print(f"Saved weight L2 info to {save_path}")

def load_weight_l2_info(weight_l2_file):
    """
    Loads the weight L2 norm information for each layer from a specified file.
    
    Args:
        weight_l2_file (str): Path to the file containing weight L2 norm information.
    
    Returns:
        dict: A nested dictionary structured as:
            {
                layer_idx: {
                    'o_proj': tensor_of_l2,
                    'down_proj': tensor_of_l2,
                    ...
                },
                ...
            }
    
    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    if not os.path.exists(weight_l2_file):
        raise FileNotFoundError(f"Weight L2 info file not found: {weight_l2_file}")
    weight_l2_data = torch.load(weight_l2_file)
    print(f"Loaded weight L2 info from {weight_l2_file}")
    return weight_l2_data

def plot_selected_neurons_activations(
    activations            : dict[str, dict],
    task_groups            : dict[str, list[str]],
    layers_to_plot         : list[int],
    neuron_indices,
    module_name            : str  = "mlp_intermediate_states",
    plot_field             : str  = "l2",
    hidden_size            : int  = 4096,
    num_heads              : int  = 32,
    head_level             : bool = False,
    normalize              : bool = True,
    tasks_per_group        : int  = 2,
    random_seed            : int  | None = 42,
    task_fontsize          : int  = 96,
    tick_fontsize          : int  = 48,
    cbar_fontsize          : int  = 48,
    wspace                 : float= 0.35,
    hspace                 : float= 0.6,
):
    """
    Rows = 组内任务       Columns = 任务组
    activations: {task: {layer: {module: {field: tensor}}}}
    """

    def zscore(x):
        std = np.std(x)
        return (x - np.mean(x)) / std if std else x - np.mean(x)

    def extract(task_act, layers, ids):
        out = []
        for l in layers:
            vec = task_act.get(l, {}).get(module_name, {}).get(plot_field, None)
            if vec is None or not isinstance(vec, torch.Tensor):
                out.append(np.zeros(len(ids)))
            else:
                if head_level and "attention" in module_name:
                    head_dim = hidden_size // num_heads
                    vec = vec.view(num_heads, head_dim).mean(dim=1)
                sel = vec[ids].cpu().numpy()
                out.append(zscore(sel) if normalize else sel)
        return np.array(out)

    # —— 随机抽 neuron —— 
    if isinstance(neuron_indices, int):
        dim = None
        for t in activations.values():
            for l in layers_to_plot:
                v = t.get(l, {}).get(module_name, {}).get(plot_field, None)
                if v is not None:
                    dim = num_heads if (head_level and "attention" in module_name) else len(v)
                    break
            if dim:
                break
        if dim is None:
            raise ValueError("Cannot infer neuron dimension.")
        random.seed(random_seed)
        neuron_indices = random.sample(range(dim), neuron_indices)

    # —— Figure 网格 & 外部 colorbar 空间 —— 
    groups = list(task_groups.keys())
    n_cols = len(groups)
    n_rows = tasks_per_group

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * len(neuron_indices) * 1.2,
                 n_rows * len(layers_to_plot) * 3),
        gridspec_kw={
            "width_ratios": [1] * n_cols,
            "wspace": wspace,
            "hspace": hspace,
        },
        constrained_layout=False
    )

    # —— 绘制 heatmaps（关掉内部 colorbar） —— 
    last_hm = None
    for col, grp in enumerate(groups):
        tasks = task_groups[grp][:tasks_per_group]
        for row, task in enumerate(tasks):
            ax = axes[row, col]
            if task not in activations:
                ax.axis("off")
                continue

            mat = extract(activations[task], layers_to_plot, neuron_indices)
            last_hm = sns.heatmap(
                mat, ax=ax, cmap="coolwarm",
                center=0 if normalize else None,
                xticklabels=[f"N{i}" for i in neuron_indices],
                yticklabels=[f"L{l}" for l in layers_to_plot],
                cbar=False
            )
            ax.set_title(task, fontsize=task_fontsize, pad=15, fontweight="bold")
            ax.set_xlabel("Neuron Index", fontsize=tick_fontsize, labelpad=8)
            ax.set_ylabel("Layer Index",  fontsize=tick_fontsize, labelpad=8)
            ax.tick_params(axis='x', labelsize=tick_fontsize, rotation=90)
            ax.tick_params(axis='y', labelsize=tick_fontsize)

    # —— 统一 external colorbar —— 
    plt.tight_layout(rect=[0, 0, 0.90, 1])
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(
        last_hm.get_children()[0], cax=cbar_ax,
        label=f"Activation {plot_field}"
    )
    cbar_ax.tick_params(labelsize=cbar_fontsize)
    cbar_ax.yaxis.label.set_size(cbar_fontsize)

    plt.show()


