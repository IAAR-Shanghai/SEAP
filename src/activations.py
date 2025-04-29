# activations.py

import os
import gc
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
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

def plot_selected_neurons_activations(
    subset1_activations: dict,
    subset2_activations: dict,
    layers_to_plot: list,
    task_names: list,
    neuron_indices,
    module_name: str = 'attention_post_aggregation',
    plot_field: str = 'mean',
    hidden_size: int = 4096,
    num_heads: int = 32,
    head_level: bool = False,
    fontsize: int = 48,
    tick_fontsize: int = 36,
    cbar_fontsize: int = 36,
    random_seed: int = None,
    hspace: float = 0.6,
    wspace: float = 0.3,
    normalize: bool = True,
    subset_names: tuple = ('Subset 1', 'Subset 2'),
):
    """
    Plots activation heatmaps for selected neurons across tasks and layers.
    Rows = subsets, Columns = tasks.
    """

    def z_score_standardize(data):
        std_val = np.std(data)
        if std_val == 0:
            return data - np.mean(data)
        return (data - np.mean(data)) / std_val

    def extract_matrix(task_acts: dict, layers, neuron_ids):
        mat = []
        for l in layers:
            vec = task_acts.get(l, {}).get(module_name, {}).get(plot_field, None)
            if vec is None or not isinstance(vec, torch.Tensor):
                mat.append(np.zeros(len(neuron_ids)))
            else:
                if head_level and 'attention' in module_name:
                    # Reshape hidden_size → [num_heads, head_dim]
                    head_dim = hidden_size // num_heads
                    if vec.shape[0] != hidden_size:
                        raise ValueError(f"Expected hidden size {hidden_size}, got {vec.shape[0]}")
                    vec = vec.view(num_heads, head_dim).mean(dim=1)  # [num_heads]

                selected = vec[neuron_ids].cpu().numpy()
                if normalize:
                    selected = z_score_standardize(selected)
                mat.append(selected)
        return np.array(mat)

    # Auto infer neuron indices
    if isinstance(neuron_indices, int):
        found = False
        for task in task_names:
            for acts in [subset1_activations.get(task, {}), subset2_activations.get(task, {})]:
                for l in layers_to_plot:
                    vec = acts.get(l, {}).get(module_name, {}).get(plot_field, None)
                    if vec is not None:
                        if head_level and 'attention' in module_name:
                            total_units = num_heads
                        else:
                            total_units = len(vec)
                        found = True
                        break
                if found: break
            if found: break

        if not found:
            raise ValueError("Unable to infer dimension from activations.")

        if random_seed is not None:
            random.seed(random_seed)
        neuron_indices = random.sample(range(total_units), neuron_indices)

    num_tasks = len(task_names)
    num_subsets = 2  # Subset 1 & Subset 2

    fig, axes = plt.subplots(num_subsets, num_tasks,
                             figsize=(len(neuron_indices) * num_tasks,
                                      len(layers_to_plot) * num_subsets * 2))
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    if num_tasks == 1:
        axes = np.array(axes).reshape(2, 1)

    for col_idx, task in enumerate(task_names):
        s1_acts = subset1_activations.get(task, {})
        s2_acts = subset2_activations.get(task, {})

        for row_idx, (acts, subset_label) in enumerate(zip([s1_acts, s2_acts], subset_names)):
            mat = extract_matrix(acts, layers_to_plot, neuron_indices)
            ax = axes[row_idx, col_idx]
            heatmap = sns.heatmap(
                mat,
                xticklabels=[f'D{i}' for i in neuron_indices],
                yticklabels=[f'L{i}' for i in layers_to_plot],
                cmap='coolwarm',
                center=0 if normalize else None,
                ax=ax,
                cbar=True,
                cbar_kws={'shrink': 0.8, 'label': f'Activation {plot_field}'}
            )
            heatmap.collections[0].colorbar.ax.tick_params(labelsize=cbar_fontsize)
            heatmap.collections[0].colorbar.set_label(f'Activation {plot_field}', size=cbar_fontsize)

            ax.set_title(f"{task} - {subset_label}", fontsize=fontsize, pad=20, fontweight='bold')
            ax.set_xlabel('Neuron Index', fontsize=fontsize, labelpad=10)
            ax.set_ylabel('Layer Index', fontsize=fontsize, labelpad=10)
            ax.tick_params(axis='x', labelsize=tick_fontsize, rotation=90)
            ax.tick_params(axis='y', labelsize=tick_fontsize)

    plt.tight_layout()
    plt.show()


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


def split_dataset_by_task(shot_inputs, task_types):
    """
    Splits the dataset into two subsets based on task types.
    
    Args:
        shot_inputs (list): List of input prompts.
        task_types (list): List of corresponding task types.
    
    Returns:
        tuple: Four lists containing:
            - subset1_shot_inputs (list): First subset of input prompts.
            - subset1_task_types (list): Corresponding task types for subset 1.
            - subset2_shot_inputs (list): Second subset of input prompts.
            - subset2_task_types (list): Corresponding task types for subset 2.
    """
    task_to_samples = defaultdict(list)
    
    for ttype, inp in zip(task_types, shot_inputs):
        task_to_samples[ttype].append(inp)
    
    s1_inps, s2_inps = [], []
    s1_types, s2_types = [], []
    
    for task, samples in task_to_samples.items():
        n = len(samples)
        split_idx = n // 2
        
        s1_inps.extend(samples[:split_idx])
        s1_types.extend([task] * split_idx)
        s2_inps.extend(samples[split_idx:])
        s2_types.extend([task] * (n - split_idx))
    
    return s1_inps, s1_types, s2_inps, s2_types


