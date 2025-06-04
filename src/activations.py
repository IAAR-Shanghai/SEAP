"""
Activation analysis utilities for transformer models.

This module provides utilities for collecting and analyzing neural network
activations, including online statistics tracking and visualization tools.

Author: why
Date: 2024
"""

# Standard library imports
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Union, Tuple

# Third-party imports
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib.gridspec as gridspec


class OnlineStats:
    """Maintains running statistics for neural network activations.
    
    Uses an online update strategy to track:
        - running_mean: Average value for each feature dimension
        - running_var: Variance approximation for each feature dimension
        - running_l2: Mean of squared values (L2) for each feature dimension
    
    The online update strategy allows dynamic estimation of statistics
    without storing all sample data in memory.
    
    Attributes:
        hidden_dim: Dimension of the feature vectors
        running_mean: Running mean of features [D]
        running_var: Running variance of features [D]
        running_l2: Mean of squared features (L2) [D]
        sample_count: Total number of samples processed
    """

    def __init__(self, hidden_dim: Optional[int] = None):
        """Initialize the online statistics tracker.
        
        Args:
            hidden_dim: Optional dimension of feature vectors
        """
        self.hidden_dim = hidden_dim
        self.running_mean = None
        self.running_var = None
        self.running_l2 = None
        self.sample_count = 0

    def update(self, x: torch.Tensor) -> None:
        """Update statistics with a new batch of data.
        
        Supports input shapes of (batch, length, dim) or (N, dim);
        internally flattens to (N, D).
        
        Args:
            x: Input tensor to update statistics with
            
        Steps:
            1. Flatten to [N, D]
            2. Convert to float32 to avoid numerical issues
            3. Initialize statistics if first update
            4. Update existing statistics using online merging
        """
        # 1. Flatten to [N, D]
        x = x.view(-1, x.shape[-1])

        # 2. Convert to float32 to avoid numerical issues
        x = x.to(torch.float32)

        # Check for numerical issues
        if torch.isinf(x).any():
            print(f"[OnlineStats] After conversion, x has Inf values, shape={x.shape}")
        if torch.isnan(x).any():
            print(f"[OnlineStats] x has NaN values after conversion, shape={x.shape}")

        n_new = x.shape[0]
        
        # 3. Initialize statistics if first update
        if self.sample_count == 0:
            self.hidden_dim = x.shape[-1]
            self.running_mean = x.mean(dim=0)
            self.running_var = torch.zeros_like(self.running_mean)
            self.running_l2 = (x ** 2).mean(dim=0)
            self.sample_count = n_new
            return

        # 4. Update existing statistics using online merging
        old_mean = self.running_mean.clone()
        total_count = self.sample_count + n_new

        alpha_old = self.sample_count / total_count
        alpha_new = n_new / total_count

        batch_mean = x.mean(dim=0)
        self.running_mean = alpha_old * self.running_mean + alpha_new * batch_mean

        # Update variance
        if total_count > 1:
            self.running_var *= (self.sample_count - 1) / (total_count - 1)

        diff_sum = ((x - self.running_mean) * (x - old_mean)).sum(dim=0)
        self.running_var += diff_sum / total_count

        # Update L2 mean
        batch_l2 = (x ** 2).mean(dim=0)
        self.running_l2 = alpha_old * self.running_l2 + alpha_new * batch_l2

        # Update sample count
        self.sample_count = total_count

    def get_stats(self) -> Dict[str, Optional[torch.Tensor]]:
        """Return current statistics.
        
        Returns:
            Dictionary containing mean, variance, and L2 statistics
        """
        if self.sample_count == 0:
            return {
                "mean": None,
                "var": None,
                "l2": None
            }
        return {
            "mean": self.running_mean,
            "var": self.running_var,
            "l2": self.running_l2
        }

    def reset(self) -> None:
        """Reset all statistics to initial state."""
        self.running_mean = None
        self.running_var = None
        self.running_l2 = None
        self.sample_count = 0


class ActivationHookManager:
    """Manages forward hooks for collecting neural network activations.
    
    Tracks activations at key points in transformer layers:
        - Attention input states
        - Post-attention aggregation
        - MLP input states
        - MLP intermediate states
    """

    def __init__(self):
        """Initialize the activation hook manager."""
        self.layer_activations = {}

    def _init_stats_dict(self) -> Dict[str, OnlineStats]:
        """Initialize statistics trackers for each activation point.
        
        Returns:
            Dictionary mapping activation points to OnlineStats objects
        """
        return {
            'attention_input_states': OnlineStats(),
            'attention_post_aggregation': OnlineStats(),
            'mlp_input_states': OnlineStats(),
            'mlp_intermediate_states': OnlineStats()
        }

    def get_layer_hooks(self, layer_idx: int, layer: torch.nn.Module) -> Tuple[callable, ...]:
        """Create hook functions for tracking layer activations.
        
        Args:
            layer_idx: Index of the layer
            layer: The transformer layer module
            
        Returns:
            Tuple of hook functions for different components
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

    def register_activation_hooks(self, model: torch.nn.Module) -> None:
        """Register hooks for collecting activations from all layers.
        
        Args:
            model: The transformer model to hook into
        """
        self.layer_activations.clear()
        for i, layer in enumerate(model.model.layers):
            self.layer_activations[i] = self._init_stats_dict()
            q_hook, o_hook, g_hook, d_hook = self.get_layer_hooks(i, layer)
            layer.self_attn.q_proj.register_forward_hook(q_hook)
            layer.self_attn.o_proj.register_forward_hook(o_hook)
            layer.mlp.gate_proj.register_forward_hook(g_hook)
            layer.mlp.down_proj.register_forward_hook(d_hook)

    def clear_activations(self) -> None:
        """Clear all collected activation statistics."""
        for layer_idx in self.layer_activations:
            for key in self.layer_activations[layer_idx]:
                self.layer_activations[layer_idx][key].reset()


def capture_activations(
    model: torch.nn.Module,
    tokenizer,
    hook_manager: ActivationHookManager,
    shot_inputs: List[str],
    task_types: List[str]
) -> Dict[str, Dict]:
    """Collect activation statistics for different tasks.
    
    Args:
        model: The transformer model to analyze
        tokenizer: Tokenizer for processing inputs
        hook_manager: Manager for activation hooks
        shot_inputs: List of input prompts
        task_types: List of corresponding task types
        
    Returns:
        Dictionary mapping tasks to their activation statistics
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
                        "var": stats["var"].float(),
                        "l2": stats["l2"].float()
                    }

            task_activations[ttype] = final_dict

    return task_activations


def save_activations_dict(
    activations_dict: Dict[str, Dict],
    output_root: str = '../activations'
) -> None:
    """Save activation statistics to disk.
    
    Args:
        activations_dict: Dictionary of activation statistics by task
        output_root: Root directory for saving files
    """
    os.makedirs(output_root, exist_ok=True)

    for task_type, task_data in activations_dict.items():
        task_dir = os.path.join(output_root, task_type)
        os.makedirs(task_dir, exist_ok=True)
        save_path = os.path.join(task_dir, 'activations.pt')
        torch.save(task_data, save_path)
        print(f"[save_activations_dict] Saved {task_type} to {save_path}")


def load_activations(root_path: str = '../activations') -> Dict[str, Dict]:
    """Load saved activation statistics from disk.
    
    Args:
        root_path: Root directory containing activation files
        
    Returns:
        Dictionary mapping task types to their activation statistics
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

def compute_and_save_weight_l2(model: torch.nn.Module, save_path: str) -> None:
    """Compute and save L2 norms of model weights.
    
    Args:
        model: The transformer model to analyze
        save_path: Path to save the weight statistics
    """
    weight_l2_info = {}
    
    for i, layer in enumerate(model.model.layers):
        layer_info = {}
        
        # Attention weights
        q_weight = layer.self_attn.q_proj.weight
        k_weight = layer.self_attn.k_proj.weight
        v_weight = layer.self_attn.v_proj.weight
        o_weight = layer.self_attn.o_proj.weight
        
        # MLP weights
        gate_weight = layer.mlp.gate_proj.weight
        up_weight = layer.mlp.up_proj.weight
        down_weight = layer.mlp.down_proj.weight
        
        layer_info["attention"] = {
            "q_proj": (q_weight ** 2).sum(dim=0).cpu(),
            "k_proj": (k_weight ** 2).sum(dim=0).cpu(),
            "v_proj": (v_weight ** 2).sum(dim=0).cpu(),
            "o_proj": (o_weight ** 2).sum(dim=0).cpu(),
        }
        
        layer_info["mlp"] = {
            "gate_proj": (gate_weight ** 2).sum(dim=0).cpu(),
            "up_proj": (up_weight ** 2).sum(dim=0).cpu(),
            "down_proj": (down_weight ** 2).sum(dim=0).cpu(),
        }
        
        weight_l2_info[i] = layer_info
    
    torch.save(weight_l2_info, save_path)
    print(f"[compute_and_save_weight_l2] Saved weight L2 info to {save_path}")

def load_weight_l2_info(weight_l2_file: str) -> Dict:
    """Load saved weight L2 statistics from disk.
    
    Args:
        weight_l2_file: Path to the saved weight statistics file
        
    Returns:
        Dictionary containing weight L2 norms by layer and component
        
    Raises:
        FileNotFoundError: If the weight statistics file doesn't exist
    """
    if not os.path.exists(weight_l2_file):
        raise FileNotFoundError(f"Weight L2 file not found: {weight_l2_file}")
    
    weight_l2_info = torch.load(weight_l2_file)
    return weight_l2_info

def plot_selected_neurons_activations(
    activations: Dict[str, Dict],
    task_groups: Dict[str, List[str]],
    layers_to_plot: List[int],
    neuron_indices: np.ndarray,
    module_name: str = "mlp_intermediate_states",
    plot_field: str = "l2",
    hidden_size: int = 4096,
    num_heads: int = 32,
    head_level: bool = False,
    normalize: bool = True,
    tasks_per_group: int = 2,
    random_seed: Optional[int] = 42,
    task_fontsize: int = 96,
    tick_fontsize: int = 48,
    cbar_fontsize: int = 48,
    wspace: float = 0.35,
    hspace: float = 0.6,
) -> None:
    """Plot activation patterns for selected neurons across tasks.
    
    Creates a visualization showing how selected neurons respond to different
    tasks, optionally normalized and grouped by task type.
    
    Args:
        activations: Dictionary of activation statistics by task
        task_groups: Mapping of task groups to their member tasks
        layers_to_plot: List of layer indices to visualize
        neuron_indices: Indices of neurons to analyze
        module_name: Name of the module to analyze activations from
        plot_field: Which activation statistic to plot ('mean', 'var', or 'l2')
        hidden_size: Size of hidden layers
        num_heads: Number of attention heads
        head_level: Whether to analyze at attention head level
        normalize: Whether to normalize activation values
        tasks_per_group: Number of tasks to sample from each group
        random_seed: Random seed for task sampling
        task_fontsize: Font size for task labels
        tick_fontsize: Font size for tick labels
        cbar_fontsize: Font size for colorbar labels
        wspace: Width spacing between subplots
        hspace: Height spacing between subplots
    """
    def zscore(x: np.ndarray) -> np.ndarray:
        """Normalize array to zero mean and unit variance."""
        return (x - x.mean()) / (x.std() + 1e-8)

    def extract(task_act: Dict, layers: List[int], ids: np.ndarray) -> np.ndarray:
        """Extract activation values for specified layers and neurons."""
        values = []
        for layer_idx in layers:
            layer_data = task_act[layer_idx][module_name][plot_field]
            values.append(layer_data[ids].cpu().numpy())
        return np.stack(values)

    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)

    # Sample tasks from each group
    selected_tasks = []
    for group_tasks in task_groups.values():
        available = list(set(group_tasks) & set(activations.keys()))
        if available:
            n_sample = min(tasks_per_group, len(available))
            selected_tasks.extend(random.sample(available, n_sample))

    # Prepare plot grid
    n_tasks = len(selected_tasks)
    n_layers = len(layers_to_plot)
    fig = plt.figure(figsize=(n_tasks * 3, n_layers * 3))
    gs = gridspec.GridSpec(n_layers, n_tasks)

    # Extract and plot activations
    all_values = []
    for task_idx, task in enumerate(selected_tasks):
        task_values = extract(activations[task], layers_to_plot, neuron_indices)
        if normalize:
            task_values = zscore(task_values)
        all_values.append(task_values)

        for layer_idx in range(n_layers):
            ax = fig.add_subplot(gs[layer_idx, task_idx])
            im = ax.imshow(task_values[layer_idx].reshape(-1, 1),
                         aspect='auto', cmap='RdBu_r')

            # Customize plot appearance
            if layer_idx == 0:
                ax.set_title(task, fontsize=task_fontsize, pad=20)
            if task_idx == 0:
                ax.set_ylabel(f"Layer {layers_to_plot[layer_idx]}", fontsize=tick_fontsize)

            ax.set_xticks([])
            ax.set_yticks([])

    # Add colorbar
    fig.colorbar(im, ax=fig.axes, location='right', shrink=0.8,
                label="Normalized Activation" if normalize else "Activation",
                orientation='vertical')

    # Adjust layout
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()


