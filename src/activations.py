# activations.py

import os
import gc
import random
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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


def save_activations(model, tokenizer, hook_manager, shot_inputs, task_types, output_root='../activations'):
    """
    Run inference on different task prompts, collect activations via hooks, and save them to files.
    """
    os.makedirs(output_root, exist_ok=True)

    # Group prompts by task type
    task_to_prompts = {}
    for prompt, ttype in zip(shot_inputs, task_types):
        task_to_prompts.setdefault(ttype, []).append(prompt)

    model.eval()
    with torch.no_grad():
        for ttype, prompts in task_to_prompts.items():
            # Clear activation statistics before processing
            hook_manager.clear_activations()

            for prompt in tqdm(prompts, desc=f"Processing {ttype}", unit="prompt"):
                inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
                _ = model(**inputs)

                # Release temporary variables to free up memory
                del inputs
                gc.collect()
                torch.cuda.empty_cache()

            # Get the statistics after processing all prompts
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

            # Clear hook manager statistics
            hook_manager.clear_activations()
            gc.collect()
            torch.cuda.empty_cache()

            # Save the results
            task_dir = os.path.join(output_root, ttype)
            os.makedirs(task_dir, exist_ok=True)
            save_path = os.path.join(task_dir, 'activations.pt')
            torch.save(final_dict, save_path)
            print(f"Saved activations for task type: {ttype} to {save_path}")

    gc.collect()
    torch.cuda.empty_cache()

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

import os
import torch
from collections import defaultdict

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

def collect_and_save_subset_activations(model, tokenizer, hook_manager, shot_inputs, task_types, output_dir):
    """
    Collects and saves activation values for a given subset of data and task types.
    
    Args:
        model (torch.nn.Module): The model used to generate activations.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to process input text.
        hook_manager (ActivationHookManager): Hook manager to collect activations.
        shot_inputs (list): A list of input prompts.
        task_types (list): A list of corresponding task types for each input prompt.
        output_dir (str): Directory path to save the collected activations.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear activation statistics to avoid contamination from previous data
    hook_manager.clear_activations()
    
    # Collect and save activations
    with torch.no_grad():
        save_activations(
            model=model,
            tokenizer=tokenizer,
            hook_manager=hook_manager,
            shot_inputs=shot_inputs,
            task_types=task_types,
            output_root=output_dir
        )

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

def load_and_merge_subsets(subset1_root, subset2_root, subset1_name='Subset_1', subset2_name='Subset_2'):
    """
    Loads activations from two subsets and merges them into a structured dictionary.
    
    Args:
        subset1_root (str): Directory path of the first subset activations.
        subset2_root (str): Directory path of the second subset activations.
        subset1_name (str, optional): Name identifier for the first subset. Defaults to 'Subset_1'.
        subset2_name (str, optional): Name identifier for the second subset. Defaults to 'Subset_2'.
    
    Returns:
        dict: A dictionary with the structure:
            {
                task_type: {
                    subset1_name: activations_for_subset1,
                    subset2_name: activations_for_subset2
                },
                ...
            }
    """
    subset1_acts = load_activations(subset1_root)
    subset2_acts = load_activations(subset2_root)
    
    merged_data = {}
    
    for task_type in subset1_acts.keys():
        merged_data[task_type] = {
            subset1_name: subset1_acts[task_type],
            subset2_name: subset2_acts[task_type]
        }
    
    return merged_data


def plot_selected_neurons_activations(
    activations_data,
    layers_to_plot,
    neuron_indices,
    task_indices=None,
    subsets=['Subset_1', 'Subset_2'],
    module_name='attention_input_states',
    plot_field='mean',
    fontsize=48,
    tick_fontsize=36,
    cbar_fontsize=36,        # Font size for colorbar labels
    random_seed=None,
    hspace=0.6,              # Height space between subplots
    wspace=0.3,              # Width space between subplots
    normalize=True           # Whether to perform z-score normalization
):
    """
    Plots activation heatmaps for selected neurons across specific layers and tasks.
    The user can choose to normalize the data using z-score normalization.
    """
    
    def z_score_standardize(data):
        std_val = np.std(data)
        if std_val == 0:
            return data - np.mean(data)
        return (data - np.mean(data)) / std_val

    task_types = list(activations_data.keys())
    if task_indices is None:
        task_indices = range(len(task_types))
    selected_task_types = [task_types[i] for i in task_indices]

    # If neuron_indices is an integer, randomly select the specified number of neurons
    if isinstance(neuron_indices, int):
        num_neurons_to_select = neuron_indices
        found_neuron_count = None
        for ttype in selected_task_types:
            for subset_name in subsets:
                layer_data = activations_data[ttype].get(subset_name, {})
                for layer_idx in layers_to_plot:
                    data_dict = layer_data.get(layer_idx, {}).get(module_name, {})
                    if isinstance(data_dict, dict) and plot_field in data_dict:
                        found_neuron_count = len(data_dict[plot_field])
                        break
                if found_neuron_count is not None:
                    break
            if found_neuron_count is not None:
                break
        if found_neuron_count is None:
            raise ValueError("Unable to find available neuron data.")
        if num_neurons_to_select > found_neuron_count:
            raise ValueError(f"Requested number of neurons {num_neurons_to_select} exceeds available neurons {found_neuron_count}.")
        if random_seed is not None:
            random.seed(random_seed)
        neuron_indices = random.sample(range(found_neuron_count), num_neurons_to_select)

    num_task_types = len(selected_task_types)
    num_subsets = len(subsets)

    # Create subplots
    fig, axes = plt.subplots(num_subsets, num_task_types,
                             figsize=(len(neuron_indices) * num_task_types,
                                      len(layers_to_plot) * num_subsets * 2))
    # Adjust space between subplots
    fig.subplots_adjust(hspace=hspace, wspace=wspace)

    # Handle axes shape to ensure proper indexing
    if num_subsets == 1 and num_task_types == 1:
        axes = np.array([[axes]])
    elif num_subsets == 1:
        axes = axes[np.newaxis, :]
    elif num_task_types == 1:
        axes = axes[:, np.newaxis]

    for col_idx, task_type in enumerate(selected_task_types):
        module_data_for_task = activations_data[task_type]
        for row_idx, subset_name in enumerate(subsets):
            layer_data = module_data_for_task.get(subset_name, {})
            data_matrix = []
            for layer_idx in layers_to_plot:
                data_dict = layer_data.get(layer_idx, {}).get(module_name, {})
                if not isinstance(data_dict, dict) or plot_field not in data_dict:
                    data_matrix.append(np.zeros(len(neuron_indices)))
                    continue
                neuron_activations = data_dict[plot_field]
                if len(neuron_activations) == 0:
                    data_matrix.append(np.zeros(len(neuron_indices)))
                    continue
                if max(neuron_indices) >= len(neuron_activations):
                    raise IndexError(f"Neuron indices {neuron_indices} exceed dimension {len(neuron_activations)}")
                
                # Get activations for the selected neurons
                selected_neurons = neuron_activations[neuron_indices].cpu().numpy()
                
                # Perform z-score standardization if needed
                if normalize:
                    selected_neurons = z_score_standardize(selected_neurons)

                data_matrix.append(selected_neurons)

            data_matrix = np.array(data_matrix)
            layer_labels = [f'L {idx}' for idx in layers_to_plot]
            neuron_labels = [f'D {idx}' for idx in neuron_indices]
            ax = axes[row_idx, col_idx]

            # Plot heatmap
            heatmap = sns.heatmap(
                data_matrix,
                xticklabels=neuron_labels,
                yticklabels=layer_labels,
                cmap='coolwarm',
                center=0 if normalize else None,  # Center heatmap if normalized
                ax=ax,
                cbar=True,
                cbar_kws={
                    'shrink': 0.8,
                    'label': f'Activation {plot_field}'
                }
            )
            # Adjust colorbar font size
            cbar = heatmap.collections[0].colorbar
            cbar.ax.tick_params(labelsize=cbar_fontsize)  # Font size for tick labels
            cbar.set_label(f'Activation {plot_field}', size=cbar_fontsize)  # Font size for label

            # Set titles and axis labels
            ax.set_title(f'{task_type} - {subset_name}', fontsize=fontsize, fontweight='bold', pad=20)
            ax.set_xlabel('Dimensions', fontsize=fontsize, labelpad=10)
            ax.set_ylabel('Layers', fontsize=fontsize, labelpad=10)
            ax.tick_params(axis='x', labelsize=tick_fontsize, rotation=90)
            ax.tick_params(axis='y', labelsize=tick_fontsize)

    plt.show()
