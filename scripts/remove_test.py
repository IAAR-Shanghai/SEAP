# remove_test.py

import os
import sys
import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Adjust project_root to point to the root of your project
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import custom modules
from src.data_utils import load_datasets, generate_tasks
from src.model_utils import load_model_and_tokenizer

from src.test_utils import (
    create_test_prompts_and_answers,
    test_model_on_prompts,
    extract_answer,
    calculate_accuracy,
    save_test_results,
    save_generation_args,
    get_config_hash,
)

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# 基本信息配置
root_path = "/home/hanyu/models/"
model_name = "Llama-3.2-3B-Instruct"
model_name_or_path = os.path.join(root_path, model_name)

# 此处仅在最外层加载一次模型用来获取num_layers等信息
# 后续实际执行剪枝测试前会再次加载模型以确保每次实验使用干净的模型状态
model_cpu, tokenizer = load_model_and_tokenizer(model_name_or_path)
model_cpu = model_cpu.cpu()

num_layers = model_cpu.config.num_hidden_layers
modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
task_types = ['mmlu', 'hellaswag', 'piqa', 'gsm8k', 'ai2_arc', 'winogrande']
subsets = ['Total']

# Define the root path for attributions data
save_root_path = './attributions/'
save_root_path = os.path.join(save_root_path, model_name)

# Load attributions data
attributions_data = load_attributions(
    task_types=task_types,
    subsets=subsets,
    modules=modules,
    save_root_path=save_root_path
)

# Extract 'Total' subset data
attributions = {task_type: attributions_data[task_type]['Total'] for task_type in task_types}

# Load test datasets
test_data_dir = './data/processed'
test_datasets = load_datasets(data_dir=test_data_dir, split='test')

# Generate test tasks
test_tasks = generate_tasks(test_datasets, sample_size=None, seed=42)

# Create test prompts and answers
test_prompts, test_answers = create_test_prompts_and_answers(test_tasks, sample_size=100)

# Specify tasks to test and generation arguments
tasks_to_test = ['mmlu', 'piqa']  # Modify as needed
generation_args = {
    'max_new_tokens': 30,
    'do_sample': False,
}

# Define sparsity levels and layers to prune
sparsity_levels = [20, 50, 100, 0]  # Modify as needed
layers_to_prune = list(range(num_layers))

# Define the base results directory
base_results_root = './remove_test'

# Initialize a list to store experiment results
experiment_results = []

# Loop over sparsity levels
for sparsity in sparsity_levels:
    # Set the results directory for the current sparsity level
    results_root = os.path.join(base_results_root, f'results_{sparsity}')
    
    # Loop over tasks to test
    for task_type in tasks_to_test:
        print(f"\nTesting task: {task_type} with sparsity {sparsity}%")
    
        prompts = test_prompts.get(task_type, [])
        actual_answers = test_answers.get(task_type, [])
    
        if not prompts:
            print(f"No test prompts found for task {task_type}. Skipping.")
            continue
    
        # Load attributions data for the task
        attributions_data = attributions[task_type]
    
        # Loop over layers to prune
        for layer_idx in layers_to_prune:
            # Define pruning configuration for the current layer and sparsity
            pruning_config = {
                layer_idx: {module: sparsity for module in modules}
            }
            # Set other layers' sparsity to 0%
            for other_layer in range(num_layers):
                if other_layer != layer_idx:
                    pruning_config[other_layer] = {module: 0 for module in modules}
    
            # Generate a unique hash for the pruning configuration
            config_hash = get_config_hash(pruning_config)
    
            # Define the experiment directory
            experiment_dir = os.path.join(
                results_root,
                task_type,
                f'layer_{layer_idx}',
                f'experiment_{config_hash}'
            )
            os.makedirs(experiment_dir, exist_ok=True)
    
            # Save pruning configuration and generation arguments
            save_pruning_config(pruning_config, experiment_dir)
            save_generation_args(generation_args, experiment_dir)
    
            # 使用随机剪枝方式确定剪枝索引
            compute_and_save_random_pruning_indices(attributions_data, pruning_config, experiment_dir)
    
            # **在这里重新加载模型以确保每次剪枝开始时是干净的模型**
            fresh_model, tokenizer = load_model_and_tokenizer(model_name_or_path)
            fresh_model = fresh_model.cpu()

            # Prune the model
            pruning_indices_path = os.path.join(experiment_dir, 'pruning_indices')
            pruned_model = prune_model_for_task(fresh_model, pruning_indices_path)
            pruned_model.to('cuda')
    
            # Test the pruned model
            pruned_outputs = test_model_on_prompts(pruned_model, prompts, tokenizer, generation_args)
    
            # Extract predicted answers and calculate accuracy
            predicted_answers = [extract_answer(output) for output in pruned_outputs]
            accuracy = calculate_accuracy(predicted_answers, actual_answers)
            print(f"Accuracy for {task_type}: {accuracy:.2f}%")
            
            save_test_results(pruned_outputs, predicted_answers, actual_answers, accuracy, experiment_dir)
    
            # Record experiment results
            experiment_results.append({
                'task_type': task_type,
                'sparsity': sparsity,
                'layer_idx': layer_idx,
                'accuracy': accuracy
            })
    
            # Clean up to free memory
            pruned_model.to('cpu')
            del pruned_model
            del fresh_model
            torch.cuda.empty_cache()

# Convert experiment results to a DataFrame
df_results = pd.DataFrame(experiment_results)

# Save the experiment results to a CSV file for future analysis
df_results.to_csv(os.path.join(base_results_root, 'experiment_results.csv'), index=False)
