#!/usr/bin/env python3
# coding: utf-8
# scripts/compute_activations.py

import sys
import os
import gc
import torch
import random
import numpy as np
from typing import List
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import custom modules
from src.activations import (
    ActivationHookManager,
    save_activations,
    compute_and_save_weight_l2
)
from src.data_utils import load_datasets, build_few_shot_prompts
from src.model_utils import load_model_and_tokenizer

def main(args):
    """Main function to compute and save activations for various tasks."""
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Construct model path and output directory for activations
    model_path = os.path.join(args.model_root_path, args.model_name)
    activations_output_dir = os.path.join(args.activations_root_path, args.model_name)
    os.makedirs(activations_output_dir, exist_ok=True)

    # Load datasets for training (or other appropriate splits)
    datasets = load_datasets(args.data_dir, split='train')

    # Build few-shot prompts based on the datasets
    shot_inputs, shot_task_types = build_few_shot_prompts(
        datasets,
        min_shot=args.min_shot,
        max_shot=args.max_shot,
        seed=args.seed,
        sample_size=args.sample_size, 
        use_corpus=True
    )

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval()

    # ===== Optionally compute and save the weight L2 norms =====
    if args.save_weight_l2:
        weight_l2_save_path = os.path.join(activations_output_dir, "weight_l2_info.pt")
        compute_and_save_weight_l2(model, weight_l2_save_path)

    # Register activation hooks to collect activations during forward pass
    hook_manager = ActivationHookManager()
    hook_manager.register_activation_hooks(model)

    # Collect and save activations using the registered hooks
    with torch.no_grad():
        save_activations(
            model=model,
            tokenizer=tokenizer,
            hook_manager=hook_manager,
            shot_inputs=shot_inputs,
            task_types=shot_task_types,
            output_root=activations_output_dir
        )

    print("Activations computation and saving completed.")

if __name__ == "__main__":
    # Argument parser for running the script
    parser = argparse.ArgumentParser(description="Compute activations for different tasks using model forward passes.")
    parser.add_argument("--data_dir", type=str, default="./data/processed", 
                        help="Directory where processed data is located.")
    parser.add_argument("--model_root_path", type=str, required=True, 
                        help="Root directory where models are stored.")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Name of the model to load.")
    parser.add_argument("--activations_root_path", type=str, default="./activations", 
                        help="Root directory to save activations.")
    parser.add_argument("--sample_size", type=int, default=200, 
                        help="Sample size for tasks to use in activation computation.")
    parser.add_argument("--min_shot", type=int, default=0, 
                        help="Minimum number of shots for few-shot prompts.")
    parser.add_argument("--max_shot", type=int, default=1, 
                        help="Maximum number of shots for few-shot prompts.")
    parser.add_argument("--shot_seed", type=int, default=44, 
                        help="Seed for generating few-shot prompts.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Seed for random number generation.")

    # Optional argument to save weight L2 norms
    parser.add_argument("--save_weight_l2", action='store_true', 
                        help="If set, compute and save weight L2 norms for the model.")

    args = parser.parse_args()
    main(args)
