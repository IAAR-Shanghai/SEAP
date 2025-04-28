#!/usr/bin/env python3
# coding: utf-8

import sys
import os
import gc
import torch
import random
import numpy as np
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.activations import (
    ActivationHookManager,
    capture_activations,
    save_activations_dict,
    compute_and_save_weight_l2
)
from data_preparation.data_utils import load_datasets, build_few_shot_prompts
from src.model_utils import load_model_and_tokenizer

def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_path = os.path.join(args.model_root_path, args.model_name)
    activations_output_dir = os.path.join(args.activations_root_path, args.model_name)
    os.makedirs(activations_output_dir, exist_ok=True)

    # Load data
    datasets = load_datasets(args.data_dir, split='train')
    shot_inputs, shot_task_types = build_few_shot_prompts(
        datasets,
        min_shot=args.min_shot,
        max_shot=args.max_shot,
        seed=args.shot_seed,
        sample_size=args.sample_size,
        use_corpus=True
    )

    # If tasks are specified, filter them
    if args.tasks:
        original_len = len(shot_inputs)
        filtered_inputs = []
        filtered_types = []
        for inp, ttype in zip(shot_inputs, shot_task_types):
            if ttype in args.tasks:
                filtered_inputs.append(inp)
                filtered_types.append(ttype)
        shot_inputs = filtered_inputs
        shot_task_types = filtered_types
        print(f"[compute_activations] Filtered tasks: {args.tasks}")
        print(f"[compute_activations] Reduced from {original_len} to {len(shot_inputs)} prompts.")

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval().to("cuda")

    if args.save_weight_l2:
        weight_l2_path = os.path.join(activations_output_dir, "weight_l2_info.pt")
        compute_and_save_weight_l2(model, weight_l2_path)

    hook_manager = ActivationHookManager()
    hook_manager.register_activation_hooks(model)

    print("[compute_activations] Capturing activations ...")
    captured_acts = capture_activations(
        model=model,
        tokenizer=tokenizer,
        hook_manager=hook_manager,
        shot_inputs=shot_inputs,
        task_types=shot_task_types
    )
    print("[compute_activations] Saving activations to disk ...")
    save_activations_dict(captured_acts, output_root=activations_output_dir)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    print("[compute_activations] Done. All activations and weights saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and save activations from model forward passes.")
    parser.add_argument("--data_dir", type=str, default="./data/processed")
    parser.add_argument("--model_root_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--activations_root_path", type=str, default="./activations")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="List of task names to compute activations for. If not set, all tasks are processed.")
    
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--min_shot", type=int, default=0)
    parser.add_argument("--max_shot", type=int, default=1)
    parser.add_argument("--shot_seed", type=int, default=44)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_weight_l2", action='store_true')

    args = parser.parse_args()
    main(args)
