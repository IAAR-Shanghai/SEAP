#!/usr/bin/env python3
# coding: utf-8

import sys
import os
import argparse
import torch
import json
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.activations import load_activations, load_weight_l2_info
from src.model_utils import load_model_and_tokenizer
from src.pruning_utils.compute_scores import compute_all_layers_scores
from src.pruning_utils.generate_masks import (
    generate_masks_for_all_layers,
    save_masks_to_file,
    compute_layerwise_sparsity
)
from src.remove_test import load_layerwise_results

def save_pruning_metadata(output_dir, task_type, method, strategy, pruning_ratio, sparsities, mask_file, prompt_type):
    folder_name = f"prompt={prompt_type}_task={task_type}_method={method}_strategy={strategy}_ratio={pruning_ratio}"
    task_output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(task_output_dir, exist_ok=True)

    sparsity_file = os.path.join(task_output_dir, "sparsity.json")
    with open(sparsity_file, 'w') as f:
        json.dump(sparsities, f, indent=4)

    metadata = {
        "task_type": task_type,
        "prompt_type": prompt_type,
        "method": method,
        "strategy": strategy,
        "pruning_ratio": pruning_ratio,
        "mask_file": mask_file,
    }
    metadata_file = os.path.join(task_output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"[✓] Metadata saved in {task_output_dir}")

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_path = os.path.join(args.model_root_path, args.model_name)
    print(f"[compute_masks] Loading model from {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval()

    weight_l2_file = os.path.join(args.activations_root_path, args.model_name, "weight_l2_info.pt")
    print(f"[compute_masks] Loading weight L2 from {weight_l2_file}")
    weight_l2_data = load_weight_l2_info(weight_l2_file)

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    intermediate_size = model.config.intermediate_size

    del model
    torch.cuda.empty_cache()

    print(f"[compute_masks] Model config: num_layers={num_layers}, hidden_size={hidden_size}, "
          f"num_heads={num_heads}, intermediate_size={intermediate_size}")

    model_output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    strategy_kwargs = {
        "protect_head": args.protect_head,
        "protect_tail": args.protect_tail
    }

    # Load layer importance data if needed
    cos_sims = remove_results = fitted_results = None
    if args.sparsity_strategy in {"cosine", "retention", "linear_fit", "logistic_fit"}:
        if not args.layer_importance_dir:
            raise ValueError(f"[!] Strategy '{args.sparsity_strategy}' requires --layer_importance_dir.")
        print(f"[INFO] Loading layer importance data from {args.layer_importance_dir}")
        _, cos_sims, remove_results, fitted_results = load_layerwise_results(
            model_name=args.model_name,
            base_dir=args.layer_importance_dir
        )

    for prompt_type in args.prompt_types:
        print(f"\n🔁 Processing prompt type: {prompt_type}")
        activations_dir = os.path.join(args.activations_root_path, args.model_name, prompt_type)
        print(f"[compute_masks] Loading activations from {activations_dir}")
        all_task_activations = load_activations(activations_dir)

        if args.tasks:
            missing = [t for t in args.tasks if t not in all_task_activations]
            if missing:
                raise ValueError(f"The following tasks were not found in activations: {missing}")
            task_list = args.tasks
        else:
            task_list = sorted(all_task_activations.keys())

        print(f"[compute_masks] Tasks for {prompt_type}: {task_list}")

        for task_type in task_list:
            print(f"\n[compute_masks] → task={task_type}")
            activation_data = all_task_activations[task_type]

            scores_dict = compute_all_layers_scores(
                activation_data=activation_data,
                weight_data=weight_l2_data,
                num_layers=num_layers,
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                method=args.method
            )

            attn_masks, mlp_masks = generate_masks_for_all_layers(
                scores_dict=scores_dict,
                strategy=args.sparsity_strategy,
                pruning_ratio=args.pruning_ratio,
                hidden_size=hidden_size,
                num_heads=num_heads,
                total_layers=num_layers,
                strategy_kwargs=strategy_kwargs,
                cos_sims=cos_sims,
                remove_results=remove_results,
                fitted_results=fitted_results
            )

            sparsities, global_sparsity = compute_layerwise_sparsity(
                attn_masks=attn_masks,
                mlp_masks=mlp_masks,
                hidden_size=hidden_size,
                num_heads=num_heads
            )

            print(f"Global sparsity: {global_sparsity:.4f}")
            for layer_idx, data in sparsities.items():
                print(f"Layer {layer_idx}: attn={data['attn_sparsity']:.3f}, mlp={data['mlp_sparsity']:.3f}")

            task_output_dir = os.path.join(
                model_output_dir,
                f"prompt={prompt_type}_task={task_type}_method={args.method}_strategy={args.sparsity_strategy}_ratio={args.pruning_ratio}"
            )
            os.makedirs(task_output_dir, exist_ok=True)

            mask_file = os.path.join(task_output_dir, f"{task_type}_masks.pt")
            save_masks_to_file(attn_masks, mlp_masks, mask_file)

            save_pruning_metadata(
                output_dir=model_output_dir,
                task_type=task_type,
                method=args.method,
                strategy=args.sparsity_strategy,
                pruning_ratio=args.pruning_ratio,
                sparsities=sparsities,
                mask_file=mask_file,
                prompt_type=prompt_type
            )

    print("\n✅ All prompt types and tasks processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute pruning masks from activations and weight L2.")
    parser.add_argument("--model_root_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--activations_root_path", type=str, default="./activations")
    parser.add_argument("--prompt_types", nargs="+", required=True,
                        choices=["zero_shot", "cot", "icl", "icl_cot", "knowledge"],
                        help="One or more prompt types to use.")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="List of task names to compute activations for. If not set, all tasks are processed.")
    parser.add_argument("--output_dir", type=str, default="./pruning_masks")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--method", type=str, default="WIFV", choices=["WIFV", "WIFN"])
    parser.add_argument("--sparsity_strategy", type=str, default="logistic",
                        choices=["uniform", "logistic", "global", "cosine", "retention", "linear_fit", "logistic_fit"])
    parser.add_argument("--pruning_ratio", type=float, default=0.2)
    parser.add_argument("--protect_head", type=int, default=2, help="Number of bottom layers to protect from pruning")
    parser.add_argument("--protect_tail", type=int, default=3, help="Number of top layers to protect from pruning")
    parser.add_argument("--layer_importance_dir", type=str, default=None,
                        help="Path to layerwise importance data (for cosine/retention/fit strategies)")

    args = parser.parse_args()
    main(args)
