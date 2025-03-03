#!/usr/bin/env python3
# coding: utf-8
# scripts/compute_masks.py

import sys
import os
import argparse
import torch
import json
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import custom modules
from src.activations import load_activations, load_weight_l2_info
from src.model_utils import load_model_and_tokenizer
from src.pruning_utils.compute_scores import compute_all_layers_scores
from src.pruning_utils.generate_masks import (
    generate_masks_for_all_layers,
    save_masks_to_file,
    compute_layerwise_sparsity
)

def compute_weighted_scores(
    scores_dicts: dict,
    wikitext2_weight: int = 3,
    task_weight: int = 2
) -> dict:
    """
    Compute weighted scores to generate a unified pruning mask across multiple tasks.
    
    Args:
        scores_dicts (dict): Dictionary of task scores. Each task contains layer scores (attn_scores and mlp_scores).
        wikitext2_weight (int, optional): Weight for the wikitext2 dataset. Default is 3.
        task_weight (int, optional): Weight for other tasks. Default is 2.
    
    Returns:
        dict: A dictionary containing the weighted scores for each layer. 
    """
    weighted_scores = {}

    # Compute weighted scores for wikitext2
    for layer_idx in scores_dicts.get('wikitext2', {}).keys():
        weighted_scores[layer_idx] = {
            'attn_scores': scores_dicts['wikitext2'][layer_idx]['attn_scores'] * wikitext2_weight,
            'mlp_scores': scores_dicts['wikitext2'][layer_idx]['mlp_scores'] * wikitext2_weight
        }

    # Compute weighted scores for other tasks
    for task_type, task_scores in scores_dicts.items():
        if task_type != 'wikitext2':
            for layer_idx, layer_scores in task_scores.items():
                if layer_idx not in weighted_scores:
                    weighted_scores[layer_idx] = {
                        'attn_scores': layer_scores['attn_scores'] * task_weight,
                        'mlp_scores': layer_scores['mlp_scores'] * task_weight
                    }
                else:
                    weighted_scores[layer_idx]['attn_scores'] += layer_scores['attn_scores'] * task_weight
                    weighted_scores[layer_idx]['mlp_scores'] += layer_scores['mlp_scores'] * task_weight

    return weighted_scores

def save_pruning_metadata(output_dir, task_type, method, structure, pruning_ratio, sparsities, mask_file):
    """
    Save metadata related to the pruning strategy, such as sparsity, model configuration, etc., for later use.
    
    Args:
        output_dir (str): Directory where metadata will be saved.
        task_type (str): The task type (e.g., "wikitext2").
        method (str): The method used for scoring (e.g., "WIFV").
        structure (str): The pruning structure (e.g., "UL-LD").
        pruning_ratio (float): The target pruning ratio.
        sparsities (dict): Layer-wise sparsity information.
        mask_file (str): Path to the saved mask file.
    """
    # Construct the folder name based on pruning metadata
    folder_name = f"task={task_type}_method={method}_structure={structure}_ratio={pruning_ratio}"
    task_output_dir = os.path.join(output_dir, folder_name)
    
    # Create the folder if it doesn't exist
    os.makedirs(task_output_dir, exist_ok=True)

    # Save sparsity information
    sparsity_file = os.path.join(task_output_dir, "sparsity.json")
    with open(sparsity_file, 'w') as f:
        json.dump(sparsities, f, indent=4)
    
    # Save other metadata like scoring method and pruning ratio
    metadata = {
        "task_type": task_type,
        "method": method,
        "structure": structure,
        "pruning_ratio": pruning_ratio,
        "mask_file": mask_file,
    }
    metadata_file = os.path.join(task_output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"[save_pruning_metadata] Metadata saved in {task_output_dir}")

def main(args):
    """Main function to compute pruning masks for all tasks."""
    # 1) Set random seed (optional)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 2) Construct model path and load the model
    model_path = os.path.join(args.model_root_path, args.model_name)
    print(f"[compute_masks] Loading model from {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval().to("cuda")

    # 3) Load activations and weight L2 data
    activations_dir = os.path.join(args.activations_root_path, args.model_name)
    print(f"[compute_masks] Loading activations from {activations_dir}")
    all_task_activations = load_activations(activations_dir)

    weight_l2_file = os.path.join(activations_dir, "weight_l2_info.pt")
    print(f"[compute_masks] Loading weight L2 from {weight_l2_file}")
    weight_l2_data = load_weight_l2_info(weight_l2_file)

    # 4) Extract model configuration
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    intermediate_size = model.config.intermediate_size

    del model
    torch.cuda.empty_cache()

    print(f"[compute_masks] Model config: num_layers={num_layers}, hidden_size={hidden_size}, "
          f"num_heads={num_heads}, intermediate_size={intermediate_size}")
    
    # 5) Collect task types (including wikitext2)
    task_list = sorted(all_task_activations.keys())
    print(f"[compute_masks] Found tasks in activation data: {task_list}")

    # Create a directory to save the results for the current model
    model_output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # 6) Generate and save masks for each task
    scores_dicts = {}

    for task_type in task_list:
        print(f"\n[compute_masks] Processing task={task_type} ...")
        activation_data = all_task_activations[task_type]
        
        # Compute task-specific scores
        scores_dict_task = compute_all_layers_scores(
            activation_data=activation_data,
            weight_data=weight_l2_data,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            method=args.method
        )
        
        scores_dicts[task_type] = scores_dict_task

        # Generate pruning masks
        attn_masks, mlp_masks = generate_masks_for_all_layers(
            scores_dict_task,
            structure=args.structure,
            pruning_ratio=args.pruning_ratio,       # Target average sparsity
            hidden_size=hidden_size, # For AL-AM structure
            num_heads=num_heads, # For AL-AM structure
            total_layers=num_layers, # For UL-LD structure
            logistic_k=args.logistic_k, # For UL-LD structure
            logistic_x0=args.logistic_x0, # For UL-LD structure
        )

        # Compute layer-wise sparsities
        sparsities = compute_layerwise_sparsity(attn_masks, mlp_masks)
        for layer_idx, data in sparsities.items():
            print(f"Layer {layer_idx}: attn_sparsity={data['attn_sparsity']:.3f}, "
                  f"mlp_sparsity={data['mlp_sparsity']:.3f}")

        # Save task-specific masks
        task_output_dir = os.path.join(
            model_output_dir, 
            f"task={task_type}_method={args.method}_structure={args.structure}_ratio={args.pruning_ratio}"
        )
        os.makedirs(task_output_dir, exist_ok=True)

        mask_file = os.path.join(task_output_dir, f"{task_type}_masks.pt")
        save_masks_to_file(attn_masks, mlp_masks, mask_file)
        print(f"[compute_masks] Saved masks for task={task_type} to {mask_file}")

        # Save pruning metadata
        save_pruning_metadata(
            output_dir=model_output_dir,
            task_type=task_type,
            method=args.method,
            structure=args.structure,
            pruning_ratio=args.pruning_ratio,
            sparsities=sparsities,
            mask_file=mask_file
        )

    # 7) If use_generic_mask is True, compute and save a generic mask based on weighted scores
    if args.use_generic_mask:
        print(f"\n[compute_masks] Using generic mask for evaluation.")
        
        weighted_scores = compute_weighted_scores(scores_dicts)
        
        # Generate the generic mask
        attn_masks, mlp_masks = generate_masks_for_all_layers(
            weighted_scores,
            structure=args.structure,
            pruning_ratio=args.pruning_ratio,  # Target average sparsity
            hidden_size=hidden_size,
            num_heads=num_heads,
            total_layers=num_layers,
            logistic_k=args.logistic_k,
            logistic_x0=args.logistic_x0,
        )

        sparsities = compute_layerwise_sparsity(attn_masks, mlp_masks)
        for layer_idx, data in sparsities.items():
            print(f"Layer {layer_idx}: attn_sparsity={data['attn_sparsity']:.3f}, "
                  f"mlp_sparsity={data['mlp_sparsity']:.3f}")

        # Save generic mask
        generic_mask_output_dir = os.path.join(
            model_output_dir, 
            f"task=generic_method={args.method}_structure={args.structure}_ratio={args.pruning_ratio}"
        )
        os.makedirs(generic_mask_output_dir, exist_ok=True)

        mask_file = os.path.join(generic_mask_output_dir, "generic_masks.pt")
        save_masks_to_file(attn_masks, mlp_masks, mask_file)
        print(f"[compute_masks] Saved generic masks to {mask_file}")

        save_pruning_metadata(
            output_dir=model_output_dir,
            task_type="generic",
            method=args.method,
            structure=args.structure,
            pruning_ratio=args.pruning_ratio,
            sparsities=sparsities,
            mask_file=mask_file
        )

    print("[compute_masks] All tasks processed successfully.")
    print(f"[compute_masks] Masks saved in: {model_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute pruning masks from activations and weight L2 for multiple tasks.")
    parser.add_argument("--model_root_path", type=str, required=True, 
                        help="Path to root dir containing the model.")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Model name.")
    parser.add_argument("--activations_root_path", type=str, default="./activations",
                        help="Path to activations root dir (which has <model_name>/<task>/activations.pt).")
    parser.add_argument("--output_dir", type=str, default="./pruning_masks",
                        help="Where to save the generated masks.")
    # Pruning strategy options
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--method", type=str, default="WIFV", 
                        choices=["WIFV", "WIFN"],
                        help="Scoring method for compute_scores. Default: WIFV")
    parser.add_argument("--structure", type=str, default="UL-LD",
                        help="Structure strategy like UL-UM, AL-AM, UL-LD. Default: UL-LD")
    parser.add_argument("--pruning_ratio", type=float, default=0.1,
                        help="Fraction of heads/channels to prune. Default=0.1")
    parser.add_argument("--logistic_k", type=float, default=0.8,
                        help="(UL-LD) Logistic function steepness parameter k.")
    parser.add_argument("--logistic_x0", type=float, default=0.3,
                        help="(UL-LD) Logistic function midpoint parameter x0.")
    parser.add_argument("--use_generic_mask", action="store_true", 
                        help="If set, generate a generic mask based on weighted scores across all tasks.")

    args = parser.parse_args()
    main(args)
