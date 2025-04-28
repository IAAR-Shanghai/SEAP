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

def save_pruning_metadata(output_dir, task_type, method, strategy, pruning_ratio, sparsities, mask_file):
    folder_name = f"task={task_type}_method={method}_strategy={strategy}_ratio={pruning_ratio}"
    task_output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(task_output_dir, exist_ok=True)

    sparsity_file = os.path.join(task_output_dir, "sparsity.json")
    with open(sparsity_file, 'w') as f:
        json.dump(sparsities, f, indent=4)

    metadata = {
        "task_type": task_type,
        "method": method,
        "strategy": strategy,
        "pruning_ratio": pruning_ratio,
        "mask_file": mask_file,
    }
    metadata_file = os.path.join(task_output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"[save_pruning_metadata] Metadata saved in {task_output_dir}")

def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_path = os.path.join(args.model_root_path, args.model_name)
    print(f"[compute_masks] Loading model from {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval().to("cuda")

    activations_dir = os.path.join(args.activations_root_path, args.model_name)
    print(f"[compute_masks] Loading activations from {activations_dir}")
    all_task_activations = load_activations(activations_dir)

    weight_l2_file = os.path.join(activations_dir, "weight_l2_info.pt")
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

    if args.tasks:
        if args.tasks not in all_task_activations:
            raise ValueError(f"Task '{args.tasks}' not found in activations directory.")
        task_list = [args.tasks]
    else:
        task_list = sorted(all_task_activations.keys())

    print(f"[compute_masks] Found tasks in activation data: {task_list}")

    model_output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Build strategy_kwargs based on selected sparsity strategy
    if args.sparsity_strategy == "logistic":
        strategy_kwargs = {
            "k": args.logistic_k,
            "x0": args.logistic_x0
        }
    else:
        strategy_kwargs = {}


    for task_type in task_list:
        print(f"\\n[compute_masks] Processing task={task_type} ...")
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
            strategy_kwargs=strategy_kwargs
        )

        sparsities, global_sparsity = compute_layerwise_sparsity(
            attn_masks=attn_masks,
            mlp_masks=mlp_masks,
            hidden_size=hidden_size,
            num_heads=num_heads
        )
        print(f"Global sparsity (cost-aware): {global_sparsity:.4f}")
        for layer_idx, data in sparsities.items():
            print(f"Layer {layer_idx}: attn_sparsity={data['attn_sparsity']:.3f}, "
                    f"mlp_sparsity={data['mlp_sparsity']:.3f}")

        task_output_dir = os.path.join(
            model_output_dir,
            f"task={task_type}_method={args.method}_strategy={args.sparsity_strategy}_ratio={args.pruning_ratio}"
        )
        os.makedirs(task_output_dir, exist_ok=True)

        mask_file = os.path.join(task_output_dir, f"{task_type}_masks.pt")
        save_masks_to_file(attn_masks, mlp_masks, mask_file)
        print(f"[compute_masks] Saved masks for task={task_type} to {mask_file}")

        save_pruning_metadata(
            output_dir=model_output_dir,
            task_type=task_type,
            method=args.method,
            strategy=args.sparsity_strategy,
            pruning_ratio=args.pruning_ratio,
            sparsities=sparsities,
            mask_file=mask_file
        )

    print("[compute_masks] All tasks processed successfully.")
    print(f"[compute_masks] Masks saved in: {model_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute pruning masks from activations and weight L2.")
    parser.add_argument("--model_root_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--activations_root_path", type=str, default="./activations")
    parser.add_argument("--tasks", type=str, default=None,
                help="Optional: specify a single task to process (e.g., wikitext2). If not set, all tasks will be processed.")
    parser.add_argument("--output_dir", type=str, default="./pruning_masks")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--method", type=str, default="WIFV", choices=["WIFV", "WIFN"])
    parser.add_argument("--sparsity_strategy", type=str, default="logistic",
                        choices=["uniform", "logistic", "global"],
                        help="Sparsity scheduling strategy: 'uniform', 'logistic', or 'global'.")
    parser.add_argument("--pruning_ratio", type=float, default=0.1)
    parser.add_argument("--logistic_k", type=float, default=1.2)
    parser.add_argument("--logistic_x0", type=float, default=0.3)

    args = parser.parse_args()
    main(args)