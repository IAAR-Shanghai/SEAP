#!/usr/bin/env python3
# coding: utf-8
# scripts/apply_pruning.py

import sys
import os
import torch
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import custom modules
from src.model_utils import load_model_and_tokenizer
from src.pruning_utils.generate_masks import load_masks_from_file
from src.pruning_utils.apply_pruning import apply_pruning_to_model
from src.activations import load_activations

def main(args):
    """Main function to apply pruning masks to the model and save the pruned model.

    Args:
        args (argparse.Namespace): The arguments parsed from the command line.
    """

    # 1) Load model
    model_path = os.path.join(args.model_root_path, args.model_name)
    print(f"[apply_pruning] Loading model from {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval().to("cuda")

    # 2) Automatically compute head_dim
    #    (No longer passed via command-line, computed based on hidden_size / num_attention_heads)
    if hasattr(model.config, "hidden_size") and hasattr(model.config, "num_attention_heads"):
        hidden_size = model.config.hidden_size
        num_heads = model.config.num_attention_heads
        if hidden_size % num_heads != 0:
            print(f"Warning: hidden_size={hidden_size} not divisible by num_heads={num_heads}; "
                  "using integer division for head_dim.")
        head_dim = hidden_size // num_heads
    else:
        # If the model lacks these attributes, raise an error
        raise ValueError("[apply_pruning] Model config missing hidden_size/num_attention_heads, "
                         "cannot compute head_dim automatically.")
    print(f"[apply_pruning] Computed head_dim={head_dim} from hidden_size={hidden_size} / num_heads={num_heads}.")

    # 3) Load baseline activations if bias compensation is enabled
    attn_mean_inps_dict = {}
    mlp_mean_inps_dict = {}
    if args.biascomp:
        if args.activations_root_path is not None:
            act_dir = os.path.join(args.activations_root_path, args.model_name)
            print(f"[apply_pruning] Bias compensation is ON. Loading activations from {act_dir}")
            if not os.path.isdir(act_dir):
                print(f"Warning: The directory {act_dir} doesn't exist. Skipping baseline_inp.")
            else:
                # Read activations for all tasks: structure is {task_type: {layer_idx: {...}}}
                all_task_activations = load_activations(act_dir)
                for ttype in args.task_types:
                    if ttype not in all_task_activations:
                        print(f"[apply_pruning] No activations for task={ttype}. Skip baseline.")
                        continue
                    attn_mean_inps_dict[ttype] = {}
                    mlp_mean_inps_dict[ttype] = {}
                    layer_dict = all_task_activations[ttype]
                    for layer_idx, layer_info in layer_dict.items():
                        # Extract "attention_post_aggregation" mean as attn_mean_inp
                        attn_mean = layer_info.get("attention_post_aggregation", {}).get("mean", None)
                        # Extract "mlp_intermediate_states" mean as mlp_mean_inp
                        mlp_mean = layer_info.get("mlp_intermediate_states", {}).get("mean", None)

                        attn_mean_inps_dict[ttype][layer_idx] = attn_mean
                        mlp_mean_inps_dict[ttype][layer_idx] = mlp_mean
        else:
            print("[apply_pruning] --biascomp is True but no --activations_root_path provided. "
                  "Skipping baseline_inp for all tasks.")
    else:
        print("[apply_pruning] Bias compensation is OFF.")

    # 4) Iterate over multiple tasks
    print(f"[apply_pruning] Tasks to process: {args.task_types}")
    for idx, task_type in enumerate(args.task_types):
        print(f"\n[apply_pruning] === Processing task={task_type} ===")

        # (A) Construct mask file path for the task
        task_mask_dir = os.path.join(args.masks_root_dir, args.model_name, 
                                     f"task={task_type}_method={args.method}_structure={args.structure}_ratio={args.pruning_ratio}")
        mask_file = os.path.join(task_mask_dir, f"{task_type}_masks.pt")
        if not os.path.exists(mask_file):
            print(f"[apply_pruning] Mask file not found for task={task_type}: {mask_file}. Skipping.")
            continue

        # (B) Load pruning masks
        print(f"[apply_pruning] Loading masks from {mask_file}")
        attn_masks, mlp_masks = load_masks_from_file(mask_file)

        # (C) Retrieve corresponding baseline_inp for bias compensation
        attn_mean_inps = None
        mlp_mean_inps  = None
        if args.biascomp:
            if task_type in attn_mean_inps_dict:
                attn_mean_inps = attn_mean_inps_dict[task_type]
                mlp_mean_inps  = mlp_mean_inps_dict[task_type]
            else:
                print(f"[apply_pruning] No baseline data for task={task_type}. No bias compensation.")

        # (D) Apply pruning to the model
        print(f"[apply_pruning] Pruning with task={task_type} ... softmask={args.softmask}, bias={args.biascomp}")
        apply_pruning_to_model(
            model=model,
            attn_masks=attn_masks,
            mlp_masks=mlp_masks,
            attn_mean_inps=attn_mean_inps,   # {layer_idx->Tensor([hidden_size])}
            mlp_mean_inps=mlp_mean_inps,     # {layer_idx->Tensor([intermediate_size])}
            device="cuda",
            bias=args.biascomp,
            unstr=args.softmask,
            head_dim=head_dim
        )
        print(f"[apply_pruning] Done applying masks for task={task_type}.")

        # (E) Save the pruned model
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f"{args.model_name}_pruned_{task_type}_method={args.method}_structure={args.structure}_ratio={args.pruning_ratio}")
        os.makedirs(save_path, exist_ok=True)
        print(f"[apply_pruning] Saving pruned model (task={task_type}) to {save_path}")

        if args.softmask:
            model.save_pretrained(save_path)
        else:
            torch.save(model, f"{save_path}/pruned_model.pt")

        # Save the tokenizer
        tokenizer.save_pretrained(save_path)

        # (F) Free memory and reload the model for the next task if needed
        if len(args.task_types) > 1 and idx < len(args.task_types) - 1:
            print("[apply_pruning] Reloading original model for next task, to avoid cumulative pruning.")
            del model
            torch.cuda.empty_cache()
            model, tokenizer = load_model_and_tokenizer(model_path)
            model.eval().to("cuda")

    print("\n[apply_pruning] Done. All tasks processed successfully.")


if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Load pruning masks for multiple tasks and apply them to the model with optional bias compensation."
    )
    parser.add_argument("--model_root_path", type=str, required=True,
                        help="Path to root dir containing the model.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name (folder name under --model_root_path).")

    parser.add_argument("--masks_root_dir", type=str, default="./pruning_masks",
                        help="Directory where the <task>_masks.pt are located.")
    parser.add_argument("--task_types", nargs='+', default=["gsm8k"],
                        help="List of task names to process, e.g.: gsm8k hellaswag mmlu")
    parser.add_argument("--output_dir", type=str, default="./pruned_models",
                        help="Where to save each pruned model (for each task).")
    
    parser.add_argument("--method", type=str, default="WIFV", 
                        choices=["WIFV", "WIFN"],
                        help="Scoring method used for pruning. Default is 'WIFV'.")

    parser.add_argument("--structure", type=str, default="UL-LD", 
                        choices=["UL-UM", "AL-AM","UL-LD"],
                        help="Pruning structure strategy. Default is 'UL-LD'.")
    
    parser.add_argument("--biascomp", action="store_true",
                        help="Enable baseline input based bias compensation if baseline_inp is available.")
    parser.add_argument("--softmask", action="store_true",
                        help="If set, only mask weights to zero (unstructured), else do structured pruning.")

    parser.add_argument("--pruning_ratio", type=float, default=0.1,
                        help="Fraction of heads/channels to prune. Default=0.1")

    # For loading baseline_inp from activations
    parser.add_argument("--activations_root_path", type=str, default=None,
                        help="Where to load layer-wise baseline_inp from if bias compensation is needed.")

    args = parser.parse_args()
    main(args)
