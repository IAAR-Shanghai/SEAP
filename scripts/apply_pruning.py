#!/usr/bin/env python3
# coding: utf-8

import sys
import os
import argparse
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.model_utils import load_model_and_tokenizer
from src.pruning_utils.generate_masks import load_masks_from_file
from src.pruning_utils.apply_pruning import apply_pruning_to_model


def main(args):
    model_path = os.path.join(args.model_root_path, args.model_name)
    print(f"[apply_pruning] Loading model from {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval().to("cuda")

    # Compute head_dim
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads
    print(f"[apply_pruning] Computed head_dim={head_dim} (hidden_size={hidden_size}, num_heads={num_heads})")

    use_softmask = not args.hardmask

    for idx, task_type in enumerate(args.tasks):
        print(f"\n[apply_pruning] === Task: {task_type} ===")

        mask_path = os.path.join(
            args.masks_root_dir,
            args.model_name,
            f"prompt={args.prompt_type}_task={task_type}_method={args.method}_strategy={args.sparsity_strategy}_ratio={args.pruning_ratio}",
            f"{task_type}_masks.pt"
        )

        if not os.path.exists(mask_path):
            print(f"[apply_pruning] Mask not found: {mask_path}. Skipping.")
            continue

        print(f"[apply_pruning] Loading masks from {mask_path}")
        attn_masks, mlp_masks = load_masks_from_file(mask_path)

        # Apply pruning
        apply_pruning_to_model(
            model=model,
            attn_masks=attn_masks,
            mlp_masks=mlp_masks,
            unstr=use_softmask,
            head_dim=head_dim
        )

        # Save pruned model
        save_dir = os.path.join(
            args.output_dir,
            f"{args.model_name}_pruned_prompt={args.prompt_type}_task={task_type}_method={args.method}_strategy={args.sparsity_strategy}_ratio={args.pruning_ratio}"
        )
        os.makedirs(save_dir, exist_ok=True)

        if use_softmask:
            model.save_pretrained(save_dir)
        else:
            torch.save(model, f"{save_dir}/pruned_model.pt")

        tokenizer.save_pretrained(save_dir)
        print(f"[apply_pruning] Saved pruned model to {save_dir}")

        # Reload original model if multiple tasks
        if len(args.tasks) > 1 and idx < len(args.tasks) - 1:
            print("[apply_pruning] Reloading original model for next task.")
            del model
            torch.cuda.empty_cache()
            model, tokenizer = load_model_and_tokenizer(model_path)
            model.eval().to("cuda")

    print("\n[apply_pruning] All tasks done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply pruning masks to a model for specified tasks.")
    parser.add_argument("--model_root_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)

    parser.add_argument("--masks_root_dir", type=str, default="./pruning_masks")
    parser.add_argument("--tasks", nargs='+', required=True,
                        help="List of tasks to prune on, e.g., gsm8k hellaswag")
    parser.add_argument("--prompt_type", type=str, required=True,
                        choices=["zero_shot", "cot", "icl", "icl_cot", "knowledge"],
                        help="Prompt type used to generate activations and masks.")
    parser.add_argument("--output_dir", type=str, default="./pruned_models")

    parser.add_argument("--method", type=str, default="WIFV", choices=["WIFV", "WIFN"])
    parser.add_argument("--sparsity_strategy", type=str, default="logistic",
                        choices=["uniform", "logistic", "global"])
    parser.add_argument("--pruning_ratio", type=float, default=0.2)

    parser.add_argument("--hardmask", action="store_true",
                        help="Use hardmask (structured pruning). Default is softmask (unstructured).")

    args = parser.parse_args()
    main(args)
