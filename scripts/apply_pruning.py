#!/usr/bin/env python3
"""
Apply pruning masks to transformer models.

This script applies pre-computed pruning masks to transformer models, supporting both
structured (hard) and unstructured (soft) pruning. It can process multiple tasks
sequentially and saves the pruned models.

Example usage:
    python apply_pruning.py \
        --model_root_path /path/to/models \
        --model_name llama2-7b \
        --tasks gsm8k arc_e \
        --prompt_type zero_shot \
        --method WIFV \
        --sparsity_strategy retention \
        --pruning_ratio 0.2

Author: why
Date: 2024
"""

# Standard library imports
import sys
import os
from typing import Tuple, Optional, Any

# Third-party imports
import torch
import argparse

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Local imports
from src.model_utils import load_model_and_tokenizer
from src.pruning_utils.generate_masks import load_masks_from_file
from src.pruning_utils.apply_pruning import apply_pruning_to_model

# Constants
PROMPT_TYPES = [
    "zero_shot",   # Direct task completion
    "cot",         # Chain-of-thought reasoning
    "icl",         # In-context learning
    "icl_cot",     # ICL with chain-of-thought
    "knowledge",   # Knowledge-enhanced prompting
    "corpus",      # Raw corpus text
    "experts"      # Expert demonstrations
]

PRUNING_METHODS = ["WIFV", "WIFN"]
SPARSITY_STRATEGIES = [
    "uniform",     # Equal sparsity across layers
    "global",      # Global threshold across components
    "cosine",      # Cosine similarity-based scheduling
    "retention",   # Retention score-based scheduling
    "linear_fit",  # Linear regression-based scheduling
    "logistic_fit" # Logistic regression-based scheduling
]


def setup_model(
    model_path: str,
    device: str = "cuda"
) -> Tuple[torch.nn.Module, Any, dict]:
    """Load and configure model for pruning.
    
    Args:
        model_path: Path to model directory
        device: Device to load model on
        
    Returns:
        Tuple containing:
            - Loaded model
            - Tokenizer
            - Model configuration dictionary
            
    Raises:
        ValueError: If model loading fails
    """
    try:
        print(f"🧠 Loading model from {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)
        model.eval().to(device)

        config = {
            "hidden_size": model.config.hidden_size,
            "num_heads": model.config.num_attention_heads,
            "head_dim": model.config.hidden_size // model.config.num_attention_heads
        }

        print(f"📐 Model config: hidden_size={config['hidden_size']}, "
              f"num_heads={config['num_heads']}, head_dim={config['head_dim']}")

        return model, tokenizer, config

    except Exception as e:
        raise ValueError(f"Failed to setup model: {e}")


def process_task(
    task_type: str,
    model: torch.nn.Module,
    tokenizer: Any,
    config: dict,
    args: argparse.Namespace
) -> None:
    """Apply pruning for a single task.
    
    Args:
        task_type: Type of task
        model: Model to prune
        tokenizer: Model tokenizer
        config: Model configuration
        args: Command line arguments
    """
    print(f"\n📋 Processing task: {task_type}")

    # Build mask path
        mask_path = os.path.join(
            args.masks_root_dir,
            args.model_name,
        f"prompt={args.prompt_type}_task={task_type}_"
        f"method={args.method}_strategy={args.sparsity_strategy}_"
        f"ratio={args.pruning_ratio}",
            f"{task_type}_masks.pt"
        )

        if not os.path.exists(mask_path):
        print(f"❌ Mask not found: {mask_path}")
        return

    try:
        # Load and apply masks
        print(f"📥 Loading masks from {mask_path}")
        attn_masks, mlp_masks = load_masks_from_file(mask_path)

        apply_pruning_to_model(
            model=model,
            attn_masks=attn_masks,
            mlp_masks=mlp_masks,
            unstr=not args.hardmask,
            head_dim=config["head_dim"]
        )

        # Save pruned model
        save_dir = os.path.join(
            args.output_dir,
            f"{args.model_name}_pruned_prompt={args.prompt_type}_"
            f"task={task_type}_method={args.method}_"
            f"strategy={args.sparsity_strategy}_ratio={args.pruning_ratio}"
        )
        os.makedirs(save_dir, exist_ok=True)

        if not args.hardmask:
            model.save_pretrained(save_dir)
        else:
            torch.save(model, os.path.join(save_dir, "pruned_model.pt"))

        tokenizer.save_pretrained(save_dir)
        print(f"✅ Saved pruned model to {save_dir}")

    except Exception as e:
        print(f"❌ Error processing task {task_type}: {e}")


def main(args: argparse.Namespace) -> None:
    """Main execution function.
    
    Args:
        args: Command line arguments
    """
    try:
        model_path = os.path.join(args.model_root_path, args.model_name)
        model, tokenizer, config = setup_model(model_path)

        # Process each task
        for idx, task_type in enumerate(args.tasks):
            process_task(task_type, model, tokenizer, config, args)

            # Reload model for next task if needed
        if len(args.tasks) > 1 and idx < len(args.tasks) - 1:
                print("🔄 Reloading original model for next task")
            del model
            torch.cuda.empty_cache()
                model, tokenizer, config = setup_model(model_path)

        print("\n✅ All tasks processed")

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply pruning masks to transformer models.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--model_root_path",
        type=str,
        required=True,
        help="Root directory containing model files"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model directory"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="List of tasks to process"
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        required=True,
        choices=PROMPT_TYPES,
        help="Prompt format used for masks"
    )

    # Optional arguments
    parser.add_argument(
        "--masks_root_dir",
        type=str,
        default="./pruning_masks",
        help="Root directory containing masks"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pruned_models",
        help="Output directory for pruned models"
    )

    # Pruning configuration
    parser.add_argument(
        "--method",
        type=str,
        default="WIFV",
        choices=PRUNING_METHODS,
        help="Method used to compute masks"
    )
    parser.add_argument(
        "--sparsity_strategy",
        type=str,
        default="retention",
        choices=SPARSITY_STRATEGIES,
        help="Strategy for sparsity scheduling"
    )
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        default=0.2,
        help="Target pruning ratio"
    )
    parser.add_argument(
        "--hardmask",
        action="store_true",
        help="Use structured pruning (default: unstructured)"
    )

    args = parser.parse_args()
    main(args)
