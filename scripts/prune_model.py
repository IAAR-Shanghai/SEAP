#!/usr/bin/env python3
"""
End-to-end model pruning script.

This script performs the complete pruning pipeline in one go:
1. Loads model and activation statistics
2. Computes importance scores
3. Generates pruning masks
4. Applies masks to model
5. Saves both masks and pruned model

Example usage:
    python prune_model.py \
        --model_root_path /path/to/models \
        --model_name llama2-7b \
        --prompt_types zero_shot cot \
        --tasks gsm8k arc_e \
        --method WIFV \
        --sparsity_strategy retention \
        --pruning_ratio 0.2

Author: why
Date: 2024
"""

# Standard library imports
import sys
import os
import json
from typing import Dict, List, Optional, Tuple, Any

# Third-party imports
import torch
import numpy as np
import argparse

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Local imports
from src.activations import load_activations, load_weight_l2_info
from src.model_utils import load_model_and_tokenizer
from src.pruning_utils.compute_scores import compute_all_layers_scores
from src.pruning_utils.generate_masks import (
    generate_masks_for_all_layers,
    save_masks_to_file,
    compute_layerwise_sparsity
)
from src.pruning_utils.apply_pruning import apply_pruning_to_model
from src.remove_test import load_layerwise_results

# Constants
PROMPT_TYPES = [
    "zero_shot",   # Direct task completion
    "cot",         # Chain-of-thought reasoning
    "icl",         # In-context learning
    "icl_cot",     # ICL with chain-of-thought
    "knowledge",   # Knowledge-enhanced prompting
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


def save_pruning_metadata(
    output_dir: str,
    task_type: str,
    method: str,
    strategy: str,
    pruning_ratio: float,
    sparsities: Dict[int, Dict[str, float]],
    mask_file: str,
    prompt_type: str
) -> None:
    """Save pruning configuration and results metadata.
    
    Args:
        output_dir: Base output directory
        task_type: Type of task
        method: Pruning method
        strategy: Sparsity scheduling strategy
        pruning_ratio: Target pruning ratio
        sparsities: Dictionary of layer-wise sparsity values
        mask_file: Path to saved mask file
        prompt_type: Type of prompt used
    """
    folder_name = f"prompt={prompt_type}_task={task_type}_method={method}_strategy={strategy}_ratio={pruning_ratio}"
    task_output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(task_output_dir, exist_ok=True)

    # Save sparsity information
    with open(os.path.join(task_output_dir, "sparsity.json"), 'w') as f:
        json.dump(sparsities, f, indent=4)

    # Save configuration metadata
    metadata = {
        "task_type": task_type,
        "prompt_type": prompt_type,
        "method": method,
        "strategy": strategy,
        "pruning_ratio": pruning_ratio,
        "mask_file": mask_file,
    }
    with open(os.path.join(task_output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Metadata saved in {task_output_dir}")


def setup_model(
    model_path: str,
    device: str = "cuda"
) -> Tuple[torch.nn.Module, Any, Dict[str, Any]]:
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
            "num_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "num_heads": model.config.num_attention_heads,
            "intermediate_size": model.config.intermediate_size,
            "head_dim": model.config.hidden_size // model.config.num_attention_heads
        }

        print(f"📐 Model config: {json.dumps(config, indent=2)}")
        return model, tokenizer, config

    except Exception as e:
        raise ValueError(f"Failed to setup model: {e}")


def load_importance_data(
    strategy: str,
    model_name: str,
    importance_dir: Optional[str]
) -> Tuple[Optional[Dict[str, Any]], ...]:
    """Load layer importance data for advanced scheduling strategies.
    
    Args:
        strategy: Sparsity scheduling strategy
        model_name: Name of the model
        importance_dir: Directory containing importance data
        
    Returns:
        Tuple of importance metrics (cos_sims, remove_results, fitted_results)
        
    Raises:
        ValueError: If required data is missing
    """
    if strategy in {"cosine", "retention", "linear_fit", "logistic_fit"}:
        if not importance_dir:
            raise ValueError(f"Strategy '{strategy}' requires --layer_importance_dir")
            
        print(f"📈 Loading layer importance from {importance_dir}")
        try:
            _, cos_sims, remove_results, fitted_results = load_layerwise_results(
                model_name=model_name,
                base_dir=importance_dir
            )
            return cos_sims, remove_results, fitted_results
        except Exception as e:
            raise ValueError(f"Failed to load importance data: {e}")
    
    return None, None, None


def process_task(
    task_type: str,
    model: torch.nn.Module,
    tokenizer: Any,
    config: Dict[str, Any],
    activation_data: Dict[str, torch.Tensor],
    weight_data: Dict[str, torch.Tensor],
    importance_data: Tuple[Optional[Dict[str, Any]], ...],
    args: argparse.Namespace,
    prompt_type: str
) -> None:
    """Process a single task for pruning.
    
    Args:
        task_type: Type of task
        model: Model to prune
        tokenizer: Model tokenizer
        config: Model configuration
        activation_data: Task activation tensors
        weight_data: Model weight information
        importance_data: Layer importance metrics
        args: Command line arguments
        prompt_type: Type of prompt used
    """
    print(f"\n📋 Processing task: {task_type}")
    
    try:
        # Compute importance scores
        scores_dict = compute_all_layers_scores(
            activation_data=activation_data,
            weight_data=weight_data,
            num_layers=config["num_layers"],
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"],
            intermediate_size=config["intermediate_size"],
            method=args.method
        )

        # Generate pruning masks
        cos_sims, remove_results, fitted_results = importance_data
        attn_masks, mlp_masks = generate_masks_for_all_layers(
            scores_dict=scores_dict,
            strategy=args.sparsity_strategy,
            pruning_ratio=args.pruning_ratio,
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"],
            total_layers=config["num_layers"],
            strategy_kwargs={
                "protect_head": args.protect_head,
                "protect_tail": args.protect_tail
            },
            cos_sims=cos_sims,
            remove_results=remove_results,
            fitted_results=fitted_results
        )

        # Compute achieved sparsity
        sparsities, global_sparsity = compute_layerwise_sparsity(
            attn_masks=attn_masks,
            mlp_masks=mlp_masks,
            hidden_size=config["hidden_size"],
            num_heads=config["num_heads"]
        )

        # Print sparsity information
        print(f"Global sparsity: {global_sparsity:.4f}")
        for layer_idx, data in sparsities.items():
            print(f"Layer {layer_idx:02d}: attn={data['attn_sparsity']:.3f}, mlp={data['mlp_sparsity']:.3f}")

        # Save masks and metadata
        task_output_dir = os.path.join(
            args.mask_output_dir,
            args.model_name,
            f"prompt={prompt_type}_task={task_type}_"
            f"method={args.method}_strategy={args.sparsity_strategy}_"
            f"ratio={args.pruning_ratio}"
        )
        os.makedirs(task_output_dir, exist_ok=True)

        mask_file = os.path.join(task_output_dir, f"{task_type}_masks.pt")
        save_masks_to_file(attn_masks, mlp_masks, mask_file)

        save_pruning_metadata(
            output_dir=os.path.join(args.mask_output_dir, args.model_name),
            task_type=task_type,
            method=args.method,
            strategy=args.sparsity_strategy,
            pruning_ratio=args.pruning_ratio,
            sparsities=sparsities,
            mask_file=mask_file,
            prompt_type=prompt_type
        )

        # Apply pruning and save model
        apply_pruning_to_model(
            model=model,
            attn_masks=attn_masks,
            mlp_masks=mlp_masks,
            unstr=not args.hardmask,
            head_dim=config["head_dim"]
        )

        pruned_save_dir = os.path.join(
            args.pruned_model_output_dir,
            f"{args.model_name}_pruned_prompt={prompt_type}_"
            f"task={task_type}_method={args.method}_"
            f"strategy={args.sparsity_strategy}_ratio={args.pruning_ratio}"
        )
        os.makedirs(pruned_save_dir, exist_ok=True)

        if not args.hardmask:
            model.save_pretrained(pruned_save_dir)
        else:
            torch.save(model, os.path.join(pruned_save_dir, "pruned_model.pt"))
        tokenizer.save_pretrained(pruned_save_dir)

        print(f"✅ Saved pruned model to {pruned_save_dir}")

    except Exception as e:
        print(f"❌ Error processing task {task_type}: {e}")


def main(args: argparse.Namespace) -> None:
    """Main execution function.
    
    Args:
        args: Command line arguments
    """
    try:
        # Set random seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        # Setup model and load data
        model_path = os.path.join(args.model_root_path, args.model_name)
        model, tokenizer, config = setup_model(model_path)

        weight_l2_file = os.path.join(args.activations_root_path, args.model_name, "weight_l2_info.pt")
        print(f"📊 Loading weight L2 from {weight_l2_file}")
        weight_l2_data = load_weight_l2_info(weight_l2_file)

        # Load layer importance data if needed
        importance_data = load_importance_data(
            strategy=args.sparsity_strategy,
            model_name=args.model_name,
            importance_dir=args.layer_importance_dir
        )

        # Create output directories
        os.makedirs(os.path.join(args.mask_output_dir, args.model_name), exist_ok=True)

        # Process each prompt type
        for prompt_type in args.prompt_types:
            print(f"\n🔁 Processing prompt type: {prompt_type}")
            
            # Load activations
            activations_dir = os.path.join(args.activations_root_path, args.model_name, prompt_type)
            print(f"📦 Loading activations from {activations_dir}")
            all_task_activations = load_activations(activations_dir)

            # Determine tasks to process
            task_list = args.tasks if args.tasks else sorted(all_task_activations.keys())

            # Process each task
            for task_type in task_list:
                process_task(
                    task_type=task_type,
                    model=model,
                    tokenizer=tokenizer,
                    config=config,
                    activation_data=all_task_activations[task_type],
                    weight_data=weight_l2_data,
                    importance_data=importance_data,
                    args=args,
                    prompt_type=prompt_type
                )

                # Reload model for next task if needed
                if len(task_list) > 1 and task_type != task_list[-1]:
                    print("🔄 Reloading model for next task")
                    del model
                    torch.cuda.empty_cache()
                    model, tokenizer, config = setup_model(model_path)

        print("\n✅ All pruning completed")

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end model pruning pipeline.",
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
        "--prompt_types",
        nargs="+",
        required=True,
        choices=PROMPT_TYPES,
        help="Prompt formats to process"
    )

    # Optional arguments
    parser.add_argument(
        "--activations_root_path",
        type=str,
        default="./activations",
        help="Root directory containing activations"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="List of tasks to process (default: all)"
    )
    parser.add_argument(
        "--mask_output_dir",
        type=str,
        default="./pruning_masks",
        help="Output directory for masks"
    )
    parser.add_argument(
        "--pruned_model_output_dir",
        type=str,
        default="./pruned_models",
        help="Output directory for pruned models"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    # Pruning configuration
    parser.add_argument(
        "--method",
        type=str,
        default="WIFV",
        choices=PRUNING_METHODS,
        help="Method for computing importance scores"
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
        help="Target ratio of weights to prune"
    )
    parser.add_argument(
        "--protect_head",
        type=int,
        default=0,
        help="Number of bottom layers to protect"
    )
    parser.add_argument(
        "--protect_tail",
        type=int,
        default=0,
        help="Number of top layers to protect"
    )
    parser.add_argument(
        "--layer_importance_dir",
        type=str,
        default=None,
        help="Directory with layer importance data"
    )
    parser.add_argument(
        "--hardmask",
        action="store_true",
        help="Use structured pruning (default: unstructured)"
    )

    args = parser.parse_args()
    main(args)
