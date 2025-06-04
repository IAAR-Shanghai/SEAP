#!/usr/bin/env python3
"""
Compute pruning masks for transformer model components.

This script generates boolean masks for pruning attention heads and MLP layers based on
importance scores computed from model activations. Supports multiple pruning methods
and sparsity scheduling strategies.

Example usage:
    python compute_masks.py \
        --model_root_path /path/to/models \
        --model_name llama2-7b \
        --prompt_types zero_shot cot \
        --method WIFV \
        --sparsity_strategy logistic \
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
from src.remove_test import load_layerwise_results

# Constants
PROMPT_TYPES = ["zero_shot", "cot", "icl", "icl_cot", "knowledge"]
PRUNING_METHODS = ["WIFV", "WIFN"]
SPARSITY_STRATEGIES = [
    "uniform",    # Equal sparsity across layers
    "logistic",   # Logistic curve-based sparsity
    "global",     # Global threshold across all components
    "cosine",     # Cosine similarity-based scheduling
    "retention",  # Retention score-based scheduling
    "linear_fit", # Linear regression-based scheduling
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
        task_type: Type of task (e.g., gsm8k, arc_e)
        method: Pruning method (WIFV/WIFN)
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
    sparsity_file = os.path.join(task_output_dir, "sparsity.json")
    with open(sparsity_file, 'w') as f:
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
    metadata_file = os.path.join(task_output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Metadata saved in {task_output_dir}")


def setup_model_config(
    model_path: str,
    activations_path: str
) -> Tuple[Dict[str, int], Dict[str, torch.Tensor]]:
    """Load model and extract configuration.
    
    Args:
        model_path: Path to model directory
        activations_path: Path to activations directory
        
    Returns:
        Tuple containing:
            - Dictionary of model configuration values
            - Dictionary of weight L2 information
            
    Raises:
        ValueError: If model loading fails
    """
    try:
        print(f"🧠 Loading model from {model_path}")
        model, _ = load_model_and_tokenizer(model_path)
        model.eval()

        config = {
            "num_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "num_heads": model.config.num_attention_heads,
            "intermediate_size": model.config.intermediate_size
        }

        del model
        torch.cuda.empty_cache()

        # Load weight L2 information
        weight_l2_file = os.path.join(activations_path, "weight_l2_info.pt")
        print(f"📊 Loading weight L2 from {weight_l2_file}")
        weight_l2_data = load_weight_l2_info(weight_l2_file)

        return config, weight_l2_data

    except Exception as e:
        raise ValueError(f"Failed to setup model configuration: {e}")


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
    activation_data: Dict[str, torch.Tensor],
    weight_data: Dict[str, torch.Tensor],
    model_config: Dict[str, int],
    method: str,
    strategy: str,
    pruning_ratio: float,
    strategy_kwargs: Dict[str, Any],
    importance_data: Tuple[Optional[Dict[str, Any]], ...],
    output_dir: str,
    prompt_type: str
) -> None:
    """Process a single task to generate pruning masks.
    
    Args:
        task_type: Type of task
        activation_data: Task activation tensors
        weight_data: Model weight information
        model_config: Model architecture configuration
        method: Pruning method
        strategy: Sparsity scheduling strategy
        pruning_ratio: Target pruning ratio
        strategy_kwargs: Additional strategy parameters
        importance_data: Layer importance metrics
        output_dir: Output directory for masks
        prompt_type: Type of prompt used
    """
    print(f"\n📋 Processing task: {task_type}")
    
    try:
        # Compute importance scores
        scores_dict = compute_all_layers_scores(
            activation_data=activation_data,
            weight_data=weight_data,
            num_layers=model_config["num_layers"],
            hidden_size=model_config["hidden_size"],
            num_heads=model_config["num_heads"],
            intermediate_size=model_config["intermediate_size"],
            method=method
        )

        # Generate pruning masks
        cos_sims, remove_results, fitted_results = importance_data
        attn_masks, mlp_masks = generate_masks_for_all_layers(
            scores_dict=scores_dict,
            strategy=strategy,
            pruning_ratio=pruning_ratio,
            hidden_size=model_config["hidden_size"],
            num_heads=model_config["num_heads"],
            total_layers=model_config["num_layers"],
            strategy_kwargs=strategy_kwargs,
            cos_sims=cos_sims,
            remove_results=remove_results,
            fitted_results=fitted_results
        )

        # Compute achieved sparsity
        sparsities, global_sparsity = compute_layerwise_sparsity(
            attn_masks=attn_masks,
            mlp_masks=mlp_masks,
            hidden_size=model_config["hidden_size"],
            num_heads=model_config["num_heads"]
        )

        # Print sparsity information
        print(f"Global sparsity: {global_sparsity:.4f}")
        for layer_idx, data in sparsities.items():
            print(f"Layer {layer_idx}: attn={data['attn_sparsity']:.3f}, mlp={data['mlp_sparsity']:.3f}")

        # Save masks and metadata
        task_output_dir = os.path.join(
            output_dir,
            f"prompt={prompt_type}_task={task_type}_method={method}_strategy={strategy}_ratio={pruning_ratio}"
        )
        os.makedirs(task_output_dir, exist_ok=True)

        mask_file = os.path.join(task_output_dir, f"{task_type}_masks.pt")
        save_masks_to_file(attn_masks, mlp_masks, mask_file)

        save_pruning_metadata(
            output_dir=output_dir,
            task_type=task_type,
            method=method,
            strategy=strategy,
            pruning_ratio=pruning_ratio,
            sparsities=sparsities,
            mask_file=mask_file,
            prompt_type=prompt_type
        )

    except Exception as e:
        print(f"❌ Error processing task {task_type}: {e}")


def main(args: argparse.Namespace) -> None:
    """Main execution function.
    
    Args:
        args: Command line arguments
    """
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        # Setup model configuration and load weight data
        model_config, weight_l2_data = setup_model_config(
            model_path=os.path.join(args.model_root_path, args.model_name),
            activations_path=os.path.join(args.activations_root_path, args.model_name)
        )

        print(f"📐 Model config: {json.dumps(model_config, indent=2)}")

        # Create output directory
        model_output_dir = os.path.join(args.output_dir, args.model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        # Load layer importance data if needed
        importance_data = load_importance_data(
            strategy=args.sparsity_strategy,
            model_name=args.model_name,
            importance_dir=args.layer_importance_dir
        )

        # Process each prompt type
        for prompt_type in args.prompt_types:
            print(f"\n🔁 Processing prompt type: {prompt_type}")
            
            # Load activations
            activations_dir = os.path.join(args.activations_root_path, args.model_name, prompt_type)
            print(f"📦 Loading activations from {activations_dir}")
            all_task_activations = load_activations(activations_dir)

            # Determine tasks to process
            if args.tasks:
                missing = [t for t in args.tasks if t not in all_task_activations]
                if missing:
                    raise ValueError(f"Tasks not found in activations: {missing}")
                task_list = args.tasks
            else:
                task_list = sorted(all_task_activations.keys())

            print(f"📋 Tasks for {prompt_type}: {task_list}")

            # Process each task
            for task_type in task_list:
                process_task(
                    task_type=task_type,
                    activation_data=all_task_activations[task_type],
                    weight_data=weight_l2_data,
                    model_config=model_config,
                    method=args.method,
                    strategy=args.sparsity_strategy,
                    pruning_ratio=args.pruning_ratio,
                    strategy_kwargs={
                        "protect_head": args.protect_head,
                        "protect_tail": args.protect_tail
                    },
                    importance_data=importance_data,
                    output_dir=model_output_dir,
                    prompt_type=prompt_type
                )

        print("\n✅ All prompt types and tasks processed")

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute pruning masks from activations and weight L2.",
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
        help="Name of the model directory under model_root_path"
    )
    parser.add_argument(
        "--prompt_types",
        nargs="+",
        required=True,
        choices=PROMPT_TYPES,
        help="One or more prompt formats to process"
    )

    # Optional arguments
    parser.add_argument(
        "--activations_root_path",
        type=str,
        default="./activations",
        help="Root directory containing activation files"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="List of task types to process (default: all)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pruning_masks",
        help="Output directory for masks and metadata"
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
        default="logistic",
        choices=SPARSITY_STRATEGIES,
        help="Strategy for scheduling sparsity across layers"
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
        default=2,
        help="Number of bottom layers to protect"
    )
    parser.add_argument(
        "--protect_tail",
        type=int,
        default=3,
        help="Number of top layers to protect"
    )
    parser.add_argument(
        "--layer_importance_dir",
        type=str,
        default=None,
        help="Directory with layer importance data"
    )

    args = parser.parse_args()
    main(args)
