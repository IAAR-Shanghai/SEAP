#!/usr/bin/env python3
"""
Compute and save model activations for different prompt types.

This script captures and saves model activations during forward passes with different
prompt formats. It supports multiple tasks and prompt types, with options for sampling
and task filtering.

Example usage:
    python compute_activations.py \
        --model_root_path /path/to/models \
        --model_name llama2-7b \
        --prompt_types zero_shot cot \
        --tasks gsm8k arc_e \
        --sample_size 100

Author: why
Date: 2024
"""

# Standard library imports
import sys
import os
import gc
import random
from typing import List, Optional

# Third-party imports
import torch
import numpy as np
import pandas as pd
import argparse

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Local imports
from src.activations import (
    ActivationHookManager,
    capture_activations,
    save_activations_dict,
    compute_and_save_weight_l2
)
from src.model_utils import load_model_and_tokenizer

# Constants
PROMPT_COLUMN_MAPPING = {
    "corpus": "corpus",
    "zero_shot": "prompt_zero_shot",
    "cot": "prompt_cot",
    "icl": "prompt_icl",
    "icl_cot": "prompt_icl_cot",
    "knowledge": "knowledge",
    "experts": "prompt_experts",
}


def setup_output_dirs(base_dir: str, model_name: str, prompt_type: Optional[str] = None) -> str:
    """Create and return output directory path.
    
    Args:
        base_dir: Base directory for outputs
        model_name: Name of the model
        prompt_type: Optional prompt type for subdirectory
        
    Returns:
        Path to created output directory
    """
    if prompt_type:
        output_dir = os.path.join(base_dir, model_name, prompt_type)
    else:
        output_dir = os.path.join(base_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def process_dataset(
    df: pd.DataFrame,
    tasks: Optional[List[str]] = None,
    sample_size: Optional[int] = None,
    seed: int = 42
) -> pd.DataFrame:
    """Filter and sample dataset based on configuration.
    
    Args:
        df: Input DataFrame
        tasks: List of task types to include
        sample_size: Number of samples per task
        seed: Random seed for sampling
        
    Returns:
        Processed DataFrame
    """
    if tasks:
        df = df[df["task_type"].isin(tasks)]

    if "split" in df.columns:
        df = df[df["split"] == "train"]

    if sample_size:
        df = df.groupby("task_type", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), sample_size), random_state=seed)
        ).reset_index(drop=True)

    return df


def main(args: argparse.Namespace) -> None:
    """Main execution function.
    
    Args:
        args: Command line arguments
    """
    # Set random seeds
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load model and tokenizer
    model_path = os.path.join(args.model_root_path, args.model_name)
    print("🧠 Loading model...")
    try:
        model, tokenizer = load_model_and_tokenizer(model_path)
        model.eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # Setup output directory and handle L2 weights
    base_output_dir = setup_output_dirs(args.activations_root_path, args.model_name)
    weight_l2_path = os.path.join(base_output_dir, "weight_l2_info.pt")
    
    if not os.path.exists(weight_l2_path) or args.overwrite_l2:
        print("💾 Computing and saving weight L2 file...")
        try:
            compute_and_save_weight_l2(model, weight_l2_path)
        except Exception as e:
            print(f"❌ Error computing weight L2: {e}")
            sys.exit(1)
    else:
        print(f"✅ weight_l2_info.pt exists at {weight_l2_path}")

    # Load and process dataset
    print("📦 Loading dataset...")
    try:
        df = pd.read_parquet(args.data_path)
        df = process_dataset(df, args.tasks, args.sample_size, args.seed)
    except Exception as e:
        print(f"❌ Error loading/processing dataset: {e}")
        sys.exit(1)

    # Process each prompt type
    for prompt_type in args.prompt_types:
        print(f"\n🚀 Processing prompt_type: {prompt_type}")
        
        # Validate prompt type
        prompt_col = PROMPT_COLUMN_MAPPING.get(prompt_type)
        if prompt_col is None or prompt_col not in df.columns:
            print(f"❌ Prompt type '{prompt_type}' not found in dataset")
            continue

        # Prepare inputs
        shot_inputs = df[prompt_col].tolist()
        shot_task_types = df["task_type"].tolist()
        print(f"Processing {len(shot_inputs)} examples...")

        # Setup output directory
        activations_output_dir = setup_output_dirs(
            args.activations_root_path,
            args.model_name,
            prompt_type
        )

        # Capture and save activations
        try:
            hook_manager = ActivationHookManager()
            hook_manager.register_activation_hooks(model)

            print(f"[compute_activations] Capturing activations...")
            captured_acts = capture_activations(
                model=model,
                tokenizer=tokenizer,
                hook_manager=hook_manager,
                shot_inputs=shot_inputs,
                task_types=shot_task_types
            )

            print("[compute_activations] Saving to disk...")
            save_activations_dict(captured_acts, output_root=activations_output_dir)

        except Exception as e:
            print(f"❌ Error processing {prompt_type}: {e}")
            continue
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    print("✅ All prompt types processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute and save model activations for different prompt types.",
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
        choices=list(PROMPT_COLUMN_MAPPING.keys()),
        help="One or more prompt formats to process"
    )
    
    # Optional arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/processed/prompts.parquet",
        help="Path to processed prompts parquet file"
    )
    parser.add_argument(
        "--activations_root_path",
        type=str,
        default="./activations",
        help="Root directory for saving activations"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="List of task types to process (default: all)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=200,
        help="Number of samples per task type"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--overwrite_l2",
        action="store_true",
        help="Force recompute weight L2 information"
    )

    args = parser.parse_args()
    main(args)