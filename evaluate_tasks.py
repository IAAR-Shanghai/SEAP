#!/usr/bin/env python3
# coding: utf-8

import os
import shutil
import subprocess
import argparse
import torch
import gc
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.activations import load_activations
from src.pruning_utils.compute_scores import compute_all_layers_scores
from src.pruning_utils.generate_masks import generate_masks_for_all_layers
from src.pruning_utils.apply_pruning import apply_pruning_to_model
from src.remove_test import load_layerwise_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment variables
env = os.environ.copy()
env["HF_ALLOW_CODE_EVAL"] = "1"

def prune_model_from_activations(
    model: AutoModelForCausalLM,
    activation_data: Dict[str, torch.Tensor],
    weight_data: Dict[str, torch.Tensor],
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    intermediate_size: int,
    method: str,
    strategy: str,
    pruning_ratio: float,
    head_dim: int,
    use_softmask: bool,
    strategy_kwargs: Dict[str, Any],
    cos_sims: Optional[torch.Tensor] = None,
    remove_results: Optional[Dict[str, Any]] = None,
    fitted_results: Optional[Dict[str, Any]] = None
) -> AutoModelForCausalLM:
    """
    Prune a transformer model based on activation data and specified strategy.

    Args:
        model: The transformer model to prune
        activation_data: Dictionary containing activation data
        weight_data: Dictionary containing weight data
        num_layers: Number of transformer layers
        hidden_size: Hidden state dimension
        num_heads: Number of attention heads
        intermediate_size: Size of intermediate layer
        method: Pruning method (e.g., 'WIFV', 'WIFN')
        strategy: Pruning strategy
        pruning_ratio: Ratio of weights to prune
        head_dim: Dimension of attention heads
        use_softmask: Whether to use soft masking
        strategy_kwargs: Additional strategy parameters
        cos_sims: Optional cosine similarities
        remove_results: Optional removal test results
        fitted_results: Optional fitted curve results

    Returns:
        The pruned model
    """
    logger.info(f"Computing scores using method: {method}")
    scores_dict = compute_all_layers_scores(
        activation_data=activation_data,
        weight_data=weight_data,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        method=method
    )

    logger.info(f"Generating masks using strategy: {strategy}")
    attn_masks, mlp_masks = generate_masks_for_all_layers(
        scores_dict=scores_dict,
        strategy=strategy,
        pruning_ratio=pruning_ratio,
        hidden_size=hidden_size,
        num_heads=num_heads,
        total_layers=num_layers,
        strategy_kwargs=strategy_kwargs,
        cos_sims=cos_sims,
        remove_results=remove_results,
        fitted_results=fitted_results
    )

    logger.info("Applying pruning masks to model")
    apply_pruning_to_model(
        model=model,
        attn_masks=attn_masks,
        mlp_masks=mlp_masks,
        unstr=use_softmask,
        head_dim=head_dim
    )
    return model

def setup_model_and_tokenizer(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load and setup the model and tokenizer.

    Args:
        model_path: Path to the model

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model and tokenizer from {model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model/tokenizer: {str(e)}")
        raise

def evaluate_model(
    model_dir: str,
    task_list: List[str],
    output_dir: str,
    env: Dict[str, str]
) -> None:
    """
    Evaluate model on specified tasks.

    Args:
        model_dir: Directory containing the model
        task_list: List of tasks to evaluate
        output_dir: Directory to save results
        env: Environment variables
    """
    task_str = ",".join(task_list)
    logger.info(f"Evaluating model on tasks: {task_str}")

    os.makedirs(output_dir, exist_ok=True)
    
    eval_cmd = [
        "python", "-m", "lm_eval.__main__",
        "--model", "hf",
        "--model_args", f"pretrained={model_dir}",
        "--tasks", task_str,
        "--batch_size", "auto",
        "--output_path", output_dir,
        "--confirm_run_unsafe_code",
    ]
    
    try:
        logger.info(f"Running evaluation command: {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        raise

def main(args: argparse.Namespace) -> None:
    """
    Main function to run the evaluation pipeline.

    Args:
        args: Command line arguments
    """
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    try:
        # Setup paths
        model_output_dir = os.path.join(args.output_base_dir, args.model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        model_path = os.path.join(args.model_root_path, args.model_name)
        weight_l2_file = os.path.join(args.activations_root_path, args.model_name, "weight_l2_info.pt")
        
        # Load weight data
        logger.info(f"Loading weight data from {weight_l2_file}")
        weight_data = torch.load(weight_l2_file)

        # Load layer importance data if needed
        cos_sims = remove_results = fitted_results = None
        if args.sparsity_strategy in {"cosine", "retention", "linear_fit", "logistic_fit"}:
            if not args.layer_importance_dir:
                raise ValueError(f"Strategy '{args.sparsity_strategy}' requires --layer_importance_dir")
            
            logger.info(f"Loading layer importance data from {args.layer_importance_dir}")
            _, cos_sims, remove_results, fitted_results = load_layerwise_results(
                model_name=args.model_name,
                base_dir=args.layer_importance_dir
            )

        # Process each prompt type
        for prompt_type in args.prompt_types:
            logger.info(f"Processing prompt type: {prompt_type}")
            
            # Load activations
            activations_dir = os.path.join(args.activations_root_path, args.model_name, prompt_type)
            all_task_activations = load_activations(activations_dir)

            if args.calibration_task not in all_task_activations:
                raise ValueError(f"Calibration task {args.calibration_task} not found in activations")

            activation_data = all_task_activations[args.calibration_task]

            # Setup temporary model directory
            temp_model_dir = os.path.join(
                args.temp_dir,
                f"{args.model_name}_calib-{args.calibration_task}_prompt-{prompt_type}"
            )

            try:
                logger.info(f"Pruning model using calibration task: {args.calibration_task}")
                
                # Load model and tokenizer
                model, tokenizer = setup_model_and_tokenizer(model_path)

                # Get model configuration
                config = model.config
                hidden_size = config.hidden_size
                num_heads = config.num_attention_heads
                head_dim = hidden_size // num_heads
                num_layers = config.num_hidden_layers
                intermediate_size = config.intermediate_size
                use_softmask = not args.hardmask

                strategy_kwargs = {
                    "protect_head": args.protect_head,
                    "protect_tail": args.protect_tail
                }

                # Prune model
                pruned_model = prune_model_from_activations(
                    model=model,
                    activation_data=activation_data,
                    weight_data=weight_data,
                    num_layers=num_layers,
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    method=args.method,
                    strategy=args.sparsity_strategy,
                    pruning_ratio=args.pruning_ratio,
                    head_dim=head_dim,
                    use_softmask=use_softmask,
                    strategy_kwargs=strategy_kwargs,
                    cos_sims=cos_sims,
                    remove_results=remove_results,
                    fitted_results=fitted_results
                )

                # Save pruned model
                if os.path.exists(temp_model_dir):
                    shutil.rmtree(temp_model_dir)
                os.makedirs(temp_model_dir, exist_ok=True)
                
                logger.info(f"Saving pruned model to {temp_model_dir}")
                pruned_model.save_pretrained(temp_model_dir)
                tokenizer.save_pretrained(temp_model_dir)

                # Clean up
                del model, pruned_model
                torch.cuda.empty_cache()
                gc.collect()

                # Setup evaluation output directory
                eval_out_dir = os.path.join(
                    model_output_dir,
                    f"calib-{args.calibration_task}_prompt-{prompt_type}_method-{args.method}_strategy-{args.sparsity_strategy}_ratio-{args.pruning_ratio}"
                )
                
                # Run evaluation
                evaluate_model(temp_model_dir, args.task_types, eval_out_dir, env)

            finally:
                if not args.keep_temp:
                    logger.info(f"Cleaning up temporary directory: {temp_model_dir}")
                    shutil.rmtree(temp_model_dir, ignore_errors=True)
                torch.cuda.empty_cache()
                gc.collect()

        logger.info("All evaluations completed successfully")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pruned transformer models on various tasks.")
    
    # Model and data paths
    parser.add_argument("--model_root_path", type=str, required=True,
                      help="Root directory containing the base model")
    parser.add_argument("--model_name", type=str, required=True,
                      help="Name of the model to evaluate")
    parser.add_argument("--activations_root_path", type=str, required=True,
                      help="Root directory containing activation data")
    
    # Task and prompt configuration
    parser.add_argument("--prompt_types", nargs='+', required=True,
                      help="List of prompt types to evaluate")
    parser.add_argument("--task_types", nargs='+', required=True,
                      help="List of tasks to evaluate on")
    parser.add_argument("--calibration_task", type=str, required=True,
                      help="Task to use for calibration")
    
    # Pruning configuration
    parser.add_argument("--method", type=str, default="WIFV",
                      choices=["WIFV", "WIFN"],
                      help="Pruning method to use")
    parser.add_argument("--sparsity_strategy", type=str, default="retention",
                      choices=["uniform", "global", "cosine", "retention", "linear_fit", "logistic_fit"],
                      help="Strategy for applying sparsity")
    parser.add_argument("--pruning_ratio", type=float, default=0.2,
                      help="Ratio of weights to prune")
    parser.add_argument("--protect_head", type=int, default=0,
                      help="Number of head layers to protect")
    parser.add_argument("--protect_tail", type=int, default=0,
                      help="Number of tail layers to protect")
    
    # Additional configuration
    parser.add_argument("--layer_importance_dir", type=str, default="./layer_importance",
                      help="Directory containing layer importance data")
    parser.add_argument("--hardmask", action="store_true",
                      help="Use hard masking instead of soft masking")
    parser.add_argument("--temp_dir", type=str, default="./tmp",
                      help="Directory for temporary files")
    parser.add_argument("--output_base_dir", type=str, default="./eval_out",
                      help="Base directory for evaluation outputs")
    parser.add_argument("--keep_temp", action="store_true",
                      help="Keep temporary files after evaluation")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    
    args = parser.parse_args()

    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    main(args)