import os
import shutil
import subprocess
import argparse
import torch
import gc
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activations import load_activations
from src.pruning_utils.generate_masks import load_masks_from_file
from src.pruning_utils.apply_pruning import apply_pruning_to_model

def prune_model_for_task(
    model: AutoModelForCausalLM,
    mask_file: str,
    activations_data: dict = None,
    biascomp: bool = True,
    softmask: bool = False,
    head_dim: int = 128,
    method: str = "WIFV",
    structure: str = "UL-LD",
    pruning_ratio: float = 0.1,
    device: str = "cuda"
):
    """
    This function prunes the given model based on a task-specific mask and performs pruning strategies.
    It applies the masks for attention layers and MLP layers.
    
    Parameters:
    model (AutoModelForCausalLM): The model to be pruned.
    mask_file (str): Path to the pruning mask file.
    activations_data (dict, optional): Activations data used for bias compensation. Default is None.
    biascomp (bool): Whether to enable bias compensation. Default is True.
    softmask (bool): Whether to use unstructured pruning (mask to 0). Default is False.
    head_dim (int): Head dimension for attention layers. Default is 128.
    method (str): Method used for pruning. Default is "WIFV".
    structure (str): Pruning structure. Default is "UL-LD".
    pruning_ratio (float): Fraction of heads/channels to prune. Default is 0.1.
    device (str): The device to run the model on. Default is "cuda".

    Returns:
    AutoModelForCausalLM: The pruned model.
    """
    attn_masks, mlp_masks = load_masks_from_file(mask_file)
    
    # Bias compensation if activations data is provided
    if biascomp and (activations_data is not None):
        attn_mean_inps = {}
        mlp_mean_inps = {}
        for layer_idx, layer_info in activations_data.items():
            attn_agg = layer_info.get("attention_post_aggregation", {})
            attn_mean = attn_agg.get("mean", None)
            mlp_inter = layer_info.get("mlp_intermediate_states", {})
            mlp_mean = mlp_inter.get("mean", None)
            attn_mean_inps[layer_idx] = attn_mean
            mlp_mean_inps[layer_idx] = mlp_mean
    else:
        attn_mean_inps = None
        mlp_mean_inps = None
    
    # Apply pruning using the provided masks and activations data
    with torch.no_grad():
        apply_pruning_to_model(
            model=model,
            attn_masks=attn_masks,
            mlp_masks=mlp_masks,
            attn_mean_inps=attn_mean_inps,
            mlp_mean_inps=mlp_mean_inps,
            device=device,
            bias=biascomp,
            unstr=softmask,
            head_dim=head_dim
        )

    return model

def main(args):
    """
    Main function that processes model pruning, evaluation, and task-specific evaluations.
    It supports various pruning strategies and bias compensation, and evaluates the pruned models
    on different tasks using lm_eval.

    Parameters:
    args (Namespace): Arguments parsed from command line input, including paths, task names, and other settings.
    """
    # Mapping of task names to internal task names used by lm_eval
    task_map = {
        "mmlu": "mmlu",
        "piqa": "piqa",
        "winogrande": "winogrande",
        "hellaswag": "hellaswag",
        "gsm8k": "gsm8k",
        "ai2_arc": "arc_easy,arc_challenge",
        "boolq": "boolq",
        "obqa": "openbookqa"
    }

    task_types = args.task_types.split(",") if args.task_types else list(task_map.keys())

    shots_map = {
        "mmlu": 0,
        "arc_easy,arc_challenge": 0,
        "hellaswag": 0,
        "gsm8k": 0,
        "piqa": 0,
        "winogrande": 0,
        "openbookqa": 0,
        "boolq": 0
    }

    # Set up output directory for model results
    model_output_dir = os.path.join(args.output_base_dir, args.model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Load activation data if available
    all_task_activations = {}
    if args.activations_root_path:
        act_dir = os.path.join(args.activations_root_path, args.model_name)
        if os.path.isdir(act_dir):
            print(f"[evaluate_multiple_tasks] Loading activations from: {act_dir}")
            all_task_activations = load_activations(act_dir)
        else:
            print(f"Warning: no activation dir {act_dir}, skip bias compensation.")
    else:
        print("[evaluate_multiple_tasks] No activations path => no biascomp data.")

    # Handle generic mask evaluation branch
    if args.use_generic_mask:
        try:
            print("[evaluate_multiple_tasks] Using generic mask for evaluation.")
            model_path = os.path.join(args.model_root_path, args.model_name)
            print(f"[evaluate_multiple_tasks] Loading model from {model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            hidden_size = getattr(model.config, "hidden_size", None)
            num_heads = getattr(model.config, "num_attention_heads", None)
            
            # Compute head dimension if not provided
            if hidden_size is None or num_heads is None:
                print("[evaluate_multiple_tasks] Cannot auto compute head_dim => using 128")
                auto_head_dim = 128
            else:
                if hidden_size % num_heads != 0:
                    print(f"Warning: hidden_size={hidden_size} not divisible by num_heads={num_heads}, using int division")
                auto_head_dim = hidden_size // num_heads
            print(f"[evaluate_multiple_tasks] auto_head_dim={auto_head_dim}")

            # Construct the mask file path
            task_generic = "generic"
            task_mask_dir = os.path.join(
                args.pruning_indices_root_dir,
                args.model_name,
                f"task={task_generic}_method={args.method}_structure={args.structure}_ratio={args.pruning_ratio}"
            )
            mask_file = os.path.join(task_mask_dir, f"{task_generic}_masks.pt")
            print(f"[evaluate_multiple_tasks] Using generic mask: {mask_file}")

            # Perform pruning for generic task
            pruned_model = prune_model_for_task(
                model=model,
                mask_file=mask_file,
                activations_data=None,  # No specific activations for generic mask
                biascomp=args.biascomp,
                softmask=args.softmask,
                head_dim=auto_head_dim,
                method=args.method,
                structure=args.structure,
                pruning_ratio=args.pruning_ratio,
                device="cuda"
            )

            # Save the pruned model to a temporary directory
            temp_model_dir = os.path.join(args.temp_dir, f"pruned_{args.model_name}_generic")
            if os.path.exists(temp_model_dir):
                shutil.rmtree(temp_model_dir)
            os.makedirs(temp_model_dir, exist_ok=True)

            print(f"[evaluate_multiple_tasks] Saving pruned model to {temp_model_dir}")
            pruned_model.save_pretrained(temp_model_dir)
            tokenizer.save_pretrained(temp_model_dir)

            # Clear memory to free up GPU resources
            del model
            del pruned_model
            gc.collect()
            torch.cuda.empty_cache()

            # Evaluate tasks using the pruned model
            for internal_task_name in task_types:
                lm_eval_task_name = task_map.get(internal_task_name, internal_task_name)
                print(f"\n=== Evaluating task: {internal_task_name} (lm_eval={lm_eval_task_name}) using generic mask ===")

                out_dir = os.path.join(
                    model_output_dir,
                    f"task-{internal_task_name}_method-{args.method}_structure-{args.structure}_ratio-{args.pruning_ratio}"
                )
                os.makedirs(out_dir, exist_ok=True)

                num_fewshot = shots_map.get(lm_eval_task_name, 0)
                model_args_str = f"pretrained={temp_model_dir}"

                eval_cmd = [
                    "python",
                    "-m", "lm_eval.__main__",
                    "--model", "hf",
                    "--model_args", model_args_str,
                    "--tasks", lm_eval_task_name,
                    "--batch_size", "auto",
                    "--output_path", out_dir,
                    "--num_fewshot", str(num_fewshot),
                ]
                print("[evaluate_multiple_tasks] Running lm_eval:", " ".join(eval_cmd))
                subprocess.run(eval_cmd, check=True)

                gc.collect()
                torch.cuda.empty_cache()

            # Remove temporary model directory if not kept
            if not args.keep_temp:
                print(f"[evaluate_multiple_tasks] Removing temporary model dir {temp_model_dir}")
                shutil.rmtree(temp_model_dir, ignore_errors=True)

        except Exception as e:
            print(f"Error while processing generic mask: {e}")
            if 'temp_model_dir' in locals() and os.path.exists(temp_model_dir):
                print(f"Removing temporary model dir: {temp_model_dir}")
                shutil.rmtree(temp_model_dir, ignore_errors=True)
            gc.collect()
            torch.cuda.empty_cache()
            raise
    
    # Handle Wiki mask evaluation branch
    if args.use_wiki_mask:
        try:
            print("[evaluate_multiple_tasks] Using wiki mask for evaluation.")
            model_path = os.path.join(args.model_root_path, args.model_name)
            print(f"[evaluate_multiple_tasks] Loading model from {model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            hidden_size = getattr(model.config, "hidden_size", None)
            num_heads = getattr(model.config, "num_attention_heads", None)
            if hidden_size is None or num_heads is None:
                print("[evaluate_multiple_tasks] Cannot auto compute head_dim => using 128")
                auto_head_dim = 128
            else:
                if hidden_size % num_heads != 0:
                    print(f"Warning: hidden_size={hidden_size} not divisible by num_heads={num_heads}, using int division")
                auto_head_dim = hidden_size // num_heads
            print(f"[evaluate_multiple_tasks] auto_head_dim={auto_head_dim}")

            # Set up Wiki task mask directory
            task_wikitext2 = "wikitext2"
            task_mask_dir = os.path.join(
                args.pruning_indices_root_dir,
                args.model_name,
                f"task={task_wikitext2}_method={args.method}_structure={args.structure}_ratio={args.pruning_ratio}"
            )
            mask_file = os.path.join(task_mask_dir, f"{task_wikitext2}_masks.pt")
            print(f"[evaluate_multiple_tasks] Using wikitext2 mask: {mask_file}")

            # Get activation data for the Wiki task if available
            activations_data_for_task = all_task_activations.get(task_wikitext2, None)
            if activations_data_for_task is None:
                print(f"[evaluate_multiple_tasks] Warning: No activations for wikitext2 mask ({task_wikitext2}). Bias compensation disabled.")
            else:
                print(f"[evaluate_multiple_tasks] Using wikitext2 activations for bias compensation.")

            # Prune the model for Wiki task
            pruned_model = prune_model_for_task(
                model=model,
                mask_file=mask_file,
                activations_data=activations_data_for_task,
                biascomp=args.biascomp,
                softmask=args.softmask,
                head_dim=auto_head_dim,
                method=args.method,
                structure=args.structure,
                pruning_ratio=args.pruning_ratio,
                device="cuda"
            )

            # Save the pruned model to a temporary directory
            temp_model_dir = os.path.join(args.temp_dir, f"pruned_{args.model_name}_wikitext2")
            if os.path.exists(temp_model_dir):
                shutil.rmtree(temp_model_dir)
            os.makedirs(temp_model_dir, exist_ok=True)

            print(f"[evaluate_multiple_tasks] Saving pruned model to {temp_model_dir}")
            pruned_model.save_pretrained(temp_model_dir)
            tokenizer.save_pretrained(temp_model_dir)

            del model
            del pruned_model
            gc.collect()
            torch.cuda.empty_cache()

            # Evaluate tasks using the pruned model
            for internal_task_name in task_types:
                lm_eval_task_name = task_map.get(internal_task_name, internal_task_name)
                print(f"\n=== Evaluating task: {internal_task_name} (lm_eval={lm_eval_task_name}) using wikitext2 mask ===")

                out_dir = os.path.join(
                    model_output_dir,
                    f"task-{internal_task_name}_method-{args.method}_structure-{args.structure}_ratio-{args.pruning_ratio}"
                )
                os.makedirs(out_dir, exist_ok=True)

                num_fewshot = shots_map.get(lm_eval_task_name, 0)
                model_args_str = f"pretrained={temp_model_dir}"

                eval_cmd = [
                    "python",
                    "-m", "lm_eval.__main__",
                    "--model", "hf",
                    "--model_args", model_args_str,
                    "--tasks", lm_eval_task_name,
                    "--batch_size", "auto",
                    "--output_path", out_dir,
                    "--num_fewshot", str(num_fewshot),
                ]
                print("[evaluate_multiple_tasks] Running lm_eval:", " ".join(eval_cmd))
                subprocess.run(eval_cmd, check=True)

                gc.collect()
                torch.cuda.empty_cache()

            if not args.keep_temp:
                print(f"[evaluate_multiple_tasks] Removing temporary model dir {temp_model_dir}")
                shutil.rmtree(temp_model_dir, ignore_errors=True)

        except Exception as e:
            print(f"Error while processing wiki mask: {e}")
            if 'temp_model_dir' in locals() and os.path.exists(temp_model_dir):
                print(f"Removing temporary model dir: {temp_model_dir}")
                shutil.rmtree(temp_model_dir, ignore_errors=True)
            gc.collect()
            torch.cuda.empty_cache()
            raise
    
    # Evaluate task-specific models
    for internal_task_name in task_types:
        lm_eval_task_name = task_map.get(internal_task_name, internal_task_name)
        print(f"\n=== Evaluating task: {internal_task_name} (lm_eval={lm_eval_task_name}) ===")

        try:
            model_path = os.path.join(args.model_root_path, args.model_name)
            print(f"[evaluate_multiple_tasks] Loading model from {model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            hidden_size = getattr(model.config, "hidden_size", None)
            num_heads = getattr(model.config, "num_attention_heads", None)
            if hidden_size is None or num_heads is None:
                print("[evaluate_multiple_tasks] Cannot auto compute head_dim => using 128")
                auto_head_dim = 128
            else:
                if hidden_size % num_heads != 0:
                    print(f"Warning: hidden_size={hidden_size} not divisible by num_heads={num_heads}, using int division")
                auto_head_dim = hidden_size // num_heads
            print(f"[evaluate_multiple_tasks] auto_head_dim={auto_head_dim}")

            # Set up task-specific mask directory
            task_mask_dir = os.path.join(args.pruning_indices_root_dir, args.model_name,
                                            f"task={internal_task_name}_method={args.method}_structure={args.structure}_ratio={args.pruning_ratio}")
            mask_file = os.path.join(task_mask_dir, f"{internal_task_name}_masks.pt")
            if not os.path.exists(mask_file):
                print(f"[evaluate_multiple_tasks] mask_file={mask_file} not found => skip {internal_task_name}.")
                del model
                gc.collect()
                torch.cuda.empty_cache()
                continue

            if args.biascomp and (internal_task_name in all_task_activations):
                activations_data_for_task = all_task_activations[internal_task_name]
            else:
                activations_data_for_task = None

            # Prune the model for the task
            pruned_model = prune_model_for_task(
                model=model,
                mask_file=mask_file,
                activations_data=activations_data_for_task,
                biascomp=args.biascomp,
                softmask=args.softmask,
                head_dim=auto_head_dim,
                method=args.method,
                structure=args.structure,
                pruning_ratio=args.pruning_ratio,
                device="cuda"
            )

            temp_model_dir = os.path.join(args.temp_dir, f"pruned_{args.model_name}_{internal_task_name}")
            if os.path.exists(temp_model_dir):
                shutil.rmtree(temp_model_dir)
            os.makedirs(temp_model_dir, exist_ok=True)

            print(f"[evaluate_multiple_tasks] Saving pruned model to {temp_model_dir}")
            pruned_model.save_pretrained(temp_model_dir)
            tokenizer.save_pretrained(temp_model_dir)

            del pruned_model
            del model
            gc.collect()
            torch.cuda.empty_cache()

            # Set up output directory and evaluate the task
            out_dir = os.path.join(model_output_dir, 
                                    f"task-{internal_task_name}_method-{args.method}_structure-{args.structure}_ratio-{args.pruning_ratio}")
            os.makedirs(out_dir, exist_ok=True)

            num_fewshot = shots_map.get(lm_eval_task_name, 0)
            model_args_str = f"pretrained={temp_model_dir}"

            eval_cmd = [
                "python",
                "-m", "lm_eval.__main__",
                "--model", "hf",
                "--model_args", model_args_str,
                "--tasks", lm_eval_task_name,
                "--batch_size", "auto",
                "--output_path", out_dir,
                "--num_fewshot", str(num_fewshot),
            ]
            print("[evaluate_multiple_tasks] Running lm_eval:", " ".join(eval_cmd))
            subprocess.run(eval_cmd, check=True)

        except Exception as e:
            print(f"Error while processing task {internal_task_name}: {e}")
            if 'temp_model_dir' in locals() and os.path.exists(temp_model_dir):
                print(f"Removing temporary model dir: {temp_model_dir}")
                shutil.rmtree(temp_model_dir, ignore_errors=True)
            gc.collect()
            torch.cuda.empty_cache()
            raise

        if not args.keep_temp and 'temp_model_dir' in locals() and os.path.exists(temp_model_dir):
            print(f"[evaluate_multiple_tasks] Removing tmp dir {temp_model_dir}")
            shutil.rmtree(temp_model_dir, ignore_errors=True)

        gc.collect()
        torch.cuda.empty_cache()

    print("\n[evaluate_multiple_tasks] All tasks done.")

if __name__ == "__main__":
    """
    Main function to handle command-line argument parsing and initiate model pruning and evaluation.

    Parses the command-line arguments and calls the `main` function to evaluate pruned or masked 
    models on multiple tasks via `lm_eval`.

    Args:
        None: The function reads command-line arguments.
    
    Returns:
        None: Executes the pruning and evaluation process based on the provided arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate pruned or masked models on multiple tasks via lm_eval.")
    parser.add_argument("--model_root_path", type=str, required=True,
                        help="Root dir of original HF model. Specify the path to the model's root directory.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="e.g. 'Llama-2-7b-hf'. The name of the model to be used for evaluation.")
    parser.add_argument("--pruning_indices_root_dir", type=str, required=True,
                        help="Where <task>_masks.pt are stored. Specify the path to the directory containing pruning masks.")
    parser.add_argument("--temp_dir", type=str, default="./tmp",
                        help="A temporary directory for pruned models. The default is './tmp'.")
    parser.add_argument("--output_base_dir", type=str, default="./eval_out",
                        help="Directory for evaluation results. The default is './eval_out'.")
    parser.add_argument("--keep_temp", action="store_true",
                        help="If set, keeps the pruned model directory after evaluation.")
    parser.add_argument("--activations_root_path", type=str, default=None,
                        help="If --biascomp is True, we load baseline_inps from here.")
    parser.add_argument("--biascomp", action="store_true",
                        help="Enable FLAP-style bias compensation (need baseline_inps).")
    parser.add_argument("--softmask", action="store_true",
                        help="Use unstructured pruning (mask to 0) instead of structured pruning.")
    parser.add_argument("--method", type=str, default="WIFV", 
                        choices=["WIFV", "WIFN"],
                        help="Scoring method used for pruning. Default is 'WIFV'.")
    parser.add_argument("--structure", type=str, default="UL-LD", 
                        choices=["UL-UM", "AL-AM", "UL-LD"],
                        help="Pruning structure strategy. Default is 'UL-LD'.")
    parser.add_argument("--pruning_ratio", type=float, default=0.1,
                        help="Fraction of heads/channels to prune. Default is 0.1.")
    parser.add_argument("--task_types", type=str, default="piqa,winogrande,hellaswag",
                        help="Comma-separated list of tasks to evaluate, e.g., 'piqa,winogrande'.")
    parser.add_argument("--use_wiki_mask", action="store_true", 
                        help="If set, use the wikitext2 mask (wikitext2_masks.pt) instead of task-specific masks.")
    parser.add_argument("--use_generic_mask", action="store_true", 
                        help="If set, use the generic mask (generic_masks.pt) instead of task-specific masks.")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Call the main function to process the model pruning and evaluation
    main(args)