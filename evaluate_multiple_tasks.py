import os
os.environ["HF_DATASETS_CACHE"] = "../.cache/huggingface/datasets/"

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
    
    attn_masks, mlp_masks = load_masks_from_file(mask_file)
    if biascomp and (activations_data is not None):
        attn_mean_inps = {}
        mlp_mean_inps  = {}
        for layer_idx, layer_info in activations_data.items():
            attn_agg = layer_info.get("attention_post_aggregation", {})
            attn_mean = attn_agg.get("mean", None)
            mlp_inter = layer_info.get("mlp_intermediate_states", {})
            mlp_mean  = mlp_inter.get("mean", None)
            attn_mean_inps[layer_idx] = attn_mean
            mlp_mean_inps[layer_idx]  = mlp_mean
    else:
        attn_mean_inps = None
        mlp_mean_inps  = None
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

    model_output_dir = os.path.join(args.output_base_dir, args.model_name)
    os.makedirs(model_output_dir, exist_ok=True)

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

    if args.use_generic_mask:
        # ----------------- 通用模型评估分支 -----------------
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
            num_heads   = getattr(model.config, "num_attention_heads", None)
            if hidden_size is None or num_heads is None:
                print("[evaluate_multiple_tasks] Cannot auto compute head_dim => using 128")
                auto_head_dim = 128
            else:
                if hidden_size % num_heads != 0:
                    print(f"Warning: hidden_size={hidden_size} not divisible by num_heads={num_heads}, using int division")
                auto_head_dim = hidden_size // num_heads
            print(f"[evaluate_multiple_tasks] auto_head_dim={auto_head_dim}")

            # 计算并使用通用模型掩码
            task_generic = "generic"
            task_mask_dir = os.path.join(
                args.pruning_indices_root_dir,
                args.model_name,
                f"task={task_generic}_method={args.method}_structure={args.structure}_ratio={args.pruning_ratio}"
            )
            mask_file = os.path.join(task_mask_dir, f"{task_generic}_masks.pt")
            print(f"[evaluate_multiple_tasks] Using generic mask: {mask_file}")

            pruned_model = prune_model_for_task(
                model=model,
                mask_file=mask_file,
                activations_data=None,  # 没有特定的激活数据
                biascomp=args.biascomp,
                softmask=args.softmask,
                head_dim=auto_head_dim,
                method=args.method,
                structure=args.structure,
                pruning_ratio=args.pruning_ratio,
                device="cuda"
            )

            temp_model_dir = os.path.join(args.temp_dir, f"pruned_{args.model_name}_generic")
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
    
    if args.use_wiki_mask:
        # ----------------- Wiki Mask 分支 -----------------
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
            num_heads   = getattr(model.config, "num_attention_heads", None)
            if hidden_size is None or num_heads is None:
                print("[evaluate_multiple_tasks] Cannot auto compute head_dim => using 128")
                auto_head_dim = 128
            else:
                if hidden_size % num_heads != 0:
                    print(f"Warning: hidden_size={hidden_size} not divisible by num_heads={num_heads}, using int division")
                auto_head_dim = hidden_size // num_heads
            print(f"[evaluate_multiple_tasks] auto_head_dim={auto_head_dim}")

            task_wikitext2 = "wikitext2"
            task_mask_dir = os.path.join(
                args.pruning_indices_root_dir,
                args.model_name,
                f"task={task_wikitext2}_method={args.method}_structure={args.structure}_ratio={args.pruning_ratio}"
            )
            mask_file = os.path.join(task_mask_dir, f"{task_wikitext2}_masks.pt")
            print(f"[evaluate_multiple_tasks] Using wikitext2 mask: {mask_file}")

            activations_data_for_task = all_task_activations.get(task_wikitext2, None)
            if activations_data_for_task is None:
                print(f"[evaluate_multiple_tasks] Warning: No activations for wikitext2 mask ({task_wikitext2}). Bias compensation disabled.")
            else:
                print(f"[evaluate_multiple_tasks] Using wikitext2 activations for bias compensation.")

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
            print(f"Error while processing generic mask: {e}")
            if 'temp_model_dir' in locals() and os.path.exists(temp_model_dir):
                print(f"Removing temporary model dir: {temp_model_dir}")
                shutil.rmtree(temp_model_dir, ignore_errors=True)
            gc.collect()
            torch.cuda.empty_cache()
            raise

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
            num_heads   = getattr(model.config, "num_attention_heads", None)
            if hidden_size is None or num_heads is None:
                print("[evaluate_multiple_tasks] Cannot auto compute head_dim => using 128")
                auto_head_dim = 128
            else:
                if hidden_size % num_heads != 0:
                    print(f"Warning: hidden_size={hidden_size} not divisible by num_heads={num_heads}, using int division")
                auto_head_dim = hidden_size // num_heads
            print(f"[evaluate_multiple_tasks] auto_head_dim={auto_head_dim}")

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
    parser = argparse.ArgumentParser(description="Evaluate pruned or masked models on multiple tasks via lm_eval.")
    parser.add_argument("--model_root_path", type=str, required=True,
                        help="Root dir of original HF model.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="e.g. 'Llama-2-7b-hf'.")
    parser.add_argument("--pruning_indices_root_dir", type=str, required=True,
                        help="Where <task>_masks.pt are stored.")
    parser.add_argument("--temp_dir", type=str, default="./tmp",
                        help="A temporary directory for pruned models.")
    parser.add_argument("--output_base_dir", type=str, default="./eval_out",
                        help="Directory for evaluation results.")
    parser.add_argument("--keep_temp", action="store_true",
                        help="If set, keep the pruned model directory after evaluation.")
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
                        help="Fraction of heads/channels to prune. Default=0.1")
    parser.add_argument("--task_types", type=str, default="piqa,winogrande,hellaswag",
                        help="Comma-separated list of tasks to evaluate, e.g., 'piqa,winogrande'.")
    parser.add_argument("--use_wiki_mask", action="store_true", 
                        help="If set, use the wikitext2 mask (wikitext2_masks.pt) instead of task-specific masks.")
    parser.add_argument("--use_generic_mask", action="store_true", 
                        help="If set, use the generic mask (generic_masks.pt) instead of task-specific masks.")
    args = parser.parse_args()
    main(args)
