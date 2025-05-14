#!/usr/bin/env python3
# coding: utf-8

import os
import shutil
import subprocess
import argparse
import torch
import gc
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.activations import load_activations
from src.pruning_utils.compute_scores import compute_all_layers_scores
from src.pruning_utils.generate_masks import generate_masks_for_all_layers
from src.pruning_utils.apply_pruning import apply_pruning_to_model
from src.remove_test import load_layerwise_results

env = os.environ.copy()
env["HF_ALLOW_CODE_EVAL"] = "1"


def prune_model_from_activations(
    model, activation_data, weight_data,
    num_layers, hidden_size, num_heads, intermediate_size, method,
    strategy, pruning_ratio, head_dim, use_softmask, strategy_kwargs,
    cos_sims=None, remove_results=None, fitted_results=None
):
    scores_dict = compute_all_layers_scores(
        activation_data=activation_data,
        weight_data=weight_data,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        method=method
    )

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

    apply_pruning_to_model(
        model=model,
        attn_masks=attn_masks,
        mlp_masks=mlp_masks,
        unstr=use_softmask,
        head_dim=head_dim
    )
    return model


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    task_list = args.task_types
    prompt_types = args.prompt_types
    model_output_dir = os.path.join(args.output_base_dir, args.model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    model_path = os.path.join(args.model_root_path, args.model_name)
    weight_l2_file = os.path.join(args.activations_root_path, args.model_name, "weight_l2_info.pt")
    weight_data = torch.load(weight_l2_file)

    # Load layerwise importance if needed
    cos_sims = remove_results = fitted_results = None
    if args.sparsity_strategy in {"cosine", "retention", "linear_fit", "logistic_fit"}:
        if not args.layer_importance_dir:
            raise ValueError(f"[!] Strategy '{args.sparsity_strategy}' requires --layer_importance_dir.")
        print(f"[INFO] Loading layer importance data from {args.layer_importance_dir}")
        _, cos_sims, remove_results, fitted_results = load_layerwise_results(
            model_name=args.model_name,
            base_dir=args.layer_importance_dir
        )

    for prompt_type in prompt_types:
        print(f"🔁 Prompt Type: {prompt_type}")
        activations_dir = os.path.join(args.activations_root_path, args.model_name, prompt_type)
        all_task_activations = load_activations(activations_dir)

        if args.calibration_task not in all_task_activations:
            raise ValueError(f"Calibration task {args.calibration_task} not found in activations.")

        activation_data = all_task_activations[args.calibration_task]

        temp_model_dir = os.path.join(
            args.temp_dir,
            f"{args.model_name}_calib-{args.calibration_task}_prompt-{prompt_type}"
        )

        try:
            print(f"🚀 Pruning model using calibration task: {args.calibration_task}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)

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

            if os.path.exists(temp_model_dir):
                shutil.rmtree(temp_model_dir)
            os.makedirs(temp_model_dir, exist_ok=True)
            pruned_model.save_pretrained(temp_model_dir)
            tokenizer.save_pretrained(temp_model_dir)

            del model, pruned_model
            torch.cuda.empty_cache()
            gc.collect()

            task_str = ",".join(task_list)
            print(f"📊 Evaluating on tasks: {task_str}")

            out_dir = os.path.join(
                model_output_dir,
                f"calib-{args.calibration_task}_prompt-{prompt_type}_method-{args.method}_strategy-{args.sparsity_strategy}_ratio-{args.pruning_ratio}"
            )
            os.makedirs(out_dir, exist_ok=True)

            eval_cmd = [
                "python", "-m", "lm_eval.__main__",
                "--model", "hf",
                "--model_args", f"pretrained={temp_model_dir}",
                "--tasks", task_str,
                "--batch_size", "auto",
                "--output_path", out_dir,
                "--confirm_run_unsafe_code",
            ]
            print("[evaluate_tasks] Running evaluation:", " ".join(eval_cmd))
            subprocess.run(eval_cmd, check=True, env=env)

        finally:
            if not args.keep_temp:
                shutil.rmtree(temp_model_dir, ignore_errors=True)
                print(f"🧹 Deleted temp model directory: {temp_model_dir}")
            torch.cuda.empty_cache()
            gc.collect()

    print("✅ All evaluations complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_root_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--activations_root_path", type=str, required=True)
    parser.add_argument("--prompt_types", nargs='+', required=True)
    parser.add_argument("--task_types", nargs='+', required=True)
    parser.add_argument("--calibration_task", type=str, required=True)
    parser.add_argument("--method", type=str, default="WIFV", choices=["WIFV", "WIFN"])
    parser.add_argument("--sparsity_strategy", type=str, default="retention",
                        choices=["uniform", "global", "cosine", "retention", "linear_fit", "logistic_fit"])
    parser.add_argument("--pruning_ratio", type=float, default=0.2)
    parser.add_argument("--protect_head", type=int, default=0)
    parser.add_argument("--protect_tail", type=int, default=0)
    parser.add_argument("--layer_importance_dir", type=str, default="./layer_importance")
    parser.add_argument("--hardmask", action="store_true")
    parser.add_argument("--temp_dir", type=str, default="./tmp")
    parser.add_argument("--output_base_dir", type=str, default="./eval_out")
    parser.add_argument("--keep_temp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)