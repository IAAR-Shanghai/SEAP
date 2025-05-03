#!/usr/bin/env python3
# coding: utf-8

import sys
import os
import gc
import torch
import random
import numpy as np
import argparse
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.activations import (
    ActivationHookManager,
    capture_activations,
    save_activations_dict,
    compute_and_save_weight_l2
)
from src.model_utils import load_model_and_tokenizer

PROMPT_COLUMN_MAPPING = {
    "zero_shot": "prompt_zero_shot",
    "cot": "prompt_cot",
    "icl": "prompt_icl",
    "icl_cot": "prompt_icl_cot",
    "knowledge": "knowledge"
}

def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    model_path = os.path.join(args.model_root_path, args.model_name)
    print("🧠 Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval()

    # 确保 L2 文件存在，或强制重新计算
    base_output_dir = os.path.join(args.activations_root_path, args.model_name)
    os.makedirs(base_output_dir, exist_ok=True)
    weight_l2_path = os.path.join(base_output_dir, "weight_l2_info.pt")
    if not os.path.exists(weight_l2_path) or args.overwrite_l2:
        print("💾 Computing and saving weight L2 file...")
        compute_and_save_weight_l2(model, weight_l2_path)
    else:
        print(f"✅ weight_l2_info.pt already exists at {weight_l2_path}, skipping computation.")

    print("📦 Loading dataset...")
    df = pd.read_parquet(args.data_path)

    if args.tasks:
        df = df[df["task_type"].isin(args.tasks)]

    if "split" in df.columns:
        df = df[df["split"] == "train"]

    if args.sample_size:
        df = df.groupby("task_type", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), args.sample_size), random_state=args.seed)
        ).reset_index(drop=True)

    for prompt_type in args.prompt_types:
        print(f"\n🚀 Processing prompt_type: {prompt_type}")
        prompt_col = PROMPT_COLUMN_MAPPING.get(prompt_type)
        if prompt_col is None or prompt_col not in df.columns:
            raise ValueError(f"Prompt type '{prompt_type}' not found in dataset.")

        shot_inputs = df[prompt_col].tolist()
        shot_task_types = df["task_type"].tolist()

        activations_output_dir = os.path.join(args.activations_root_path, args.model_name, prompt_type)
        os.makedirs(activations_output_dir, exist_ok=True)

        hook_manager = ActivationHookManager()
        hook_manager.register_activation_hooks(model)

        print(f"[compute_activations] Capturing activations for prompt_type={prompt_type} ...")
        print(len(shot_inputs))
        captured_acts = capture_activations(
            model=model,
            tokenizer=tokenizer,
            hook_manager=hook_manager,
            shot_inputs=shot_inputs,
            task_types=shot_task_types
        )

        print("[compute_activations] Saving activations to disk...")
        save_activations_dict(captured_acts, output_root=activations_output_dir)

        torch.cuda.empty_cache()
        gc.collect()

    print("✅ All prompt types processed and saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute and save activations from model forward passes.")
    parser.add_argument("--model_root_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="./data/processed/prompts.parquet")
    parser.add_argument("--activations_root_path", type=str, default="./activations")
    parser.add_argument("--prompt_types", nargs="+", required=True,
                        choices=["zero_shot", "cot", "icl", "icl_cot", "knowledge"],
                        help="One or more prompt formats to use.")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="List of task names to compute activations for. If not set, all tasks are processed.")
    parser.add_argument("--sample_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite_l2", action="store_true",
                        help="Force recompute and overwrite weight_l2_info.pt file.")

    args = parser.parse_args()
    main(args)
