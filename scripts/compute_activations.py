#!/usr/bin/env python3
# coding: utf-8
# scripts/compute_activations.py

import sys
import os
import gc
import torch
import random
import numpy as np
from typing import List
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# 引入我们定义的函数和类
from src.activations import (
    ActivationHookManager,
    save_activations,
    compute_and_save_weight_l2
)
from src.data_utils import load_datasets, build_few_shot_prompts
from src.model_utils import load_model_and_tokenizer

def main(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # 构建 model_path / activations_output_dir
    model_path = os.path.join(args.model_root_path, args.model_name)
    activations_output_dir = os.path.join(args.activations_root_path, args.model_name)
    os.makedirs(activations_output_dir, exist_ok=True)

    # 加载数据
    datasets = load_datasets(args.data_dir, split='train')

    shot_inputs, shot_task_types = build_few_shot_prompts(
        datasets,
        min_shot=args.min_shot,
        max_shot=args.max_shot,
        seed=args.seed,
        sample_size=args.sample_size, 
        use_corpus=True
    )
    # 加载模型 / 分词器
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval()

    # ===== 一次性地计算并保存权重的 L2 范数 =====
    if args.save_weight_l2:
        weight_l2_save_path = os.path.join(activations_output_dir, "weight_l2_info.pt")
        compute_and_save_weight_l2(model, weight_l2_save_path)

    # 注册hook（在线统计激活值）
    hook_manager = ActivationHookManager()
    hook_manager.register_activation_hooks(model)

    # 收集并保存激活值
    with torch.no_grad():
        save_activations(
            model=model,
            tokenizer=tokenizer,
            hook_manager=hook_manager,
            shot_inputs=shot_inputs,
            task_types=shot_task_types,
            output_root=activations_output_dir
        )

    print("Activations computation and saving completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute activations for different tasks using model forward passes.")
    parser.add_argument("--data_dir", type=str, default="./data/processed", help="Directory of processed data.")
    parser.add_argument("--model_root_path", type=str, required=True, help="Root directory of models.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name, used to build full model path.")
    parser.add_argument("--activations_root_path", type=str, default="./activations", help="Root path for saving activations.")
    parser.add_argument("--sample_size", type=int, default=200, help="Sample size of tasks to use.")
    parser.add_argument("--min_shot", type=int, default=0, help="Min shot for few-shot prompts.")
    parser.add_argument("--max_shot", type=int, default=1, help="Max shot for few-shot prompts.")
    parser.add_argument("--shot_seed", type=int, default=44, help="Seed for few-shot prompt generation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # 可选参数，用于决定是否保存权重L2
    parser.add_argument("--save_weight_l2", action='store_true', 
                        help="Whether to compute and save weight L2 norms once.")

    args = parser.parse_args()
    main(args)
