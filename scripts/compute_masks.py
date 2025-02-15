#!/usr/bin/env python3
# coding: utf-8
# scripts/compute_masks.py

import sys
import os
import argparse
import torch
import json
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# 引入自定义模块
from src.activations import load_activations, load_weight_l2_info
from src.model_utils import load_model_and_tokenizer
from src.pruning_utils.compute_scores import compute_all_layers_scores
from src.pruning_utils.generate_masks import (
    generate_masks_for_all_layers,
    save_masks_to_file,
    compute_layerwise_sparsity
)

def compute_weighted_scores(
    scores_dicts: dict,
    wikitext2_weight: int = 3,
    task_weight: int = 2
) -> dict:
    """
    计算加权得分，用于生成通用的剪枝掩码。
    
    Args:
        scores_dicts: 任务及其对应分数的字典。每个任务包含层级分数（attn_scores 和 mlp_scores）。
        wikitext2_weight: wikitext2 数据集的权重。
        task_weight: 其他任务的权重。
    
    Returns:
        加权得分字典，每层的加权分数。
    """
    weighted_scores = {}

    # 计算wikitext2的加权得分
    for layer_idx in scores_dicts.get('wikitext2', {}).keys():
        weighted_scores[layer_idx] = {
            'attn_scores': scores_dicts['wikitext2'][layer_idx]['attn_scores'] * wikitext2_weight,
            'mlp_scores': scores_dicts['wikitext2'][layer_idx]['mlp_scores'] * wikitext2_weight
        }

    # 计算其他任务的加权得分
    for task_type, task_scores in scores_dicts.items():
        if task_type != 'wikitext2':
            for layer_idx, layer_scores in task_scores.items():
                if layer_idx not in weighted_scores:
                    weighted_scores[layer_idx] = {
                        'attn_scores': layer_scores['attn_scores'] * task_weight,
                        'mlp_scores': layer_scores['mlp_scores'] * task_weight
                    }
                else:
                    weighted_scores[layer_idx]['attn_scores'] += layer_scores['attn_scores'] * task_weight
                    weighted_scores[layer_idx]['mlp_scores'] += layer_scores['mlp_scores'] * task_weight

    return weighted_scores

def save_pruning_metadata(output_dir, task_type, method, structure, pruning_ratio, sparsities, mask_file):
    """
    保存剪枝策略的元数据（如稀疏度、模型配置等），以便后续调用。
    
    Args:
        output_dir (str): 保存路径。
        task_type (str): 任务类型。
        method (str): 评分方法。
        structure (str): 剪枝结构。
        pruning_ratio (float): 剪枝比例。
        sparsities (dict): 每层的稀疏度信息。
        mask_file (str): 保存的掩码文件路径。
    """
    # 构建保存文件夹的路径
    folder_name = f"task={task_type}_method={method}_structure={structure}_ratio={pruning_ratio}"
    task_output_dir = os.path.join(output_dir, folder_name)
    
    # 创建文件夹（如果不存在）    
    os.makedirs(task_output_dir, exist_ok=True)

    # 保存稀疏度信息
    sparsity_file = os.path.join(task_output_dir, "sparsity.json")
    with open(sparsity_file, 'w') as f:
        json.dump(sparsities, f, indent=4)
    
    # 保存其他信息，比如计算方法、剪枝比例等
    metadata = {
        "task_type": task_type,
        "method": method,
        "structure": structure,
        "pruning_ratio": pruning_ratio,
        "mask_file": mask_file,
    }
    metadata_file = os.path.join(task_output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"[save_pruning_metadata] Metadata saved in {task_output_dir}")

def main(args):
    # 1) 设置随机种子（可选）
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 2) 构建模型路径并加载模型
    model_path = os.path.join(args.model_root_path, args.model_name)
    print(f"[compute_masks] Loading model from {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval().to("cuda")

    # 3) 加载激活数据和权重L2
    activations_dir = os.path.join(args.activations_root_path, args.model_name)
    print(f"[compute_masks] Loading activations from {activations_dir}")
    all_task_activations = load_activations(activations_dir)

    weight_l2_file = os.path.join(activations_dir, "weight_l2_info.pt")
    print(f"[compute_masks] Loading weight L2 from {weight_l2_file}")
    weight_l2_data = load_weight_l2_info(weight_l2_file)

    # 4) 从模型中获取关键信息
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    intermediate_size = model.config.intermediate_size

    del model
    torch.cuda.empty_cache()

    print(f"[compute_masks] Model config: num_layers={num_layers}, hidden_size={hidden_size}, "
          f"num_heads={num_heads}, intermediate_size={intermediate_size}")
    
    # 5) 拿到所有任务(含 wikitext2)
    task_list = sorted(all_task_activations.keys())
    print(f"[compute_masks] Found tasks in activation data: {task_list}")

    # 创建包含模型名称的文件夹路径
    model_output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # 6) 针对all_task_activations中的每个任务类型，循环生成并保存掩码
    scores_dicts = {}

    for task_type in task_list:
        print(f"\n[compute_masks] Processing task={task_type} ...")
        activation_data = all_task_activations[task_type]
        
        # 计算任务的分数
        scores_dict_task = compute_all_layers_scores(
            activation_data=activation_data,
            weight_data=weight_l2_data,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            method=args.method
        )
        
        scores_dicts[task_type] = scores_dict_task

        # 生成掩码
        attn_masks, mlp_masks = generate_masks_for_all_layers(
            scores_dict_task,
            structure=args.structure,
            pruning_ratio=args.pruning_ratio,       # 这里表示目标平均稀疏度
            hidden_size=hidden_size, # AL-AM 需要
            num_heads=num_heads, # AL-AM 需要
            total_layers=num_layers, # UL-LD 需要
            logistic_k=args.logistic_k, # UL-LD 需要
            logistic_x0=args.logistic_x0, # UL-LD 需要
        )

        sparsities = compute_layerwise_sparsity(attn_masks, mlp_masks)
        for layer_idx, data in sparsities.items():
            print(f"Layer {layer_idx}: attn_sparsity={data['attn_sparsity']:.3f}, "
                  f"mlp_sparsity={data['mlp_sparsity']:.3f}")

        # 保存其他任务的掩码
        task_output_dir = os.path.join(
            model_output_dir, 
            f"task={task_type}_method={args.method}_structure={args.structure}_ratio={args.pruning_ratio}"
        )
        os.makedirs(task_output_dir, exist_ok=True)

        mask_file = os.path.join(task_output_dir, f"{task_type}_masks.pt")
        save_masks_to_file(attn_masks, mlp_masks, mask_file)
        print(f"[compute_masks] Saved masks for task={task_type} to {mask_file}")

        # 保存元数据
        save_pruning_metadata(
            output_dir=model_output_dir,
            task_type=task_type,
            method=args.method,
            structure=args.structure,
            pruning_ratio=args.pruning_ratio,
            sparsities=sparsities,
            mask_file=mask_file
        )

    # 7) 如果use_generic_mask为True，计算并保存加权通用掩码
    if args.use_generic_mask:
        print(f"\n[compute_masks] Using generic mask for evaluation.")
        
        weighted_scores = compute_weighted_scores(scores_dicts)
        
        # 生成通用掩码
        attn_masks, mlp_masks = generate_masks_for_all_layers(
            weighted_scores,
            structure=args.structure,
            pruning_ratio=args.pruning_ratio,  # 这里表示目标平均稀疏度
            hidden_size=hidden_size,
            num_heads=num_heads,
            total_layers=num_layers,
            logistic_k=args.logistic_k,
            logistic_x0=args.logistic_x0,
        )

        sparsities = compute_layerwise_sparsity(attn_masks, mlp_masks)
        for layer_idx, data in sparsities.items():
            print(f"Layer {layer_idx}: attn_sparsity={data['attn_sparsity']:.3f}, "
                  f"mlp_sparsity={data['mlp_sparsity']:.3f}")

        # 保存通用掩码
        generic_mask_output_dir = os.path.join(
            model_output_dir, 
            f"task=generic_method={args.method}_structure={args.structure}_ratio={args.pruning_ratio}"
        )
        os.makedirs(generic_mask_output_dir, exist_ok=True)

        mask_file = os.path.join(generic_mask_output_dir, "generic_masks.pt")
        save_masks_to_file(attn_masks, mlp_masks, mask_file)
        print(f"[compute_masks] Saved generic masks to {mask_file}")

        save_pruning_metadata(
            output_dir=model_output_dir,
            task_type="generic",
            method=args.method,
            structure=args.structure,
            pruning_ratio=args.pruning_ratio,
            sparsities=sparsities,
            mask_file=mask_file
        )

    print("[compute_masks] All tasks processed successfully.")
    print(f"[compute_masks] Masks saved in: {model_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute pruning masks from activations and weight L2 for multiple tasks.")
    parser.add_argument("--model_root_path", type=str, required=True, 
                        help="Path to root dir containing the model.")
    parser.add_argument("--model_name", type=str, required=True, 
                        help="Model name.")
    parser.add_argument("--activations_root_path", type=str, default="./activations",
                        help="Path to activations root dir (which has <model_name>/<task>/activations.pt).")
    parser.add_argument("--output_dir", type=str, default="./pruning_masks",
                        help="Where to save the generated masks.")
    # 剪枝策略相关
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--method", type=str, default="WIFV", 
                        choices=["WIFV", "WIFN"],
                        help="Scoring method for compute_scores. Default: WIFV")
    parser.add_argument("--structure", type=str, default="UL-LD",
                        help="Structure strategy like UL-UM, AL-AM, UL-LD. Default: UL-LD")
    parser.add_argument("--pruning_ratio", type=float, default=0.1,
                        help="Fraction of heads/channels to prune. Default=0.1")
    parser.add_argument("--logistic_k", type=float, default=0.8,
                        help="(UL-LD) Logistic function steepness parameter k.")
    parser.add_argument("--logistic_x0", type=float, default=0.3,
                        help="(UL-LD) Logistic function midpoint parameter x0.")
    parser.add_argument("--use_generic_mask", action="store_true", 
                        help="If set, generate a generic mask based on weighted scores across all tasks.")

    args = parser.parse_args()
    main(args)
