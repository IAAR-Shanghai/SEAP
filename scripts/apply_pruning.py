#!/usr/bin/env python3
# coding: utf-8
# scripts/apply_pruning.py

import sys
import os
import torch
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.model_utils import load_model_and_tokenizer
from src.pruning_utils.generate_masks import load_masks_from_file
from src.pruning_utils.apply_pruning import apply_pruning_to_model
from src.activations import load_activations

def main(args):
    # 1) 加载模型
    model_path = os.path.join(args.model_root_path, args.model_name)
    print(f"[apply_pruning] Loading model from {model_path}")
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.eval().to("cuda")

    # 2) 自动计算 head_dim
    #    (不再从命令行传入，而是统一根据 hidden_size / num_attention_heads)
    if hasattr(model.config, "hidden_size") and hasattr(model.config, "num_attention_heads"):
        hidden_size = model.config.hidden_size
        num_heads = model.config.num_attention_heads
        if hidden_size % num_heads != 0:
            print(f"Warning: hidden_size={hidden_size} not divisible by num_heads={num_heads}; "
                  "using integer division for head_dim.")
        head_dim = hidden_size // num_heads
    else:
        # 如果模型缺少相关信息，可以报错或给定默认值
        raise ValueError("[apply_pruning] Model config missing hidden_size/num_attention_heads, "
                         "cannot compute head_dim automatically.")
    print(f"[apply_pruning] Computed head_dim={head_dim} from hidden_size={hidden_size} / num_heads={num_heads}.")

    # 3) 如果需要 bias 补偿 => 从激活文件中加载 baseline_inp (注意力+MLP)
    attn_mean_inps_dict = {}
    mlp_mean_inps_dict = {}
    if args.biascomp:
        if args.activations_root_path is not None:
            act_dir = os.path.join(args.activations_root_path, args.model_name)
            print(f"[apply_pruning] Bias compensation is ON. Loading activations from {act_dir}")
            if not os.path.isdir(act_dir):
                print(f"Warning: The directory {act_dir} doesn't exist. Skip baseline_inp.")
            else:
                # 读取所有任务激活：结构形如 {task_type: {layer_idx: {...}}}
                all_task_activations = load_activations(act_dir)
                # 这里从 each_task_activations[task_type][layer_idx]
                # 中提取 "attention_post_aggregation"]["mean"] 作为 attn_mean_inp
                # 以及 "mlp_intermediate_states"]["mean"] 作为 mlp_mean_inp
                for ttype in args.task_types:
                    if ttype not in all_task_activations:
                        print(f"[apply_pruning] No activations for task={ttype}. Skip baseline.")
                        continue
                    attn_mean_inps_dict[ttype] = {}
                    mlp_mean_inps_dict[ttype] = {}
                    layer_dict = all_task_activations[ttype]  # => {layer_idx: {...}}
                    for layer_idx, layer_info in layer_dict.items():
                        # 取 attention_post_aggregation["mean"] => shape=[hidden_size]
                        if ("attention_post_aggregation" in layer_info and
                            "mean" in layer_info["attention_post_aggregation"]):
                            attn_mean = layer_info["attention_post_aggregation"]["mean"]
                        else:
                            attn_mean = None

                        # 取 mlp_intermediate_states["mean"] => shape=[intermediate_size]
                        if ("mlp_intermediate_states" in layer_info and
                            "mean" in layer_info["mlp_intermediate_states"]):
                            mlp_mean = layer_info["mlp_intermediate_states"]["mean"]
                        else:
                            mlp_mean = None

                        attn_mean_inps_dict[ttype][layer_idx] = attn_mean
                        mlp_mean_inps_dict[ttype][layer_idx]  = mlp_mean

        else:
            print("[apply_pruning] --biascomp is True but no --activations_root_path provided. "
                  "Skipping baseline_inp for all tasks.")
    else:
        print("[apply_pruning] Bias compensation is OFF.")

    # 4) 遍历多个任务
    print(f"[apply_pruning] Tasks to process: {args.task_types}")
    for idx, task_type in enumerate(args.task_types):
        print(f"\n[apply_pruning] === Processing task={task_type} ===")

        # (A) 构造mask文件路径
        task_mask_dir = os.path.join(args.masks_root_dir, args.model_name, f"task={task_type}_method={args.method}_structure={args.structure}_ratio={args.pruning_ratio}")
        mask_file = os.path.join(task_mask_dir, f"{task_type}_masks.pt")
        if not os.path.exists(mask_file):
            print(f"[apply_pruning] Mask file not found for task={task_type}: {mask_file}. Skip.")
            continue

        # (B) 加载mask
        print(f"[apply_pruning] Loading masks from {mask_file}")
        attn_masks, mlp_masks = load_masks_from_file(mask_file)

        # (C) 获取对应 baseline_inp
        attn_mean_inps = None
        mlp_mean_inps  = None
        if args.biascomp:
            if task_type in attn_mean_inps_dict:
                attn_mean_inps = attn_mean_inps_dict[task_type]
                mlp_mean_inps  = mlp_mean_inps_dict[task_type]
            else:
                print(f"[apply_pruning] No baseline data for task={task_type}. No bias compensation.")

        # (D) 调用 apply_pruning_to_model
        print(f"[apply_pruning] Pruning with task={task_type} ... softmask={args.softmask}, bias={args.biascomp}")
        apply_pruning_to_model(
            model=model,
            attn_masks=attn_masks,
            mlp_masks=mlp_masks,
            attn_mean_inps=attn_mean_inps,   # {layer_idx->Tensor([hidden_size])}
            mlp_mean_inps=mlp_mean_inps,     # {layer_idx->Tensor([intermediate_size])}
            device="cuda",
            bias=args.biascomp,
            unstr=args.softmask,
            head_dim=head_dim
        )
        print(f"[apply_pruning] Done applying masks for task={task_type}.")

        # (E) 保存剪枝后模型
        os.makedirs(args.output_dir, exist_ok=True)
        save_path = os.path.join(args.output_dir, f"{args.model_name}_pruned_{task_type}_method={args.method}_ratio={args.pruning_ratio}")
        os.makedirs(save_path, exist_ok=True)
        print(f"[apply_pruning] Saving pruned model (task={task_type}) to {save_path}")

        if args.softmask:
            # 如果是 softmask，则继续使用原有的保存逻辑
            model.save_pretrained(save_path)
        else:
            # 如果不是 softmask，则使用 torch.save 保存模型
            torch.save(model, f"{save_path}/pruned_model.pt")

        # 保存 tokenizer
        tokenizer.save_pretrained(save_path)

        # (F) 清理内存，重新加载模型
        if len(args.task_types) > 1 and idx < len(args.task_types) - 1:
            print("[apply_pruning] Reloading original model for next task, to avoid cumulative pruning.")
            del model
            torch.cuda.empty_cache()
            model, tokenizer = load_model_and_tokenizer(model_path)
            model.eval().to("cuda")

    print("\n[apply_pruning] Done. All tasks processed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load pruning masks for multiple tasks and apply them to the model with optional bias compensation."
    )
    parser.add_argument("--model_root_path", type=str, required=True,
                        help="Path to root dir containing the model.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name (folder name under --model_root_path).")

    parser.add_argument("--masks_root_dir", type=str, default="./pruning_masks",
                        help="Directory where the <task>_masks.pt are located.")
    parser.add_argument("--task_types", nargs='+', default=["gsm8k"],
                        help="List of task names to process, e.g.: gsm8k hellaswag mmlu")
    parser.add_argument("--output_dir", type=str, default="./pruned_models",
                        help="Where to save each pruned model (for each task).")
    
    parser.add_argument("--method", type=str, default="WIFV", 
                        choices=["WIFV", "WIFN"],
                        help="Scoring method used for pruning. Default is 'WIFV'.")

    parser.add_argument("--structure", type=str, default="UL-LD", 
                        choices=["UL-UM", "AL-AM","UL-LD"],
                        help="Pruning structure strategy. Default is 'UL-LD'.")
    
    parser.add_argument("--biascomp", action="store_true",
                        help="Enable baseline input based bias compensation if baseline_inp is available.")
    parser.add_argument("--softmask", action="store_true",
                        help="If set, only mask weights to zero (unstructured), else do structured pruning.")

    parser.add_argument("--pruning_ratio", type=float, default=0.1,
                        help="Fraction of heads/channels to prune. Default=0.1")

    # 用于从激活中加载 baseline_inp
    parser.add_argument("--activations_root_path", type=str, default=None,
                        help="Where to load layer-wise baseline_inp from if bias compensation is needed.")

    args = parser.parse_args()
    main(args)
