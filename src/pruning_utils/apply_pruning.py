#!/usr/bin/env python3
# coding: utf-8
"""
/src/pruning_utils/apply_pruning.py

此文件中，我们主要完成：
1. compress_layer: 针对单层，根据 attn_mask / mlp_mask 做剪枝或mask；
2. apply_pruning_to_model: 针对全模型多层，逐层调用 compress_layer；并可选结合 baseline_inp 做 bias 补偿。

改进点：在 structured prune 时，如果 bias=True，
我们会先提取 "待移除" 的行/列对应的权重，乘以 baseline_inp 得到对输出的贡献，再加到 bias 上，
然后再正式删除相应行/列。这样可以贴近 FLAP 的原生做法。
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union

def compress_layer(
    layer: nn.Module,
    attn_mask: Optional[torch.Tensor] = None,
    mlp_mask: Optional[torch.Tensor] = None,
    attn_mean_inp: Optional[torch.Tensor] = None,
    mlp_mean_inp: Optional[torch.Tensor] = None,
    device: Union[str, torch.device] = "cuda",
    bias: bool = True,
    unstr: bool = False,
    head_dim: int = 128
):
    """
    Compress (prune or mask) a single Transformer layer based on provided masks.
    Includes robust dtype casting for bias compensation.

    Args:
        layer (nn.Module): The layer to compress, with submodules:
            layer.self_attn.q_proj, .k_proj, .v_proj, .o_proj,
            layer.mlp.up_proj, .down_proj, .gate_proj, etc.
        attn_mask (torch.Tensor): shape=[num_heads], bool or 0/1 mask for attention heads
        mlp_mask (torch.Tensor): shape=[intermediate_size], bool or 0/1 mask for MLP channels
        attn_mean_inp (torch.Tensor): baseline input for attention (shape=[hidden_size])
        mlp_mean_inp (torch.Tensor): baseline input for MLP (shape=[intermediate_size])
        device: device used for computations
        bias: whether to do bias compensation
        unstr: whether to do unstructured (soft) mask or structured pruning
        head_dim: dimension of each attention head (default=128)

    Returns:
        None (in-place modifications)
    """

    # ---------------------------------------------------------------------------
    # A) Handle Attention heads
    # ---------------------------------------------------------------------------
    if attn_mask is not None:
        retain_heads = attn_mask.sum().item()
        expanded_attn_mask = attn_mask.repeat_interleave(head_dim)  # shape=[num_heads * head_dim]

        q_proj = layer.self_attn.q_proj
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj

        # Check if q_proj and k_proj have the same output dimension
        qk_match = q_proj.out_features == k_proj.out_features

        # unstructured = True => soft mask
        if unstr:
            # Only do pruning on q_proj, k_proj, and v_proj if q_proj and k_proj match
            if qk_match:
                # Multiply rows in q/k/v
                q_proj.weight.data *= expanded_attn_mask.unsqueeze(-1).to(device)
                k_proj.weight.data *= expanded_attn_mask.unsqueeze(-1).to(device)
                v_proj.weight.data *= expanded_attn_mask.unsqueeze(-1).to(device)

            # 2) o_proj => multiply columns
            o_weight = o_proj.weight.data  # shape=[out_features, hidden_size]

            if bias and (attn_mean_inp is not None):
                removed_mask = (~expanded_attn_mask.bool()).float().to(device)

                weight_dtype = o_weight.dtype
                attn_mean_inp_ = attn_mean_inp.to(device).to(weight_dtype)
                removed_mask_  = removed_mask.to(weight_dtype)

                output_bias = (attn_mean_inp_ * removed_mask_) @ o_weight.T
                o_proj.bias.data = output_bias

            o_proj.weight.data = o_weight

        else:
            # structured pruning
            keep_indices = torch.where(expanded_attn_mask.to(device) > 0)[0]

            # -- bias compensation (FLAP style) before actually remove --
            if bias and (attn_mean_inp is not None):
                removed_indices = torch.where(expanded_attn_mask.to(device) == 0)[0]
                if removed_indices.numel() > 0:
                    old_o_weight = o_proj.weight.data  # [out_features, hidden_size]

                    # 1) Convert baseline input to same dtype
                    weight_dtype = old_o_weight.dtype
                    attn_mean_inp_ = attn_mean_inp.to(device).to(weight_dtype)
                    removed_inp    = attn_mean_inp_[removed_indices]

                    # partial_removed_w => shape=[out_features, num_removed]
                    partial_removed_w = old_o_weight[:, removed_indices]

                    delta_bias = removed_inp @ partial_removed_w.T
                    o_proj.bias.data += delta_bias

            # Actually remove
            if qk_match:
                # Prune q/k/v if they match
                q_proj.weight.data = q_proj.weight.data[keep_indices]
                k_proj.weight.data = k_proj.weight.data[keep_indices]
                v_proj.weight.data = v_proj.weight.data[keep_indices]

                q_proj.out_features = keep_indices.size(0)
                k_proj.out_features = keep_indices.size(0)
                v_proj.out_features = keep_indices.size(0)

            # 2) o_proj => remove columns
            o_weight = o_proj.weight.data
            o_weight = o_weight[:, keep_indices]
            o_proj.weight.data = o_weight
            o_proj.in_features = keep_indices.size(0)

            # Update self_attn attributes
            layer.self_attn.num_heads = retain_heads
            layer.self_attn.hidden_size = int(retain_heads * head_dim)

    # ---------------------------------------------------------------------------
    # B) Handle MLP channels
    # ---------------------------------------------------------------------------
    if mlp_mask is not None:
        retain_mlp = mlp_mask.sum().item()

        up_proj   = layer.mlp.up_proj
        gate_proj = layer.mlp.gate_proj
        down_proj = layer.mlp.down_proj

        if unstr:
            up_proj.weight.data   *= mlp_mask.unsqueeze(-1).to(device)
            gate_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)

            dw = down_proj.weight.data
            if bias and (mlp_mean_inp is not None):
                removed_mask = (~mlp_mask.bool()).float().to(device)
                weight_dtype = dw.dtype
                mlp_mean_inp_ = mlp_mean_inp.to(device).to(weight_dtype)
                removed_mask_  = removed_mask.to(weight_dtype)

                delta_bias = (mlp_mean_inp_ * removed_mask_) @ dw.T
                down_proj.bias.data = delta_bias

            down_proj.weight.data = dw

        else:
            keep_mlp = torch.where(mlp_mask.to(device) > 0)[0]

            if bias and (mlp_mean_inp is not None):
                removed_mlp = torch.where(mlp_mask.to(device) == 0)[0]
                if removed_mlp.numel() > 0:
                    old_dw = down_proj.weight.data
                    weight_dtype = old_dw.dtype
                    mlp_mean_inp_ = mlp_mean_inp.to(device).to(weight_dtype)
                    removed_inp   = mlp_mean_inp_[removed_mlp]

                    partial_removed_w = old_dw[:, removed_mlp]
                    delta_bias = removed_inp @ partial_removed_w.T
                    down_proj.bias.data += delta_bias

            # Real prune
            up_proj.weight.data = up_proj.weight.data[keep_mlp]
            gate_proj.weight.data = gate_proj.weight.data[keep_mlp]
            up_proj.out_features   = keep_mlp.size(0)
            gate_proj.out_features = keep_mlp.size(0)

            dw = down_proj.weight.data
            dw = dw[:, keep_mlp]
            down_proj.weight.data = dw
            down_proj.in_features = keep_mlp.size(0)

            layer.mlp.intermediate_size = keep_mlp.size(0)

    torch.cuda.empty_cache()



def force_add_bias_if_none(linear_module: nn.Linear, device=None):
    """
    如果 linear_module.bias is None，就给它添加一个新的 bias Parameter。
    """
    if linear_module.bias is None:
        out_features = linear_module.weight.size(0)
        weight_dtype = linear_module.weight.dtype
        # 用 0 初始化新的 bias
        new_bias = nn.Parameter(torch.zeros(out_features, dtype=weight_dtype, device=device))
        linear_module.bias = new_bias  # 替换原先 None
        print(f"[force_add_bias_if_none] Added bias param to {linear_module}")


def apply_pruning_to_model(
    model: nn.Module,
    attn_masks: Dict[int, torch.Tensor],
    mlp_masks: Dict[int, torch.Tensor],
    attn_mean_inps: Optional[Dict[int, torch.Tensor]] = None,
    mlp_mean_inps: Optional[Dict[int, torch.Tensor]] = None,
    device: Union[str, torch.device] = "cuda",
    bias: bool = True,
    unstr: bool = False,
    head_dim: int = 128
):
    """
    对模型的所有层应用剪枝或mask操作。若需bias补偿，则传入 baseline_inps.

    Args:
        model: 大模型，如 LLaMA / GPT
        attn_masks: {layer_idx: Tensor([num_heads])}, bool
        mlp_masks:  {layer_idx: Tensor([intermediate_size])}, bool
        attn_mean_inps: {layer_idx: Tensor([hidden_size])} or None
        mlp_mean_inps:  {layer_idx: Tensor([intermediate_size])} or None
        bias: 是否做bias补偿
        unstr: 是否只做soft mask
        head_dim: 每个head多少维

    Returns:
        None, in-place修改
    """

    for layer_idx, layer in enumerate(model.model.layers):
        a_mask = attn_masks.get(layer_idx, None)
        m_mask = mlp_masks.get(layer_idx, None)

        a_mean = attn_mean_inps[layer_idx] if (attn_mean_inps and layer_idx in attn_mean_inps) else None
        m_mean = mlp_mean_inps[layer_idx]  if (mlp_mean_inps and layer_idx in mlp_mean_inps)  else None

        # 如果需要bias补偿，就先检查 o_proj / down_proj 是否有bias，没有则给它加上
        if bias:
            # self_attn.o_proj
            force_add_bias_if_none(layer.self_attn.o_proj, device=device)
            # mlp.down_proj
            force_add_bias_if_none(layer.mlp.down_proj, device=device)

        compress_layer(
            layer=layer,
            attn_mask=a_mask,
            mlp_mask=m_mask,
            attn_mean_inp=a_mean,
            mlp_mean_inp=m_mean,
            device=device,
            bias=bias,
            unstr=unstr,
            head_dim=head_dim
        )

    torch.cuda.empty_cache()