"""
/src/pruning_utils/compute_scores.py

该文件主要用于从已保存的激活数据 (activation_data) 和权重统计信息 (weight_data) 中，
计算用于结构化剪枝的“重要性分数”(pruning scores)，
只保留 FLAP 中常见的 WIFV 和 WIFN 两种方法。

- WIFV: Weighted Input Feature Variance
- WIFN: Weighted Input Feature Norm

思路:
  对Attention:
    - WIFV = \sum_{d=1}^{head_dim} (variance_{head,d} * weight_{head,d})
    - WIFN = mean_{d=1..head_dim}( sqrt(variance_{head,d}) * weight_{head,d} )
  对MLP:
    - WIFV = variance[channel] * weight[channel]
    - WIFN = sqrt(variance[channel]) * weight[channel]
    
返回 [num_heads] 或 [intermediate_size] 的聚合结果.
"""

import torch
from typing import Dict, Any

# 只保留这两种方法
SUPPORTED_METHODS = ["WIFV", "WIFN"]

def compute_attention_head_scores(
    layer_idx: int,
    activation_info: Dict[str, Any],
    weight_info: Dict[int, Dict[str, torch.Tensor]],
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    method: str = "WIFV"
) -> torch.Tensor:
    """
    计算单层注意力 heads 的重要性分数（head-level），支持 WIFV、WIFN。
    与 prune_flap 保持一致的逻辑:
      - 先拿到 [hidden_size] 的激活统计 (var 或 l2/sqrt(l2))
      - 再和 [hidden_size] 的权重统计做元素相乘，仍是 [hidden_size]
      - 最后 reshape -> [num_heads, head_dim] 并根据方法 (WIFV/WIFN) 做 sum/mean 等聚合
      - 返回 shape=[num_heads] 的最终分数
    """

    # 1) 从激活信息中获取 var 或 l2
    post_agg_dict = activation_info["attention_post_aggregation"]
    var_activations = post_agg_dict.get("var", None)
    if var_activations is None:
        raise ValueError(f"[compute_attention_head_scores] 'var' not found for layer={layer_idx}, method={method}")

    if var_activations.shape[0] != hidden_size:
        raise ValueError(
            f"Layer {layer_idx} shape mismatch: expected {hidden_size}, got {var_activations.shape[0]}"
        )
    
    # 若同时存在 l2
    l2_activations = post_agg_dict.get("l2", None)
    if l2_activations is not None and l2_activations.shape[0] != hidden_size:
        raise ValueError(
            f"Layer {layer_idx} l2 shape mismatch: expected {hidden_size}, got {l2_activations.shape[0]}"
        )

    # 2) 读取权重信息 [hidden_size]
    w_layer_dict = weight_info.get(layer_idx, {})
    w_o_proj = w_layer_dict.get('o_proj', None)
    if w_o_proj is None:
        raise ValueError(f"Method {method} requires weight_info[{layer_idx}]['o_proj'] to exist")

    if w_o_proj.shape[0] != hidden_size:
        raise ValueError(
            f"o_proj shape mismatch: expected {hidden_size}, got {w_o_proj.shape[0]}"
        )

    # 3) 根据 WIFV / WIFN 计算, 得到 shape=[hidden_size]
    #    (先在 hidden_size 维上做 element-wise，再最后 reshape)
    if method == "WIFV":
        # WIFV = var * weight (element-wise)
        raw_scores = var_activations * w_o_proj
    elif method == "WIFN":
        # WIFN = sqrt(var) * weight, 如果有 l2 则 sqrt(l2)
        if l2_activations is not None:
            raw_scores = torch.sqrt(l2_activations) * w_o_proj
        else:
            raw_scores = torch.sqrt(var_activations) * w_o_proj
    else:
        raise NotImplementedError(f"Only WIFV/WIFN are supported, got {method}")

    # 4) 现在 raw_scores.shape=[hidden_size], reshape-> [num_heads, head_dim]
    raw_scores_2d = raw_scores.view(num_heads, head_dim)

    scores = raw_scores_2d.mean(dim=1)

    return scores

def compute_mlp_channel_scores(
    layer_idx: int,
    activation_info: Dict[str, Any],
    weight_info: Dict[int, Dict[str, torch.Tensor]],
    hidden_size: int,
    intermediate_size: int,
    method: str = "WIFV"
) -> torch.Tensor:
    """
    计算单层 MLP 通道 (4*hidden_size) 的重要性分数 (WIFV / WIFN).
    对应 down_proj 的输入, 即 "mlp_intermediate_states" 字段.

    流程:
      1) 读取 var / l2
      2) 读取 down_proj 的统计
      3) 做 elementwise 乘并得到 final scores: shape=[intermediate_size]

    Args:
        layer_idx: 当前层索引
        activation_info: e.g. activation_info[layer_idx]["mlp_intermediate_states"] = {
            "var": [...], "l2": [...]
        }
        weight_info: weight_info[layer_idx]["down_proj"] (e.g. shape=[intermediate_size])
        method: "WIFV" or "WIFN"

    Returns:
        scores: shape=[intermediate_size], 每个 channel 的分数
    """
    mlp_dict = activation_info["mlp_intermediate_states"]
    var_activations = mlp_dict.get("var", None)
    if var_activations is None:
        raise ValueError(f"[compute_mlp_channel_scores] 'var' missing for layer={layer_idx}, method={method}")

    # 防止数据维度>intermediate_size
    if var_activations.shape[0] != intermediate_size:
        var_activations = var_activations[:intermediate_size]

    l2_activations = mlp_dict.get("l2", None)
    if l2_activations is not None:
        l2_activations = l2_activations[:intermediate_size]

    # 读取 down_proj 的统计
    w_layer_dict = weight_info.get(layer_idx, {})
    w_down_proj = w_layer_dict.get('down_proj', None)
    if w_down_proj is None:
        raise ValueError(f"Method {method} requires weight_info[{layer_idx}]['down_proj'] to exist")

    if w_down_proj.shape[0] != intermediate_size:
        w_down_proj = w_down_proj[:intermediate_size]

    # 4) 依据 WIFV / WIFN 计算
    if method == "WIFV":
        # Weighted Input Feature Variance
        scores = var_activations * w_down_proj

    elif method == "WIFN":
        # Weighted Input Feature Norm
        if l2_activations is not None:
            scores = torch.sqrt(l2_activations) * w_down_proj
        else:
            scores = torch.sqrt(var_activations) * w_down_proj
    else:
        raise NotImplementedError(f"Only WIFV/WIFN are supported, got {method}")

    return scores

def compute_layer_scores(
    layer_idx: int,
    activation_data: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    weight_data: Dict[int, Dict[str, torch.Tensor]],
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    intermediate_size: int,
    method: str = "WIFV"
):
    """
    计算单层 (注意力 heads 的分数, MLP 通道分数) 并返回 (attn_scores, mlp_scores).

    在本实现里:
      attn_scores = shape [num_heads]
      mlp_scores  = shape [intermediate_size]

    后续可以直接对 attn_scores 做排序阈值 (如 UL-UM, AL-AM)
    也可对 mlp_scores 做同样处理.
    """
    if layer_idx not in activation_data:
        raise ValueError(f"Missing activation data for layer {layer_idx}")

    layer_info = activation_data[layer_idx]

    # 1) 注意力 heads
    attn_scores = compute_attention_head_scores(
        layer_idx=layer_idx,
        activation_info=layer_info,
        weight_info=weight_data,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        method=method
    )

    # 2) MLP channels
    mlp_scores = compute_mlp_channel_scores(
        layer_idx=layer_idx,
        activation_info=layer_info,
        weight_info=weight_data,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        method=method
    )

    return attn_scores, mlp_scores

def compute_all_layers_scores(
    activation_data: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    weight_data: Dict[int, Dict[str, torch.Tensor]],
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    intermediate_size: int,
    method: str = "WIFV"
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    针对整个模型(多层) 计算注意力与 MLP 的分数, 返回:
      scores_dict[layer_idx] = {
          "attn_scores": shape=[num_heads],
          "mlp_scores":  shape=[intermediate_size]
      }

    仅支持 "WIFV" / "WIFN" 两种方法:
      - WIFV: Weighted Input Feature Variance
      - WIFN: Weighted Input Feature Norm
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method={method} not in supported list: {SUPPORTED_METHODS}")

    head_dim = hidden_size // num_heads
    scores_dict = {}

    for layer_idx in range(num_layers):
        if layer_idx not in activation_data:
            raise ValueError(f"Missing activation data for layer={layer_idx}")

        attn_scores, mlp_scores = compute_layer_scores(
            layer_idx=layer_idx,
            activation_data=activation_data,
            weight_data=weight_data,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            method=method
        )
        scores_dict[layer_idx] = {
            "attn_scores": attn_scores,
            "mlp_scores":  mlp_scores
        }

    return scores_dict
