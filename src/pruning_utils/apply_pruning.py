import torch
import torch.nn as nn
from typing import Dict, Optional, Union

def compress_layer(
    layer: nn.Module,
    attn_mask: Optional[torch.Tensor] = None,
    mlp_mask: Optional[torch.Tensor] = None,
    unstr: bool = False,
    head_dim: int = 128,
):
    """
    对单个 Transformer 层进行剪枝或掩码处理（支持多卡分布）。

    Args:
        layer (nn.Module): 单个 Transformer Layer，需要有 self_attn 和 mlp。
        attn_mask (Optional[torch.Tensor]): Attention head的mask，[num_heads]。
        mlp_mask (Optional[torch.Tensor]): MLP hidden单元的mask，[intermediate_size]。
        unstr (bool): 是否使用 unstructured masking（即软掩码，不改变维度）。
        head_dim (int): 每个 attention head 的特征维度，默认128。

    注：
        - 如果 unstr=True，只对 weight做掩码乘0，不改参数shape；
        - 如果 unstr=False，会硬删参数，改shape。
        - 自动识别当前 layer 所在 device，支持多GPU模型分布。
    """
    # 自动根据当前layer拿device
    device = next(layer.parameters()).device

    # ---------------------------------------------
    # A) Attention Heads 部分剪枝
    # ---------------------------------------------
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)
        expanded_attn_mask = attn_mask.repeat_interleave(head_dim)  # 展开成 [hidden_size]

        q_proj, k_proj, v_proj, o_proj = layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.self_attn.o_proj

        if unstr:
            # soft masking：权重乘掩码，但不改变shape
            for proj in [q_proj, k_proj, v_proj]:
                proj.weight.data *= expanded_attn_mask.unsqueeze(-1)
        else:
            # hard pruning：真的删掉无关的head
            keep_indices = torch.where(expanded_attn_mask > 0)[0]
            for proj in [q_proj, k_proj, v_proj]:
                proj.weight.data = proj.weight.data[keep_indices]
                proj.out_features = keep_indices.size(0)

            o_proj.weight.data = o_proj.weight.data[:, keep_indices]
            o_proj.in_features = keep_indices.size(0)

            # 更新模型结构信息
            layer.self_attn.num_heads = int(attn_mask.sum().item())
            layer.self_attn.hidden_size = layer.self_attn.num_heads * head_dim

    # ---------------------------------------------
    # B) MLP Channels 部分剪枝
    # ---------------------------------------------
    if mlp_mask is not None:
        mlp_mask = mlp_mask.to(device)

        up_proj, gate_proj, down_proj = layer.mlp.up_proj, layer.mlp.gate_proj, layer.mlp.down_proj

        if unstr:
            # soft masking
            up_proj.weight.data *= mlp_mask.unsqueeze(-1)
            gate_proj.weight.data *= mlp_mask.unsqueeze(-1)
        else:
            # hard pruning
            keep_indices = torch.where(mlp_mask > 0)[0]
            up_proj.weight.data = up_proj.weight.data[keep_indices]
            gate_proj.weight.data = gate_proj.weight.data[keep_indices]

            up_proj.out_features = keep_indices.size(0)
            gate_proj.out_features = keep_indices.size(0)

            down_proj.weight.data = down_proj.weight.data[:, keep_indices]
            down_proj.in_features = keep_indices.size(0)

            layer.mlp.intermediate_size = keep_indices.size(0)

    torch.cuda.empty_cache()


def apply_pruning_to_model(
    model: nn.Module,
    attn_masks: Dict[int, torch.Tensor],
    mlp_masks: Dict[int, torch.Tensor],
    unstr: bool = True,
    head_dim: int = 128,
):
    """
    给整个 Transformer 模型（多层）应用剪枝或掩码。

    Args:
        model (nn.Module): Transformer 模型，要求 model.model.layers 能拿到所有层。
        attn_masks (Dict[int, torch.Tensor]): 每层对应的 attention mask。
        mlp_masks (Dict[int, torch.Tensor]): 每层对应的 mlp hidden mask。
        unstr (bool): 是否用 unstructured masking（默认为软剪枝）。
        head_dim (int): 每个 attention head 的特征维度，默认128。
    """
    layers = model.model.layers

    for layer_idx, layer in enumerate(layers):
        a_mask = attn_masks.get(layer_idx, None)
        m_mask = mlp_masks.get(layer_idx, None)

        compress_layer(
            layer=layer,
            attn_mask=a_mask,
            mlp_mask=m_mask,
            unstr=unstr,
            head_dim=head_dim,
        )

    torch.cuda.empty_cache()
