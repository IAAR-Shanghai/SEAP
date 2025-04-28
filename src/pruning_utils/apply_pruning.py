import torch
import torch.nn as nn
from typing import Dict, Optional, Union

def compress_layer(
    layer: nn.Module,
    attn_mask: Optional[torch.Tensor] = None,
    mlp_mask: Optional[torch.Tensor] = None,
    device: Union[str, torch.device] = "cuda",
    unstr: bool = False,
    head_dim: int = 128,
):
    """
    Compress (prune or mask) a single Transformer layer based on provided masks.
    This function prunes attention heads and MLP channels in-place.

    Args:
        layer (nn.Module): Transformer layer with attributes like:
            - self_attn.{q_proj, k_proj, v_proj, o_proj}
            - mlp.{up_proj, gate_proj, down_proj}
        attn_mask (torch.Tensor): Mask for attention heads (shape: [num_heads]).
        mlp_mask (torch.Tensor): Mask for MLP intermediate channels (shape: [intermediate_size]).
        device (Union[str, torch.device]): Device to run pruning on.
        unstr (bool): Whether to apply unstructured (soft) masking.
        head_dim (int): Dimension per attention head.
    """
    # -------------------------------------------------------------
    # A) Attention pruning
    # -------------------------------------------------------------
    if attn_mask is not None:
        expanded_attn_mask = attn_mask.repeat_interleave(head_dim)  # [num_heads * head_dim]
        q_proj = layer.self_attn.q_proj
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj

        qk_match = q_proj.out_features == k_proj.out_features

        if unstr:
            if qk_match:
                q_proj.weight.data *= expanded_attn_mask.unsqueeze(-1).to(device)
                k_proj.weight.data *= expanded_attn_mask.unsqueeze(-1).to(device)
                v_proj.weight.data *= expanded_attn_mask.unsqueeze(-1).to(device)
        else:
            keep_indices = torch.where(expanded_attn_mask.to(device) > 0)[0]

            if qk_match:
                q_proj.weight.data = q_proj.weight.data[keep_indices]
                k_proj.weight.data = k_proj.weight.data[keep_indices]
                v_proj.weight.data = v_proj.weight.data[keep_indices]

                q_proj.out_features = keep_indices.size(0)
                k_proj.out_features = keep_indices.size(0)
                v_proj.out_features = keep_indices.size(0)

            o_proj.weight.data = o_proj.weight.data[:, keep_indices]
            o_proj.in_features = keep_indices.size(0)

        layer.self_attn.num_heads = attn_mask.sum().item()
        layer.self_attn.hidden_size = int(layer.self_attn.num_heads * head_dim)

    # -------------------------------------------------------------
    # B) MLP pruning
    # -------------------------------------------------------------
    if mlp_mask is not None:
        up_proj   = layer.mlp.up_proj
        gate_proj = layer.mlp.gate_proj
        down_proj = layer.mlp.down_proj

        if unstr:
            up_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
            gate_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
        else:
            keep_mlp = torch.where(mlp_mask.to(device) > 0)[0]

            up_proj.weight.data = up_proj.weight.data[keep_mlp]
            gate_proj.weight.data = gate_proj.weight.data[keep_mlp]

            up_proj.out_features = keep_mlp.size(0)
            gate_proj.out_features = keep_mlp.size(0)

            down_proj.weight.data = down_proj.weight.data[:, keep_mlp]
            down_proj.in_features = keep_mlp.size(0)

            layer.mlp.intermediate_size = keep_mlp.size(0)

    torch.cuda.empty_cache()


def apply_pruning_to_model(
    model: nn.Module,
    attn_masks: Dict[int, torch.Tensor],
    mlp_masks: Dict[int, torch.Tensor],
    device: Union[str, torch.device] = "cuda",
    unstr: bool = False,
    head_dim: int = 128
):
    """
    Apply pruning or masking operations to all layers of the model.

    Args:
        model (nn.Module): The model (e.g., LLaMA, GPT) with .model.layers attribute.
        attn_masks (Dict[int, torch.Tensor]): Layer-wise attention masks.
        mlp_masks (Dict[int, torch.Tensor]): Layer-wise MLP masks.
        device (str or torch.device): Device to move weights/masks to.
        unstr (bool): Whether to use unstructured pruning.
        head_dim (int): Per-head hidden dimension.
    """
    for layer_idx, layer in enumerate(model.model.layers):
        a_mask = attn_masks.get(layer_idx, None)
        m_mask = mlp_masks.get(layer_idx, None)

        compress_layer(
            layer=layer,
            attn_mask=a_mask,
            mlp_mask=m_mask,
            device=device,
            unstr=unstr,
            head_dim=head_dim
        )

    torch.cuda.empty_cache()