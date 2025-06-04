"""
Pruning application utilities for transformer models.

This module provides functions for applying pruning masks to transformer models,
supporting both structured and unstructured pruning of attention heads and MLP units.

Author: why
Date: 2024
"""

# Standard library imports
from typing import Dict, Optional, Union

# Third-party imports
import torch
import torch.nn as nn

def compress_layer(
    layer: nn.Module,
    attn_mask: Optional[torch.Tensor] = None,
    mlp_mask: Optional[torch.Tensor] = None,
    unstr: bool = False,
    head_dim: int = 128,
) -> None:
    """Apply pruning or masking to a single transformer layer.
    
    Supports distributed model across multiple GPUs. Can perform either
    structured pruning (actually removing parameters) or unstructured
    masking (zeroing weights without changing dimensions).
    
    Args:
        layer: Single transformer layer with self_attn and mlp components
        attn_mask: Mask for attention heads, shape [num_heads]
        mlp_mask: Mask for MLP hidden units, shape [intermediate_size]
        unstr: Whether to use unstructured masking (soft masks without dimension change)
        head_dim: Feature dimension of each attention head
        
    Notes:
        - If unstr=True, weights are masked by multiplication with 0s
        - If unstr=False, parameters are actually removed, changing shapes
        - Automatically detects device of current layer for multi-GPU support
    """
    # Get device from current layer
    device = next(layer.parameters()).device

    # ---------------------------------------------
    # A) Prune Attention Heads
    # ---------------------------------------------
    if attn_mask is not None:
        attn_mask = attn_mask.to(device)
        expanded_attn_mask = attn_mask.repeat_interleave(head_dim)  # Expand to [hidden_size]

        q_proj = layer.self_attn.q_proj
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
        o_proj = layer.self_attn.o_proj

        if unstr:
            # Soft masking: multiply weights by mask without changing shape
            for proj in [q_proj, k_proj, v_proj]:
                proj.weight.data *= expanded_attn_mask.unsqueeze(-1)
        else:
            # Hard pruning: actually remove irrelevant heads
            keep_indices = torch.where(expanded_attn_mask > 0)[0]
            for proj in [q_proj, k_proj, v_proj]:
                proj.weight.data = proj.weight.data[keep_indices]
                proj.out_features = keep_indices.size(0)

            o_proj.weight.data = o_proj.weight.data[:, keep_indices]
            o_proj.in_features = keep_indices.size(0)

            # Update model structure information
            layer.self_attn.num_heads = int(attn_mask.sum().item())
            layer.self_attn.hidden_size = layer.self_attn.num_heads * head_dim

    # ---------------------------------------------
    # B) Prune MLP Channels
    # ---------------------------------------------
    if mlp_mask is not None:
        mlp_mask = mlp_mask.to(device)

        up_proj = layer.mlp.up_proj
        gate_proj = layer.mlp.gate_proj
        down_proj = layer.mlp.down_proj

        if unstr:
            # Soft masking
            up_proj.weight.data *= mlp_mask.unsqueeze(-1)
            gate_proj.weight.data *= mlp_mask.unsqueeze(-1)
        else:
            # Hard pruning
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
) -> None:
    """Apply pruning or masking to entire transformer model.
    
    Args:
        model: Transformer model with accessible model.model.layers
        attn_masks: Dictionary mapping layer indices to attention masks
        mlp_masks: Dictionary mapping layer indices to MLP hidden masks
        unstr: Whether to use unstructured masking (default is soft pruning)
        head_dim: Feature dimension of each attention head
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
