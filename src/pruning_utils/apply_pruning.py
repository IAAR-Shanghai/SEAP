# /src/pruning_utils/apply_pruning.py

"""
This file primarily implements the following functions:
1. compress_layer: Prune or mask a single layer based on the provided attn_mask / mlp_mask.
2. apply_pruning_to_model: Apply pruning to all layers of the model, calling compress_layer layer by layer. Optionally, bias compensation can be performed using the baseline_inp.

Improvement: During structured pruning, if bias=True, we first extract the weights corresponding to the rows/columns to be removed, multiply them by the baseline_inp to compute their contribution to the output, then add that contribution to the bias. Afterward, we formally delete the corresponding rows/columns. This approach closely follows the original FLAP method.
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
        attn_mask (torch.Tensor): shape=[num_heads], bool or 0/1 mask for attention heads.
        mlp_mask (torch.Tensor): shape=[intermediate_size], bool or 0/1 mask for MLP channels.
        attn_mean_inp (torch.Tensor): baseline input for attention (shape=[hidden_size]).
        mlp_mean_inp (torch.Tensor): baseline input for MLP (shape=[intermediate_size]).
        device (Union[str, torch.device]): device used for computations.
        bias (bool): whether to perform bias compensation.
        unstr (bool): whether to perform unstructured (soft) masking or structured pruning.
        head_dim (int): dimension of each attention head (default=128).

    Returns:
        None: Modifies the layer in place.
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
            # Only prune q_proj, k_proj, and v_proj if q_proj and k_proj match
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
    If the `linear_module.bias` is None, add a new bias parameter initialized to zeros.

    Args:
        linear_module (nn.Linear): The linear layer to check.
        device: The device for the new bias parameter.
    
    Returns:
        None: The function adds the bias in place if necessary.
    """
    if linear_module.bias is None:
        out_features = linear_module.weight.size(0)
        weight_dtype = linear_module.weight.dtype
        # Initialize the new bias to zero
        new_bias = nn.Parameter(torch.zeros(out_features, dtype=weight_dtype, device=device))
        linear_module.bias = new_bias  # Replace the original None with the new bias
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
    Apply pruning or masking operations to all layers of the model. If bias compensation is needed, 
    provide the baseline inputs.

    Args:
        model (nn.Module): The model, such as LLaMA or GPT.
        attn_masks (Dict[int, torch.Tensor]): A dictionary where keys are layer indices, 
            and values are tensors of shape [num_heads] representing the attention mask.
        mlp_masks (Dict[int, torch.Tensor]): A dictionary where keys are layer indices,
            and values are tensors of shape [intermediate_size] representing the MLP mask.
        attn_mean_inps (Optional[Dict[int, torch.Tensor]]): A dictionary where keys are layer indices, 
            and values are tensors representing baseline input for attention.
        mlp_mean_inps (Optional[Dict[int, torch.Tensor]]): A dictionary where keys are layer indices, 
            and values are tensors representing baseline input for MLP.
        bias (bool): Whether to perform bias compensation.
        unstr (bool): Whether to apply only soft masking (unstructured pruning).
        head_dim (int): The dimension of each attention head (default=128).

    Returns:
        None: In-place modifications of the model.
    """
    for layer_idx, layer in enumerate(model.model.layers):
        a_mask = attn_masks.get(layer_idx, None)
        m_mask = mlp_masks.get(layer_idx, None)

        a_mean = attn_mean_inps[layer_idx] if (attn_mean_inps and layer_idx in attn_mean_inps) else None
        m_mean = mlp_mean_inps[layer_idx]  if (mlp_mean_inps and layer_idx in mlp_mean_inps)  else None

        # If bias compensation is required, check and add bias if necessary.
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
