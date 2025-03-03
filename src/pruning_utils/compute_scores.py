# /src/pruning_utils/compute_scores.py

"""
This file is responsible for computing the "pruning scores" used for structured pruning from saved activation data (activation_data) and weight statistics (weight_data).
Only two methods commonly used in FLAP are retained: **WIFV** and **WIFN**.

- **WIFV**: Weighted Input Feature Variance
- **WIFN**: Weighted Input Feature Norm

Approach:
For Attention:
  - WIFV = \sum_{d=1}^{head_dim} (variance_{head,d} * weight_{head,d})
  - WIFN = mean_{d=1..head_dim}( sqrt(variance_{head,d}) * weight_{head,d} )

For MLP:
  - WIFV = variance[channel] * weight[channel]
  - WIFN = sqrt(variance[channel]) * weight[channel]
  
Returns aggregated results of shape [num_heads] or [intermediate_size].
"""

import torch
from typing import Dict, Any

# Only these two methods are supported
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
    Compute importance scores for attention heads in a given layer, supporting both WIFV and WIFN methods.
    Logic:
      - First, get activation statistics (variance or l2/sqrt(l2)) of shape [hidden_size]
      - Then, element-wise multiply with weight statistics of shape [hidden_size]
      - Finally, reshape to [num_heads, head_dim] and aggregate (sum/mean) based on method (WIFV/WIFN)
      - Return final score of shape [num_heads]

    Args:
        layer_idx (int): Layer index
        activation_info (Dict[str, Any]): Activation data for the layer
        weight_info (Dict[int, Dict[str, torch.Tensor]]): Weight statistics for the layer
        hidden_size (int): Hidden size dimension
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each head
        method (str): Method for scoring ("WIFV" or "WIFN")
    
    Returns:
        torch.Tensor: Computed attention head scores of shape [num_heads]
    """

    # 1) Get variance or l2 from the activation information
    post_agg_dict = activation_info["attention_post_aggregation"]
    var_activations = post_agg_dict.get("var", None)
    if var_activations is None:
        raise ValueError(f"[compute_attention_head_scores] 'var' not found for layer={layer_idx}, method={method}")

    if var_activations.shape[0] != hidden_size:
        raise ValueError(
            f"Layer {layer_idx} shape mismatch: expected {hidden_size}, got {var_activations.shape[0]}"
        )
    
    # Check if l2 activations are present
    l2_activations = post_agg_dict.get("l2", None)
    if l2_activations is not None and l2_activations.shape[0] != hidden_size:
        raise ValueError(
            f"Layer {layer_idx} l2 shape mismatch: expected {hidden_size}, got {l2_activations.shape[0]}"
        )

    # 2) Retrieve weight information for the layer
    w_layer_dict = weight_info.get(layer_idx, {})
    w_o_proj = w_layer_dict.get('o_proj', None)
    if w_o_proj is None:
        raise ValueError(f"Method {method} requires weight_info[{layer_idx}]['o_proj'] to exist")

    if w_o_proj.shape[0] != hidden_size:
        raise ValueError(
            f"o_proj shape mismatch: expected {hidden_size}, got {w_o_proj.shape[0]}"
        )

    # 3) Compute scores based on WIFV / WIFN
    if method == "WIFV":
        # WIFV = variance * weight (element-wise multiplication)
        raw_scores = var_activations * w_o_proj
    elif method == "WIFN":
        # WIFN = sqrt(variance) * weight (or sqrt(l2) if l2 is available)
        if l2_activations is not None:
            raw_scores = torch.sqrt(l2_activations) * w_o_proj
        else:
            raw_scores = torch.sqrt(var_activations) * w_o_proj
    else:
        raise NotImplementedError(f"Only WIFV/WIFN are supported, got {method}")

    # 4) Reshape raw scores to [num_heads, head_dim] and calculate mean across head_dim
    raw_scores_2d = raw_scores.view(num_heads, head_dim)
    scores = raw_scores_2d.mean(dim=1)  # Mean aggregation

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
    Compute importance scores for MLP channels (4*hidden_size) in a given layer (WIFV / WIFN).
    This corresponds to the "down_proj" projection in MLP.

    Process:
      1) Retrieve variance or l2
      2) Retrieve statistics for down_proj weights
      3) Perform element-wise multiplication and return final scores of shape [intermediate_size]

    Args:
        layer_idx (int): Current layer index
        activation_info (Dict[str, Any]): Activation data (including var, l2)
        weight_info (Dict[int, Dict[str, torch.Tensor]]): Weight statistics for the layer
        hidden_size (int): Hidden size
        intermediate_size (int): Intermediate size for the MLP
        method (str): "WIFV" or "WIFN"
    
    Returns:
        torch.Tensor: Computed MLP channel scores of shape [intermediate_size]
    """
    mlp_dict = activation_info["mlp_intermediate_states"]
    var_activations = mlp_dict.get("var", None)
    if var_activations is None:
        raise ValueError(f"[compute_mlp_channel_scores] 'var' missing for layer={layer_idx}, method={method}")

    # Handle dimension mismatch by truncating to intermediate_size
    if var_activations.shape[0] != intermediate_size:
        var_activations = var_activations[:intermediate_size]

    l2_activations = mlp_dict.get("l2", None)
    if l2_activations is not None:
        l2_activations = l2_activations[:intermediate_size]

    # Retrieve down_proj statistics
    w_layer_dict = weight_info.get(layer_idx, {})
    w_down_proj = w_layer_dict.get('down_proj', None)
    if w_down_proj is None:
        raise ValueError(f"Method {method} requires weight_info[{layer_idx}]['down_proj'] to exist")

    if w_down_proj.shape[0] != intermediate_size:
        w_down_proj = w_down_proj[:intermediate_size]

    # Compute scores based on WIFV / WIFN
    if method == "WIFV":
        # WIFV = variance * weight
        scores = var_activations * w_down_proj
    elif method == "WIFN":
        # WIFN = sqrt(variance) * weight (or sqrt(l2) if l2 is available)
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
    Compute scores for a single layer: attention heads scores and MLP channels scores.
    Return both attention and MLP scores.

    - attn_scores: shape [num_heads]
    - mlp_scores: shape [intermediate_size]
    
    Later, you can directly apply thresholding (e.g., UL-UM, AL-AM) on these scores.

    Args:
        layer_idx (int): Layer index
        activation_data (Dict[int, Dict[str, Dict[str, torch.Tensor]]): Activation data
        weight_data (Dict[int, Dict[str, torch.Tensor]]): Weight data
        hidden_size (int): Hidden size
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
        intermediate_size (int): Intermediate size of MLP
        method (str): Scoring method ("WIFV" or "WIFN")
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Attention scores and MLP scores
    """
    if layer_idx not in activation_data:
        raise ValueError(f"Missing activation data for layer {layer_idx}")

    layer_info = activation_data[layer_idx]

    # 1) Compute attention scores
    attn_scores = compute_attention_head_scores(
        layer_idx=layer_idx,
        activation_info=layer_info,
        weight_info=weight_data,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        method=method
    )

    # 2) Compute MLP channel scores
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
    Compute the scores for all layers in the model: both attention and MLP scores.
    Returns a dictionary with scores for each layer.

    Args:
        activation_data (Dict[int, Dict[str, Dict[str, torch.Tensor]]): Activation data
        weight_data (Dict[int, Dict[str, torch.Tensor]]): Weight data
        num_layers (int): Number of layers
        hidden_size (int): Hidden size
        num_heads (int): Number of attention heads
        intermediate_size (int): Intermediate size of MLP
        method (str): Scoring method ("WIFV" or "WIFN")
    
    Returns:
        Dict[int, Dict[str, torch.Tensor]]: Layer-wise scores for attention and MLP
    """
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"method={method} not in supported list: {SUPPORTED_METHODS}")

    head_dim = hidden_size // num_heads
    scores_dict = {}

    # Loop over each layer and compute scores
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
