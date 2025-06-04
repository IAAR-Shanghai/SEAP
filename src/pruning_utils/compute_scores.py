"""
Importance score computation utilities for model pruning.

This module provides functions for computing importance scores of attention heads
and MLP channels in transformer models, supporting different scoring methods like
WIFV (Weighted Input Feature Variance) and WIFN (Weighted Input Feature Norm).

Author: why
Date: 2024
"""

# Standard library imports
from typing import Dict, Any, Tuple

# Third-party imports
import torch

# ======================== #
# Scoring Method Registry  #
# ======================== #

def score_wifv(stats: Dict[str, torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """Compute Weighted Input Feature Variance (WIFV) scores.
    
    Args:
        stats: Dictionary containing 'var' and optionally 'l2' activations
        weights: Corresponding weight tensor
        
    Returns:
        Tensor with importance scores based on variance
    """
    return stats["var"] * weights


def score_wifn(stats: Dict[str, torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    """Compute Weighted Input Feature Norm (WIFN) scores.
    
    Args:
        stats: Dictionary containing 'var' and optionally 'l2' activations
        weights: Corresponding weight tensor
        
    Returns:
        Tensor with importance scores based on L2 norm
    """
    source = stats["l2"] if stats["l2"] is not None else stats["var"]
    return torch.sqrt(source) * weights


SCORE_METHODS = {
    "WIFV": score_wifv,
    "WIFN": score_wifn
}


def compute_attention_head_scores(
    layer_idx: int,
    activation_info: Dict[str, Any],
    weight_info: Dict[int, Dict[str, torch.Tensor]],
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    method: str = "WIFV"
) -> torch.Tensor:
    """Compute importance scores for attention heads in a given layer.
    
    Args:
        layer_idx: Index of the current transformer layer
        activation_info: Dictionary with activation statistics
        weight_info: Dictionary with weight statistics for each layer
        hidden_size: Total hidden dimension of the model
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        method: Scoring method to use ("WIFV" or "WIFN")
        
    Returns:
        Tensor of shape [num_heads] with scores for each head
        
    Raises:
        ValueError: If method is unsupported or required data is missing/mismatched
    """
    if method not in SCORE_METHODS:
        raise ValueError(f"Unsupported method: {method}")

    post_agg = activation_info["attention_post_aggregation"]
    stats = {
        "var": post_agg.get("var"),
        "l2": post_agg.get("l2")
    }

    if stats["var"] is None or stats["var"].shape[0] != hidden_size:
        raise ValueError(f"Missing or mismatched 'var' for attention layer {layer_idx}")

    if stats["l2"] is not None and stats["l2"].shape[0] != hidden_size:
        stats["l2"] = None  # fallback to variance

    weights = weight_info[layer_idx].get("o_proj")
    if weights is None or weights.shape[0] != hidden_size:
        raise ValueError(f"Missing or mismatched o_proj weights for layer {layer_idx}")

    raw_scores = SCORE_METHODS[method](stats, weights)
    return raw_scores.view(num_heads, head_dim).mean(dim=1)


def compute_mlp_channel_scores(
    layer_idx: int,
    activation_info: Dict[str, Any],
    weight_info: Dict[int, Dict[str, torch.Tensor]],
    hidden_size: int,
    intermediate_size: int,
    method: str = "WIFV"
) -> torch.Tensor:
    """Compute importance scores for MLP channels.
    
    Args:
        layer_idx: Index of the current transformer layer
        activation_info: Dictionary with MLP activation statistics
        weight_info: Dictionary with weight statistics for each layer
        hidden_size: Total hidden dimension of the model
        intermediate_size: Size of the MLP intermediate representation
        method: Scoring method to use ("WIFV" or "WIFN")
        
    Returns:
        Tensor of shape [intermediate_size] with scores for MLP channels
        
    Raises:
        ValueError: If method is unsupported or required data is missing
    """
    if method not in SCORE_METHODS:
        raise ValueError(f"Unsupported method: {method}")

    mlp_dict = activation_info["mlp_intermediate_states"]
    stats = {
        "var": mlp_dict.get("var"),
        "l2": mlp_dict.get("l2")
    }

    if stats["var"] is None:
        raise ValueError(f"Missing 'var' for MLP layer {layer_idx}")

    stats["var"] = stats["var"][:intermediate_size]
    if stats["l2"] is not None:
        stats["l2"] = stats["l2"][:intermediate_size]

    weights = weight_info[layer_idx].get("down_proj")
    if weights is None:
        raise ValueError(f"Missing down_proj weights for layer {layer_idx}")

    weights = weights[:intermediate_size]

    return SCORE_METHODS[method](stats, weights)


def compute_layer_scores(
    layer_idx: int,
    activation_data: Dict[int, Dict[str, Dict[str, torch.Tensor]]],
    weight_data: Dict[int, Dict[str, torch.Tensor]],
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    intermediate_size: int,
    method: str = "WIFV"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute both attention and MLP scores for a specific layer.
    
    Args:
        layer_idx: Layer index
        activation_data: Activation statistics for all layers
        weight_data: Weight statistics for all layers
        hidden_size: Hidden size of the model
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        intermediate_size: Size of the MLP intermediate layer
        method: Scoring method
        
    Returns:
        Tuple containing:
            - Attention scores tensor of shape [num_heads]
            - MLP scores tensor of shape [intermediate_size]
            
    Raises:
        ValueError: If activation data is missing for the specified layer
    """
    if layer_idx not in activation_data:
        raise ValueError(f"Missing activation data for layer {layer_idx}")

    layer_info = activation_data[layer_idx]

    attn_scores = compute_attention_head_scores(
        layer_idx=layer_idx,
        activation_info=layer_info,
        weight_info=weight_data,
        hidden_size=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        method=method
    )

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
    """Compute pruning scores for all layers in the model.
    
    Args:
        activation_data: Activation statistics for all layers
        weight_data: Weight statistics for all layers
        num_layers: Total number of transformer layers
        hidden_size: Model hidden size
        num_heads: Number of attention heads
        intermediate_size: MLP hidden size
        method: Scoring method to apply
        
    Returns:
        Dictionary mapping each layer index to its attention and MLP scores
        
    Raises:
        ValueError: If method is unsupported or activation data is missing
    """
    if method not in SCORE_METHODS:
        raise ValueError(f"Method '{method}' not supported. Supported: {list(SCORE_METHODS.keys())}")

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
            "mlp_scores": mlp_scores
        }

    return scores_dict
