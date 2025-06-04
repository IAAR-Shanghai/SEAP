# /src/pruning_utils/generate_masks.py

"""
Mask generation utilities for model pruning.

This module provides functions for generating boolean masks to prune attention heads
and MLP channels in transformer models. The masks are based on importance scores
and user-defined pruning strategies. Currently supports FLAP pruning strategies,
specifically "UL-UM" (layer-wise pruning).

Author: why
Date: 2024
"""

# Standard library imports
import os
from typing import Dict, Tuple, Any, List

# Third-party imports
import torch
import math

# Local imports
from src.pruning_utils.sparsity_scheduler import get_layerwise_sparsity_map


def generate_layerwise_masks_from_scores(
    attn_scores: torch.Tensor,
    mlp_scores: torch.Tensor,
    attn_sparsity: float,
    mlp_sparsity: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate pruning masks for a single layer using per-module sparsity values.
    
    Prunes the lowest-scoring attention heads and MLP neurons by score ranking.
    
    Args:
        attn_scores: Scores for attention heads, shape [num_heads]
        mlp_scores: Scores for MLP channels, shape [intermediate_size]
        attn_sparsity: Fraction of attention heads to prune (0 ≤ s ≤ 1)
        mlp_sparsity: Fraction of MLP channels to prune (0 ≤ s ≤ 1)
        
    Returns:
        Tuple containing boolean masks (True = keep, False = prune):
            - attn_mask: shape [num_heads]
            - mlp_mask: shape [intermediate_size]
    """
    device = attn_scores.device
    num_heads = attn_scores.numel()
    num_mlp = mlp_scores.numel()

    # Initialize keep masks
    attn_mask = torch.ones(num_heads, dtype=torch.bool, device=device)
    mlp_mask = torch.ones(num_mlp, dtype=torch.bool, device=device)

    # Generate attention mask
    num_attn_prune = int(num_heads * attn_sparsity)
    if num_attn_prune > 0:
        _, idx = torch.topk(attn_scores, k=num_attn_prune, largest=False)
        attn_mask[idx] = False

    # Generate MLP mask
    num_mlp_prune = int(num_mlp * mlp_sparsity)
    if num_mlp_prune > 0:
        _, idx = torch.topk(mlp_scores, k=num_mlp_prune, largest=False)
        mlp_mask[idx] = False

    return attn_mask, mlp_mask


def generate_masks_for_all_layers(
    scores_dict: Dict[int, Dict[str, torch.Tensor]],
    strategy: str,
    pruning_ratio: float,
    hidden_size: int = None,
    num_heads: int = None,
    total_layers: int = None,
    strategy_kwargs: Dict[str, Any] = None,
    cos_sims: List[float] = None,
    remove_results: Dict[int, List[Tuple[float, float]]] = None,
    fitted_results: Dict[str, Dict[str, float]] = None
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Generate pruning masks for all layers based on selected sparsity strategy.
    
    Supports strategies:
        - "uniform": Uniform sparsity across layers
        - "global": Global importance ranking
        - "cosine": Based on layer-wise cosine similarity
        - "retention": Based on layer-wise retention tests
        - "linear_fit": Based on linear regression fit
        - "logistic_fit": Based on logistic regression fit
    
    Args:
        scores_dict: Per-layer importance scores
        strategy: Strategy name
        pruning_ratio: Target global sparsity
        hidden_size: Required for global strategy
        num_heads: Required for global strategy
        total_layers: Total number of layers (optional)
        strategy_kwargs: Dict containing protect_head, protect_tail, etc.
        cos_sims: List of cosine similarities per layer (for "cosine" strategy)
        remove_results: Per-layer similarity retention test (for "retention")
        fitted_results: Dict with linear/logistic fit params
        
    Returns:
        Tuple containing:
            - attn_masks: Dict mapping layer index to attention boolean masks
            - mlp_masks: Dict mapping layer index to MLP boolean masks
    """
    strategy_kwargs = strategy_kwargs or {}
    num_layers = total_layers or len(scores_dict)

    linear_params = None
    logistic_params = None
    if fitted_results is not None:
        linear_params = fitted_results.get("linear_params")
        logistic_params = fitted_results.get("logistic_params")

    sparsity_map = get_layerwise_sparsity_map(
        strategy=strategy,
        num_layers=num_layers,
        pruning_ratio=pruning_ratio,
        cos_sims=cos_sims,
        remove_results=remove_results,
        linear_params=linear_params,
        logistic_params=logistic_params,
        scores_dict=scores_dict,
        hidden_size=hidden_size,
        num_heads=num_heads,
        strategy_kwargs=strategy_kwargs
    )

    attn_masks = {}
    mlp_masks = {}

    for layer_idx, sparsity in sparsity_map.items():
        attn_scores = scores_dict[layer_idx]["attn_scores"]
        mlp_scores = scores_dict[layer_idx]["mlp_scores"]

        attn_ratio = sparsity["attn_sparsity"]
        mlp_ratio = sparsity["mlp_sparsity"]

        attn_mask, mlp_mask = generate_layerwise_masks_from_scores(
            attn_scores=attn_scores,
            mlp_scores=mlp_scores,
            attn_sparsity=attn_ratio,
            mlp_sparsity=mlp_ratio
        )

        attn_masks[layer_idx] = attn_mask
        mlp_masks[layer_idx] = mlp_mask

    return attn_masks, mlp_masks


def save_masks_to_file(
    attn_masks: Dict[int, torch.Tensor],
    mlp_masks: Dict[int, torch.Tensor],
    file_path: str
) -> None:
    """Save attention and MLP mask dictionaries to a file.
    
    Args:
        attn_masks: Attention masks to save
        mlp_masks: MLP masks to save
        file_path: Path where masks will be saved (e.g., "mask_dir/masks.pt")
    """
    to_save = {
        "attn_masks": {k: v.cpu() for k, v in attn_masks.items()},
        "mlp_masks": {k: v.cpu() for k, v in mlp_masks.items()}
    }
    torch.save(to_save, file_path)
    print(f"[save_masks_to_file] Saved masks to {file_path}")


def load_masks_from_file(file_path: str) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """Load attention and MLP masks from a file.
    
    Args:
        file_path: Path to the saved mask file
        
    Returns:
        Tuple containing:
            - attn_masks: Dictionary of attention masks
            - mlp_masks: Dictionary of MLP masks
            
    Raises:
        FileNotFoundError: If mask file is not found
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Mask file not found: {file_path}")
    loaded = torch.load(file_path)
    attn_masks = loaded["attn_masks"]
    mlp_masks = loaded["mlp_masks"]
    print(f"[load_masks_from_file] Loaded masks from {file_path}")
    return attn_masks, mlp_masks


def compute_layerwise_sparsity(
    attn_masks: Dict[int, torch.Tensor],
    mlp_masks: Dict[int, torch.Tensor],
    hidden_size: int,
    num_heads: int
) -> Tuple[Dict[int, Dict[str, float]], float]:
    """Compute sparsity ratios for each layer's attention heads and MLP channels.
    
    Calculates per-layer sparsity and aggregates a global cost-weighted sparsity.
    
    Args:
        attn_masks: Attention masks for each layer
        mlp_masks: MLP masks for each layer
        hidden_size: Hidden size of the transformer
        num_heads: Number of attention heads
        
    Returns:
        Tuple containing:
            - Dictionary mapping layer indices to sparsity metrics
            - Global sparsity ratio (cost-weighted)
    """
    def compression_factor(hidden_size: int, num_heads: int) -> float:
        return (4.0 / 3.0) * (hidden_size / num_heads)

    results = {}
    total_cost = 0.0
    kept_cost = 0.0

    head_cost = compression_factor(hidden_size, num_heads)
    mlp_cost = 1.0

    for layer_idx in sorted(attn_masks.keys()):
        attn_mask = attn_masks[layer_idx]
        mlp_mask = mlp_masks[layer_idx]

        total_heads = attn_mask.numel()
        kept_heads = attn_mask.sum().item()
        attn_sparsity = 1.0 - (kept_heads / total_heads)

        total_mlp = mlp_mask.numel()
        kept_mlp = mlp_mask.sum().item()
        mlp_sparsity = 1.0 - (kept_mlp / total_mlp)

        # Cost-based accumulation
        total_cost += head_cost * total_heads + mlp_cost * total_mlp
        kept_cost += head_cost * kept_heads + mlp_cost * kept_mlp

        results[layer_idx] = {
            "attn_sparsity": attn_sparsity,
            "mlp_sparsity": mlp_sparsity
        }

    global_sparsity = 1.0 - (kept_cost / total_cost)
    global_sparsity = round(global_sparsity, 6)
    return results, global_sparsity

