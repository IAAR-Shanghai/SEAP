# /src/pruning_utils/generate_masks.py

"""
This file is responsible for generating the final Boolean masks (attn_mask and mlp_mask) for pruning attention heads and MLP channels.
These masks are based on pruning scores (scores_dict) calculated in compute_scores.py, combined with user-defined pruning strategies.
For this version, only the FLAP pruning strategies, specifically "UL-UM" (layer-wise pruning), are supported.
The masks can also be saved or loaded to/from files.

"""

import torch
import math
import os
from typing import Dict, Tuple, Any
from src.pruning_utils.sparsity_scheduler import get_layerwise_sparsity_map

def generate_layerwise_masks_from_scores(
    attn_scores: torch.Tensor,
    mlp_scores: torch.Tensor,
    attn_sparsity: float,
    mlp_sparsity: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate pruning masks for a single layer using per-module sparsity values.

    This function prunes the lowest-scoring attention heads and MLP neurons by score ranking.

    Args:
        attn_scores (torch.Tensor): Scores for attention heads, shape [num_heads].
        mlp_scores (torch.Tensor): Scores for MLP channels, shape [intermediate_size].
        attn_sparsity (float): Fraction of attention heads to prune (0 ≤ s ≤ 1).
        mlp_sparsity (float): Fraction of MLP channels to prune (0 ≤ s ≤ 1).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Boolean masks (True = keep, False = prune):
            - attn_mask: shape [num_heads]
            - mlp_mask: shape [intermediate_size]
    """
    device = attn_scores.device
    num_heads = attn_scores.numel()
    num_mlp = mlp_scores.numel()

    # Init all keep masks
    attn_mask = torch.ones(num_heads, dtype=torch.bool, device=device)
    mlp_mask = torch.ones(num_mlp, dtype=torch.bool, device=device)

    # ---- Attention mask ----
    num_attn_prune = int(num_heads * attn_sparsity)
    if num_attn_prune > 0:
        _, idx = torch.topk(attn_scores, k=num_attn_prune, largest=False)
        attn_mask[idx] = False

    # ---- MLP mask ----
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
    strategy_kwargs: Dict[str, Any] = None
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Generate pruning masks for all layers based on the selected sparsity strategy.
    
    Args:
        scores_dict (Dict[int, Dict[str, torch.Tensor]]): Scores for each layer/module.
        strategy (str): Sparsity strategy. One of: "uniform", "logistic", "al-am".
        pruning_ratio (float): Global pruning ratio or sparsity target.
        hidden_size (int, optional): Required for AL-AM strategy.
        num_heads (int, optional): Required for AL-AM strategy.
        total_layers (int, optional): Required for logistic strategy.
        strategy_kwargs (Dict[str, Any], optional): Additional strategy-specific args.
    
    Returns:
        Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]: attention and MLP masks.
    """
    if strategy_kwargs is None:
        strategy_kwargs = {}

    sparsity_map = get_layerwise_sparsity_map(
        strategy=strategy,
        num_layers=total_layers or len(scores_dict),
        pruning_ratio=pruning_ratio,
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
    mlp_masks:  Dict[int, torch.Tensor],
    file_path: str
):
    """
    Save the attention and MLP mask dictionaries to a file.

    Args:
        attn_masks (Dict[int, torch.Tensor]): Attention masks to save.
        mlp_masks (Dict[int, torch.Tensor]): MLP masks to save.
        file_path (str): The file path where the masks will be saved (e.g., "mask_dir/masks.pt").
    """
    to_save = {
        "attn_masks": {k: v.cpu() for k, v in attn_masks.items()},
        "mlp_masks":  {k: v.cpu() for k, v in mlp_masks.items()}
    }
    torch.save(to_save, file_path)
    print(f"[save_masks_to_file] Saved masks to {file_path}")

def load_masks_from_file(file_path: str) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Load attention and MLP masks from a file.

    Args:
        file_path (str): The path to the saved mask file.

    Returns:
        Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
            - attn_masks: A dictionary of attention masks.
            - mlp_masks: A dictionary of MLP masks.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Mask file not found: {file_path}")
    loaded = torch.load(file_path)
    attn_masks = loaded["attn_masks"]
    mlp_masks  = loaded["mlp_masks"]
    print(f"[load_masks_from_file] Loaded masks from {file_path}")
    return attn_masks, mlp_masks

def compute_layerwise_sparsity(
    attn_masks: Dict[int, torch.Tensor],
    mlp_masks: Dict[int, torch.Tensor],
    hidden_size: int,
    num_heads: int
) -> Dict[int, Dict[str, float]]:
    """
    Compute the sparsity (pruning ratio) for each layer's attention heads and MLP channels,
    and aggregate a global cost-weighted sparsity.

    Args:
        attn_masks (Dict[int, torch.Tensor]): Attention masks for each layer.
        mlp_masks (Dict[int, torch.Tensor]): MLP masks for each layer.
        hidden_size (int): Hidden size of the transformer.
        num_heads (int): Number of attention heads.

    Returns:
        Dict[int, Dict[str, float]]: Layer-wise sparsities plus a "global_sparsity" key for total.
    """
    def compression_factor(hidden_size, num_heads):
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

