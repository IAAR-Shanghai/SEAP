"""
Sparsity scheduling utilities for model pruning.

This module provides functions for computing layer-wise sparsity ratios based on
different strategies, including uniform, cosine similarity-based, retention-based,
and fitted curve-based approaches.

Author: why
Date: 2024
"""

# Standard library imports
import math
from typing import List, Dict, Tuple

# Third-party imports
import numpy as np
import torch

# ======================== #
# General Utility Functions
# ======================== #

def normalize_importance_to_sparsity(
    importance: List[float],
    target_sparsity: float,
    total_layers: int
) -> List[float]:
    """Map importance scores to sparsity ratios ensuring global target sparsity.
    
    Only distributes sparsity over non-protected layers. Protected layers should
    be merged externally.
    
    Args:
        importance: Importance values for effective layers only
        target_sparsity: Global target sparsity ratio (distributed proportionally)
        total_layers: Total number of model layers (including protected ones)
        
    Returns:
        List of sparsity ratios with length equal to len(importance)
    """
    eps = 1e-10
    L_eff = len(importance)
    imp = np.array(importance, dtype=np.float32) + eps

    # Key change: Budget based on total_layers not effective layers
    total_target_sparsity = target_sparsity * total_layers
    total_retention = L_eff - total_target_sparsity

    imp_sum = imp.sum()
    if imp_sum < 1e-10:
        normed_retain = np.ones_like(imp) * (total_retention / L_eff)
    else:
        normed_retain = imp / imp_sum * total_retention

    sparsity = 1.0 - normed_retain
    sparsity = np.clip(sparsity, 0.0, 1.0)
    return sparsity.tolist()


def make_layerwise_sparsity_map(sparsity_list: List[float]) -> Dict[int, Dict[str, float]]:
    """Convert list of sparsity values to layer-wise sparsity dictionary.
    
    Args:
        sparsity_list: List of sparsity ratios
        
    Returns:
        Dictionary mapping layer indices to attention and MLP sparsity ratios
    """
    return {
        i: {
            "attn_sparsity": round(s, 6),
            "mlp_sparsity": round(s, 6)
        } for i, s in enumerate(sparsity_list)
    }


def print_sparsity_summary(sparsity_map: Dict[int, Dict[str, float]]) -> None:
    """Print summary of average attention and MLP sparsity ratios.
    
    Args:
        sparsity_map: Dictionary mapping layer indices to sparsity ratios
    """
    attn_vals = [v["attn_sparsity"] for v in sparsity_map.values()]
    mlp_vals = [v["mlp_sparsity"] for v in sparsity_map.values()]
    avg_attn = sum(attn_vals) / len(attn_vals)
    avg_mlp = sum(mlp_vals) / len(mlp_vals)
    print(f"[Sparsity Summary]  Avg ATT: {avg_attn:.4f} | Avg MLP: {avg_mlp:.4f}")


# ======================== #
# Strategy Implementations
# ======================== #

def uniform_sparsity(
    num_layers: int,
    sparsity: float
) -> Dict[int, Dict[str, float]]:
    """Generate uniform sparsity ratios across all layers.
    
    Args:
        num_layers: Total number of model layers
        sparsity: Target sparsity ratio to apply uniformly
        
    Returns:
        Dictionary mapping layer indices to uniform sparsity ratios
    """
    return {
        layer: {"attn_sparsity": sparsity, "mlp_sparsity": sparsity}
        for layer in range(num_layers)
    }


def cosine_sparsity(
    cos_sims: List[float],
    target_sparsity: float,
    protect_head: int = 0,
    protect_tail: int = 0
) -> Dict[int, Dict[str, float]]:
    """Generate sparsity ratios based on cosine similarities.
    
    Computes importance from cosine similarities and maps to sparsity ratios.
    Protected layers are excluded from normalization.
    
    Args:
        cos_sims: List of cosine similarities per layer
        target_sparsity: Global target sparsity ratio
        protect_head: Number of initial layers to protect
        protect_tail: Number of final layers to protect
        
    Returns:
        Dictionary mapping layer indices to sparsity ratios
    """
    num_layers = len(cos_sims)
    
    full_importance = [1 - s for s in cos_sims]

    protected = set(range(protect_head)) | set(range(num_layers - protect_tail, num_layers))
    effective_indices = [i for i in range(num_layers) if i not in protected]
    effective_importance = [full_importance[i] for i in effective_indices]

    eff_sparsity = normalize_importance_to_sparsity(
        importance=effective_importance,
        target_sparsity=target_sparsity,
        total_layers=num_layers
    )

    sparsity = [0.0] * num_layers
    for i, s in zip(effective_indices, eff_sparsity):
        sparsity[i] = s

    return make_layerwise_sparsity_map(sparsity)


def retention_sparsity(
    remove_results: Dict[int, List[Tuple[float, float]]],
    num_layers: int,
    target_sparsity: float,
    protect_head: int = 0,
    protect_tail: int = 0
) -> Dict[int, Dict[str, float]]:
    """Generate sparsity ratios based on retention test results.
    
    Uses average retention rate (1 - similarity) as importance measure.
    Protected layers are excluded from sparsity allocation but included
    in global sparsity constraint.
    
    Args:
        remove_results: Dictionary mapping layer indices to retention test results
        num_layers: Total number of model layers
        target_sparsity: Global target sparsity ratio
        protect_head: Number of initial layers to protect
        protect_tail: Number of final layers to protect
        
    Returns:
        Dictionary mapping layer indices to sparsity ratios
    """
    full_importance = []
    for i in range(num_layers):
        if i in remove_results:
            sims = [sim for _, sim in remove_results[i]]
            imp = 1.0 - float(np.mean(sims))
        else:
            imp = 0.0
        full_importance.append(imp)

    protected = set(range(protect_head)) | set(range(num_layers - protect_tail, num_layers))
    effective_indices = [i for i in range(num_layers) if i not in protected]
    effective_importance = [full_importance[i] for i in effective_indices]

    eff_sparsity = normalize_importance_to_sparsity(
        importance=effective_importance,
        target_sparsity=target_sparsity,
        total_layers=num_layers
    )

    sparsity = [0.0] * num_layers
    for i, s in zip(effective_indices, eff_sparsity):
        sparsity[i] = s

    return make_layerwise_sparsity_map(sparsity)


def fitted_sparsity(
    fitted_meta: Dict[str, float],
    num_layers: int,
    target_sparsity: float,
    fit_type: str = "linear",  # or "logistic"
    protect_head: int = 0,
    protect_tail: int = 0,
) -> Dict[int, Dict[str, float]]:
    """Generate sparsity ratios based on fitted curve parameters.
    
    Uses linear or logistic function parameters to estimate similarities,
    then maps to importance and sparsity ratios.
    
    Args:
        fitted_meta: Dictionary containing fit parameters 'a' and 'b'
        num_layers: Total number of model layers
        target_sparsity: Global target sparsity ratio
        fit_type: Type of fit ("linear" or "logistic")
        protect_head: Number of initial layers to protect
        protect_tail: Number of final layers to protect
        
    Returns:
        Dictionary mapping layer indices to sparsity ratios
        
    Raises:
        AssertionError: If fit_type is invalid or parameters are missing
    """
    assert fit_type in {"linear", "logistic"}
    assert "a" in fitted_meta and "b" in fitted_meta
    a, b = fitted_meta["a"], fitted_meta["b"]

    protected = set(range(protect_head)) | set(range(num_layers - protect_tail, num_layers))
    effective_indices = [i for i in range(num_layers) if i not in protected]

    full_importance = []
    for i in range(num_layers):
        if fit_type == "linear":
            sim = a * i + b
        else:
            sim = 1 / (1 + math.exp(-(a * i + b)))
        imp = 1.0 - sim
        full_importance.append(imp)

    effective_importance = [full_importance[i] for i in effective_indices]

    eff_sparsity = normalize_importance_to_sparsity(
        importance=effective_importance,
        target_sparsity=target_sparsity,
        total_layers=num_layers
    )

    sparsity = [0.0] * num_layers
    for i, s in zip(effective_indices, eff_sparsity):
        sparsity[i] = s

    return make_layerwise_sparsity_map(sparsity)


def global_sparsity(
    scores_dict: Dict[int, Dict[str, torch.Tensor]],
    hidden_size: int,
    num_heads: int,
    pruning_ratio: float,
) -> Dict[int, Dict[str, float]]:
    """Generate sparsity ratios based on global importance ranking.
    
    Computes cost-weighted global ranking of attention heads and MLP channels,
    then determines sparsity ratios to meet global pruning target.
    
    Args:
        scores_dict: Dictionary of layer-wise importance scores
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        pruning_ratio: Target global pruning ratio
        
    Returns:
        Dictionary mapping layer indices to sparsity ratios
    """
    def compression_factor(hidden_size: int, num_heads: int) -> float:
        return (4.0 / 3.0) * (hidden_size / num_heads)

    def robust_standardize(x: torch.Tensor, eps: float = 1e-9, clip_threshold: float = 3.0) -> torch.Tensor:
        med = x.median()
        iqr = torch.quantile(x, 0.75) - torch.quantile(x, 0.25)
        return torch.clamp((x - med) / (iqr + eps), -clip_threshold, clip_threshold)

    head_cost = compression_factor(hidden_size, num_heads)
    chn_cost = 1.0

    all_scores, all_costs, all_index = [], [], []
    attn_cnt, mlp_cnt = {}, {}

    for l, d in scores_dict.items():
        attn_s = robust_standardize(d["attn_scores"])
        mlp_s = robust_standardize(d["mlp_scores"])
        attn_cnt[l], mlp_cnt[l] = attn_s.numel(), mlp_s.numel()

        all_scores += attn_s.tolist() + mlp_s.tolist()
        all_costs += [head_cost] * attn_s.numel() + [chn_cost] * mlp_s.numel()
        all_index += [(True, l, i) for i in range(attn_s.numel())]
        all_index += [(False, l, i) for i in range(mlp_s.numel())]

    all_scores = torch.tensor(all_scores)
    all_costs = torch.tensor(all_costs)
    sorted_idx = torch.argsort(all_scores, descending=True)
    cumsum_costs = torch.cumsum(all_costs[sorted_idx], dim=0)
    target_cost = (1 - pruning_ratio) * cumsum_costs[-1].item()
    cutoff = torch.searchsorted(cumsum_costs, target_cost).item()

    keep_mask = torch.zeros_like(sorted_idx, dtype=torch.bool)
    keep_mask[:cutoff] = True

    kept_attn = {l: 0 for l in scores_dict}
    kept_mlp = {l: 0 for l in scores_dict}

    for rank, sid in enumerate(sorted_idx):
        is_attn, l, _ = all_index[sid.item()]
        if keep_mask[rank]:
            if is_attn:
                kept_attn[l] += 1
            else:
                kept_mlp[l] += 1

    sparsity_map = {}
    for l in scores_dict:
        sa = 1.0 - (kept_attn[l] / attn_cnt[l])
        sm = 1.0 - (kept_mlp[l] / mlp_cnt[l])
        sparsity_map[l] = {
            "attn_sparsity": round(min(max(sa, 0.0), 1.0), 6),
            "mlp_sparsity": round(min(max(sm, 0.0), 1.0), 6)
        }
    return sparsity_map


# ======================== #
# Main Entry Point
# ======================== #

def get_layerwise_sparsity_map(
    strategy: str,
    num_layers: int,
    pruning_ratio: float,
    cos_sims: List[float] = None,
    remove_results: Dict[int, List[Tuple[float, float]]] = None,
    linear_params: Dict[str, float] = None,
    logistic_params: Dict[str, float] = None,
    scores_dict: Dict[int, Dict[str, torch.Tensor]] = None,
    hidden_size: int = None,
    num_heads: int = None,
    strategy_kwargs: Dict = None
) -> Dict[int, Dict[str, float]]:
    """Generate layer-wise sparsity map based on selected strategy.
    
    Unified entry point for all sparsity scheduling strategies.
    
    Args:
        strategy: Name of sparsity strategy to use
        num_layers: Total number of model layers
        pruning_ratio: Target global pruning ratio
        cos_sims: List of cosine similarities (for cosine strategy)
        remove_results: Layer-wise retention test results
        linear_params: Linear fit parameters
        logistic_params: Logistic fit parameters
        scores_dict: Layer-wise importance scores
        hidden_size: Model hidden dimension
        num_heads: Number of attention heads
        strategy_kwargs: Additional strategy-specific parameters
        
    Returns:
        Dictionary mapping layer indices to sparsity ratios
        
    Raises:
        ValueError: If strategy is invalid or required parameters are missing
    """
    strategy_kwargs = strategy_kwargs or {}
    protect_head = strategy_kwargs.get("protect_head", 0)
    protect_tail = strategy_kwargs.get("protect_tail", 0)

    if strategy == "uniform":
        return uniform_sparsity(num_layers, pruning_ratio)

    elif strategy == "cosine":
        if cos_sims is None:
            raise ValueError("cos_sims required for cosine strategy")
        return cosine_sparsity(
            cos_sims=cos_sims,
            target_sparsity=pruning_ratio,
            protect_head=protect_head,
            protect_tail=protect_tail
        )

    elif strategy == "retention":
        if remove_results is None:
            raise ValueError("remove_results required for retention strategy")
        return retention_sparsity(
            remove_results=remove_results,
            num_layers=num_layers,
            target_sparsity=pruning_ratio,
            protect_head=protect_head,
            protect_tail=protect_tail
        )

    elif strategy == "linear_fit":
        if linear_params is None:
            raise ValueError("linear_params required for linear_fit strategy")
        return fitted_sparsity(
            fitted_meta=linear_params,
            num_layers=num_layers,
            target_sparsity=pruning_ratio,
            fit_type="linear",
            protect_head=protect_head,
            protect_tail=protect_tail
        )

    elif strategy == "logistic_fit":
        if logistic_params is None:
            raise ValueError("logistic_params required for logistic_fit strategy")
        return fitted_sparsity(
            fitted_meta=logistic_params,
            num_layers=num_layers,
            target_sparsity=pruning_ratio,
            fit_type="logistic",
            protect_head=protect_head,
            protect_tail=protect_tail
        )

    elif strategy == "global":
        if scores_dict is None or hidden_size is None or num_heads is None:
            raise ValueError("scores_dict, hidden_size, num_heads required for global strategy")
        return global_sparsity(
            scores_dict=scores_dict,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pruning_ratio=pruning_ratio
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


