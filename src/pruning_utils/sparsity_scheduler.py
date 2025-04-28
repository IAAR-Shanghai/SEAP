import torch
import math
from typing import Dict, Callable


def uniform_sparsity(
    num_layers: int,
    sparsity: float
) -> Dict[int, Dict[str, float]]:
    """
    Assigns the same sparsity to all layers uniformly.

    Args:
        num_layers (int): Total number of layers.
        sparsity (float): Uniform sparsity ratio (0 ~ 1).

    Returns:
        Dict[int, Dict[str, float]]: Layer-wise sparsity mapping.
    """
    return {
        layer: {
            "attn_sparsity": sparsity,
            "mlp_sparsity": sparsity
        } for layer in range(num_layers)
    }


def logistic_sparsity(
    num_layers: int,
    sparsity: float,
    m: int = 2,
    n: int = 2,
    k: float = 1.2,
    x0: float = 0.3
) -> Dict[int, Dict[str, float]]:
    """
    Computes layerwise sparsity using a logistic decay pattern, adjusting for protected layers (m and n)
    to ensure the overall model meets the global target sparsity.

    Args:
        num_layers (int): Total number of layers.
        sparsity (float): Target global sparsity (0 ~ 1).
        m (int): Number of bottom layers not to prune.
        n (int): Number of top layers not to prune.
        k (float): Logistic curve steepness.
        x0 (float): Logistic curve midpoint.

    Returns:
        Dict[int, Dict[str, float]]: Layer-wise sparsity map.
    """
    effective_layers = num_layers - m - n
    if effective_layers <= 0:
        raise ValueError("Too many protected layers: m + n must be less than total number of layers.")

    # Adjusted target sparsity for effective layers
    adjusted_target = sparsity * num_layers / effective_layers
    adjusted_target = min(adjusted_target, 1.0)

    # Solve for lambda to meet adjusted target
    def get_lambda():
        def avg_sp(lambda_):
            total = 0.0
            for i in range(num_layers):
                if i < m or i >= num_layers - n:
                    continue
                x = (i - m) / (effective_layers - 1)
                rho = lambda_ / (1 + math.exp(-k * (x - x0)))
                total += rho
            return total / effective_layers

        lo, hi = 0.0, 2.0
        for _ in range(100):
            mid = (lo + hi) / 2.0
            if abs(avg_sp(mid) - adjusted_target) < 1e-4:
                return mid
            if avg_sp(mid) > adjusted_target:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2.0

    lambda_star = get_lambda()
    sparsity_map = {}

    for i in range(num_layers):
        if i < m or i >= num_layers - n:
            rho = 0.0
        else:
            x = (i - m) / (effective_layers - 1)
            rho = lambda_star / (1 + math.exp(-k * (x - x0)))
        sparsity_map[i] = {
            "attn_sparsity": round(min(max(rho, 0.0), 1.0), 6),
            "mlp_sparsity":  round(min(max(rho, 0.0), 1.0), 6)
        }

    return sparsity_map


def global_sparsity(
    scores_dict: Dict[int, Dict[str, torch.Tensor]],
    hidden_size: int,
    num_heads: int,
    pruning_ratio: float,
) -> Dict[int, Dict[str, float]]:
    """
    Calculates sparsity per layer based on global cost-aware strategy.

    Args:
        scores_dict (Dict[int, Dict[str, torch.Tensor]]): Computed scores from compute_scores.py.
        hidden_size (int): Transformer hidden size.
        num_heads (int): Number of attention heads.
        pruning_ratio (float): Global resource pruning ratio (0 ~ 1).

    Returns:
        Dict[int, Dict[str, float]]: Per-layer sparsity dict.
    """
    def compression_factor(hidden_size, num_heads):
        return (4.0 / 3.0) * (hidden_size / num_heads)

    def robust_standardize(x: torch.Tensor, eps=1e-9, clip_threshold=3.0):
        if x.dim() == 1:
            med = x.median()
            iqr = torch.quantile(x, 0.75) - torch.quantile(x, 0.25)
            return torch.clamp((x - med) / (iqr + eps), -clip_threshold, clip_threshold)
        elif x.dim() == 2:
            med = torch.median(x, dim=1, keepdim=True).values
            q1 = torch.quantile(x, 0.25, dim=1, keepdim=True)
            q3 = torch.quantile(x, 0.75, dim=1, keepdim=True)
            iqr = q3 - q1
            return torch.clamp((x - med) / (iqr + eps), -clip_threshold, clip_threshold)
        else:
            raise ValueError("Input must be 1D or 2D tensor")

    head_cost = compression_factor(hidden_size, num_heads)
    chn_cost = 1.0

    all_scores = []
    all_costs = []
    all_index = []

    attn_cnt = {}
    mlp_cnt = {}

    for l, d in scores_dict.items():
        attn_s = d["attn_scores"]
        mlp_s = d["mlp_scores"]
        attn_cnt[l] = attn_s.numel()
        mlp_cnt[l] = mlp_s.numel()

        attn_s = robust_standardize(attn_s)
        mlp_s = robust_standardize(mlp_s)

        for i, s in enumerate(attn_s):
            all_scores.append(s.item())
            all_costs.append(head_cost)
            all_index.append((True, l, i))

        for i, s in enumerate(mlp_s):
            all_scores.append(s.item())
            all_costs.append(chn_cost)
            all_index.append((False, l, i))

    all_scores = torch.tensor(all_scores)
    all_costs = torch.tensor(all_costs)
    sorted_idx = torch.argsort(all_scores, descending=True)
    sorted_costs = all_costs[sorted_idx]
    cumsum_costs = torch.cumsum(sorted_costs, dim=0)
    target_cost = (1 - pruning_ratio) * cumsum_costs[-1].item()

    cutoff = torch.searchsorted(cumsum_costs, target_cost).item()
    keep_mask = torch.zeros_like(sorted_idx, dtype=torch.bool)
    keep_mask[:cutoff] = True

    kept_attn = {l: 0 for l in scores_dict}
    kept_mlp = {l: 0 for l in scores_dict}

    for rank, sid in enumerate(sorted_idx):
        is_attn, l, idx = all_index[sid.item()]
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

def get_layerwise_sparsity_map(
    strategy: str,
    num_layers: int,
    pruning_ratio: float,
    scores_dict: Dict[int, Dict[str, torch.Tensor]] = None,
    hidden_size: int = None,
    num_heads: int = None,
    strategy_kwargs: Dict = None
) -> Dict[int, Dict[str, float]]:
    """
    Main entry point to compute layerwise sparsity configuration.

    Args:
        strategy (str): One of "uniform", "logistic", or "al-am".
        num_layers (int): Total number of layers in model.
        pruning_ratio (float): Global pruning ratio (0 ~ 1).
        scores_dict (Dict, optional): Required for 'al-am' strategy.
        hidden_size (int, optional): Required for 'al-am'.
        num_heads (int, optional): Required for 'al-am'.
        strategy_kwargs (Dict, optional): Extra kwargs for strategy functions.

    Returns:
        Dict[int, Dict[str, float]]: Per-layer sparsity map.
    """
    strategy_kwargs = strategy_kwargs or {}

    if strategy == "uniform":
        return uniform_sparsity(num_layers, pruning_ratio)

    elif strategy == "logistic":
        return logistic_sparsity(
            num_layers=num_layers,
            sparsity=pruning_ratio,
            **strategy_kwargs
        )

    elif strategy == "global":
        if scores_dict is None or hidden_size is None or num_heads is None:
            raise ValueError("al-am requires scores_dict, hidden_size, and num_heads.")
        return global_sparsity(
            scores_dict=scores_dict,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pruning_ratio=pruning_ratio,
            **strategy_kwargs
        )

    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

if __name__ == "__main__":
    import random

    torch.manual_seed(42)
    random.seed(42)

    num_layers = 32
    hidden_size = 1024
    num_heads = 32
    intermediate_size = 4096
    pruning_ratio = 0.3

    # ---- Build dummy scores_dict ----
    scores_dict = {}
    for layer in range(num_layers):
        attn_scores = torch.randn(num_heads)
        mlp_scores = torch.randn(intermediate_size)
        scores_dict[layer] = {
            "attn_scores": attn_scores,
            "mlp_scores": mlp_scores
        }

    # ---- Test uniform ----
    print("\n=== [uniform] ===")
    uniform_map = get_layerwise_sparsity_map(
        strategy="uniform",
        num_layers=num_layers,
        pruning_ratio=pruning_ratio
    )
    for k, v in uniform_map.items():
        print(f"Layer {k}: {v}")

    # ---- Test logistic ----
    print("\n=== [logistic] ===")
    logistic_map = get_layerwise_sparsity_map(
        strategy="logistic",
        num_layers=num_layers,
        pruning_ratio=pruning_ratio,
        strategy_kwargs={"k": 1.2, "x0": 0.3}
    )
    for k, v in logistic_map.items():
        print(f"Layer {k}: {v}")

    # ---- Test al-am ----
    print("\n=== [global] ===")
    al_am_map = get_layerwise_sparsity_map(
        strategy="global",
        num_layers=num_layers,
        pruning_ratio=pruning_ratio,
        scores_dict=scores_dict,
        hidden_size=hidden_size,
        num_heads=num_heads
    )
    for k, v in al_am_map.items():
        print(f"Layer {k}: {v}")
