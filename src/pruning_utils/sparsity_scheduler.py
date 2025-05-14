import numpy as np
import torch
import math
from typing import List, Dict, Tuple


# ==== 通用工具函数 ====

def normalize_importance_to_sparsity(
    importance: List[float],
    target_sparsity: float,
    total_layers: int
) -> List[float]:
    """
    将 importance 映射到 sparsity，确保全局稀疏度 ≈ target_sparsity。
    仅对传入的 importance 列表（非保护层）做分配，保护层应在外部合并。
    
    Args:
        importance: 仅包含有效层的 importance 值。
        target_sparsity: 全局目标稀疏度（会按比例映射到有效层）
        total_layers: 模型总层数（包含保护层）
    
    Returns:
        List[float]: 长度等于 len(importance)，表示对应层的稀疏度
    """
    eps = 1e-10
    L_eff = len(importance)
    imp = np.array(importance, dtype=np.float32) + eps

    # 关键修改：预算基于 total_layers 而非有效层数
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
    return {
        i: {
            "attn_sparsity": round(s, 6),
            "mlp_sparsity": round(s, 6)
        } for i, s in enumerate(sparsity_list)
    }


def print_sparsity_summary(sparsity_map: Dict[int, Dict[str, float]]):
    attn_vals = [v["attn_sparsity"] for v in sparsity_map.values()]
    mlp_vals = [v["mlp_sparsity"] for v in sparsity_map.values()]
    avg_attn = sum(attn_vals) / len(attn_vals)
    avg_mlp = sum(mlp_vals) / len(mlp_vals)
    print(f"[Sparsity Summary]  Avg ATT: {avg_attn:.4f} | Avg MLP: {avg_mlp:.4f}")


# ==== 策略实现 ====

def uniform_sparsity(num_layers: int, sparsity: float) -> Dict[int, Dict[str, float]]:
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
    """
    基于 cosine 相似度计算 importance，并映射 sparsity。
    保护层不参与归一化分配。
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
    """
    根据每层平均保留率 (1 - similarity) 反映重要性，映射到稀疏度。
    保护层不参与稀疏度分配，但计入目标 sparsity 的约束。
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
    """
    根据线性/逻辑函数的拟合参数生成各层的相似度估计，映射到 importance → sparsity。
    保护层不参与归一化，但计入全局 sparsity 要求。
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
    def compression_factor(hidden_size, num_heads):
        return (4.0 / 3.0) * (hidden_size / num_heads)

    def robust_standardize(x: torch.Tensor, eps=1e-9, clip_threshold=3.0):
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


# ==== 总统一致入口 ====

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
    strategy_kwargs = strategy_kwargs or {}
    protect_head = strategy_kwargs.get("protect_head", 0)
    protect_tail = strategy_kwargs.get("protect_tail", 0)

    if strategy == "uniform":
        return uniform_sparsity(num_layers, pruning_ratio)

    elif strategy == "global":
        if scores_dict is None or hidden_size is None or num_heads is None:
            raise ValueError("global strategy requires scores_dict, hidden_size, and num_heads.")
        return global_sparsity(scores_dict, hidden_size, num_heads, pruning_ratio)

    elif strategy == "cosine":
        if cos_sims is None:
            raise ValueError("cos_sims must be provided for cosine strategy.")
        return cosine_sparsity(cos_sims, pruning_ratio, protect_head, protect_tail)

    elif strategy == "retention":
        if remove_results is None:
            raise ValueError("remove_results must be provided for retention strategy.")
        return retention_sparsity(remove_results, num_layers, pruning_ratio, protect_head, protect_tail)

    elif strategy == "linear_fit":
        if linear_params is None:
            raise ValueError("linear_params must be provided for linear_fit strategy.")
        return fitted_sparsity(linear_params, num_layers, pruning_ratio, "linear", protect_head, protect_tail)

    elif strategy == "logistic_fit":
        if logistic_params is None:
            raise ValueError("logistic_params must be provided for logistic_fit strategy.")
        return fitted_sparsity(logistic_params, num_layers, pruning_ratio, "logistic", protect_head, protect_tail)

    else:
        raise ValueError(f"Unsupported strategy: {strategy}")


