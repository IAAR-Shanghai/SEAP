"""
/src/pruning_utils/generate_masks.py

该文件用于根据 compute_scores.py 中计算好的剪枝分数 (scores_dict)，
结合用户设定的结构化剪枝策略 (这里仅保留 FLAP 中的 UL-UM, AL-AM),
来生成最终的注意力头和 MLP 通道的布尔掩码 (mask) 并可将其保存/加载到文件。
"""

import torch
import math
import os
from typing import Dict, Tuple, Any

import math

import math

def layer_sparsities_with_logistic(L, G, m=2, n=2, k=1.0, x0=0.3, 
                                   lambda_lower=0.0, lambda_upper=2.0, 
                                   max_iter=100, tol=1e-4):
    """
    通过对 Lambda 做数值搜索, 使得层稀疏度的离散平均值 ≈ G。
    同时前面 m 层和后面 n 层的稀疏度设为 0。
    
    参数说明：
    ----------
    L : int
        模型层数（假设层索引从 1 到 L）
    G : float
        全局期望的平均稀疏度目标 (0 <= G <= 1)
    m : int, 缺省=0
        不参与稀疏化的前 m 层
    n : int, 缺省=0
        不参与稀疏化的后 n 层
    k : float, 缺省=1.0
        Logistic 函数曲线的陡峭度参数
    x0 : float, 缺省=0.3
        Logistic 函数的拐点位置 (0 <= x0 <= 1)
    lambda_lower : float, 缺省=0.0
        搜索 Lambda 时的下界
    lambda_upper : float, 缺省=2.0
        搜索 Lambda 时的上界
    max_iter : int, 缺省=100
        二分搜索的最大迭代次数
    tol : float, 缺省=1e-4
        收敛阈值，如果平均稀疏度与 G 的差值小于该阈值则停止
    
    返回：
    ----------
    rho_list : list of float
        长度为 L 的列表, 表示每层的稀疏度 [rho_1, rho_2, ..., rho_L]
    Lambda_star : float
        求得的 Lambda 值
    """

    # 定义一个函数：给定 Lambda, 返回离散平均稀疏度
    def average_sparsity(Lambda_):
        """ 计算在当前 Lambda_ 下, 离散层稀疏度的平均值 """
        rho_vals = []
        active_layers = L - m - n  # 不包含保护层的有效层数
        for ell in range(1, L+1):
            # 跳过前 m 层和后 n 层
            if ell <= m or ell > (L - n):
                rho_vals.append(0.0)  # 保护层稀疏度为0
                continue
            
            # 归一化坐标 x_ell
            x_ell = (ell - 1 - m) / (L - m - n - 1) if active_layers > 1 else 0.0
            # Logistic 函数值
            rho_val = Lambda_ / (1.0 + math.exp(-k * (x_ell - x0)))
            rho_vals.append(rho_val)
        
        return sum(rho_vals) / active_layers  # 只计算有效层的平均稀疏度
    
    # 如果 G 为 0 或 1, 可以直接返回全 0 或全 1
    if abs(G) < 1e-12:
        return [0.0]*L, 0.0
    if abs(G - 1.0) < 1e-12:
        return [1.0]*L, 1.0
    
    # 开始二分搜索, 在 [lambda_lower, lambda_upper] 区间
    low, high = lambda_lower, lambda_upper
    best_Lambda = (low + high) / 2.0
    
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        avg_sp = average_sparsity(mid)
        
        if abs(avg_sp - G) < tol:
            best_Lambda = mid
            break
        
        if avg_sp > G:
            # 平均稀疏度过高, 需要减小 Lambda
            high = mid
        else:
            # 平均稀疏度过低, 需要增大 Lambda
            low = mid
        
        best_Lambda = mid
    
    # 最终用 best_Lambda 计算每层稀疏度
    rho_list = []
    active_layers = L - m - n  # 不包含保护层的有效层数
    for ell in range(1, L+1):
        if ell <= m or ell > (L - n):
            rho_list.append(0.0)  # 保护层稀疏度为0
        else:
            x_ell = (ell - 1 - m) / (L - m - n - 1) if active_layers > 1 else 0.0
            rho_val = best_Lambda / (1.0 + math.exp(-k * (x_ell - x0)))
            rho_list.append(rho_val)
    
    return rho_list, best_Lambda

def generate_ul_um_masks(
    attn_scores: torch.Tensor,
    mlp_scores: torch.Tensor,
    pruning_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    针对“UL-UM”(逐层局部剪枝)模式：
      对单层的注意力头分数 (attn_scores) 和 MLP 通道分数 (mlp_scores)
      分别做局部排序，按 pruning_ratio 裁剪最低分数。
    即：保留前 (1-pruning_ratio)*num_heads / (1-pruning_ratio)*inter_dim 的最高分数。

    Args:
        attn_scores: shape=[num_heads],  注意力头得分
        mlp_scores:  shape=[intermediate_size], MLP通道得分
        pruning_ratio: 剪枝比例 (0~1)

    Returns:
        (attn_mask, mlp_mask): 均为 bool 向量 (True=保留, False=剪)
    """
    device = attn_scores.device
    num_heads = attn_scores.shape[0]
    inter_dim = mlp_scores.shape[0]

    attn_mask = torch.ones(num_heads, dtype=torch.bool, device=device)
    mlp_mask  = torch.ones(inter_dim, dtype=torch.bool, device=device)

    # 对注意力头：选出分数最低的 head_prune_num 个来剪
    head_prune_num = int(num_heads * pruning_ratio)
    sorted_heads, idx_heads = torch.sort(attn_scores)  # ascending
    heads_to_prune = idx_heads[:head_prune_num]
    attn_mask[heads_to_prune] = False

    # 对MLP通道：选出分数最低的 mlp_prune_num 个来剪
    mlp_prune_num = int(inter_dim * pruning_ratio)
    sorted_mlp, idx_mlp = torch.sort(mlp_scores)  # ascending
    mlp_to_prune = idx_mlp[:mlp_prune_num]
    mlp_mask[mlp_to_prune] = False

    return attn_mask, mlp_mask

def standardize_scores(x: torch.Tensor, eps=1e-9, clip_threshold=3.0) -> torch.Tensor:
    """
    对一个 1D 或 2D Tensor 使用中位数和IQR进行稳健的标准化，并进行截断，
    确保所有标准化后的值在[-clip_threshold, clip_threshold]范围内。
    
    - 如果是 1D: 就直接用全局 median/IQR。
    - 如果是 2D: [num_layers, num_items]，对每一行(对应每层)单独做稳健标准化:
        row = (row - median(row)) / IQR(row)
    
    Parameters:
    x: torch.Tensor - 输入的 1D 或 2D Tensor
    eps: float - 防止除以零的很小值（默认为1e-9）
    clip_threshold: float - 截断的最大值，标准化后的值不会超过这个值（默认为3.0）
    
    Returns:
    torch.Tensor - 稳健标准化并截断后的 Tensor
    """
    if x.dim() == 1:
        # 1D 情况，仍然使用全局 median/IQR
        median = x.median()
        q1 = torch.quantile(x, 0.25)
        q3 = torch.quantile(x, 0.75)
        iqr = q3 - q1
        standardized_x = (x - median) / (iqr + eps)
    
    elif x.dim() == 2:
        # 2D: shape = [num_layers, num_items]
        # 按行逐层做稳健标准化
        median = torch.median(x, axis=1, keepdim=True).values
        q1 = torch.quantile(x, 0.25, axis=1, keepdim=True)
        q3 = torch.quantile(x, 0.75, axis=1, keepdim=True)
        iqr = q3 - q1
        standardized_x = (x - median) / (iqr + eps)
    
    else:
        # 如果有其它情况，如更多维度，可根据需求自行扩展
        raise ValueError(f"Expected 1D or 2D tensor, got shape {x.shape}")

    # 对标准化后的值进行截断，确保它们在[-clip_threshold, clip_threshold]范围内
    standardized_x = torch.clamp(standardized_x, min=-clip_threshold, max=clip_threshold)
    
    return standardized_x

def compute_compression_factor(hidden_size: int, num_heads: int, up_gate_down: int = 3) -> float:
    """
    根据 FLAP 思路计算“剪掉一个 head”与“剪掉一个 MLP neuron”之间的参数量比值 (cost)。
    - 注意力头: 大致 param ~ 4 * hidden_size^2 / num_heads  (Q,K,V,O)
    - MLP neuron: ~ 3 * hidden_size (up, gate, down)
    => factor = (4 * hidden_size^2 / num_heads) / (3 * hidden_size)
              = (4/3) * (hidden_size/num_heads)
    """
    return (4.0 / 3.0) * (hidden_size / num_heads)

def generate_al_am_masks_global(
    scores_dict: Dict[int, Dict[str, torch.Tensor]],
    hidden_size: int,
    num_heads: int,
    pruning_ratio: float
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    复刻 FLAP 在“AL-AM”（Across Layers + Across Modules）全局剪枝的思路：
      - 对 score 做标准化(可选)并收集
      - 分离 score 与 cost
      - 按 score descending 排序
      - 在此顺序下 cumsum cost, 直到保留 (1 - pruning_ratio)*总cost
      - 超过的部分即剪除

    Args:
        scores_dict: {layer_idx: {"attn_scores": [num_heads], "mlp_scores": [intermediate_size]}}
        hidden_size: Transformer 的 hidden_size
        num_heads:   注意力头数
        pruning_ratio: 要剪除多少比例的“加权资源”(0~1)

    Returns:
        (attn_masks, mlp_masks):
          - attn_masks[layer_idx] => bool tensor, shape=[num_heads]
          - mlp_masks[layer_idx]  => bool tensor, shape=[intermediate_size]
    """
    layer_indices = sorted(scores_dict.keys())
    attn_list = []
    mlp_list  = []

    layer_head_count = {}
    layer_mlp_count  = {}

    # 1) 先把各层 attn_scores, mlp_scores 收集到 2D 矩阵
    for layer_idx_ in layer_indices:
        attn_s = scores_dict[layer_idx_]["attn_scores"]  # shape=[num_heads]
        mlp_s  = scores_dict[layer_idx_]["mlp_scores"]   # shape=[intermediate_size]

        layer_head_count[layer_idx_] = attn_s.shape[0]
        layer_mlp_count[layer_idx_]  = mlp_s.shape[0]

        attn_list.append(attn_s)
        mlp_list.append(mlp_s)

    attn_mat = torch.stack(attn_list)  # shape [L, num_heads]
    mlp_mat  = torch.stack(mlp_list)   # shape [L, mlp_size]
    # has_inf = torch.isinf(mlp_mat).any()
    # print(mlp_mat.shape,mlp_mat)

    # 2) 分别标准化
    attn_mat = standardize_scores(attn_mat)
    mlp_mat  = standardize_scores(mlp_mat)
    # print(attn_mat.shape,attn_mat)
    # print(mlp_mat.shape,mlp_mat)

    # 3) 设置 cost
    # 注意力 head cost: ~ (4/3)*(hidden_size/num_heads)
    # MLP channel cost: ~ 1.0 (或另一个常量)
    head_cost = compute_compression_factor(hidden_size, num_heads)
    channel_cost = 1.0

    # 4) 构建大列表: big_scores, big_costs, big_indices
    big_scores = []
    big_costs  = []
    big_indices= []

    # print("attn scores:",)
    # 收集注意力
    for i, row_vec in enumerate(attn_mat):
        l_idx = layer_indices[i]
        num_h = layer_head_count[l_idx]
        for local_h in range(num_h):
            s_val = row_vec[local_h].item()  # score
            # print("layer:", l_idx, "haed:",local_h, "score:",s_val)
            big_scores.append(s_val)
            big_costs.append(head_cost)      # cost
            big_indices.append((True, l_idx, local_h))  # 记录(是attn, 层id, head_idx)

    # 收集 MLP
    # print("mlp scores:",)
    for i, row_vec in enumerate(mlp_mat):
        l_idx = layer_indices[i]
        num_c = layer_mlp_count[l_idx]
        for local_m in range(num_c):
            s_val = row_vec[local_m].item()
            # print("layer:", l_idx, "mlp conut:",local_m, "score:",s_val)
            big_scores.append(s_val)
            big_costs.append(channel_cost)
            big_indices.append((False, l_idx, local_m))

    big_scores = torch.tensor(big_scores)
    big_costs  = torch.tensor(big_costs)

    # 5) 按 score descending 排序
    sorted_scores, sorted_ids = torch.sort(big_scores, descending=True)
    sorted_costs = big_costs[sorted_ids]

    # 6) cumsum cost，保留 (1 - pruning_ratio)*total_cost
    cumsum_cost = torch.cumsum(sorted_costs, dim=0)
    total_cost  = cumsum_cost[-1]
    target_cost = (1.0 - pruning_ratio) * total_cost

    keep_idx = torch.searchsorted(cumsum_cost, target_cost)
    if keep_idx >= len(sorted_scores):
        keep_idx = len(sorted_scores) - 1

    # 前 keep_idx+1 都保留
    cost_threshold_rank = keep_idx.item()

    # 7) 生成 mask
    attn_masks = {}
    mlp_masks  = {}
    for l_ in layer_indices:
        attn_masks[l_] = torch.ones(layer_head_count[l_], dtype=torch.bool)
        mlp_masks[l_]  = torch.ones(layer_mlp_count[l_], dtype=torch.bool)

    # rank_i > cost_threshold_rank 的剪掉
    for rank_i, sid in enumerate(sorted_ids):
        is_attn, l_idx, local_idx = big_indices[sid.item()]
        if rank_i > cost_threshold_rank:
            if is_attn:
                attn_masks[l_idx][local_idx] = False
            else:
                mlp_masks[l_idx][local_idx] = False

    return attn_masks, mlp_masks

def generate_masks_for_all_layers(
    scores_dict: Dict[int, Dict[str, torch.Tensor]],
    structure: str,
    pruning_ratio: float,
    hidden_size: int = None,
    num_heads: int = None,
    total_layers: int = None,
    logistic_k: float = 1.2,
    logistic_x0: float = 0.3,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    主函数：根据结构化剪枝模式 (UL-UM, AL-AM, UL-LD)，
    对所有层的分数进行剪枝并返回注意力和MLP的mask字典。

    Args:
        scores_dict: 由 compute_scores.py 返回的分数字典,
          格式: {layer_idx: {"attn_scores": [...], "mlp_scores": [...]}}

        structure: 剪枝模式:
          - "UL-UM": 逐层局部剪枝
          - "AL-AM": 跨层跨模块全局剪枝
          - "UL-LD": 逐层局部剪枝，但每层稀疏度不同（示例实现）
        pruning_ratio: 若结构是 "UL-UM" 或 "AL-AM" 的统一剪枝比例 (0~1)。
                       若是 "UL-LD"，则可理解为“全局平均目标” G。
        hidden_size: 若结构是 AL-AM，需要计算 cost, 必须给定 hidden_size
        num_heads:   若结构是 AL-AM，需要计算 cost, 必须给定 num_heads

        total_layers: 仅在 "UL-LD" 用到，告诉我们实际有多少层，用于 logistic 函数
        logistic_k, logistic_x0: Logistic 函数参数

    Returns:
        (attn_masks, mlp_masks):
          - attn_masks[layer_idx]: shape=[num_heads], bool
          - mlp_masks[layer_idx]:  shape=[intermediate_size], bool
    """
    # 初始化空
    attn_masks = {}
    mlp_masks  = {}

    if structure == "UL-UM":
        # 逐层局部剪枝（全局同一 ratio）
        for layer_idx, data in scores_dict.items():
            a_scores = data["attn_scores"]
            m_scores = data["mlp_scores"]
            a_mask, m_mask = generate_ul_um_masks(a_scores, m_scores, pruning_ratio)
            attn_masks[layer_idx] = a_mask
            mlp_masks[layer_idx]  = m_mask

    elif structure == "AL-AM":
        # 跨层、跨模块全局剪枝 (FLAP)
        if hidden_size is None or num_heads is None:
            raise ValueError("AL-AM requires hidden_size and num_heads to compute cost.")
        attn_masks, mlp_masks = generate_al_am_masks_global(
            scores_dict, hidden_size, num_heads, pruning_ratio
        )
    
    elif structure == "UL-LD":
        # 新增：逐层局部 + 每层不同剪枝比例

        # 1) 假设 pruning_ratio 这里表示全局平均目标 G
        G = pruning_ratio

        # 2) 我们需要事先知道总层数 total_layers
        if total_layers is None:
            # 也可尝试用 len(scores_dict.keys()) 获取，但有时模型可能并非 1~L 连续索引
            raise ValueError("UL-LD requires `total_layers` to compute layerwise ratio.")

        # 3) 用 logistic 函数生成各层的稀疏度
        #    例如 layer_idx 是从 0 开始，logistic 里是 1..L，则需要做一点映射
        from math import isclose
        rho_list, _ = layer_sparsities_with_logistic(
            L=total_layers,
            G=G,
            k=logistic_k,
            x0=logistic_x0
        )
        # 记得 clip 到 [0,1]
        rho_list = [max(0.0, min(r, 1.0)) for r in rho_list]

        # 4) 逐层应用 local 剪枝
        #    假设 scores_dict 的 key 正好是 0 ~ total_layers-1
        for layer_idx, data in scores_dict.items():
            a_scores = data["attn_scores"]
            m_scores = data["mlp_scores"]

            # layer_idx 可能从 0 开始，所以对应 rho_list[layer_idx]
            local_ratio = rho_list[layer_idx]

            # 用本地函数按 local_ratio 剪枝
            a_mask, m_mask = generate_ul_um_masks(a_scores, m_scores, local_ratio)

            attn_masks[layer_idx] = a_mask
            mlp_masks[layer_idx]  = m_mask

    else:
        raise ValueError(f"Unsupported structure: {structure}. "
                         f"Only 'UL-UM', 'AL-AM' or 'UL-LD' are allowed.")

    return attn_masks, mlp_masks

def save_masks_to_file(
    attn_masks: Dict[int, torch.Tensor],
    mlp_masks:  Dict[int, torch.Tensor],
    file_path: str
):
    """
    将注意力和 MLP 的 mask 字典保存到文件。
    file_path 例如 "mask_dir/masks.pt"
    """
    to_save = {
        "attn_masks": {k: v.cpu() for k, v in attn_masks.items()},
        "mlp_masks":  {k: v.cpu() for k, v in mlp_masks.items()}
    }
    torch.save(to_save, file_path)
    print(f"[save_masks_to_file] Saved masks to {file_path}")

def load_masks_from_file(file_path: str) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    从文件加载 mask 信息，返回 (attn_masks, mlp_masks) 两个字典。
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
    mlp_masks: Dict[int, torch.Tensor]
) -> Dict[int, Dict[str, float]]:
    """
    计算每一层注意力头和 MLP 通道的稀疏度(剪枝比例)。
    
    Args:
        attn_masks: {layer_idx: torch.Tensor(bool), shape=[num_heads] or [heads_kept]}
        mlp_masks:  {layer_idx: torch.Tensor(bool), shape=[intermediate_size]}
    
    Returns:
        一个字典，每个 layer_idx 对应一个子字典:
          {
              "attn_sparsity": <float, 0~1>,
              "mlp_sparsity":  <float, 0~1>
          }
        其中 attn_sparsity 表示该层注意力头被剪掉的比例，mlp_sparsity 表示 MLP 通道被剪掉的比例。
    """
    results = {}
    layer_ids = sorted(attn_masks.keys())
    for layer_idx in layer_ids:
        # 注意力头 mask
        attn_mask = attn_masks[layer_idx]
        total_heads = attn_mask.numel()
        kept_heads = attn_mask.sum().item()   # True 的数量
        attn_sparsity = 1.0 - (kept_heads / total_heads)  # 被剪比例

        # MLP 通道 mask
        mlp_mask = mlp_masks[layer_idx]
        total_mlp = mlp_mask.numel()
        kept_mlp = mlp_mask.sum().item()
        mlp_sparsity = 1.0 - (kept_mlp / total_mlp)

        results[layer_idx] = {
            "attn_sparsity": attn_sparsity,
            "mlp_sparsity":  mlp_sparsity
        }
    return results
