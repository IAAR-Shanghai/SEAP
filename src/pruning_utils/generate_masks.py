# /src/pruning_utils/generate_masks.py

"""
This file is responsible for generating the final Boolean masks (attn_mask and mlp_mask) for pruning attention heads and MLP channels.
These masks are based on pruning scores (scores_dict) calculated in compute_scores.py, combined with user-defined pruning strategies.
For this version, only the FLAP pruning strategies, specifically "UL-UM" (layer-wise pruning), are supported.
The masks can also be saved or loaded to/from files.

Key pruning strategy:
- UL-UM: Unstructured pruning at the layer level, pruning lowest-scoring attention heads and MLP channels.
"""

import torch
import math
import os
from typing import Dict, Tuple, Any

def layer_sparsities_with_logistic(L, G, m=2, n=2, k=1.0, x0=0.3, 
                                   lambda_lower=0.0, lambda_upper=2.0, 
                                   max_iter=100, tol=1e-4):
    """
    Performs numerical search for Lambda to make the discrete average sparsity of layers approximately equal to G.
    The first 'm' and last 'n' layers have their sparsity set to 0 (not pruned).
    
    Args:
    L (int): Total number of layers (assumed to be indexed from 1 to L).
    G (float): Global target average sparsity (0 <= G <= 1).
    m (int): Number of initial layers that are not pruned (default is 2).
    n (int): Number of last layers that are not pruned (default is 2).
    k (float): Steepness of the logistic function curve (default is 1.0).
    x0 (float): The midpoint of the logistic curve (0 <= x0 <= 1).
    lambda_lower (float): Lower bound of Lambda during the search (default is 0.0).
    lambda_upper (float): Upper bound of Lambda during the search (default is 2.0).
    max_iter (int): Maximum number of iterations for binary search (default is 100).
    tol (float): Convergence tolerance, the search stops when the difference between average sparsity and G is smaller than this value (default is 1e-4).

    Returns:
    ----------
    rho_list (list of float): List of sparsity values for each layer [rho_1, rho_2, ..., rho_L].
    Lambda_star (float): The optimal Lambda value that results in the target average sparsity.
    """
    
    def average_sparsity(Lambda_):
        """
        Given a Lambda value, compute the average sparsity across the layers, excluding the first 'm' and last 'n' layers.
        
        Args:
        Lambda_ (float): The current Lambda value for the logistic function.

        Returns:
        float: The average sparsity of layers.
        """
        rho_vals = []
        active_layers = L - m - n  # Number of effective layers excluding protected layers
        for ell in range(1, L + 1):
            # Skip the first 'm' layers and the last 'n' layers
            if ell <= m or ell > (L - n):
                rho_vals.append(0.0)  # Protected layers have sparsity of 0
                continue
            
            # Normalize the layer index 'ell' for the logistic function
            x_ell = (ell - 1 - m) / (L - m - n - 1) if active_layers > 1 else 0.0
            rho_val = Lambda_ / (1.0 + math.exp(-k * (x_ell - x0)))  # Apply logistic function
            rho_vals.append(rho_val)
        
        # Return the average sparsity over the active layers
        return sum(rho_vals) / active_layers
    
    # If G is very close to 0 or 1, return all 0s or all 1s for sparsity
    if abs(G) < 1e-12:
        return [0.0] * L, 0.0
    if abs(G - 1.0) < 1e-12:
        return [1.0] * L, 1.0
    
    # Perform binary search for Lambda between lambda_lower and lambda_upper
    low, high = lambda_lower, lambda_upper
    best_Lambda = (low + high) / 2.0
    
    for _ in range(max_iter):
        mid = (low + high) / 2.0
        avg_sp = average_sparsity(mid)
        
        # Check if average sparsity is close enough to target G
        if abs(avg_sp - G) < tol:
            best_Lambda = mid
            break
        
        # Adjust search bounds based on the computed sparsity
        if avg_sp > G:
            high = mid
        else:
            low = mid
        
        best_Lambda = mid
    
    # Final computation of sparsity for each layer using the best Lambda found
    rho_list = []
    active_layers = L - m - n  # Number of effective layers excluding protected layers
    for ell in range(1, L + 1):
        if ell <= m or ell > (L - n):
            rho_list.append(0.0)  # Protected layers have sparsity of 0
        else:
            x_ell = (ell - 1 - m) / (L - m - n - 1) if active_layers > 1 else 0.0
            rho_val = best_Lambda / (1.0 + math.exp(-k * (x_ell - x0)))  # Apply logistic function
            rho_list.append(rho_val)
    
    return rho_list, best_Lambda

def generate_ul_um_masks(
    attn_scores: torch.Tensor,
    mlp_scores: torch.Tensor,
    pruning_ratio: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates the "UL-UM" (layer-wise unstructured pruning) masks:
      - Prunes the attention heads and MLP channels by sorting the scores and selecting the lowest-pruned scores
      - Retains the top (1-pruning_ratio)*num_heads for attention and (1-pruning_ratio)*intermediate_size for MLP.

    Args:
        attn_scores (torch.Tensor): Tensor of attention head scores with shape [num_heads].
        mlp_scores (torch.Tensor): Tensor of MLP channel scores with shape [intermediate_size].
        pruning_ratio (float): The proportion of layers to prune, in the range [0, 1].

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two Boolean tensors representing the masks for attention heads and MLP channels.
            - attn_mask (bool): Mask for attention heads (True = keep, False = prune).
            - mlp_mask (bool): Mask for MLP channels (True = keep, False = prune).
    """
    device = attn_scores.device
    num_heads = attn_scores.shape[0]
    inter_dim = mlp_scores.shape[0]

    # Initialize masks to keep all heads and channels
    attn_mask = torch.ones(num_heads, dtype=torch.bool, device=device)
    mlp_mask = torch.ones(inter_dim, dtype=torch.bool, device=device)

    # Select the number of heads to prune
    head_prune_num = int(num_heads * pruning_ratio)
    sorted_heads, idx_heads = torch.sort(attn_scores)  # Sort in ascending order
    heads_to_prune = idx_heads[:head_prune_num]  # Select the lowest-scoring heads to prune
    attn_mask[heads_to_prune] = False  # Set them as False in the mask

    # Select the number of MLP channels to prune
    mlp_prune_num = int(inter_dim * pruning_ratio)
    sorted_mlp, idx_mlp = torch.sort(mlp_scores)  # Sort in ascending order
    mlp_to_prune = idx_mlp[:mlp_prune_num]  # Select the lowest-scoring MLP channels to prune
    mlp_mask[mlp_to_prune] = False  # Set them as False in the mask

    return attn_mask, mlp_mask

def standardize_scores(x: torch.Tensor, eps=1e-9, clip_threshold=3.0) -> torch.Tensor:
    """
    Perform robust standardization on a 1D or 2D tensor using the median and IQR, with optional clipping.
    After standardization, ensure all values fall within the range [-clip_threshold, clip_threshold].
    
    - For 1D tensors: Global median/IQR is used.
    - For 2D tensors: Each row (corresponding to a layer) is standardized individually.

    Args:
        x (torch.Tensor): Input tensor of shape [num_layers, num_items] (2D) or [num_items] (1D).
        eps (float): Small value to avoid division by zero (default 1e-9).
        clip_threshold (float): Maximum allowed value after standardization (default 3.0).

    Returns:
        torch.Tensor: Standardized and clipped tensor.
    """
    if x.dim() == 1:
        # For 1D tensor, standardize globally using median and IQR
        median = x.median()
        q1 = torch.quantile(x, 0.25)
        q3 = torch.quantile(x, 0.75)
        iqr = q3 - q1
        standardized_x = (x - median) / (iqr + eps)
    
    elif x.dim() == 2:
        # For 2D tensor, standardize each row individually (per layer)
        median = torch.median(x, axis=1, keepdim=True).values
        q1 = torch.quantile(x, 0.25, axis=1, keepdim=True)
        q3 = torch.quantile(x, 0.75, axis=1, keepdim=True)
        iqr = q3 - q1
        standardized_x = (x - median) / (iqr + eps)
    
    else:
        raise ValueError(f"Expected 1D or 2D tensor, got shape {x.shape}")

    # Clip the standardized values to be within [-clip_threshold, clip_threshold]
    standardized_x = torch.clamp(standardized_x, min=-clip_threshold, max=clip_threshold)
    
    return standardized_x

def compute_compression_factor(hidden_size: int, num_heads: int, up_gate_down: int = 3) -> float:
    """
    Computes the compression factor (cost) between "pruning one head" and "pruning one MLP neuron" 
    based on the FLAP pruning approach.
    
    - Attention head cost: approximately ~ 4 * hidden_size^2 / num_heads (Q, K, V, O).
    - MLP neuron cost: approximately ~ 3 * hidden_size (up, gate, down).
    
    Formula: 
    cost_factor = (4 * hidden_size^2 / num_heads) / (3 * hidden_size) = (4/3) * (hidden_size / num_heads)

    Args:
        hidden_size (int): The hidden size of the transformer model.
        num_heads (int): The number of attention heads.

    Returns:
        float: The cost factor representing the ratio of pruning an attention head to pruning an MLP neuron.
    """
    return (4.0 / 3.0) * (hidden_size / num_heads)

def generate_al_am_masks_global(
    scores_dict: Dict[int, Dict[str, torch.Tensor]],
    hidden_size: int,
    num_heads: int,
    pruning_ratio: float
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    """
    Implements the FLAP global pruning approach for "AL-AM" (Across Layers + Across Modules).
    - Standardizes the scores (optional), collects and separates scores from their costs.
    - Sorts the scores in descending order.
    - Computes the cumulative cost and retains (1 - pruning_ratio) * total cost.
    - Prunes the rest.

    Args:
        scores_dict (Dict[int, Dict[str, torch.Tensor]]): A dictionary containing the attention and MLP scores for each layer.
            Example format: {layer_idx: {"attn_scores": [num_heads], "mlp_scores": [intermediate_size]}}
        hidden_size (int): The hidden size of the transformer model.
        num_heads (int): The number of attention heads.
        pruning_ratio (float): The proportion of weighted resources to prune (0~1).

    Returns:
        Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
            - attn_masks: A dictionary of attention masks for each layer (bool tensor of shape [num_heads]).
            - mlp_masks: A dictionary of MLP masks for each layer (bool tensor of shape [intermediate_size]).
    """
    layer_indices = sorted(scores_dict.keys())
    attn_list = []
    mlp_list  = []

    layer_head_count = {}
    layer_mlp_count  = {}

    # 1) Collect attention and MLP scores into 2D matrices.
    for layer_idx_ in layer_indices:
        attn_s = scores_dict[layer_idx_]["attn_scores"]
        mlp_s  = scores_dict[layer_idx_]["mlp_scores"]

        layer_head_count[layer_idx_] = attn_s.shape[0]
        layer_mlp_count[layer_idx_]  = mlp_s.shape[0]

        attn_list.append(attn_s)
        mlp_list.append(mlp_s)

    attn_mat = torch.stack(attn_list)
    mlp_mat  = torch.stack(mlp_list)

    # 2) Standardize the scores
    attn_mat = standardize_scores(attn_mat)
    mlp_mat  = standardize_scores(mlp_mat)

    # 3) Set the cost factors
    head_cost = compute_compression_factor(hidden_size, num_heads)
    channel_cost = 1.0

    # 4) Build big lists: big_scores, big_costs, big_indices
    big_scores = []
    big_costs  = []
    big_indices= []

    # Collect attention scores and their costs
    for i, row_vec in enumerate(attn_mat):
        l_idx = layer_indices[i]
        num_h = layer_head_count[l_idx]
        for local_h in range(num_h):
            s_val = row_vec[local_h].item()
            big_scores.append(s_val)
            big_costs.append(head_cost)
            big_indices.append((True, l_idx, local_h))

    # Collect MLP scores and their costs
    for i, row_vec in enumerate(mlp_mat):
        l_idx = layer_indices[i]
        num_c = layer_mlp_count[l_idx]
        for local_m in range(num_c):
            s_val = row_vec[local_m].item()
            big_scores.append(s_val)
            big_costs.append(channel_cost)
            big_indices.append((False, l_idx, local_m))

    big_scores = torch.tensor(big_scores)
    big_costs  = torch.tensor(big_costs)

    # 5) Sort scores in descending order
    sorted_scores, sorted_ids = torch.sort(big_scores, descending=True)
    sorted_costs = big_costs[sorted_ids]

    # 6) Cumulative sum of costs, keeping (1 - pruning_ratio) * total_cost
    cumsum_cost = torch.cumsum(sorted_costs, dim=0)
    total_cost  = cumsum_cost[-1]
    target_cost = (1.0 - pruning_ratio) * total_cost

    keep_idx = torch.searchsorted(cumsum_cost, target_cost)
    if keep_idx >= len(sorted_scores):
        keep_idx = len(sorted_scores) - 1

    cost_threshold_rank = keep_idx.item()

    # 7) Generate the masks
    attn_masks = {}
    mlp_masks  = {}
    for l_ in layer_indices:
        attn_masks[l_] = torch.ones(layer_head_count[l_], dtype=torch.bool)
        mlp_masks[l_]  = torch.ones(layer_mlp_count[l_], dtype=torch.bool)

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
    Main function: Prune attention and MLP masks across all layers according to the given pruning structure (UL-UM, AL-AM, UL-LD).
    
    Args:
        scores_dict (Dict[int, Dict[str, torch.Tensor]]): The dictionary of scores (attn_scores and mlp_scores) returned by compute_scores.py.
        structure (str): The pruning structure to use:
          - "UL-UM": Unstructured pruning per layer.
          - "AL-AM": Across layers and across modules (global pruning).
          - "UL-LD": Unstructured pruning per layer with layer-specific sparsity.
        pruning_ratio (float): The global pruning ratio for "UL-UM" and "AL-AM" (0~1), or the global sparsity target G for "UL-LD".
        hidden_size (int, optional): The transformer hidden size, required for "AL-AM" pruning.
        num_heads (int, optional): The number of attention heads, required for "AL-AM" pruning.
        total_layers (int, optional): Total number of layers, required for "UL-LD" pruning.
        logistic_k (float, optional): The steepness parameter for the logistic function (default 1.2).
        logistic_x0 (float, optional): The midpoint for the logistic function (default 0.3).

    Returns:
        Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
            - attn_masks: A dictionary of attention masks (bool tensor of shape [num_heads]).
            - mlp_masks: A dictionary of MLP masks (bool tensor of shape [intermediate_size]).
    """
    attn_masks = {}
    mlp_masks  = {}

    if structure == "UL-UM":
        # Local pruning per layer with the same pruning ratio across all layers
        for layer_idx, data in scores_dict.items():
            a_scores = data["attn_scores"]
            m_scores = data["mlp_scores"]
            a_mask, m_mask = generate_ul_um_masks(a_scores, m_scores, pruning_ratio)
            attn_masks[layer_idx] = a_mask
            mlp_masks[layer_idx]  = m_mask

    elif structure == "AL-AM":
        # Global pruning across layers and modules (FLAP)
        if hidden_size is None or num_heads is None:
            raise ValueError("AL-AM requires hidden_size and num_heads to compute cost.")
        attn_masks, mlp_masks = generate_al_am_masks_global(
            scores_dict, hidden_size, num_heads, pruning_ratio
        )
    
    elif structure == "UL-LD":
        # Layer-wise unstructured pruning with different pruning ratios per layer (using logistic function)
        if total_layers is None:
            raise ValueError("UL-LD requires `total_layers` to compute layerwise ratio.")
        
        rho_list, _ = layer_sparsities_with_logistic(
            L=total_layers,
            G=pruning_ratio,
            k=logistic_k,
            x0=logistic_x0
        )
        rho_list = [max(0.0, min(r, 1.0)) for r in rho_list]

        for layer_idx, data in scores_dict.items():
            a_scores = data["attn_scores"]
            m_scores = data["mlp_scores"]

            local_ratio = rho_list[layer_idx]
            a_mask, m_mask = generate_ul_um_masks(a_scores, m_scores, local_ratio)

            attn_masks[layer_idx] = a_mask
            mlp_masks[layer_idx]  = m_mask

    else:
        raise ValueError(f"Unsupported structure: {structure}. Only 'UL-UM', 'AL-AM', or 'UL-LD' are allowed.")

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
    mlp_masks: Dict[int, torch.Tensor]
) -> Dict[int, Dict[str, float]]:
    """
    Compute the sparsity (pruning ratio) for each layer's attention heads and MLP channels.

    Args:
        attn_masks (Dict[int, torch.Tensor]): Attention masks for each layer.
        mlp_masks (Dict[int, torch.Tensor]): MLP masks for each layer.

    Returns:
        Dict[int, Dict[str, float]]: A dictionary for each layer index containing:
          {
              "attn_sparsity": <float, 0~1>,  # proportion of attention heads pruned.
              "mlp_sparsity":  <float, 0~1>   # proportion of MLP channels pruned.
          }
    """
    results = {}
    layer_ids = sorted(attn_masks.keys())
    for layer_idx in layer_ids:
        # Calculate attention sparsity
        attn_mask = attn_masks[layer_idx]
        total_heads = attn_mask.numel()
        kept_heads = attn_mask.sum().item()
        attn_sparsity = 1.0 - (kept_heads / total_heads)

        # Calculate MLP sparsity
        mlp_mask = mlp_masks[layer_idx]
        total_mlp = mlp_mask.numel()
        kept_mlp = mlp_mask.sum().item()
        mlp_sparsity = 1.0 - (kept_mlp / total_mlp)

        results[layer_idx] = {
            "attn_sparsity": attn_sparsity,
            "mlp_sparsity":  mlp_sparsity
        }
    return results
