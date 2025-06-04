"""
Model pruning and layer analysis utilities.

This module provides utilities for analyzing transformer model layers through
pruning experiments, including layer-wise similarity analysis and visualization
of pruning effects.

Author: why
Date: 2024
"""

# Standard library imports
import os
import sys
import json
from typing import Dict, List, Tuple, Any, Optional

# Third-party imports
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Add project root to sys.path for module imports
project_root = os.path.abspath("..")
sys.path.append(project_root)

# Local imports
from src.pruning_utils.apply_pruning import apply_pruning_to_model


def compute_layer_cos_sims(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str]
) -> List[float]:
    """Compute average cosine similarity between layer inputs and outputs.
    
    Args:
        model: Loaded Huggingface model (e.g., LlamaForCausalLM)
        tokenizer: Corresponding tokenizer
        texts: List of input texts
        
    Returns:
        List of average cosine similarities for each layer
    """
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    inputs = tokenizer(texts, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # (embeddings, layer1, ..., layerN)

    layer_sims = []
    for i in range(len(hidden_states) - 1):
        h_in = hidden_states[i][:, -1, :]
        h_out = hidden_states[i + 1][:, -1, :]
        sim = F.cosine_similarity(h_in, h_out, dim=-1).mean().item()
        layer_sims.append(sim)

    return layer_sims


def generate_random_masks_for_layer(
    model: torch.nn.Module,
    layer_idx: int,
    target_pruning_ratio: float,
    cumulative_attn_mask: Optional[torch.Tensor] = None,
    cumulative_mlp_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random pruning masks for attention heads and MLP units.
    
    Args:
        model: Model instance
        layer_idx: Current layer index
        target_pruning_ratio: Pruning ratio (0 to 1)
        cumulative_attn_mask: Cumulative attention mask from previous iterations
        cumulative_mlp_mask: Cumulative MLP mask from previous iterations
        
    Returns:
        Tuple of boolean masks for attention heads and MLP units
    """
    layer = model.model.layers[layer_idx]
    device = next(layer.parameters()).device

    num_heads = model.config.num_attention_heads
    intermediate_size = layer.mlp.gate_proj.weight.shape[0]

    attn_mask = (cumulative_attn_mask.clone() if cumulative_attn_mask is not None 
                 else torch.ones(num_heads, dtype=torch.bool, device=device))
    mlp_mask = (cumulative_mlp_mask.clone() if cumulative_mlp_mask is not None 
                else torch.ones(intermediate_size, dtype=torch.bool, device=device))

    num_attn_to_keep = int(num_heads * (1.0 - target_pruning_ratio))
    num_mlp_to_keep = int(intermediate_size * (1.0 - target_pruning_ratio))

    # Update attention mask
    current_attn_indices = torch.nonzero(attn_mask, as_tuple=True)[0]
    if current_attn_indices.numel() > num_attn_to_keep:
        prune_indices = current_attn_indices[torch.randperm(
            current_attn_indices.numel())[:current_attn_indices.numel() - num_attn_to_keep]]
        attn_mask[prune_indices] = False

    # Update MLP mask
    current_mlp_indices = torch.nonzero(mlp_mask, as_tuple=True)[0]
    if current_mlp_indices.numel() > num_mlp_to_keep:
        prune_indices = current_mlp_indices[torch.randperm(
            current_mlp_indices.numel())[:current_mlp_indices.numel() - num_mlp_to_keep]]
        mlp_mask[prune_indices] = False

    return attn_mask, mlp_mask


@torch.no_grad()
def run_layerwise_remove_test(
    model: torch.nn.Module,
    tokenizer,
    text_list: List[str],
    prune_ratios: List[float],
    use_softmask: bool = True
) -> Dict[int, List[Tuple[float, float]]]:
    """Run layer-wise pruning tests and record output similarities.
    
    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        text_list: List of input texts
        prune_ratios: List of pruning ratios to test
        use_softmask: Whether to use soft masking (unstructured)
        
    Returns:
        Dictionary mapping layer indices to lists of (ratio, similarity) pairs
    """
    inputs = tokenizer(text_list, return_tensors="pt", padding=True)
    position_ids = inputs["attention_mask"].cumsum(dim=1) - 1  # [B, L]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        original_hidden_states = outputs.hidden_states
        original_final_output = original_hidden_states[-1][:, -1, :]  # [B, D]

    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    head_dim = hidden_size // num_heads

    all_results = {}

    for layer_idx in range(num_layers):
        print(f"\n=== Remove Test for Layer {layer_idx} ===")
        results = []

        layer_input = original_hidden_states[layer_idx].detach()
        cumulative_attn_mask = None
        cumulative_mlp_mask = None

        for ratio in prune_ratios:
            attn_mask, mlp_mask = generate_random_masks_for_layer(
                model=model,
                layer_idx=layer_idx,
                target_pruning_ratio=ratio,
                cumulative_attn_mask=cumulative_attn_mask,
                cumulative_mlp_mask=cumulative_mlp_mask
            )

            cumulative_attn_mask = attn_mask
            cumulative_mlp_mask = mlp_mask

            apply_pruning_to_model(
                model=model,
                attn_masks={layer_idx: attn_mask},
                mlp_masks={layer_idx: mlp_mask},
                unstr=use_softmask,
                head_dim=head_dim
            )

            # Forward from current layer to final layer
            h = layer_input
            for i in range(layer_idx, num_layers):
                h = model.model.layers[i](
                    hidden_states=h,
                    position_ids=position_ids
                )[0]

            final_output = h[:, -1, :]
            ref = original_final_output.to(final_output.device)
            sim = F.cosine_similarity(final_output, ref, dim=-1).mean().item()

            print(f"[Layer {layer_idx} | Ratio={ratio:.1f}] CosSim = {sim:.4f} | "
                  f"Kept heads: {attn_mask.sum().item()}, MLP: {mlp_mask.sum().item()}")
            results.append((ratio, sim))

        all_results[layer_idx] = results

    return all_results


def plot_remove_test(
    results_by_layer: Dict[int, List[Tuple[float, float]]],
    ratios_to_plot: List[float]
) -> None:
    """Plot line graphs of layer similarities at different pruning ratios.
    
    Args:
        results_by_layer: Dictionary of pruning results by layer
        ratios_to_plot: List of pruning ratios to include in plot
    """
    layers = sorted(results_by_layer.keys())
    for ratio in ratios_to_plot:
        sims = []
        for layer in layers:
            sim = next((s for r, s in results_by_layer[layer] 
                       if abs(r - ratio) < 1e-6), None)
            sims.append(sim)
        plt.plot(layers, sims, marker='o', label=f"Ratio={ratio:.1f}")

    plt.xlabel("Layer Index")
    plt.ylabel("Cosine Similarity")
    plt.title("Layerwise Remove Test (Softmask)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def logistic_fn(x: float, a: float, b: float) -> float:
    """Logistic function for curve fitting.
    
    Args:
        x: Input value
        a: Slope parameter
        b: Offset parameter
        
    Returns:
        Logistic function value
    """
    return 1 / (1 + np.exp(-a * x + b))


def remove_outliers(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "iqr",
    threshold: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove outliers from data using specified method.
    
    Args:
        x: Input array
        y: Output array
        method: Outlier detection method ('iqr' or 'std')
        threshold: Threshold for outlier detection
        
    Returns:
        Tuple of cleaned x and y arrays
        
    Raises:
        ValueError: If invalid method is specified
    """
    mask = ~np.isnan(y)
    x, y = x[mask], y[mask]

    if method == "iqr":
        q1, q3 = np.percentile(y, [25, 75])
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (y >= lower) & (y <= upper)
    elif method == "std":
        mean = y.mean()
        std = y.std()
        mask = (y >= mean - threshold * std) & (y <= mean + threshold * std)
    else:
        raise ValueError("Invalid method. Use 'iqr' or 'std'.")

    return x[mask], y[mask]


def fit_logistic_curves(
    remove_results: Dict[int, List[Tuple[float, float]]],
    protect_head_layers: int = 2,
    protect_tail_layers: int = 3,
    outlier_method: str = "iqr",
    outlier_threshold: float = 1.5,
    fit_linear: bool = True
) -> Dict[str, Any]:
    """Fit logistic and/or linear curves to layer-wise pruning results.
    
    Args:
        remove_results: Dictionary of pruning results by layer
        protect_head_layers: Number of initial layers to exclude
        protect_tail_layers: Number of final layers to exclude
        outlier_method: Method for outlier detection
        outlier_threshold: Threshold for outlier detection
        fit_linear: Whether to fit linear curve
        
    Returns:
        Dictionary containing:
            - x: Cleaned x values
            - y: Cleaned y values
            - logistic: Logistic fit values (if successful)
            - logistic_params: Logistic parameters (if successful)
            - linear: Linear fit values (if requested)
            - linear_params: Linear parameters (if requested)
    """
    all_layers = sorted(remove_results.keys())
    num_layers = len(all_layers)
    min_layer = protect_head_layers
    max_layer = num_layers - protect_tail_layers - 1

    x_vals, y_vals = [], []
    for layer in range(num_layers):
        if layer < min_layer or layer > max_layer:
            continue
        if layer not in remove_results:
            continue
        sims = [sim for _, sim in remove_results[layer]]
        if len(sims) > 0:
            x_vals.append(layer)
            y_vals.append(np.mean(sims))

    x = np.array(x_vals)
    y = np.array(y_vals)

    # Remove outliers
    x_clean, y_clean = remove_outliers(x, y, method=outlier_method,
                                     threshold=outlier_threshold)
    result = {'x': x_clean, 'y': y_clean}

    # Fit logistic curve
    try:
        popt, _ = curve_fit(logistic_fn, x_clean, y_clean, p0=(0.3, 10))
        a_log, b_log = popt
        y_logistic_pred = logistic_fn(x_clean, a_log, b_log)
        result['logistic'] = y_logistic_pred
        result['logistic_params'] = {"a": float(a_log), "b": float(b_log)}
        print(f"[✓] Logistic Fit: a={a_log:.4f}, b={b_log:.4f}")
    except RuntimeError:
        print("[⚠️] Logistic fit failed")
        result['logistic'] = None
        result['logistic_params'] = None

    # Fit linear curve
    if fit_linear:
        model = LinearRegression().fit(x_clean.reshape(-1, 1), y_clean)
        coef, intercept = model.coef_[0], model.intercept_
        y_linear_pred = model.predict(x_clean.reshape(-1, 1))
        result['linear'] = y_linear_pred
        result['linear_params'] = {"a": float(coef), "b": float(intercept)}
        print(f"[✓] Linear Fit: y = {coef:.4f} * x + {intercept:.4f}")
    else:
        result['linear'] = None
        result['linear_params'] = None

    return result


def plot_fitted_curves(
    fitted_result: Dict[str, Any],
    show_data: bool = True,
    show_logistic: bool = True,
    show_linear: bool = True,
    title: str = "Fitted Curve from Remove Test"
) -> None:
    """Plot fitted curves and data points.
    
    Args:
        fitted_result: Result dictionary from fit_logistic_curves
        show_data: Whether to plot original data points
        show_logistic: Whether to plot logistic curve
        show_linear: Whether to plot linear curve
        title: Plot title
    """
    x = fitted_result["x"]
    y = fitted_result["y"]

    if show_data:
        plt.scatter(x, y, s=30, alpha=0.5, label="Data")

    if show_logistic and fitted_result.get("logistic") is not None:
        plt.plot(x, fitted_result["logistic"], linestyle='--', label="Logistic Fit")

    if show_linear and fitted_result.get("linear") is not None:
        plt.plot(x, fitted_result["linear"], linestyle='-', label="Linear Fit")

    plt.xlabel("Layer Index")
    plt.ylabel("Cosine Similarity")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_layerwise_results(
    model_name: str,
    cos_sims: List[float],
    remove_results: Dict[int, List[Tuple[float, float]]],
    fitted_result: Optional[Dict[str, Any]] = None,
    base_dir: str = "../layer_importance"
) -> None:
    """Save layer-wise analysis results to disk.
    
    Args:
        model_name: Name of the model
        cos_sims: List of cosine similarities
        remove_results: Dictionary of pruning results
        fitted_result: Optional dictionary of curve fitting results
        base_dir: Base directory for saving results
    """
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save cosine similarities
    with open(os.path.join(model_dir, "layer_cos_similarity.json"), "w") as f:
        json.dump({
            "model_name": model_name,
            "cos_sims": {f"layer_{i}": sim for i, sim in enumerate(cos_sims)}
        }, f, indent=2)

    # Save remove test results
    with open(os.path.join(model_dir, "remove_test_results.json"), "w") as f:
        json.dump({
            "model_name": model_name,
            "remove_results": {
                str(layer): [(float(r), float(s)) for r, s in data]
                for layer, data in remove_results.items()
            }
        }, f, indent=2)

    # Save fitted parameters if provided
    if fitted_result:
        if "linear_params" in fitted_result and fitted_result["linear_params"]:
            with open(os.path.join(model_dir, "fitted_meta_linear.json"), "w") as f:
                json.dump({
                    "model_name": model_name,
                    "fit_type": "linear",
                    **fitted_result["linear_params"]
                }, f, indent=2)

        if "logistic_params" in fitted_result and fitted_result["logistic_params"]:
            with open(os.path.join(model_dir, "fitted_meta_logistic.json"), "w") as f:
                json.dump({
                    "model_name": model_name,
                    "fit_type": "logistic",
                    **fitted_result["logistic_params"]
                }, f, indent=2)

    print(f"[✓] Saved all data under: {os.path.abspath(model_dir)}")


def load_layerwise_results(
    model_name: str,
    base_dir: str = "../layer_importance"
) -> Tuple[str, List[float], Dict[int, List[Tuple[float, float]]], Dict[str, Dict[str, float]]]:
    """Load layer-wise analysis results from disk.
    
    Args:
        model_name: Name of the model
        base_dir: Base directory containing results
        
    Returns:
        Tuple containing:
            - Model name
            - List of cosine similarities
            - Dictionary of pruning results
            - Dictionary of fitted parameters
    """
    model_dir = os.path.join(base_dir, model_name)

    # Load cosine similarities
    with open(os.path.join(model_dir, "layer_cos_similarity.json"), "r") as f:
        data = json.load(f)
        model_name_check = data["model_name"]
        cos_sims = [data["cos_sims"][f"layer_{i}"] 
                   for i in range(len(data["cos_sims"]))]

    # Load remove test results
    with open(os.path.join(model_dir, "remove_test_results.json"), "r") as f:
        data = json.load(f)
        assert data["model_name"] == model_name_check
        remove_results = {
            int(layer): [(float(r), float(s)) for r, s in lst]
            for layer, lst in data["remove_results"].items()
        }

    # Load fitted parameters
    fitted_params = {}
    for fit_type in ["linear", "logistic"]:
        fpath = os.path.join(model_dir, f"fitted_meta_{fit_type}.json")
        if os.path.exists(fpath):
            with open(fpath, "r") as f:
                meta = json.load(f)
                fitted_params[f"{fit_type}_params"] = {
                    "a": float(meta["a"]),
                    "b": float(meta["b"])
                }

    print(f"[✓] Loaded layerwise results for model: {model_name_check}")
    return model_name_check, cos_sims, remove_results, fitted_params
