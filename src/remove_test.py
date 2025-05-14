import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

# 添加项目根目录到 sys.path 以便导入模块
project_root = os.path.abspath("..")
sys.path.append(project_root)

# 模型工具 & 剪枝应用函数
from src.pruning_utils.apply_pruning import apply_pruning_to_model


# -----------------------------
# 1. 计算每层隐藏状态之间的余弦相似度
# -----------------------------
def compute_layer_cos_sims(
    model,
    tokenizer,
    texts: List[str],
) -> List[float]:
    """
    计算模型每一层 hidden_states 的输入和输出之间的平均余弦相似度。

    Args:
        model: 已加载的 Huggingface 模型（如 LlamaForCausalLM）
        tokenizer: 对应 tokenizer
        texts (List[str]): 输入文本列表

    Returns:
        List[float]: 每一层 input/output 的平均 cosine 相似度
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


# -----------------------------
# 2. 随机剪枝 mask 生成函数
# -----------------------------
def generate_random_masks_for_layer(
    model,
    layer_idx: int,
    target_pruning_ratio: float,
    cumulative_attn_mask: torch.Tensor = None,
    cumulative_mlp_mask: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    为特定层生成随机 attention head 和 MLP 单元的剪枝 mask。

    Args:
        model: 模型实例
        layer_idx: 当前层索引
        target_pruning_ratio: 剪枝比例
        cumulative_attn_mask: 累积的 attention mask
        cumulative_mlp_mask: 累积的 MLP mask

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: attention 和 MLP 的布尔 mask
    """
    layer = model.model.layers[layer_idx]
    device = next(layer.parameters()).device

    num_heads = model.config.num_attention_heads
    intermediate_size = layer.mlp.gate_proj.weight.shape[0]

    attn_mask = cumulative_attn_mask.clone() if cumulative_attn_mask is not None else torch.ones(num_heads, dtype=torch.bool, device=device)
    mlp_mask = cumulative_mlp_mask.clone() if cumulative_mlp_mask is not None else torch.ones(intermediate_size, dtype=torch.bool, device=device)

    num_attn_to_keep = int(num_heads * (1.0 - target_pruning_ratio))
    num_mlp_to_keep = int(intermediate_size * (1.0 - target_pruning_ratio))

    # 更新 attention mask
    current_attn_indices = torch.nonzero(attn_mask, as_tuple=True)[0]
    if current_attn_indices.numel() > num_attn_to_keep:
        prune_indices = current_attn_indices[torch.randperm(current_attn_indices.numel())[:current_attn_indices.numel() - num_attn_to_keep]]
        attn_mask[prune_indices] = False

    # 更新 MLP mask
    current_mlp_indices = torch.nonzero(mlp_mask, as_tuple=True)[0]
    if current_mlp_indices.numel() > num_mlp_to_keep:
        prune_indices = current_mlp_indices[torch.randperm(current_mlp_indices.numel())[:current_mlp_indices.numel() - num_mlp_to_keep]]
        mlp_mask[prune_indices] = False

    return attn_mask, mlp_mask


# -----------------------------
# 3. 层级剪枝测试主函数
# -----------------------------
@torch.no_grad()
def run_layerwise_remove_test(
    model,
    tokenizer,
    text_list: List[str],
    prune_ratios: List[float],
    use_softmask: bool = True
) -> Dict[int, List[Tuple[float, float]]]:
    """
    对模型逐层剪枝测试，记录每一层不同剪枝比例下的最终输出与原始输出的相似度。

    Args:
        model: 模型实例
        tokenizer: tokenizer 实例
        text_list: 输入文本列表
        prune_ratios: 剪枝比例列表
        use_softmask: 是否使用 softmask（unstructured）

    Returns:
        Dict[int, List[Tuple[float, float]]]: 每层剪枝后的相似度结果
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

            # 从该层 forward 到最后一层
            h = layer_input
            for i in range(layer_idx, num_layers):
                h = model.model.layers[i](
                    hidden_states=h,
                    position_ids=position_ids
                )[0]

            final_output = h[:, -1, :]
            ref = original_final_output.to(final_output.device)
            sim = F.cosine_similarity(final_output, ref, dim=-1).mean().item()

            print(f"[Layer {layer_idx} | Ratio={ratio:.1f}] CosSim = {sim:.4f} | Kept heads: {attn_mask.sum().item()}, MLP: {mlp_mask.sum().item()}")
            results.append((ratio, sim))

        all_results[layer_idx] = results

    return all_results

def plot_remove_test(results_by_layer: Dict[int, List[Tuple[float, float]]],
                     ratios_to_plot: List[float]):
    """
    绘制各剪枝比例下各层相似度的折线图。
    """
    layers = sorted(results_by_layer.keys())
    for ratio in ratios_to_plot:
        sims = []
        for layer in layers:
            sim = next((s for r, s in results_by_layer[layer] if abs(r - ratio) < 1e-6), None)
            sims.append(sim)
        plt.plot(layers, sims, marker='o', label=f"Ratio={ratio:.1f}")

    plt.xlabel("Layer Index")
    plt.ylabel("Cosine Similarity")
    plt.title("Layerwise Remove Test (Softmask)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def logistic_fn(x, a, b):
    return 1 / (1 + np.exp(-a * x + b))


def remove_outliers(x: np.ndarray, y: np.ndarray, method: str = "iqr", threshold: float = 1.5):
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
    """
    使用 remove_results 中每层多个剪枝率下的平均相似度来拟合 logistic 和/或 linear 曲线。
    
    Returns:
        {
            'x': np.ndarray,
            'y': np.ndarray,
            'logistic': np.ndarray or None,
            'logistic_params': Dict[str, float] or None,
            'linear': np.ndarray or None,
            'linear_params': Dict[str, float] or None,
        }
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

    # 过滤离群点
    x_clean, y_clean = remove_outliers(x, y, method=outlier_method, threshold=outlier_threshold)
    result = {'x': x_clean, 'y': y_clean}

    # logistic 拟合
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

    # 线性拟合
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
):
    """
    绘制单个拟合结果的曲线与数据点。
    
    Args:
        fitted_result: 来自 fit_logistic_curves_from_remove_results 的结果字典
        show_data: 是否绘制散点原始点
        show_logistic: 是否绘制 logistic 曲线
        show_linear: 是否绘制线性曲线
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
    fitted_result: Dict[str, Any] = None,
    base_dir: str = "../layer_importance"
):
    import os, json
    model_dir = os.path.join(base_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    # 保存 cos_sims
    with open(os.path.join(model_dir, "layer_cos_similarity.json"), "w") as f:
        json.dump({
            "model_name": model_name,
            "cos_sims": {f"layer_{i}": sim for i, sim in enumerate(cos_sims)}
        }, f, indent=2)

    # 保存 remove_results
    with open(os.path.join(model_dir, "remove_test_results.json"), "w") as f:
        json.dump({
            "model_name": model_name,
            "remove_results": {
                str(layer): [(float(r), float(s)) for r, s in data]
                for layer, data in remove_results.items()
            }
        }, f, indent=2)

    # 保存拟合参数（如果提供）
    if fitted_result:
        if "linear_params" in fitted_result and fitted_result["linear_params"] is not None:
            with open(os.path.join(model_dir, "fitted_meta_linear.json"), "w") as f:
                json.dump({
                    "model_name": model_name,
                    "fit_type": "linear",
                    **fitted_result["linear_params"]
                }, f, indent=2)

        if "logistic_params" in fitted_result and fitted_result["logistic_params"] is not None:
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
    import os, json
    model_dir = os.path.join(base_dir, model_name)

    # 加载 cos_sims
    with open(os.path.join(model_dir, "layer_cos_similarity.json"), "r") as f:
        data = json.load(f)
        model_name_check = data["model_name"]
        cos_sims = [data["cos_sims"][f"layer_{i}"] for i in range(len(data["cos_sims"]))]

    # 加载 remove_results
    with open(os.path.join(model_dir, "remove_test_results.json"), "r") as f:
        data = json.load(f)
        assert data["model_name"] == model_name_check
        remove_results = {
            int(layer): [(float(r), float(s)) for r, s in lst]
            for layer, lst in data["remove_results"].items()
        }

    # 加载拟合参数（logistic 和 linear）
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
