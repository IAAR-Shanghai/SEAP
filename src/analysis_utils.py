"""
Analysis utilities for evaluating language model performance.

This module provides utilities for analyzing and visualizing language model
evaluation results across different tasks and prompting strategies.

Author: why
Date: 2024
"""

# Standard library imports
import os
import json
from pathlib import Path
from glob import glob
from typing import Dict, List, Optional, Union

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task group definitions
TASK_GROUPS = {
    "code_gen":       ["humaneval", "mbpp"],
    "math_reasoning": ["gsm8k", "mathqa"],
    "comparison":     ["boolq", "race"],
    "knowledge_qa":   ["arc_challenge", "arc_easy", "openbookqa"],
    "commonsense":    ["piqa", "winogrande", "hellaswag"],
    "language_model": ["c4", "wikitext2"],
}

# Prompt type mapping
PROMPT_MAP = {
    "corpus": "corpus",
    "knowledge": "knowledge",
    "zero_shot": "prompt_zero_shot",
    "cot": "prompt_cot",
    "icl": "prompt_icl",
}

def parse_config_name(config_str: str) -> Dict[str, str]:
    """Parse configuration string into a dictionary.
    
    Args:
        config_str: Configuration string with key-value pairs separated by underscores
        
    Returns:
        Dictionary of configuration parameters
    """
    parts = config_str.split("_")
    config_dict = {}
    current_key = None
    for part in parts:
        if "-" in part:
            key, val = part.split("-", 1)
            config_dict[key] = val
            current_key = key
        elif current_key:
            config_dict[current_key] += "_" + part
    return config_dict

def lm_eval_results_to_df(src: Union[str, Path, Dict]) -> pd.DataFrame:
    """Convert language model evaluation results to a DataFrame.
    
    Args:
        src: Source of results (file path or dictionary)
        
    Returns:
        DataFrame containing evaluation metrics
        
    Raises:
        TypeError: If src is neither a file path nor a dictionary
    """
    if isinstance(src, (str, Path)):
        data = json.loads(Path(src).read_text())
    elif isinstance(src, dict):
        data = src
    else:
        raise TypeError("src must be a file path or dict")

    rows = []
    order = {t: i for i, t in enumerate(data["results"].keys())}
    for task, res in data["results"].items():
        ver = data.get("versions", {}).get(task, "")
        shot = data.get("n-shot", {}).get(task, "")
        hib = data.get("higher_is_better", {}).get(task, {})
        for key, val in res.items():
            if "_stderr" in key or "," not in key:
                continue
            metric, flt = key.split(",", 1)
            stderr_key = f"{metric}_stderr,{flt}"
            stderr_val = res.get(stderr_key, None)
            value_str = f"{val:.4f}" if stderr_val is None else f"{val:.4f} ± {stderr_val:.4f}"
            rows.append(dict(
                Task=task, Version=ver, Filter=flt, Shot=shot, Metric=metric,
                Arrow="↑" if hib.get(metric, True) else "↓", Value=val,
                Stderr=stderr_val, Display=value_str, _order=order[task]
            ))
    return pd.DataFrame(rows).sort_values(["_order", "Metric"]).drop(columns="_order").reset_index(drop=True)

def collect_eval_results(root_dir: Union[str, Path], patterns: Optional[List[str]] = None) -> pd.DataFrame:
    """Collect evaluation results from multiple files.
    
    Supports multiple matching patterns (e.g., __tmp__ and origan directories).
    Only keeps the latest JSON file (by filename) in each directory.
    
    Args:
        root_dir: Root directory to search for result files
        patterns: List of glob patterns to match result files
        
    Returns:
        DataFrame containing combined evaluation results
        
    Raises:
        FileNotFoundError: If no matching files are found
    """
    if patterns is None:
        patterns = [
            "**/.__tmp__*/results_*.json",  # Calibration results
            "**/origan/**/results_*.json",  # Original model results
        ]

    root_dir = Path(root_dir)
    all_candidates = []

    for pat in patterns:
        matched = sorted(glob(str(root_dir / pat), recursive=True))
        all_candidates += matched

    if not all_candidates:
        raise FileNotFoundError(f"No matching result files found in {root_dir}")

    # 1. Group by directory
    files_by_dir = {}
    for f in all_candidates:
        dir_path = str(Path(f).parent)
        files_by_dir.setdefault(dir_path, []).append(f)

    # 2. Keep only the file with latest name in each directory
    latest_files = []
    for file_list in files_by_dir.values():
        latest_file = max(file_list, key=lambda p: Path(p).name)
        latest_files.append(latest_file)

    all_frames = []
    for fp in latest_files:
        try:
            df = lm_eval_results_to_df(fp)
        except Exception as e:
            print(f"⚠️ Skipping invalid file {fp}: {e}")
            continue

        parts = Path(fp).parts
        try:
            model_name = parts[-4]
            config_name = parts[-3]
        except IndexError:
            model_name, config_name = "", ""

        config_parts = {"prompt": "original"} if "origan" in parts else parse_config_name(config_name)

        df["Model"] = model_name
        for k, v in config_parts.items():
            df[k] = v

        all_frames.append(df)

    result = pd.concat(all_frames, ignore_index=True)
    return result.sort_values(by=["Task", "Metric", "Value"], ascending=[True, True, False]).reset_index(drop=True)

def normalize_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'Metric_for_use' column to standardize evaluation metrics across tasks.
    
    Args:
        df: Input DataFrame with evaluation results
        
    Returns:
        DataFrame with added normalized metric column
    """
    df = df.copy()
    df["Metric_for_use"] = df.apply(
        lambda r: (
            "pass_at_1" if r["Task"] == "mbpp" else
            "pass@1"    if r["Task"] == "humaneval" else
            "exact_match" if r["Task"] == "gsm8k" else
            r["Metric"]
        ), axis=1
    )
    return df

def build_top_configs(df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Get top-k calibration tasks for each TaskGroup + prompt combination.
    
    Args:
        df: Input DataFrame with evaluation results
        top_k: Number of top configurations to keep
        
    Returns:
        DataFrame with top configurations and sampling weights
    """
    # Map tasks to groups
    task_groups = TASK_GROUPS
    task_to_group = {t: g for g, ts in task_groups.items() for t in ts}

    df = df.copy()
    df["TaskGroup"] = df["Task"].map(task_to_group)
    df = normalize_metric(df)
    df_valid = df[df["Metric_for_use"].isin(["acc", "exact_match", "pass@1", "pass_at_1"])]

    grouped = (
        df_valid
        .groupby(["TaskGroup", "calib", "prompt"])["Value"]
        .mean()
        .reset_index()
        .sort_values(["TaskGroup", "Value"], ascending=[True, False])
    )

    topk = grouped.groupby(["TaskGroup"]).head(top_k).reset_index(drop=True)
    topk["Weight"] = topk.groupby(["TaskGroup"])["Value"].transform(lambda x: x / x.sum())
    return topk

def build_sample_dataset(
    data_df: pd.DataFrame,
    config_df: pd.DataFrame,
    samples_per_group: int = 128,
    random_seed: int = 42
) -> pd.DataFrame:
    """Build expert dataset by weighted sampling based on top configurations.
    
    Args:
        data_df: Input data DataFrame
        config_df: Configuration DataFrame with weights
        samples_per_group: Number of samples per task group
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame containing sampled expert dataset
    """
    all_samples = []
    for _, row in config_df.iterrows():
        group, prompt, calib, weight = row["TaskGroup"], row["prompt"], row["calib"], row["Weight"]
        col = PROMPT_MAP.get(prompt)
        if col is None or col not in data_df.columns:
            print(f"[Skip] Invalid prompt column: {prompt}")
            continue

        source_df = data_df[data_df["task_type"] == calib]
        if source_df.empty:
            print(f"[Skip] No data for task: {calib}")
            continue

        n_sample = int(np.round(samples_per_group * weight))
        sampled = source_df.sample(n=n_sample, random_state=random_seed, replace=True).copy()
        sampled["prompt_experts"] = sampled[col]
        sampled["task_type"] = group
        all_samples.append(sampled)

    return pd.concat(all_samples, ignore_index=True)

def plot_prompt_heatmap(
    df: pd.DataFrame,
    prompt_type: str = "knowledge",
    normalize_mode: str = "best",
    task_groups: Optional[Dict[str, List[str]]] = None,
    figsize: tuple = (8, 6),
    model_name: Optional[str] = None
) -> None:
    """Plot heatmap of prompt performance across tasks.
    
    Args:
        df: Input DataFrame with evaluation results
        prompt_type: Type of prompt to analyze
        normalize_mode: Normalization mode ('best' or 'self')
        task_groups: Dictionary mapping task groups to tasks
        figsize: Figure size (width, height)
        model_name: Optional model name to filter results
    """
    task_display = {
        "gsm8k": "GSM8K", "mathqa": "MathQA", "arc_challenge": "ARC-C",
        "arc_easy": "ARC-E", "openbookqa": "OBQA", "hellaswag": "HellaSwag",
        "winogrande": "WinoG.", "piqa": "PIQA", "boolq": "BoolQ",
        "race": "RACE", "wikitext2": "WikiText-2", "c4": "C4",
        "mbpp": "MBPP", "humaneval": "HumanEval"
    }
    prompt_display = {
        "knowledge": "Knowledge", "cot": "COT", "zero_shot": "Zero-Shot",
        "icl": "ICL", "corpus": "Corpus"
    }
    
    if model_name:
        df = df[df["Model"] == model_name]
    df = df[df["prompt"] == prompt_type].copy()
    
    if task_groups is None:
        task_groups = TASK_GROUPS
    ordered_tasks = [t for g in task_groups.values() for t in g]
    
    df["Metric_for_use"] = df.apply(
        lambda r: "pass_at_1" if r.Task == "mbpp" else (
                  "pass@1" if r.Task == "humaneval" else (
                  "exact_match" if r.Task == "gsm8k" else r.Metric)), axis=1
    )
    df = df[df["Metric_for_use"].isin(["acc", "exact_match", "pass@1", "pass_at_1"])]
    
    pivot = df.pivot_table(index="Task", columns="calib", values="Value", aggfunc="mean")
    common = list(set(pivot.index) & set(pivot.columns))
    aligned = [t for t in ordered_tasks if t in common]
    pivot = pivot.reindex(index=aligned, columns=aligned)
    
    norm = pivot.copy()
    for task in pivot.index:
        base = pivot.at[task, task] if normalize_mode == "self" else pivot.loc[task].max(skipna=True)
        for calib in pivot.columns:
            val = pivot.at[task, calib]
            norm.at[task, calib] = np.nan if pd.isna(val) or pd.isna(base) else (val - base) / base if base != 0 else -1.0
    
    norm_disp = norm.copy()
    norm_disp.index = [task_display.get(t, t) for t in norm.index]
    norm_disp.columns = [task_display.get(t, t) for t in norm.columns]
    
    plt.figure(figsize=figsize)
    ax = sns.heatmap(norm_disp, annot=True, fmt=".2f", cmap="Blues", vmin=-0.2, vmax=0,
                     cbar_kws={"label": "Relative Drop"})
    ax.set_xlabel("Calibration Task")
    ax.set_ylabel("Test Task")
    plt.title(f"Prompt: {prompt_display.get(prompt_type, prompt_type)}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()