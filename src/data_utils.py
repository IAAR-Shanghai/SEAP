# src/data_utils.py

import os
import pandas as pd
import random
from typing import Dict, List, Any, Tuple, Optional

def load_datasets(data_dir: str, split: str = 'train') -> Dict[str, pd.DataFrame]:
    """
    Load datasets from the specified directory and split type, categorizing them by task type.

    Args:
        data_dir (str): Path to the data directory (e.g., 'data/processed').
        split (str): The data split to load, such as 'train' or 'test'.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of datasets categorized by task type.
    """
    datasets = {}
    split_path = os.path.join(data_dir, split)
    
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split directory '{split_path}' not found in '{data_dir}'")

    for file_name in os.listdir(split_path):
        if file_name.endswith(f'_{split}_processed.parquet'):
            task_name = file_name.split(f'_{split}_processed.parquet')[0]
            file_path = os.path.join(split_path, file_name)
            datasets[task_name] = pd.read_parquet(file_path)
            print(f"Loaded {task_name} dataset from {split} split, shape: {datasets[task_name].shape}")
    
    return datasets

def build_few_shot_prompts(
    datasets: Dict[str, pd.DataFrame],
    min_shot: int,
    max_shot: int,
    seed: int = 42,
    sample_size: Optional[int] = None,
    use_corpus: bool = False
) -> Tuple[List[str], List[str]]:
    """
    针对多个任务，逐层生成 prompts。

    Args:
        datasets (Dict[str, pd.DataFrame]): A dictionary of datasets categorized by task type.
        min_shot (int): The minimum number of supporting examples for each task.
        max_shot (int): The maximum number of supporting examples for each task.
        seed (int): The random seed for reproducibility.
        sample_size (Optional[int]): The number of samples to select from each dataset.
        use_corpus (bool): Whether to use the corpus column directly (instead of few-shot logic).

    Returns:
        Tuple[List[str], List[str]]: A tuple containing a list of prompts and a list of task types corresponding to each prompt.
    """
    rng = random.Random(seed)
    new_inputs: List[str] = []
    new_task_types: List[str] = []

    for task_type, df in datasets.items():
        if df.empty:
            continue
        
        # Sample the dataset if sample_size is specified
        if sample_size is not None and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

        # Convert DataFrame to a list of records (dicts)
        records = df.to_dict('records')
        if not records:
            continue

        if use_corpus:
            # Use the 'corpus' column directly
            for row in records:
                corpus_text = row.get('corpus', "")
                new_inputs.append(corpus_text)
                new_task_types.append(task_type)
        else:
            # Generate few-shot prompts
            for row in records:
                k = rng.randint(min_shot, max_shot)

                # Randomly sample k support examples from the same task
                support_samples = rng.sample(records, k)

                # Create the few-shot prompt by concatenating support examples and the target question
                prompt_parts = [sup.get('input_with_gold', "") for sup in support_samples]
                prompt_parts.append(row.get('input', ""))  # Target question (no answer)
                prompt = "\n".join(prompt_parts)
                new_inputs.append(prompt)
                new_task_types.append(task_type)

    return new_inputs, new_task_types

def create_balanced_tasks(datasets: Dict[str, pd.DataFrame], balanced: bool = False, seed: int = None) -> List[Dict[str, Any]]:
    """
    Create a list of tasks, optionally balancing the number of examples across datasets.

    Args:
        datasets (Dict[str, pd.DataFrame]): A dictionary of datasets categorized by task type.
        balanced (bool): Whether to balance the number of samples across all tasks.
        seed (int): The random seed for reproducibility.

    Returns:
        List[Dict[str, Any]]: A list of task dictionaries, each containing an 'id', 'task_type', and 'input'.
    """
    tasks = []
    task_id = 1
    random.seed(seed)
    
    # Determine the minimum count across all datasets if balancing is enabled
    min_count = min(len(df) for df in datasets.values()) if balanced else None
    
    for dataset_name, df in datasets.items():
        df_sampled = df.sample(n=min_count, random_state=seed) if balanced else df
        for _, row in df_sampled.iterrows():
            tasks.append({"id": task_id, "task_type": dataset_name, "input": row['input']})
            task_id += 1
    
    return tasks
