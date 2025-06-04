"""
Data loading and processing utilities.

This module provides functions for loading datasets, building few-shot prompts,
and creating balanced task sets for machine learning experiments.

Author: why
Date: 2024
"""

# Standard library imports
import os
import random
from typing import Dict, List, Any, Tuple, Optional

# Third-party imports
import pandas as pd


def load_datasets(data_dir: str, split: str = 'train') -> Dict[str, pd.DataFrame]:
    """Load datasets from specified directory and split type.
    
    Loads and categorizes datasets by task type from parquet files.
    
    Args:
        data_dir: Path to the data directory (e.g., 'data/processed')
        split: Data split to load ('train' or 'test')
        
    Returns:
        Dictionary mapping task types to corresponding DataFrames
        
    Raises:
        FileNotFoundError: If split directory does not exist
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
    """Build few-shot prompts for multiple tasks.
    
    Generates prompts by either using direct corpus text or creating few-shot examples
    with support samples.
    
    Args:
        datasets: Dictionary mapping task types to DataFrames
        min_shot: Minimum number of supporting examples per task
        max_shot: Maximum number of supporting examples per task
        seed: Random seed for reproducibility
        sample_size: Number of samples to select from each dataset
        use_corpus: Whether to use corpus column directly instead of few-shot logic
        
    Returns:
        Tuple containing:
            - List of generated prompts
            - List of corresponding task types
    """
    rng = random.Random(seed)
    new_inputs: List[str] = []
    new_task_types: List[str] = []

    for task_type, df in datasets.items():
        if df.empty:
            continue
        
        # Sample dataset if size specified
        if sample_size is not None and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

        # Convert DataFrame to list of records
        records = df.to_dict('records')
        if not records:
            continue

        if use_corpus:
            # Use corpus column directly
            for row in records:
                corpus_text = row.get('corpus', "")
                new_inputs.append(corpus_text)
                new_task_types.append(task_type)
        else:
            # Generate few-shot prompts
            for row in records:
                k = rng.randint(min_shot, max_shot)

                # Sample k support examples from same task
                support_samples = rng.sample(records, k)

                # Concatenate support examples and target question
                prompt_parts = [sup.get('input_with_gold', "") for sup in support_samples]
                prompt_parts.append(row.get('input', ""))  # Target question without answer
                prompt = "\n".join(prompt_parts)
                new_inputs.append(prompt)
                new_task_types.append(task_type)

    return new_inputs, new_task_types


def create_balanced_tasks(
    datasets: Dict[str, pd.DataFrame],
    balanced: bool = False,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Create list of tasks with optional balancing across datasets.
    
    Args:
        datasets: Dictionary mapping task types to DataFrames
        balanced: Whether to balance sample counts across tasks
        seed: Random seed for reproducibility
        
    Returns:
        List of task dictionaries containing:
            - id: Unique task identifier
            - task_type: Type of task
            - corpus: Task corpus text
    """
    tasks = []
    task_id = 1
    random.seed(seed)
    
    # Find minimum count across datasets if balancing
    min_count = min(len(df) for df in datasets.values()) if balanced else None
    
    for dataset_name, df in datasets.items():
        df_sampled = df.sample(n=min_count, random_state=seed) if balanced else df
        for _, row in df_sampled.iterrows():
            tasks.append({
                "id": task_id,
                "task_type": dataset_name,
                "corpus": row['corpus']
            })
            task_id += 1
    
    return tasks
