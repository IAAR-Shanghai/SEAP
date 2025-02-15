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
    
    如果 use_corpus=False:
      - 对每个任务 task_type:
        1) 若 sample_size 不为 None，则从 df 中采样 sample_size 行 (random_state=seed).
        2) 遍历该任务采样后的所有样本 (records):
           当前行作为“目标问题”(row['input'] 不含答案),
           随机选 k in [min_shot, max_shot] 条“支持示例”(来自同 task_type 的 df)，
             每个支持示例用 row['input_with_gold'] (含答案),
           最终串接成一个多段文本 => few-shot prompt。
        3) 不跨任务(同一个 task_type 内部做 few-shot)。
    
    如果 use_corpus=True:
      - 不做 few-shot 逻辑, 直接遍历 df 的每一条, 取 row['corpus'] 列即可。
      - 同样支持 sample_size 采样, 但不会拼接 few-shot, 只输出 corpus。
    
    Returns:
        new_inputs (List[str]): prompt 的列表
        new_task_types (List[str]): 与 new_inputs 等长，每个 prompt 所属的任务类型
    """
    rng = random.Random(seed)

    new_inputs: List[str] = []
    new_task_types: List[str] = []

    for task_type, df in datasets.items():
        if df.empty:
            continue
        
        # 1) 如果指定了 sample_size，对 df 做随机抽取
        if sample_size is not None and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

        # 将 DataFrame 转成列表，每条是一个 dict
        # 假设 df 至少含有下列列:
        #   - 'input'           (few-shot目标问题)
        #   - 'input_with_gold' (few-shot带答案示例)
        #   - 'corpus'          (若 use_corpus=True 时，使用此列)
        records = df.to_dict('records')
        if not records:
            continue

        if use_corpus:
            # 直接使用 corpus 列
            for row in records:
                corpus_text = row.get('corpus', "")
                new_inputs.append(corpus_text)
                new_task_types.append(task_type)
        else:
            # 保持原先 few-shot 拼接逻辑
            for row in records:
                k = rng.randint(min_shot, max_shot)

                # 从同task的records里随机抽k个支持示例
                support_samples = rng.sample(records, k)

                # 拼接 few-shot prompt
                prompt_parts = []
                for sup in support_samples:
                    prompt_parts.append(sup.get('input_with_gold', ""))  # 含答案

                # 目标问题(不含答案)
                prompt_parts.append(row.get('input', ""))

                prompt = "\n".join(prompt_parts)
                new_inputs.append(prompt)
                new_task_types.append(task_type)

    return new_inputs, new_task_types

def create_balanced_tasks(datasets: Dict[str, pd.DataFrame], balanced: bool = False, seed: int = None) -> List[Dict[str, Any]]:
    tasks = []
    task_id = 1
    random.seed(seed)
    min_count = min(len(df) for df in datasets.values()) if balanced else None
    for dataset_name, df in datasets.items():
        df_sampled = df.sample(n=min_count, random_state=seed) if balanced else df
        for _, row in df_sampled.iterrows():
            tasks.append({"id": task_id, "task_type": dataset_name, "input": row['input']})
            task_id += 1
    return tasks