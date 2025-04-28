import sys
import os
import argparse
import pandas as pd
from typing import Dict

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.data_preparation.task_processors import TASK_PROCESSORS

def load_task_data(task_path: str) -> Dict[str, pd.DataFrame]:
    train_dfs, validation_dfs, test_dfs = [], [], []

    for file_name in os.listdir(task_path):
        if file_name.endswith('.parquet'):
            file_path = os.path.join(task_path, file_name)
            df = pd.read_parquet(file_path)
            if 'train' in file_name:
                train_dfs.append(df)
            elif 'validation' in file_name:
                validation_dfs.append(df)
            elif 'test' in file_name:
                test_dfs.append(df)

    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    test_df = pd.concat(validation_dfs, ignore_index=True) if validation_dfs else (
        pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame())
    return {"train": train_df, "test": test_df}

def main(args):
    datasets = {}
    for task_name in os.listdir(args.raw_data_dir):
        task_path = os.path.join(args.raw_data_dir, task_name)
        if os.path.isdir(task_path):
            datasets[task_name] = load_task_data(task_path)
    
    all_data = []
    for dataset_name, splits in datasets.items():
        processor = TASK_PROCESSORS.get(dataset_name)
        if processor is None:
            print(f"⚠️ Warning: Unknown dataset '{dataset_name}', skipping.")
            continue

        for split_name, df in splits.items():
            if df.empty:
                print(f"⚠️ Skipping empty split: {dataset_name} - {split_name}")
                continue

            if args.sample_size and len(df) > args.sample_size:
                df = df.sample(n=args.sample_size, random_state=args.seed).reset_index(drop=True)

            processed = processor(df).copy()
            processed["task_type"] = dataset_name
            processed["split"] = split_name
            all_data.append(processed)

    final_df = pd.concat(all_data, ignore_index=True)
    os.makedirs(args.output_dir, exist_ok=True)
    final_path = os.path.join(args.output_dir, "base.parquet")
    final_df.to_parquet(final_path, index=False)
    print("✅ Saved base dataset to", final_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", required=True, help="Directory of raw dataset files")
    parser.add_argument("--output_dir", required=True, help="Directory to save the processed base dataset")
    parser.add_argument("--sample_size", type=int, default=512, help="Sample size per split per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(args)
