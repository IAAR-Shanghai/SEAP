#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_preparation.task_processors import TASK_PROCESSORS
from src.data_preparation.prompt_builders import build_batch_prompts
from src.data_preparation.utils_openai_call import (
    call_openai,
    build_rationale_prompt,
    build_knowledge_prompt
)

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

def build_base_dataset(raw_data_dir, sample_size, seed, tasks):
    datasets = {}
    for task_name in os.listdir(raw_data_dir):
        task_path = os.path.join(raw_data_dir, task_name)
        if os.path.isdir(task_path):
            datasets[task_name] = load_task_data(task_path)

    all_data = []
    for task_name, splits in datasets.items():
        if tasks and task_name not in tasks:
            continue

        processor = TASK_PROCESSORS.get(task_name)
        if processor is None:
            print(f"⚠️ Warning: Unknown task '{task_name}', skipping.")
            continue

        for split_name, df in splits.items():
            if df.empty:
                print(f"⚠️ Skipping empty split: {task_name} - {split_name}")
                continue
            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

            processed = processor(df).copy()
            processed["task_type"] = task_name
            processed["split"] = split_name
            all_data.append(processed)

    return pd.concat(all_data, ignore_index=True)


def generate_column(df, column, dry_run, temperature, max_tokens, force, args):
    def generate_single(i, row):
        try:
            if not force and pd.notna(row[column]) and not str(row[column]).startswith("[BAD]"):
                return i, row[column]
            if column == "rationale":
                prompt = build_rationale_prompt(row)
            elif column == "knowledge":
                prompt = build_knowledge_prompt(row)
            else:
                raise ValueError(f"Unknown column: {column}")
            result = "[DRY RUN]" if dry_run else call_openai(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=args.api_key,
                base_url=args.api_base_url
            )
            return i, result
        except Exception as e:
            print(f"❌ Error on row {i}: {e}")
            return i, "[BAD] Exception occurred"

    print(f"🔄 Generating '{column}' with {args.num_workers} threads...")
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(generate_single, i, row): i for i, row in df.iterrows()}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating {column}"):
            i, value = future.result()
            df.at[i, column] = value


def generate_prompts(df, min_shot, max_shot, seed):
    df["knowledge"] = df["knowledge"].astype(str).str.replace(r"^Knowledge Statement:\s*", "", regex=True)
    df["prompt_zero_shot"] = build_batch_prompts(df, "zero_shot", seed=seed)
    df["prompt_cot"] = build_batch_prompts(df, "cot", seed=seed)
    df["prompt_icl"] = build_batch_prompts(df, "icl", min_shot=min_shot, max_shot=max_shot, seed=seed)
    df["prompt_icl_cot"] = build_batch_prompts(df, "icl_cot", min_shot=min_shot, max_shot=max_shot, seed=seed)


def main(args):
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    if not os.path.exists(args.output_path) or args.generate_base:
        print("🛠 Building base dataset...")
        df = build_base_dataset(
            raw_data_dir=args.raw_data_dir,
            sample_size=args.sample_size,
            seed=args.seed,
            tasks=args.tasks
        )
    else:
        print("📥 Loading existing dataset...")
        df = pd.read_parquet(args.output_path)

    if args.generate_rationale:
        print("✏️ Generating rationale column...")
        generate_column(
            df, column="rationale", dry_run=args.dry_run,
            temperature=args.temperature, max_tokens=args.max_tokens,
            force="rationale" in args.overwrite_column,
            args=args
        )

    if args.generate_knowledge:
        print("📚 Generating knowledge column...")
        generate_column(
            df, column="knowledge", dry_run=args.dry_run,
            temperature=args.temperature, max_tokens=args.max_tokens,
            force="knowledge" in args.overwrite_column,
            args=args
        )

    if args.build_prompts:
        print("🧱 Building prompt columns...")
        generate_prompts(df, min_shot=args.min_shot, max_shot=args.max_shot, seed=args.seed)

    df.to_parquet(args.output_path, index=False)
    print(f"✅ Saved final dataset to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--sample_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--generate_base", action="store_true")
    parser.add_argument("--generate_rationale", action="store_true")
    parser.add_argument("--generate_knowledge", action="store_true")
    parser.add_argument("--build_prompts", action="store_true")
    parser.add_argument("--overwrite_column", nargs="*", default=[])

    parser.add_argument("--api_key", type=str, default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--api_base_url", type=str, default=os.getenv("OPENAI_BASE_URL", "https://api.claudeplus.top/v1"))

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--min_shot", type=int, default=3)
    parser.add_argument("--max_shot", type=int, default=5)
    parser.add_argument("--tasks", nargs="+", default=None)

    args = parser.parse_args()
    main(args)
