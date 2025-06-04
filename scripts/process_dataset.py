#!/usr/bin/env python3
"""
Process and prepare datasets for transformer model analysis.

This script handles dataset loading, processing, and prompt generation for multiple
task types. It supports:
1. Loading and processing raw datasets
2. Generating rationales and knowledge statements using LLMs
3. Building various prompt formats (zero-shot, chain-of-thought, in-context learning)

Example usage:
    python process_dataset.py \
        --raw_data_dir data/raw \
        --output_path data/processed/prompts.parquet \
        --tasks gsm8k arc_e \
        --generate_base \
        --generate_rationale \
        --build_prompts

Author: why
Date: 2024
"""

# Standard library imports
import os
import re
import sys
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Third-party imports
import pandas as pd
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Local imports
from src.data_preparation.task_processors import TASK_PROCESSORS
from src.data_preparation.prompt_builders import build_batch_prompts
from src.data_preparation.utils_openai_call import (
    call_openai,
    build_rationale_prompt,
    build_knowledge_prompt
)

# Constants
TASK_TYPE_MAP = {
    "gsm8k": "gsm8k",
    "math_qa": "mathqa",
    "arc_e": "arc_easy",
    "arc_c": "arc_challenge",
    "obqa": "openbookqa",
    "winogrande": "winogrande",
    "piqa": "piqa",
    "hellaswag": "hellaswag",
    "boolq": "boolq",
    "wikitext2": "wikitext2",
    "c4": "c4"
}


def load_task_data(task_path: str) -> Dict[str, pd.DataFrame]:
    """Load train and test data for a task.
    
    Args:
        task_path: Path to task directory containing parquet files
        
    Returns:
        Dictionary mapping split names to DataFrames
    """
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


def build_base_dataset(
    raw_data_dir: str,
    sample_size: Optional[int] = None,
    seed: int = 42,
    tasks: Optional[List[str]] = None
) -> pd.DataFrame:
    """Build base dataset from raw data files.
    
    Args:
        raw_data_dir: Directory containing raw data files
        sample_size: Number of samples per task (optional)
        seed: Random seed for sampling
        tasks: List of task types to process (optional)
        
    Returns:
        Combined DataFrame with processed data
        
    Raises:
        ValueError: If no valid data is found
    """
    # Load raw data
    datasets = {}
    for task_name in os.listdir(raw_data_dir):
        task_path = os.path.join(raw_data_dir, task_name)
        if os.path.isdir(task_path):
            datasets[task_name] = load_task_data(task_path)

    # Process each task
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

            try:
                # Process and clean data
                processed = processor(df).copy()
                processed = processed[processed["question"].notna()]
                processed = processed[processed["question"].str.strip() != ""]
                
                if processed.empty:
                    print(f"⚠️ No valid questions after cleanup: {task_name} - {split_name}")
                    continue
                
                # Sample if needed
                if sample_size and len(processed) > sample_size:
                    processed = processed.sample(
                        n=sample_size,
                        random_state=seed
                    ).reset_index(drop=True)

                # Add metadata
                processed["task_type"] = TASK_TYPE_MAP.get(task_name, task_name)
                processed["split"] = split_name
                all_data.append(processed)

            except Exception as e:
                print(f"❌ Error processing {task_name} - {split_name}: {e}")
                continue

    if not all_data:
        raise ValueError("No valid data found after processing")

    return pd.concat(all_data, ignore_index=True)


def clean_output(text: str, column: str) -> str:
    """Clean generated text output.
    
    Args:
        text: Raw generated text
        column: Column type (rationale/knowledge)
        
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return text

    if column == "rationale":
        return re.sub(
            r"Explanation\s*(\(.*?\))?:?",
            "",
            text,
            flags=re.IGNORECASE
        ).strip()
    elif column == "knowledge":
        text = re.sub(
            r"Knowledge\s+Statement\s*(\(.*?\))?:?",
            "",
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(
            r"Demo\s*(\(.*?\))?:?",
            "",
            text,
            flags=re.IGNORECASE
        )
        return text.strip()
    return text.strip()


def generate_single_text(
    i: int,
    row: pd.Series,
    column: str,
    dry_run: bool,
    force: bool,
    temperature: float,
    max_tokens: int,
    api_key: str,
    base_url: str
) -> Tuple[int, str]:
    """Generate text for a single row.
    
    Args:
        i: Row index
        row: DataFrame row
        column: Column to generate (rationale/knowledge)
        dry_run: Whether to skip actual API calls
        force: Whether to overwrite existing values
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        api_key: API key for LLM service
        base_url: Base URL for API
        
    Returns:
        Tuple of (row index, generated text)
    """
    try:
        # Check if generation needed
        if not force and pd.notna(row[column]) and not str(row[column]).startswith("[BAD]"):
            return i, row[column]

        # Build prompt
        if column == "rationale":
            prompt = build_rationale_prompt(row)
        elif column == "knowledge":
            prompt = build_knowledge_prompt(row)
        else:
            raise ValueError(f"Unknown column: {column}")

        # Generate text
        result = "[DRY RUN]" if dry_run else call_openai(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url
        )

        return i, clean_output(result, column)

    except Exception as e:
        print(f"❌ Error on row {i}: {e}")
        return i, "[BAD] Exception occurred"


def generate_column(
    df: pd.DataFrame,
    column: str,
    dry_run: bool,
    temperature: float,
    max_tokens: int,
    force: bool,
    args: argparse.Namespace
) -> None:
    """Generate text for an entire column using parallel processing.
    
    Args:
        df: Input DataFrame
        column: Column to generate
        dry_run: Whether to skip API calls
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        force: Whether to overwrite existing values
        args: Additional arguments
    """
    print(f"🔄 Generating '{column}' with {args.num_workers} threads...")
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                generate_single_text,
                i,
                row,
                column,
                dry_run,
                force,
                temperature,
                max_tokens,
                args.api_key,
                args.api_base_url
            ): i for i, row in df.iterrows()
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating {column}"):
            i, value = future.result()
            df.at[i, column] = value


def generate_prompts(
    df: pd.DataFrame,
    min_shot: int,
    max_shot: int,
    seed: int
) -> None:
    """Generate different prompt formats for the dataset.
    
    Args:
        df: Input DataFrame
        min_shot: Minimum number of shots for ICL
        max_shot: Maximum number of shots for ICL
        seed: Random seed
    """
    # Clean knowledge statements
    df["knowledge"] = df["knowledge"].astype(str).str.replace(
        r"^Knowledge Statement:\s*",
        "",
        regex=True
    )

    # Build different prompt formats
    df["prompt_zero_shot"] = build_batch_prompts(df, "zero_shot", seed=seed)
    df["prompt_cot"] = build_batch_prompts(df, "cot", seed=seed)
    df["prompt_icl"] = build_batch_prompts(
        df,
        "icl",
        min_shot=min_shot,
        max_shot=max_shot,
        seed=seed
    )
    df["prompt_icl_cot"] = build_batch_prompts(
        df,
        "icl_cot",
        min_shot=min_shot,
        max_shot=max_shot,
        seed=seed
    )


def main(args: argparse.Namespace) -> None:
    """Main execution function.
    
    Args:
        args: Command line arguments
    """
    try:
        # Create output directory
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

        # Load or build base dataset
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

        # Filter by split if needed
        if args.subset_split:
            print(f"📊 Filtering to split: {args.subset_split}")
            df = df[df["split"] == args.subset_split].reset_index(drop=True)

        # Generate rationales if requested
        if args.generate_rationale:
            print("✏️ Generating rationale column...")
            generate_column(
                df,
                column="rationale",
                dry_run=args.dry_run,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                force="rationale" in args.overwrite_column,
                args=args
            )

        # Generate knowledge statements if requested
        if args.generate_knowledge:
            print("📚 Generating knowledge column...")
            generate_column(
                df,
                column="knowledge",
                dry_run=args.dry_run,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                force="knowledge" in args.overwrite_column,
                args=args
            )

        # Build prompts if requested
        if args.build_prompts:
            print("🧱 Building prompt columns...")
            generate_prompts(
                df,
                min_shot=args.min_shot,
                max_shot=args.max_shot,
                seed=args.seed
            )

        # Save final dataset
        df.to_parquet(args.output_path, index=False)
        print(f"✅ Saved final dataset to {args.output_path}")

    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process datasets and generate prompts for model analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        required=True,
        help="Directory containing raw data files"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save processed dataset"
    )

    # Dataset configuration
    parser.add_argument(
        "--sample_size",
        type=int,
        default=128,
        help="Number of samples per task"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--subset_split",
        type=str,
        choices=["train", "test"],
        default=None,
        help="Only process specified split"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="List of tasks to process"
    )

    # Processing flags
    parser.add_argument(
        "--generate_base",
        action="store_true",
        help="Generate base dataset"
    )
    parser.add_argument(
        "--generate_rationale",
        action="store_true",
        help="Generate rationales"
    )
    parser.add_argument(
        "--generate_knowledge",
        action="store_true",
        help="Generate knowledge statements"
    )
    parser.add_argument(
        "--build_prompts",
        action="store_true",
        help="Build prompt variations"
    )
    parser.add_argument(
        "--overwrite_column",
        nargs="*",
        default=[],
        help="Columns to overwrite"
    )

    # API configuration
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="API key for LLM service"
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        default=os.getenv("OPENAI_BASE_URL"),
        help="Base URL for API"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers"
    )

    # Generation parameters
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Skip API calls"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--min_shot",
        type=int,
        default=1,
        help="Minimum shots for ICL"
    )
    parser.add_argument(
        "--max_shot",
        type=int,
        default=3,
        help="Maximum shots for ICL"
    )

    args = parser.parse_args()
    main(args)