# scripts/build_prompts.py

import sys
import os
import argparse
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from src.data_preparation.prompt_builders import build_batch_prompts

def main(args):
    print(f"📥 Loading input: {args.input_path}")
    df = pd.read_parquet(args.input_path)

    # ✅ 清洗 knowledge 前缀
    df["knowledge"] = df["knowledge"].astype(str).str.replace(r"^Knowledge Statement:\s*", "", regex=True)

    print("🧱 Building prompts...")
    df["prompt_zero_shot"] = build_batch_prompts(df, prompt_type="zero_shot", seed=args.seed)
    df["prompt_cot"] = build_batch_prompts(df, prompt_type="cot", seed=args.seed)
    df["prompt_icl"] = build_batch_prompts(df, prompt_type="icl", min_shot=args.min_shot, max_shot=args.max_shot, seed=args.seed)
    df["prompt_icl_cot"] = build_batch_prompts(df, prompt_type="icl_cot", min_shot=args.min_shot, max_shot=args.max_shot, seed=args.seed)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_parquet(args.output_path, index=False)
    print(f"✅ Saved final dataset with prompts to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="Path to input .parquet with rationale & knowledge")
    parser.add_argument("--output_path", required=True, help="Full path to save the final prompt-enriched .parquet")
    parser.add_argument("--min_shot", type=int, default=3)
    parser.add_argument("--max_shot", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args)
