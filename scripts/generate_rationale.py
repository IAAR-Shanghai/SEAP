import os
import time
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI(
    base_url="https://api.claudeplus.top/v1",
    api_key="sk-u0sQrMp7kfkhpN5QLrYjwqWUQp0YIAlGHMUzeuTFpt8lH7Dn"  # 请替换成你的 API KEY
)

def call_openai(prompt: str, temperature: float = 0.3, max_tokens: int = 512, retries: int = 3, delay: float = 2.0) -> str:
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-2024-11-20",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"[Retry {attempt+1}] Error: {e}")
            time.sleep(delay)
    return "[Error] Failed after retries"

def build_rationale_prompt(row) -> str:
    return (
        "You are a reasoning assistant. Think through the question below step by step, "
        "using common sense or domain knowledge where needed. Then provide the final answer.\n\n"
        f"Question:\n{row['question']}\n\n"
        "Please respond in the following format:\n\n"
        "Explanation:\n<your step-by-step reasoning here>\n\n"
        f"Final Answer: {row['label']}) {row['gold']}"
    )

def build_knowledge_prompt(row) -> str:
    return (
        "You are converting natural text into structured factual knowledge. "
        "Given the raw input below, extract its key fact or principle and rewrite it as a clear, textbook-style statement.\n\n"
        f"Raw input:\n{row['corpus']}\n\n"
        "Please respond in the following format:\n\n"
        "Knowledge Statement: <your rewritten sentence>"
    )

def process_row(i, row, args):
    if args.dry_run:
        rationale = "[DRY RUN] explanation...\nFinal Answer: ..."
        knowledge = "Knowledge Statement: [DRY RUN] ..."
    else:
        rationale = call_openai(build_rationale_prompt(row), args.temperature, args.max_tokens)
        knowledge = call_openai(build_knowledge_prompt(row), args.temperature, args.max_tokens)
    return i, rationale, knowledge

def main(args):
    input_path = args.input_path
    output_path = args.output_path or os.path.join(os.path.dirname(input_path), "with_rationale.parquet")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"📥 Loading input file: {input_path}")
    df = pd.read_parquet(input_path)

    if os.path.exists(output_path):
        print(f"🔁 Found existing output file: {output_path}")
        result_df = pd.read_parquet(output_path)
        if len(result_df) != len(df):
            if args.force_restart:
                print("⚠️ Row count mismatch. Force restart enabled → Overwriting output.")
                result_df = df.copy()
                result_df["rationale"] = None
                result_df["knowledge"] = None
            else:
                raise ValueError("❌ Row count mismatch. Use --force_restart to overwrite.")
    else:
        result_df = df.copy()
        result_df["rationale"] = None
        result_df["knowledge"] = None

    remaining_indices = result_df[
        result_df["rationale"].isna() | result_df["knowledge"].isna()
    ].index.tolist()

    print(f"🧠 Remaining samples to process: {len(remaining_indices)}")

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_row, i, result_df.loc[i], args): i
            for i in remaining_indices
        }

        for count, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
            i, rationale, knowledge = future.result()
            result_df.at[i, "rationale"] = rationale
            result_df.at[i, "knowledge"] = knowledge

            if (count + 1) % args.save_every == 0:
                result_df.to_parquet(output_path, index=False)
                print(f"💾 Saved checkpoint at {count+1} samples.")

    result_df.to_parquet(output_path, index=False)
    print(f"✅ Saved final dataset with rationale & knowledge to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path")
    parser.add_argument("--save_every", type=int, default=30)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_restart", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8, help="Number of threads for concurrent requests")
    args = parser.parse_args()

    main(args)
