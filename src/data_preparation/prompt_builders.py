import random
import pandas as pd
from typing import Literal

PromptType = Literal["zero_shot", "icl", "cot", "icl_cot"]

def build_answer(row: dict) -> str:
    label, gold = row.get("label", ""), row.get("gold", "")
    return f"{label}) {gold}" if label and gold and isinstance(label, str) else str(label or gold)

def build_zero_shot_prompt(row: dict) -> str:
    answer = build_answer(row)

    if row["task_format"] == "language_modeling":
        return f"{row['question']}"

    elif row["task_format"] == "generative":
        return f"# Task:\n{row['question']}\n\n# Solution:\n{answer}"

    else:
        return f"Question: {row['question']}\nAnswer: {answer}"

def build_cot_prompt(row: dict) -> str:
    rationale = row.get("rationale", "").strip()
    if row["task_format"] == "language_modeling":
        return row["question"]
    elif row["task_format"] == "generative":
        return f"# Task:\n{row['question']}\n\n# Let's think step by step.\n{rationale}\n# Final Answer:"
    elif rationale:
        return f"Question: {row['question']}\nAnswer: Let's think step by step.\n{rationale}"
    else:
        return build_zero_shot_prompt(row)

def build_icl_examples(
    support_records: list,
    exclude_index: int,
    task_format: str,
    use_cot: bool,
    min_shot: int,
    max_shot: int,
    rng: random.Random
) -> str:
    pool = support_records[:exclude_index] + support_records[exclude_index + 1:]
    k = rng.randint(min_shot, max_shot)
    examples = rng.sample(pool, min(k, len(pool)))

    shots = []
    for row in examples:
        rationale = row.get("rationale", "").strip()
        answer = build_answer(row)
        q = row["question"]

        if task_format == "generative":
            if use_cot and rationale:
                shots.append(f"# Task:\n{q}\n\n# Let's think step by step.\n{rationale}\n# Final Answer:\n{answer}")
            else:
                shots.append(f"# Task:\n{q}\n\n# Solution:\n{answer}")
        elif task_format == "language_modeling":
            shots.append(q)
        else:
            if use_cot and rationale:
                shots.append(f"Question: {q}\nAnswer: Let's think step by step.\n{rationale}")
            else:
                shots.append(f"Question: {q}\nAnswer: {answer}")

    return "\n\n".join(shots)

def build_batch_prompts(
    df: pd.DataFrame,
    prompt_type: PromptType = "zero_shot",
    min_shot: int = 3,
    max_shot: int = 5,
    seed: int = 42
) -> pd.Series:
    rng = random.Random(seed)
    prompts = pd.Series(index=df.index, dtype=str)

    for task_type, task_df in df.groupby("task_type"):
        task_records = task_df.to_dict("records")
        task_format = task_records[0].get("task_format", "unknown")

        task_prompts = []
        for i, row in enumerate(task_records):
            if prompt_type == "zero_shot":
                prompt = build_zero_shot_prompt(row)

            elif prompt_type == "cot":
                prompt = build_cot_prompt(row)

            elif prompt_type in {"icl", "icl_cot"}:
                icl_intro = build_icl_examples(
                    support_records=task_records,
                    exclude_index=i,
                    task_format=task_format,
                    use_cot=(prompt_type == "icl_cot"),
                    min_shot=min_shot,
                    max_shot=max_shot,
                    rng=rng
                )

                q = row["question"]

                if task_format == "generative":
                    if prompt_type == "icl_cot":
                        prompt = f"{icl_intro}\n\n# Task:\n{q}\n\n# Let's think step by step."
                    else:
                        prompt = f"{icl_intro}\n\n# Task:\n{q}\n\n# Solution:"
                elif task_format == "language_modeling":
                    prompt = f"{icl_intro}\n\n{q}"
                else:
                    if prompt_type == "icl_cot":
                        prompt = f"{icl_intro}\n\nQuestion: {q}\nAnswer: Let's think step by step."
                    else:
                        prompt = f"{icl_intro}\n\nQuestion: {q}\nAnswer:"

            task_prompts.append(prompt)

        prompts.loc[task_df.index] = task_prompts

    return prompts.rename("prompt")
