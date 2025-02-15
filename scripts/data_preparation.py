#!/usr/bin/env python3
# coding: utf-8
# scripts/data_preparation.py

import os
import pandas as pd
from typing import Dict
import argparse

def load_task_data(task_path: str) -> Dict[str, pd.DataFrame]:
    """
    Load and separate training, validation, and testing data from a specified task folder.
    
    Args:
        task_path (str): Path to the task folder.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary with 'train' and 'test' DataFrames for the task.
    """
    train_dfs = []
    validation_dfs = []
    test_dfs = []
    
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
    
    # Combine train, validation, and test data
    train_df = pd.concat(train_dfs, ignore_index=True) if train_dfs else pd.DataFrame()
    # 如果有validation数据集则用validation当作test集，如果没有则用test_dfs合并当作test集
    test_df = pd.concat(validation_dfs, ignore_index=True) if validation_dfs else (
        pd.concat(test_dfs, ignore_index=True) if test_dfs else pd.DataFrame())
    return {"train": train_df, "test": test_df}

def generate_input_prompts(datasets: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Generate standardized input prompts, labels, and gold answers for each dataset.
    We'll add a 'gold' column to store the correct answer's text representation,
    an 'input_with_gold' column, and now a new 'corpus' column to construct a corpus.

    Args:
        datasets (Dict[str, Dict[str, pd.DataFrame]]): 
            Nested dictionary with task names (dataset_name) and split data (train/test DataFrames).

    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: 
            Updated datasets with 'task_type', 'input', 'label', 'gold', 'input_with_gold', and 'corpus'.
    """
    for dataset_name, splits in datasets.items():
        for split_name, df in splits.items():
            if df.empty:
                continue  # Skip empty DataFrames

            if dataset_name == 'mmlu':
                # 原有处理
                df['input'] = df.apply(
                    lambda row: f"Input: {row['question']}\nOptions:\n{format_choices(row['choices'])}\nAnswer:",
                    axis=1
                )
                df['label'] = df['answer'].apply(lambda x: convert_to_letter(x))
                df['gold'] = df.apply(
                    lambda r: r['choices'][int(r['answer'])] 
                              if (not pd.isna(r['answer'])) and (int(r['answer']) < len(r['choices'])) 
                              else "", 
                    axis=1
                )
                df['input_with_gold'] = df.apply(
                    lambda row: f"{row['input']} {row['label']})  {row['gold']}", 
                    axis=1
                )

                # 新增：corpus列
                def build_corpus_mmlu(r):
                    if pd.isna(r['answer']):
                        return r['question']
                    ans_idx = int(r['answer'])
                    if ans_idx < len(r['choices']):
                        return r['question'] + " " + r['choices'][ans_idx]
                    else:
                        return r['question']
                
                df['corpus'] = df.apply(build_corpus_mmlu, axis=1)

            elif dataset_name == 'hellaswag':
                original_label = df['label'].copy()
                df['input'] = df.apply(
                    lambda row: f"Input: {row['ctx']}\nOptions:\n{format_choices(row['endings'])}\nAnswer:",
                    axis=1
                )
                df['label'] = df['label'].apply(lambda x: convert_to_letter(x))
                df['gold'] = df.apply(lambda r: r['endings'][int(original_label.loc[r.name])], axis=1)
                df['input_with_gold'] = df.apply(
                    lambda row: f"{row['input']} {row['label']}) , {row['gold']}", 
                    axis=1
                )

                # corpus = ctx + 正确endings
                def build_corpus_hellaswag(r):
                    idx = int(original_label.loc[r.name])
                    if idx < len(r['endings']):
                        return r['ctx'] + " " + r['endings'][idx]
                    else:
                        return r['ctx']
                df['corpus'] = df.apply(build_corpus_hellaswag, axis=1)

            elif dataset_name == 'piqa':
                original_label = df['label'].copy()
                df['input'] = df.apply(
                    lambda row: f"Input: {row['goal']}\nOptions:\nA) {row['sol1']}\nB) {row['sol2']}\nAnswer:",
                    axis=1
                )
                df['label'] = original_label.apply(lambda x: 'A' if x == 0 else 'B')
                df['gold'] = df.apply(
                    lambda r: r['sol1'] if original_label.loc[r.name] == 0 else r['sol2'], 
                    axis=1
                )
                df['input_with_gold'] = df.apply(
                    lambda row: f"{row['input']} {row['label']}) {row['gold']}", 
                    axis=1
                )

                # corpus = goal + correct solution
                def build_corpus_piqa(r):
                    if original_label.loc[r.name] == 0:
                        return r['goal'] + " " + r['sol1']
                    else:
                        return r['goal'] + " " + r['sol2']
                df['corpus'] = df.apply(build_corpus_piqa, axis=1)

            elif dataset_name == 'gsm8k':
                df['input'] = df.apply(
                    lambda row: f"Input: {row['question']}\nAnswer:",
                    axis=1
                )
                df['label'] = df['answer']
                df['gold'] = df['answer']
                df['input_with_gold'] = df.apply(lambda row: f"{row['input']} {row['gold']}", axis=1)

                # corpus = question + answer
                df['corpus'] = df.apply(lambda r: r['question'] + " " + r['answer'], axis=1)

            elif dataset_name == 'ai2_arc':
                mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
                df['input'] = df.apply(
                    lambda row: f"Input: {row['question']}\nOptions:\n{format_choices(row['choices']['text'])}\nAnswer:",
                    axis=1
                )

                def get_label_and_gold(r):
                    ak = r['answerKey']
                    label = None
                    gold_index = None
                    if isinstance(ak, str):
                        ak = ak.strip()
                        if ak.isalpha():
                            label = ak
                            gold_index = mapping.get(ak, None)
                        elif ak.isdigit():
                            num = int(ak)
                            label = chr(64 + num)
                            gold_index = num - 1
                    elif isinstance(ak, int):
                        label = chr(64 + ak)
                        gold_index = ak - 1

                    if gold_index is not None and gold_index < len(r['choices']['text']):
                        gold_text = r['choices']['text'][gold_index]
                    else:
                        gold_text = ""
                    return label, gold_text

                df[['label', 'gold']] = df.apply(lambda r: pd.Series(get_label_and_gold(r)), axis=1)
                df['input_with_gold'] = df.apply(lambda row: f"{row['input']} {row['label']}) {row['gold']}", axis=1)

                # corpus = question + correct choice
                def build_corpus_ai2(r):
                    label, gold = r['label'], r['gold']
                    return r['question'] + " " + gold if gold else r['question']

                df['corpus'] = df.apply(build_corpus_ai2, axis=1)

            elif dataset_name == 'winogrande':
                original_label = df['answer'].copy()
                df['input'] = df.apply(
                    lambda row: f"Input: {row['sentence']}\nOptions:\nA) {row['option1']}\nB) {row['option2']}\nAnswer:",
                    axis=1
                )
                df['label'] = df['answer'].apply(lambda x: 'A' if x == '1' else 'B')
                df['gold'] = df.apply(
                    lambda r: r['option1'] if original_label.loc[r.name] == '1' else r['option2'], 
                    axis=1
                )
                df['input_with_gold'] = df.apply(
                    lambda row: f"{row['input']} {row['label']}) {row['gold']}", 
                    axis=1
                )

                # corpus = sentence with '_' replaced by correct option
                def build_corpus_winogrande(r):
                    if original_label.loc[r.name] == '1':
                        return r['sentence'].replace('_', r['option1'])
                    else:
                        return r['sentence'].replace('_', r['option2'])

                df['corpus'] = df.apply(build_corpus_winogrande, axis=1)

            elif dataset_name == 'wikitext2':
                df['input'] = df['text']
                df['label'] = df['text']
                df['gold'] = df['text']
                df['input_with_gold'] = df.apply(lambda row: f"{row['input']} {row['gold']}", axis=1)

                # corpus = text
                df['corpus'] = df['text']

            elif dataset_name == 'boolq':
                df['input'] = df.apply(
                    lambda row: f"Question: {row['question'].capitalize()}\nPassage:\n{row['passage']}\nAnswer:",
                    axis=1
                )
                df['label'] = df['answer'].apply(lambda x: 'Yes' if x else 'No')
                df['gold'] = df['answer'].apply(lambda x: 'Yes' if x else 'No')
                df['input_with_gold'] = df.apply(
                    lambda row: f"{row['input']} {row['label']}) {row['gold']}", axis=1
                )

                # corpus = passage + question + answer
                df['corpus'] = df.apply(
                    lambda r: r['passage'] + " " + r['question'].capitalize() + "? " + r['gold'], axis=1
                )

            elif dataset_name == 'obqa':
                df['input'] = df.apply(
                    lambda row: f"Question: {row['question_stem']}\nOptions:\n{format_choices(row['choices']['text'])}\nAnswer:",
                    axis=1
                )

                def get_label_and_gold_obqa(r):
                    ak = r['answerKey']
                    gold_index = ord(ak) - ord('A')  # Convert 'A', 'B', 'C', 'D' to index 0, 1, 2, 3
                    return ak, r['choices']['text'][gold_index]

                df[['label', 'gold']] = df.apply(lambda r: pd.Series(get_label_and_gold_obqa(r)), axis=1)
                df['input_with_gold'] = df.apply(lambda row: f"{row['input']} {row['label']}) {row['gold']}", axis=1)

                # corpus = question_stem + correct choice
                df['corpus'] = df.apply(lambda r: r['fact1'].capitalize() + ". " + r['question_stem'] + " " + r['gold'], axis=1)

            else:
                print(f"Warning: Unknown dataset {dataset_name}. Skipping prompt generation.")
                continue

            df['task_type'] = dataset_name
            df = df[['task_type', 'input', 'label', 'gold', 'input_with_gold', 'corpus']]  # Add 'corpus' column
            splits[split_name] = df

    return datasets

def format_choices(choices) -> str:
    """
    Formats a list of choices into a lettered multiple-choice format (A, B, C, D).

    Args:
        choices (iterable): List of answer choices.

    Returns:
        str: Formatted string of choices with letters.
    """
    labels = ["A)", "B)", "C)", "D)", "E)"]
    formatted_choices = [f"{labels[i]} {choice}" for i, choice in enumerate(choices)]
    return "\n".join(formatted_choices)

def convert_to_letter(label) -> str:
    """
    Convert a numeric label to a letter (A, B, C, ...), or leave it as a letter if already a letter.

    Args:
        label (int or str): Original label.

    Returns:
        str: Converted letter label.
    """
    if isinstance(label, (int, float)) and not pd.isna(label):
        return chr(65 + int(label))  # Convert numeric to letter
    elif isinstance(label, str) and label.isnumeric():
        # 假设数字字符串代表0->A,1->B,2->C,3->D
        return chr(65 + int(label))
    return str(label)  # Keep as is if already letter

def save_processed_data(datasets: Dict[str, Dict[str, pd.DataFrame]], output_dir: str):
    """
    Save processed datasets to separate directories for train and test data.
    
    Args:
        datasets (Dict[str, Dict[str, pd.DataFrame]]): Nested dictionary of processed datasets.
        output_dir (str): Directory to save processed datasets.
    """
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    
    # Create directories for train and test splits
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for dataset_name, splits in datasets.items():
        for split_name, df in splits.items():
            if split_name == "train":
                output_path = os.path.join(train_dir, f"{dataset_name}_train_processed.parquet")
            elif split_name == "test":
                output_path = os.path.join(test_dir, f"{dataset_name}_test_processed.parquet")
            else:
                continue
            
            df.to_parquet(output_path, index=False)
            print(f"Saved processed {split_name} data for {dataset_name} to {output_path}")

def main(args):
    raw_data_dir = args.raw_data_dir
    processed_data_dir = args.processed_data_dir

    # Load datasets
    datasets = {}
    for task_name in os.listdir(raw_data_dir):
        task_path = os.path.join(raw_data_dir, task_name)
        if os.path.isdir(task_path):
            datasets[task_name] = load_task_data(task_path)
            print(f"Loaded data for task: {task_name}, train shape: {datasets[task_name]['train'].shape}, test shape: {datasets[task_name]['test'].shape}")

    # Generate input prompts, label, and gold columns
    datasets = generate_input_prompts(datasets)

    # Save processed datasets
    save_processed_data(datasets, processed_data_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and preprocess raw datasets.")
    parser.add_argument(
        "--raw_data_dir", type=str, default="data/raw",
        help="Directory containing raw task datasets"
    )
    parser.add_argument(
        "--processed_data_dir", type=str, default="data/processed",
        help="Directory to save processed datasets"
    )
    args = parser.parse_args()
    main(args)
