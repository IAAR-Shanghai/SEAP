"""
任务处理模块：负责将不同来源的数据集统一格式化处理为标准格式。

包含字段：
- task_type：任务名称
- task_format：任务格式类型（multiple_choice / free_form / generative / language_modeling）
- question：用于生成 Prompt 的问题内容
- label：原始标签（字母/自由文本）
- gold：答案文本
- corpus：作为知识语料的内容，可用于模型剪枝或自监督训练
- rationale：推理过程（如有）

格式分类说明：
- multiple_choice：带选项的任务，如 arc_e, obqa 等
- free_form：自由回答型问答，如 gsm8k
- generative：代码生成类，如 humaneval
- language_modeling：语言建模，如 wikitext2
"""

import pandas as pd
import re

def format_choices(choices) -> str:
    labels = [f"{chr(65 + i)})" for i in range(len(choices))]  # 65 是 'A'
    return "\n".join([f"{labels[i]} {choice}" for i, choice in enumerate(choices)])

def convert_to_letter(label) -> str:
    if isinstance(label, (int, float)) and not pd.isna(label):
        return chr(65 + int(label))
    elif isinstance(label, str) and label.isnumeric():
        return chr(65 + int(label))
    return str(label)

def process_gsm8k(df: pd.DataFrame) -> pd.DataFrame:
    df['question'] = df['question']
    df['label'] = ''
    df['gold'] = df['answer']
    df['rationale'] = ''
    df['corpus'] = df['question'] + "\n" + df['answer']
    df['task_type'] = 'gsm8k'
    df['task_format'] = 'generative'
    df['knowledge'] = ''
    return df[['task_type', 'task_format', 'question', 'label', 'gold', 'corpus', 'rationale', 'knowledge']]


def process_math_qa(df: pd.DataFrame) -> pd.DataFrame:
    def parse_options(option_str):
        pattern = r"[a-eA-E]\s*\)\s*(.*?)(?=\s*[a-eA-E]\s*\)|$)"
        return [opt.strip(" ,") for opt in re.findall(pattern, option_str)]

    df['choices'] = df['options'].apply(parse_options)
    df['question'] = df.apply(
        lambda row: f"{row['Problem']}\nOptions:\n{format_choices(row['choices'])}", axis=1)
    
    df['label'] = df['correct'].apply(lambda x: x.upper() if isinstance(x, str) else "")

    def get_gold(r):
        try:
            idx = ord(r['correct'].lower()) - ord('a')
            return r['choices'][idx].strip() if 0 <= idx < len(r['choices']) else ""
        except Exception:
            return ""

    df['gold'] = df.apply(get_gold, axis=1)
    df['corpus'] = df['Problem'] + " " + df['gold']
    df['rationale'] = df.get('Rationale', "").fillna("")
    df['task_type'] = 'math_qa'
    df['task_format'] = 'multiple_choice'
    df['knowledge'] = ''
    return df[['task_type', 'task_format', 'question', 'label', 'gold', 'corpus', 'rationale', 'knowledge']]

def process_arc_e(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    df['raw_question'] = df['question']
    df['question'] = df.apply(lambda row: f"{row['raw_question']}\nOptions:\n{format_choices(row['choices']['text'])}", axis=1)

    def get_label_and_gold(r):
        ak = r['answerKey']
        choices = r['choices']['text']
        if isinstance(ak, str) and ak.strip().isalpha():
            label = ak.strip().upper()
        elif isinstance(ak, str) and ak.strip().isdigit():
            label = chr(64 + int(ak))
        elif isinstance(ak, int):
            label = chr(64 + ak)
        else:
            label = ""
        gold_index = mapping.get(label, None)
        gold = choices[gold_index] if gold_index is not None and 0 <= gold_index < len(choices) else ""
        return label, gold

    df[['label', 'gold']] = df.apply(lambda r: pd.Series(get_label_and_gold(r)), axis=1)
    df['corpus'] = df['raw_question'] + " " + df['gold']
    df['rationale'] = ''
    df['task_type'] = 'arc_e'
    df['task_format'] = 'multiple_choice'
    df['knowledge'] = ''
    return df[['task_type', 'task_format', 'question', 'label', 'gold', 'corpus', 'rationale', 'knowledge']]

def process_arc_c(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    df['raw_question'] = df['question']
    df['question'] = df.apply(lambda row: f"{row['raw_question']}\nOptions:\n{format_choices(row['choices']['text'])}", axis=1)

    def get_label_and_gold(r):
        ak = r['answerKey']
        choices = r['choices']['text']
        if isinstance(ak, str) and ak.strip().isalpha():
            label = ak.strip().upper()
        elif isinstance(ak, str) and ak.strip().isdigit():
            label = chr(64 + int(ak))
        elif isinstance(ak, int):
            label = chr(64 + ak)
        else:
            label = ""
        gold_index = mapping.get(label, None)
        gold = choices[gold_index] if gold_index is not None and 0 <= gold_index < len(choices) else ""
        return label, gold

    df[['label', 'gold']] = df.apply(lambda r: pd.Series(get_label_and_gold(r)), axis=1)
    df['corpus'] = df['raw_question'] + " " + df['gold']
    df['rationale'] = ''
    df['task_type'] = 'arc_c'
    df['task_format'] = 'multiple_choice'
    df['knowledge'] = ''
    return df[['task_type', 'task_format', 'question', 'label', 'gold', 'corpus', 'rationale', 'knowledge']]

def process_obqa(df: pd.DataFrame) -> pd.DataFrame:
    df['question'] = df.apply(lambda row: f"{row['question_stem']}\nOptions:\n{format_choices(row['choices']['text'])}", axis=1)
    df[['label', 'gold']] = df.apply(lambda r: pd.Series([r['answerKey'], r['choices']['text'][ord(r['answerKey']) - ord('A')]]), axis=1)
    df['corpus'] = df['fact1'].str.capitalize() + ". " + df['question_stem'] + " " + df['gold']
    df['rationale'] = ''
    df['task_type'] = 'obqa'
    df['task_format'] = 'multiple_choice'
    df['knowledge'] = ''
    return df[['task_type', 'task_format', 'question', 'label', 'gold', 'corpus', 'rationale', 'knowledge']]

def process_piqa(df: pd.DataFrame) -> pd.DataFrame:
    original_label = df['label'].copy()
    df['question'] = df.apply(lambda row: f"{row['goal']}\nOptions:\nA) {row['sol1']}\nB) {row['sol2']}", axis=1)
    df['label'] = original_label.apply(lambda x: 'A' if x == 0 else 'B')
    df['gold'] = df.apply(lambda r: r['sol1'] if original_label.loc[r.name] == 0 else r['sol2'], axis=1)
    df['corpus'] = df['goal'] + " " + df['gold'].str.capitalize()
    df['rationale'] = ''
    df['task_type'] = 'piqa'
    df['task_format'] = 'multiple_choice'
    df['knowledge'] = ''
    return df[['task_type', 'task_format', 'question', 'label', 'gold', 'corpus', 'rationale', 'knowledge']]

def process_hellaswag(df: pd.DataFrame) -> pd.DataFrame:
    original_label = df['label'].copy()
    df['question'] = df.apply(lambda row: f"{row['ctx']}\nOptions:\n{format_choices(row['endings'])}", axis=1)
    df['label'] = original_label.apply(lambda x: convert_to_letter(x))
    df['gold'] = df.apply(lambda r: r['endings'][int(original_label.loc[r.name])], axis=1)
    df['corpus'] = df['ctx'] + " " + df['gold']
    df['rationale'] = ''
    df['task_type'] = 'hellaswag'
    df['task_format'] = 'multiple_choice'
    df['knowledge'] = ''
    return df[['task_type', 'task_format', 'question', 'label', 'gold', 'corpus', 'rationale', 'knowledge']]

def process_winogrande(df: pd.DataFrame) -> pd.DataFrame:
    df['question'] = df.apply(lambda row: f"{row['sentence']}\nOptions:\nA) {row['option1']}\nB) {row['option2']}", axis=1)
    df['label'] = df['answer'].apply(lambda x: 'A' if x == '1' else 'B')
    df['gold'] = df.apply(lambda r: r['option1'] if r['answer'] == '1' else r['option2'], axis=1)
    df['corpus'] = df.apply(lambda r: r['sentence'].replace('_', r['gold']), axis=1)
    df['rationale'] = ''
    df['task_type'] = 'winogrande'
    df['task_format'] = 'multiple_choice'
    df['knowledge'] = ''
    return df[['task_type', 'task_format', 'question', 'label', 'gold', 'corpus', 'rationale', 'knowledge']]

def process_boolq(df: pd.DataFrame) -> pd.DataFrame:
    df['question'] = df.apply(lambda row: f"{row['passage']}\n{row['question'].capitalize()}?", axis=1)
    df['label'] = df['answer'].apply(lambda x: 'Yes' if x else 'No')
    df['gold'] = ''
    df['corpus'] = df.apply(lambda r: f"{r['question']} {r['label']}", axis=1)
    df['rationale'] = ""
    df['task_type'] = 'boolq'
    df['task_format'] = 'multiple_choice'
    df['knowledge'] = ''
    return df[['task_type', 'task_format', 'question', 'label', 'gold', 'corpus', 'rationale', 'knowledge']]

def process_humaneval(df: pd.DataFrame) -> pd.DataFrame:
    df['question'] = df['entry_point'].apply(
        lambda name: f"Implement the function `{name}` as specified below:\n\n"
    ) + df['prompt']
    df['label'] = ''
    df['gold'] = df['prompt'] + df['canonical_solution']
    df['corpus'] = df['gold']

    df['rationale'] = ""
    df['task_type'] = 'humaneval'
    df['task_format'] = 'generative'
    df['knowledge'] = ''
    return df[['task_type', 'task_format', 'question', 'label', 'gold', 'corpus', 'rationale', 'knowledge']]

def process_mbpp(df: pd.DataFrame) -> pd.DataFrame:
    df['question'] = df['prompt']
    df['label'] = ''
    df['gold'] = df['code']
    df['corpus'] = df['prompt'] + "\n" + df['code']
    df['rationale'] = ''
    df['task_type'] = 'mbpp'
    df['task_format'] = 'generative'
    df['knowledge'] = ''
    return df[['task_type', 'task_format', 'question', 'label', 'gold', 'corpus', 'rationale', 'knowledge']]

def process_wikitext2(df: pd.DataFrame, max_length: int = 512) -> pd.DataFrame:
    df['text'] = df['text'].astype(str).str.slice(0, max_length)
    df['question'] = df['text']
    df['label'] = df['text']
    df['gold'] = df['text']
    df['corpus'] = df['text']
    df['rationale'] = ''
    df['task_type'] = 'wikitext2'
    df['task_format'] = 'language_modeling'
    df['knowledge'] = ''
    return df[['task_type', 'task_format', 'question', 'label', 'gold', 'corpus', 'rationale', 'knowledge']]

def process_c4(df: pd.DataFrame, max_length: int = 512) -> pd.DataFrame:
    df['text'] = df['text'].astype(str).str.slice(0, max_length)
    df['question'] = df['text']
    df['label'] = df['text']
    df['gold'] = df['text']
    df['corpus'] = df['text']
    df['rationale'] = ''
    df['task_type'] = 'c4'
    df['task_format'] = 'language_modeling'
    df['knowledge'] = ''
    return df[['task_type', 'task_format', 'question', 'label', 'gold', 'corpus', 'rationale', 'knowledge']]

# 注册处理函数
TASK_PROCESSORS = {
    "gsm8k": process_gsm8k,
    "math_qa": process_math_qa,
    "arc_e": process_arc_e,
    "arc_c": process_arc_c,
    "obqa": process_obqa,
    "piqa": process_piqa,
    "hellaswag": process_hellaswag,
    "winogrande": process_winogrande,
    "boolq": process_boolq,
    "humaneval": process_humaneval,
    "mbpp": process_mbpp,
    "wikitext2": process_wikitext2,
    "c4": process_c4,
}
