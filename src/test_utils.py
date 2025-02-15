# src/test_utils.py

import os
import re
import torch
from typing import List, Dict, Any, Tuple
import random
import json
import hashlib
from tqdm import tqdm

def create_test_prompts_and_answers(
    inputs: List[str],
    labels: List[str],
    task_types: List[str],
    sample_size: int = 10,
    seed: int = 42
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    从 (inputs, labels, task_types) 并行列表中抽取样本并构建测试提示和答案字典。

    Args:
        inputs (List[str]): 输入文本列表，与 labels 和 task_types 一一对应。
        labels (List[str]): 对应输入的答案列表，与 inputs 和 task_types 一一对应。
        task_types (List[str]): 对应输入的任务类型列表。
        sample_size (int): 每类任务抽样数量。
        seed (int): 随机种子，确保抽样一致性。

    Returns:
        (dict, dict):
            test_prompts: {task_type: [prompt, ...]}
            test_answers: {task_type: [answer, ...]}
    """
    # 设置随机种子
    random.seed(seed)

    test_prompts = {}
    test_answers = {}

    # 按任务类型分组
    task_groups = {}
    for inp, lab, ttype in zip(inputs, labels, task_types):
        if ttype not in task_groups:
            task_groups[ttype] = []
        task_groups[ttype].append((inp, lab))
    
    # 从每类任务中抽取样本
    for ttype, items in task_groups.items():
        sampled_items = random.sample(items, min(sample_size, len(items)))
        prompts = [x[0] for x in sampled_items]
        ans = [x[1] for x in sampled_items]

        test_prompts[ttype] = prompts
        test_answers[ttype] = ans
    
    return test_prompts, test_answers

def test_model_on_prompts(model, prompts: List[str], tokenizer, generation_args: dict = None) -> List[str]:
    """
    逐个处理输入提示，生成输出并返回结果。

    Args:
        model: 预训练模型
        prompts (List[str]): 测试提示列表
        tokenizer: 模型的分词器
        generation_args (dict): 生成参数字典

    Returns:
        List[str]: 生成的模型输出文本列表
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    outputs = []
    model.eval()

    for prompt in tqdm(prompts, desc="Testing prompts", unit="prompt"):
        inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
        attention_mask = inputs['attention_mask']
        with torch.no_grad():
            output_ids = model.generate(
                inputs['input_ids'],
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                **(generation_args or {})
            )
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        outputs.append(output_text)

    return outputs

def extract_answer(output_text: str) -> str:
    """
    从模型输出中提取答案选项，假设答案可能在 "Choice:" 或 "Answer:" 后。

    Args:
        output_text (str): 模型生成的文本输出。

    Returns:
        str: 提取的答案选项，若未找到则返回空字符串。
    """
    match = re.search(r'(Choice|Answer):\s*([A-Z])', output_text)
    return match.group(2) if match else ""

def calculate_accuracy(predictions: List[str], actuals: List[str]) -> float:
    """
    计算预测答案的准确率。

    Args:
        predictions (List[str]): 预测的答案列表。
        actuals (List[str]): 实际的答案列表。

    Returns:
        float: 准确率（百分比）。
    """
    correct = sum(p == a for p, a in zip(predictions, actuals))
    return (correct / len(actuals) * 100) if actuals else 0.0

def get_config_hash(pruning_config: dict) -> str:
    """
    计算稀疏配置的哈希值。

    Args:
        pruning_config (dict): 剪枝配置字典。

    Returns:
        str: 配置的哈希值。
    """
    config_str = json.dumps(pruning_config, sort_keys=True)
    return hashlib.md5(config_str.encode('utf-8')).hexdigest()

def save_test_results(
    outputs: List[str],
    predicted_answers: List[str],
    actual_answers: List[str],
    accuracy: float,
    experiment_dir: str
):
    """
    保存测试结果，包括模型输出、预测答案、正确答案和准确率。

    Args:
        outputs (List[str]): 模型生成的输出。
        predicted_answers (List[str]): 模型预测的答案。
        actual_answers (List[str]): 正确的答案。
        accuracy (float): 测试准确率。
        experiment_dir (str): 实验目录路径。
    """
    detailed_results = []
    for i, output in enumerate(outputs):
        detailed_results.append({
            'output': output,
            'predicted_answer': predicted_answers[i],
            'actual_answer': actual_answers[i]
        })

    detailed_results_file = os.path.join(experiment_dir, 'detailed_results.json')
    with open(detailed_results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=4)

    test_results = {
        'accuracy': accuracy,
        'total_samples': len(outputs)
    }
    test_results_file = os.path.join(experiment_dir, 'test_results.json')
    with open(test_results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=4)

def save_generation_args(generation_args: dict, experiment_dir: str):
    """
    保存生成参数到实验目录中。

    Args:
        generation_args (dict): 生成参数。
        experiment_dir (str): 实验目录路径。
    """
    generation_args_file = os.path.join(experiment_dir, 'generation_args.json')
    with open(generation_args_file, 'w', encoding='utf-8') as f:
        json.dump(generation_args, f, indent=4)
