# src/model_utils.py

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict, Any

def load_model_and_tokenizer(model_name_or_path: str, device='cuda'):
    """
    Load a model and tokenizer from the specified path.

    Args:
        model_name_or_path (str): Path to the model.

    Returns:
        model, tokenizer: Loaded model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        device_map=device, 
        torch_dtype="auto", 
        trust_remote_code=True, 
        output_hidden_states=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def get_hidden_states(prompt: str, model, tokenizer, device: str = 'cuda') -> np.ndarray:
    """
    Get hidden states for a single prompt without batching or padding.

    Args:
        prompt (str): Text input.
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        device (str): Device to run the model on.

    Returns:
        np.ndarray: Hidden states for each layer, averaged across sequence length.
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    
    # Extract and process hidden states for each layer
    hidden_states = outputs.hidden_states  # tuple of shape (num_layers, 1, seq_len, hidden_size)
    num_layers = len(hidden_states)
    
    # Average pool over the sequence length and convert to float32 before moving to numpy array
    pooled_hidden_states = np.array([
        layer_hidden_state.mean(dim=1).squeeze().to(torch.float32).cpu().numpy()  # shape: (hidden_size,)
        for layer_hidden_state in hidden_states
    ])  # Final shape: (num_layers, hidden_size)
    
    return pooled_hidden_states

def collect_hidden_states(inputs: List[str], task_types: List[str], model, tokenizer, device='cuda') -> Tuple[List[np.ndarray], List[str]]:
    hidden_states_list = []
    labels = []

    for inp, ttype in tqdm(zip(inputs, task_types), total=len(inputs), desc="Collecting Hidden States"):
        hidden_states = get_hidden_states(inp, model, tokenizer, device)
        hidden_states_list.append(hidden_states)
        labels.append(ttype)

    return hidden_states_list, labels

def create_task_type_mapping(task_types: List[str]) -> Dict[str, int]:
    """
    Given a list of task_types, create a mapping from task_type string to an integer label.
    """
    task_type_to_label = {}
    label_counter = 0
    for ttype in task_types:
        if ttype not in task_type_to_label:
            task_type_to_label[ttype] = label_counter
            label_counter += 1
    return task_type_to_label

def get_embeddings(
    inputs: List[str],
    task_types: List[str],
    model,
    tokenizer,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Compute embeddings for a list of inputs along with their task types.
    
    Args:
        inputs (List[str]): A list of input text strings.
        task_types (List[str]): A list of corresponding task types, same length as inputs.
        model: The loaded model with embeddings.
        tokenizer: The loaded tokenizer.
        device (str): The device on which to run computations.
        
    Returns:
        Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
            embeddings: (N, hidden_size) numpy array of embeddings.
            labels: (N,) numpy array of integer labels corresponding to task_types.
            task_type_to_label: A dictionary mapping from task_type string to integer label.
    """
    # 创建任务类型的标签映射
    task_type_to_label = create_task_type_mapping(task_types)
    
    embeddings = []
    labels = []
    
    for inp, ttype in tqdm(zip(inputs, task_types), total=len(inputs), desc="Generating Embeddings", unit="input"):
        label = task_type_to_label[ttype]
        
        # Tokenize单个输入，不需要填充
        encoded = tokenizer(inp, return_tensors='pt').to(device)
        input_ids = encoded['input_ids']

        with torch.no_grad():
            token_embeddings = model.get_input_embeddings()(input_ids)
        
        # 对序列长度维度求平均，并移至CPU
        embedding = token_embeddings.mean(dim=1).float().cpu().numpy()

        embeddings.append(embedding)
        labels.append(label)

    # 将嵌入和标签转换为NumPy数组
    embeddings = np.array(embeddings).squeeze()
    labels = np.array(labels)

    return embeddings, labels, task_type_to_label


