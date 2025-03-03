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
        model_name_or_path (str): Path to the model or model identifier.
        device (str): Device to load the model on ('cuda' or 'cpu').

    Returns:
        model: Loaded pre-trained model.
        tokenizer: Loaded tokenizer corresponding to the model.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        device_map=device,  # Automatically decides where to load model layers
        torch_dtype="auto", 
        trust_remote_code=True,  # To trust model code (optional)
        output_hidden_states=True  # Ensures hidden states are returned
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return model, tokenizer


def get_hidden_states(prompt: str, model, tokenizer, device: str = 'cuda') -> np.ndarray:
    """
    Get hidden states for a single prompt without batching or padding.

    Args:
        prompt (str): Text input for the model.
        model: Pretrained model (e.g., AutoModelForCausalLM).
        tokenizer: Pretrained tokenizer.
        device (str): Device to run the model on, e.g., 'cuda' or 'cpu'.

    Returns:
        np.ndarray: Hidden states for each layer, averaged across the sequence length.
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    
    # Extract hidden states (tuple of shape (num_layers, 1, seq_len, hidden_size))
    hidden_states = outputs.hidden_states  
    num_layers = len(hidden_states)
    
    # Average pooling over the sequence length (dim=1) and convert to numpy
    pooled_hidden_states = np.array([
        layer_hidden_state.mean(dim=1).squeeze().to(torch.float32).cpu().numpy()  # shape: (hidden_size,)
        for layer_hidden_state in hidden_states
    ])  # Final shape: (num_layers, hidden_size)
    
    return pooled_hidden_states


def collect_hidden_states(inputs: List[str], task_types: List[str], model, tokenizer, device='cuda') -> Tuple[List[np.ndarray], List[str]]:
    """
    Collect hidden states for a list of input prompts along with their task types.

    Args:
        inputs (List[str]): List of input text prompts.
        task_types (List[str]): List of task types corresponding to each input.
        model: Pretrained model.
        tokenizer: Pretrained tokenizer.
        device (str): Device to run the model on.

    Returns:
        hidden_states_list: List of hidden states for each input prompt.
        labels: List of task type labels corresponding to each input.
    """
    hidden_states_list = []
    labels = []

    for inp, ttype in tqdm(zip(inputs, task_types), total=len(inputs), desc="Collecting Hidden States"):
        hidden_states = get_hidden_states(inp, model, tokenizer, device)
        hidden_states_list.append(hidden_states)
        labels.append(ttype)

    return hidden_states_list, labels


def create_task_type_mapping(task_types: List[str]) -> Dict[str, int]:
    """
    Create a mapping from task type string to integer labels.

    Args:
        task_types (List[str]): List of task type strings.

    Returns:
        task_type_to_label (Dict[str, int]): Mapping of task type to integer labels.
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
    Compute embeddings for a list of input texts along with their task types.

    Args:
        inputs (List[str]): List of input text strings.
        task_types (List[str]): List of corresponding task types (labels).
        model: The pre-trained model used to generate embeddings.
        tokenizer: The pre-trained tokenizer.
        device (str): The device for computation ('cuda' or 'cpu').

    Returns:
        Tuple containing:
            - embeddings: A numpy array of shape (N, hidden_size) representing the embeddings for each input.
            - labels: A numpy array of integer labels corresponding to task types.
            - task_type_to_label: A dictionary mapping task types to integer labels.
    """
    task_type_to_label = create_task_type_mapping(task_types)
    
    embeddings = []
    labels = []
    
    for inp, ttype in tqdm(zip(inputs, task_types), total=len(inputs), desc="Generating Embeddings", unit="input"):
        label = task_type_to_label[ttype]
        
        # Tokenize the input text
        encoded = tokenizer(inp, return_tensors='pt').to(device)
        input_ids = encoded['input_ids']

        with torch.no_grad():
            token_embeddings = model.get_input_embeddings()(input_ids)
        
        # Average over sequence length (dim=1) to obtain a fixed-size embedding per input
        embedding = token_embeddings.mean(dim=1).float().cpu().numpy()

        embeddings.append(embedding)
        labels.append(label)

    # Convert lists to numpy arrays
    embeddings = np.array(embeddings).squeeze()  # Remove unnecessary dimensions
    labels = np.array(labels)

    return embeddings, labels, task_type_to_label
