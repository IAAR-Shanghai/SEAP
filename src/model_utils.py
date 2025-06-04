"""
Model and tokenizer utilities for handling transformer models.

This module provides utility functions for loading, managing and extracting
features from transformer models. It includes functions for getting hidden states
and computing embeddings from input text.

Author: why
Date: 2024
"""

# Standard library imports
from typing import List, Tuple, Dict, Any

# Third-party imports
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Constants
DEFAULT_DEVICE = 'cuda'
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

def load_model_and_tokenizer(
    model_name_or_path: str,
    device: str = 'auto'
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load pretrained model and corresponding tokenizer.
    
    Args:
        model_name_or_path: Model identifier or path
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        Tuple containing loaded model and tokenizer
        
    Example:
        >>> model, tokenizer = load_model_and_tokenizer("gpt2", device="cuda")
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        device_map=device,  # Automatically decides where to load model layers
        torch_dtype="auto", 
        trust_remote_code=True,  # To trust model code (optional)
        output_hidden_states=True  # Ensures hidden states are returned
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def get_hidden_states(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str = DEFAULT_DEVICE
) -> np.ndarray:
    """Get hidden states for a single input text.
    
    Encodes input text and retrieves hidden states from each layer,
    then applies mean pooling over sequence length.
    
    Args:
        prompt: Input text
        model: Pretrained model
        tokenizer: Corresponding tokenizer
        device: Computation device
        
    Returns:
        Array of shape (num_layers, hidden_size) containing hidden states
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    
    # Extract hidden states (tuple of shape (num_layers, 1, seq_len, hidden_size))
    hidden_states = outputs.hidden_states  
    num_layers = len(hidden_states)
    
    # Average pooling over sequence length (dim=1) and convert to numpy
    pooled_hidden_states = np.array([
        layer_hidden_state.mean(dim=1).squeeze().to(torch.float32).cpu().numpy()  # shape: (hidden_size,)
        for layer_hidden_state in hidden_states
    ])  # Final shape: (num_layers, hidden_size)
    
    return pooled_hidden_states


def collect_hidden_states(
    inputs: List[str],
    task_types: List[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str = DEFAULT_DEVICE
) -> Tuple[List[np.ndarray], List[str]]:
    """Collect hidden states for multiple input texts.
    
    Args:
        inputs: List of input texts
        task_types: List of corresponding task types
        model: Pretrained model
        tokenizer: Corresponding tokenizer
        device: Computation device
        
    Returns:
        Tuple containing:
            - List of hidden states arrays, each of shape (num_layers, hidden_size)
            - List of corresponding task type labels
    """
    hidden_states_list = []
    labels = []

    for inp, ttype in zip(inputs, task_types):
        hidden_states = get_hidden_states(inp, model, tokenizer, device)
        hidden_states_list.append(hidden_states)
        labels.append(ttype)

    return hidden_states_list, labels


def create_task_type_mapping(task_types: List[str]) -> Dict[str, int]:
    """Create mapping from task types to integer labels.
    
    Args:
        task_types: List of task type strings
        
    Returns:
        Dictionary mapping task types to integer labels
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
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str = DEFAULT_DEVICE
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Compute embeddings for input texts.
    
    For each input text, gets token embeddings through model's embedding layer,
    then applies mean pooling over sequence length to get fixed-size representation.
    
    Args:
        inputs: List of input texts
        task_types: List of corresponding task types
        model: Pretrained model
        tokenizer: Corresponding tokenizer
        device: Computation device
        
    Returns:
        Tuple containing:
            - Embedding matrix of shape (N, hidden_size)
            - Integer label array of shape (N,)
            - Dictionary mapping task types to integer labels
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
