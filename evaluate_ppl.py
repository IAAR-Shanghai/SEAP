import os
import random
import torch
import torch.nn as nn
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

def load_parquet_data(file_path):
    """
    Load data from a parquet file.

    Args:
        file_path (str): Path to the parquet file.

    Returns:
        DataFrame: Loaded data in pandas DataFrame format.
    """
    # Load the parquet file into a pandas DataFrame
    data = pd.read_parquet(file_path)
    
    print(f"Loaded data from {file_path}, shape: {data.shape}")
    
    return data

def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the Wikitext-2 dataset for training and testing.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input-target pairs) and encoded test dataset.
    """
    # Load train and test datasets from HuggingFace
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    
    # Tokenize the text data
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate training samples
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100  # Ignore target for first token in each sequence
        trainloader.append((inp, tar))
    
    return trainloader, testenc

def eval_ppl(model, tokenizer, seqlen, device=torch.device("cuda:0")):
    """
    Evaluate perplexity (PPL) on a specified model and tokenizer.

    Args:
        model (torch.nn.Module): The language model to be evaluated.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.
        seqlen (int): Sequence length for the input samples.
        device (torch.device): Device to move data onto (e.g., 'cuda:0' or 'cpu').

    Returns:
        float: The perplexity of the language model on the test dataset.
    """
    # Print evaluation start message
    print(f"Evaluating perplexity on wikitext2 dataset")

    # Get the test loader (encoded test data)
    _, testloader = get_wikitext2(128, seed=0, seqlen=seqlen, tokenizer=tokenizer)

    # Evaluate perplexity using no_grad() to avoid unnecessary computation of gradients
    with torch.no_grad():
        ppl = eval_ppl_wikitext(model, testloader, 1, seqlen, device)
    return ppl

def eval_ppl_wikitext(model, testenc, bs=1, seqlen=None, device=None):
    """
    Evaluate perplexity (PPL) specifically on the Wikitext dataset.

    Args:
        model (torch.nn.Module): The language model to be evaluated.
        testenc (TokenizerWrapper): Encoded input IDs from test set.
        bs (int): Batch size for evaluation.
        seqlen (int): Sequence length for the input samples.
        device (torch.device): Device to move data onto (e.g., 'cuda:0' or 'cpu').

    Returns:
        float: The perplexity of the language model on the wikitext test dataset.
    """
    # Extract input IDs from the TokenizerWrapper instance
    testenc = testenc.input_ids

    # Calculate the total number of samples based on sequence length
    nsamples = testenc.numel() // seqlen

    # List to store negative log likelihood values for perplexity calculation
    nlls = []
    print(f"Evaluating {nsamples} samples...")

    # Use tqdm to display a progress bar while processing the samples
    for i in tqdm(range(0, nsamples, bs), desc="Processing samples"):
        # Calculate the end index for the current batch
        j = min(i + bs, nsamples)

        # Prepare the inputs and move them to the specified device (CPU/GPU)
        inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
        inputs = inputs.reshape(j-i, seqlen)
        
        # Perform a forward pass through the model to get logits
        lm_logits = model(inputs).logits    

        # Shift logits and labels for predicting the next token
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss using cross-entropy loss function
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate the negative log likelihood for this batch
        neg_log_likelihood = loss.float() * seqlen * (j - i)

        # Append the negative log likelihood to the list
        nlls.append(neg_log_likelihood)

    # Compute the final perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

    # Empty the CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()

def main(args):
    """
    Main function to load the model, evaluate perplexity, and print results.

    Args:
        args: The command-line arguments containing model paths and other configurations.
    """
    # Load the model and tokenizer from the specified directory
    model_path = os.path.join(args.model_root_path, args.model_name)
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Set desired sequence length for evaluation
    seqlen = 512  # Example: sequence length for evaluation

    # Evaluate perplexity on the dataset
    ppl = eval_ppl(model, tokenizer, seqlen)
    print(f"Perplexity on the dataset: {ppl}")

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Evaluate pruned models' PPL.")
    parser.add_argument("--model_root_path", type=str, required=True,
                        help="Root directory of the pruned model.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the pruned model (e.g., 'pruned_model_name').")
    args = parser.parse_args()

    # Execute main function
    main(args)
