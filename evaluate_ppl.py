import os
import random
import torch
import torch.nn as nn
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm  # 引入 tqdm 库来显示进度条

def load_parquet_data(file_path):
    """
    Load data from a parquet file.

    Args:
        file_path (str): Path to the parquet file.

    Returns:
        DataFrame: Loaded data.
    """
    # Load the parquet file into a pandas DataFrame
    data = pd.read_parquet(file_path)
    
    print(f"Loaded data from {file_path}, shape: {data.shape}")
    
    return data

def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the Wikitext-2 dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded test dataset.
    """
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    
    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def eval_ppl(model, tokenizer, seqlen, device=torch.device("cuda:0")):
    """
    Evaluate perplexity (ppl) on a specified model and tokenizer.

    Args:
        model (torch.nn.Module): The language model to be evaluated.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.
        seqlen (int): Sequence length for the input samples.
        device (torch.device): Device to move data onto (e.g., 'cuda:0' or 'cpu').

    Returns:
        float: The perplexity of the language model on the test dataset.
    """

    # Print status
    print(f"Evaluating on wikitext2")

    # Get the test loader
    _, testloader = get_wikitext2(
        128, seed=0, seqlen=seqlen, tokenizer=tokenizer 
    )

    # Evaluate perplexity in no grad context to avoid updating the model
    with torch.no_grad():
        ppl = eval_ppl_wikitext(model, testloader, 1, seqlen, device)
    return ppl 

def eval_ppl_wikitext(model, testenc, bs=1, seqlen=None, device=None):
    """
    Evaluate perplexity (ppl) specifically on the wikitext dataset.

    Args:
        model (torch.nn.Module): The language model to be evaluated.
        testenc (TokenizerWrapper): Encoded input IDs from test set.
        bs (int): Batch size for evaluation.
        seqlen (int): Sequence length for the input samples.
        device (torch.device): Device to move data onto (e.g., 'cuda:0' or 'cpu').

    Returns:
        float: The perplexity of the language model on the wikitext test dataset.
    """
    # Get input IDs from the TokenizerWrapper instance
    testenc = testenc.input_ids

    # Calculate number of samples
    nsamples = testenc.numel() // seqlen

    # List to store negative log likelihoods
    nlls = []
    print(f"Evaluating {nsamples} samples...")

    # Use tqdm to display progress bar
    for i in tqdm(range(0, nsamples, bs), desc="Processing samples"):
        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
        inputs = inputs.reshape(j-i, seqlen)
        
        # Forward pass through the model
        lm_logits = model(inputs).logits    

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * seqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return ppl.item()


def main(args):
    # Load the model and tokenizer
    model_path = os.path.join(args.model_root_path, args.model_name)
    print(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Set your desired sequence length
    seqlen = 512  # Example: sequence length to use for evaluation

    # Evaluate perplexity
    ppl = eval_ppl(model, tokenizer, seqlen)
    print(f"Perplexity on the dataset: {ppl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pruned models' PPL.")
    parser.add_argument("--model_root_path", type=str, required=True,
                        help="Root dir of the pruned model.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the pruned model (e.g., 'pruned_model_name').")
    args = parser.parse_args()

    main(args)
