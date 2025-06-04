import os
import random
import torch
import torch.nn as nn
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_parquet_data(file_path):
    """
    Load data from a parquet file.

    Args:
        file_path (str): Path to the parquet file.

    Returns:
        DataFrame: Loaded data in pandas DataFrame format.
    
    Raises:
        FileNotFoundError: If the parquet file doesn't exist.
        pd.errors.EmptyDataError: If the parquet file is empty.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Parquet file not found: {file_path}")
    
    data = pd.read_parquet(file_path)
    
    if data.empty:
        raise pd.errors.EmptyDataError(f"Parquet file is empty: {file_path}")
    
    logger.info(f"Loaded data from {file_path}, shape: {data.shape}")
    
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
    
    Raises:
        ValueError: If nsamples or seqlen is invalid.
    """
    if nsamples <= 0:
        raise ValueError("nsamples must be positive")
    if seqlen <= 0:
        raise ValueError("seqlen must be positive")

    logger.info(f"Loading Wikitext-2 dataset with {nsamples} samples, sequence length {seqlen}")
    
    try:
        # Load train and test datasets
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
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        
        return trainloader, testenc
    
    except Exception as e:
        logger.error(f"Error loading Wikitext-2 dataset: {str(e)}")
        raise

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
    logger.info(f"Evaluating perplexity on wikitext2 dataset (seqlen={seqlen})")

    try:
        # Get the test loader
        _, testloader = get_wikitext2(128, seed=0, seqlen=seqlen, tokenizer=tokenizer)

        # Evaluate perplexity
        with torch.no_grad():
            ppl = eval_ppl_wikitext(model, testloader, 1, seqlen, device)
        return ppl
    
    except Exception as e:
        logger.error(f"Error during perplexity evaluation: {str(e)}")
        raise

def eval_ppl_wikitext(model, testenc, bs=1, seqlen=None, device=None):
    """
    Evaluate perplexity (PPL) specifically on the Wikitext dataset.

    Args:
        model (torch.nn.Module): The language model to be evaluated.
        testenc (TokenizerWrapper): Encoded input IDs from test set.
        bs (int): Batch size for evaluation.
        seqlen (int): Sequence length for the input samples.
        device (torch.device): Device to move data onto.

    Returns:
        float: The perplexity of the language model on the wikitext test dataset.
    """
    if not seqlen:
        raise ValueError("seqlen must be specified")
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Extract input IDs
        testenc = testenc.input_ids
        nsamples = testenc.numel() // seqlen
        nlls = []
        
        logger.info(f"Evaluating {nsamples} samples with batch size {bs}")

        # Process samples in batches
        for i in tqdm(range(0, nsamples, bs), desc="Processing samples"):
            j = min(i + bs, nsamples)
            inputs = testenc[:, (i * seqlen):(j * seqlen)].to(device)
            inputs = inputs.reshape(j-i, seqlen)
            
            # Get model predictions
            lm_logits = model(inputs).logits
            
            # Prepare for loss calculation
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]
            
            # Calculate loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), 
                          shift_labels.reshape(-1))
            
            neg_log_likelihood = loss.float() * seqlen * (j - i)
            nlls.append(neg_log_likelihood)

        # Calculate final perplexity
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        
        # Clean up
        torch.cuda.empty_cache()
        
        return ppl.item()

    except Exception as e:
        logger.error(f"Error during Wikitext perplexity evaluation: {str(e)}")
        raise

def main(args):
    """
    Main function to load the model, evaluate perplexity, and print results.

    Args:
        args: The command-line arguments containing model paths and other configurations.
    """
    try:
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Load model and tokenizer
        model_path = os.path.join(args.model_root_path, args.model_name)
        logger.info(f"Loading model from {model_path}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Ensure tokenizer has necessary tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Evaluate perplexity
        seqlen = args.seqlen or 512
        ppl = eval_ppl(model, tokenizer, seqlen, device)
        logger.info(f"Perplexity on Wikitext-2: {ppl:.2f}")

        # Save results if output path provided
        if args.output_file:
            os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
            with open(args.output_file, 'w') as f:
                f.write(f"Model: {args.model_name}\n")
                f.write(f"Perplexity: {ppl:.2f}\n")
            logger.info(f"Results saved to {args.output_file}")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate pruned models' PPL.")
    parser.add_argument("--model_root_path", type=str, required=True,
                      help="Root directory of the pruned model.")
    parser.add_argument("--model_name", type=str, required=True,
                      help="Name of the pruned model.")
    parser.add_argument("--seqlen", type=int, default=512,
                      help="Sequence length for evaluation (default: 512)")
    parser.add_argument("--output_file", type=str,
                      help="Path to save evaluation results")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")

    args = parser.parse_args()

    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    main(args)
