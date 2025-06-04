import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import time
import logging
from typing import List, Dict, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GenerationConfig:
    """Configuration class for text generation parameters."""
    def __init__(self, **kwargs):
        self.max_new_tokens: int = kwargs.get('max_new_tokens', 300)
        self.min_new_tokens: int = kwargs.get('min_new_tokens', 200)
        self.temperature: float = kwargs.get('temperature', 0.1)
        self.do_sample: bool = kwargs.get('do_sample', False)
        self.top_k: int = kwargs.get('top_k', 3)
        self.penalty_alpha: float = kwargs.get('penalty_alpha', 0.6)
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            'max_new_tokens': self.max_new_tokens,
            'min_new_tokens': self.min_new_tokens,
            'temperature': self.temperature,
            'do_sample': self.do_sample,
            'top_k': self.top_k,
            'penalty_alpha': self.penalty_alpha
        }

def setup_model_and_tokenizer(model_root_path: str, model_name: str) -> tuple:
    """
    Load and setup the model and tokenizer.
    
    Args:
        model_root_path: Root directory where models are stored
        model_name: Name of the specific model to load
        
    Returns:
        tuple: (model, tokenizer)
        
    Raises:
        Exception: If model loading fails
    """
    model_path = os.path.join(model_root_path, model_name)
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Try loading with AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        logger.info("Model loaded successfully using AutoModelForCausalLM")
    except Exception as e:
        logger.warning(f"Failed to load model using AutoModelForCausalLM: {e}")
        logger.info("Attempting to load using torch.load...")
        
        try:
            model = torch.load(
                os.path.join(model_path, "pruned_model.pt"),
                map_location=torch.device('cuda')
            )
            logger.info("Model loaded successfully using torch.load")
        except Exception as e2:
            raise Exception(f"Failed to load model using both methods: {e2}")
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        raise Exception(f"Failed to load tokenizer: {e}")
    
    return model, tokenizer

def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    config: GenerationConfig,
    device: torch.device
) -> List[str]:
    """
    Generate text for a list of prompts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompts: List of input prompts
        config: Generation configuration
        device: Device to use for generation
        
    Returns:
        List[str]: Generated texts
    """
    model.to(device)
    model.eval()
    
    generated_results = []
    total_tokens = 0
    
    # Process all prompts and generate input_ids
    input_ids_list = []
    for prompt in prompts:
        with torch.no_grad():
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            if len(input_ids) != 1:
                logger.warning(f"Unexpected input_ids shape: {input_ids.shape}")
            if input_ids[0][-1] == tokenizer.eos_token_id:
                input_ids = input_ids[:, :-1]
            input_ids_list.append(input_ids.to(device))
    
    # Generate text for each prompt
    start_time = time.time()
    
    for i, input_ids in enumerate(input_ids_list):
        try:
            generated_ids = model.generate(
                input_ids,
                **config.to_dict()
            )
            result = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0]
            generated_results.append(result)
            
            # Count tokens
            total_tokens += len(tokenizer.encode(result))
            
            logger.info(f"Generated text for prompt {i+1}/{len(prompts)}")
            
        except Exception as e:
            logger.error(f"Error generating text for prompt {i+1}: {e}")
            generated_results.append(f"[ERROR] Failed to generate text: {str(e)}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate statistics
    tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
    
    logger.info(f"Generation complete:")
    logger.info(f"- Total tokens: {total_tokens}")
    logger.info(f"- Time taken: {elapsed_time:.2f} seconds")
    logger.info(f"- Speed: {tokens_per_sec:.2f} tokens/second")
    
    return generated_results

def save_results(
    output_dir: str,
    model_name: str,
    prompts: List[str],
    generated_texts: List[str],
    config: GenerationConfig
) -> None:
    """
    Save generation results and configuration to files.
    
    Args:
        output_dir: Base output directory
        model_name: Name of the model
        prompts: Input prompts
        generated_texts: Generated texts
        config: Generation configuration
    """
    # Create output directory
    model_output_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Save individual results
    for i, (prompt, generated) in enumerate(zip(prompts, generated_texts)):
        output_file = os.path.join(model_output_dir, f"generation_{i+1}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Prompt: {prompt}\n\n")
            f.write(f"Generated Text:\n{generated}\n")
    
    # Save configuration
    config_file = os.path.join(model_output_dir, "generation_config.json")
    with open(config_file, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    logger.info(f"Results saved to {model_output_dir}")

def main(args: argparse.Namespace) -> None:
    """
    Main function to run text generation.
    
    Args:
        args: Command line arguments
    """
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(args.model_root_path, args.model_name)
        
        # Define generation configuration
        config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_k=args.top_k,
            penalty_alpha=args.penalty_alpha
        )
        
        # Define prompts
        prompts = [
            "AI can create a logo in seconds.",
            "What is McDonald's?",
        ]
        
        if args.prompts:
            prompts = args.prompts
        
        # Generate text
        generated_texts = generate_text(model, tokenizer, prompts, config, device)
        
        # Save results
        save_results(args.output_dir, args.model_name, prompts, generated_texts, config)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a pretrained language model.")
    
    # Model configuration
    parser.add_argument("--model_root_path", type=str, required=True,
                      help="Root directory where the model is stored")
    parser.add_argument("--model_name", type=str, required=True,
                      help="Model name or directory under model_root_path")
    
    # Generation configuration
    parser.add_argument("--max_new_tokens", type=int, default=300,
                      help="Maximum number of tokens to generate")
    parser.add_argument("--min_new_tokens", type=int, default=200,
                      help="Minimum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                      help="Sampling temperature")
    parser.add_argument("--do_sample", action="store_true",
                      help="Whether to use sampling for generation")
    parser.add_argument("--top_k", type=int, default=3,
                      help="Top-k sampling parameter")
    parser.add_argument("--penalty_alpha", type=float, default=0.6,
                      help="Penalty alpha for generation")
    
    # Input/Output configuration
    parser.add_argument("--prompts", nargs="+", type=str,
                      help="List of prompts for generation")
    parser.add_argument("--output_dir", type=str, default="./generated_outputs",
                      help="Directory to save generated text outputs")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    main(args)
