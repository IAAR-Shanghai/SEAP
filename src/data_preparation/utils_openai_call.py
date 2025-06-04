"""
OpenAI API utilities for generating rationales and knowledge statements.

This module provides functions for making API calls to OpenAI models to generate
explanations and knowledge statements for different task formats.

Author: why
Date: 2024
"""

# Standard library imports
import os
import time
from typing import Tuple

# Third-party imports
from openai import OpenAI

# Local imports
from src.data_preparation.prompt_templates import TASK_FORMAT_TEMPLATES


def call_openai(
    prompt: str,
    temperature: float = 0.3,
    max_tokens: int = 512,
    retries: int = 3,
    delay: float = 2.0,
    api_key: str = None,
    base_url: str = None
) -> str:
    """Make API call to OpenAI with retry logic.
    
    Args:
        prompt: Input text to send to the model
        temperature: Sampling temperature (higher = more random)
        max_tokens: Maximum number of tokens to generate
        retries: Number of retry attempts on failure
        delay: Delay between retries in seconds
        api_key: OpenAI API key (optional, can use env var)
        base_url: Base URL for API (optional, can use env var)
        
    Returns:
        Generated text from the model
        
    Raises:
        ValueError: If API key is not provided
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("Missing OpenAI API key. Set it via argument or environment variable OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[call_openai] Retry {attempt+1}/{retries} failed: {e}")
            time.sleep(delay)
    return "[Error] Failed after retries"


def build_rationale_prompt(row: dict) -> str:
    """Build prompt for generating task rationale.
    
    Args:
        row: Dictionary containing task data
        
    Returns:
        Formatted prompt for rationale generation
    """
    fmt = row["task_format"]
    return TASK_FORMAT_TEMPLATES[fmt]["rationale"](row)


def build_knowledge_prompt(row: dict) -> str:
    """Build prompt for generating knowledge statement.
    
    Args:
        row: Dictionary containing task data
        
    Returns:
        Formatted prompt for knowledge statement generation
    """
    fmt = row["task_format"]
    return TASK_FORMAT_TEMPLATES[fmt]["knowledge"](row)


def process_row(
    i: int,
    row: dict,
    temperature: float = 0.3,
    max_tokens: int = 512,
    dry_run: bool = False
) -> Tuple[int, str, str]:
    """Process a single row to generate rationale and knowledge.
    
    Args:
        i: Row index
        row: Dictionary containing task data
        temperature: Sampling temperature for API calls
        max_tokens: Maximum tokens for API calls
        dry_run: Whether to skip actual API calls
        
    Returns:
        Tuple containing:
            - Row index
            - Generated rationale
            - Generated knowledge statement
    """
    if dry_run:
        rationale = "[DRY RUN] explanation...\nFinal Answer: ..."
        knowledge = "Knowledge Statement: [DRY RUN] ..."
    else:
        rationale = call_openai(build_rationale_prompt(row), temperature, max_tokens)
        knowledge = call_openai(build_knowledge_prompt(row), temperature, max_tokens)
    return i, rationale, knowledge
