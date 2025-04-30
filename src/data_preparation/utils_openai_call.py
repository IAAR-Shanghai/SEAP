import os
import time
from openai import OpenAI
from src.data_preparation.prompt_templates import TASK_FORMAT_TEMPLATES

def call_openai(prompt: str, temperature: float = 0.3, max_tokens: int = 512,
                retries: int = 3, delay: float = 2.0,
                api_key: str = None, base_url: str = None) -> str:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.claudeplus.top/v1")

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


def build_rationale_prompt(row):
    fmt = row["task_format"]
    return TASK_FORMAT_TEMPLATES[fmt]["rationale"](row)

def build_knowledge_prompt(row):
    fmt = row["task_format"]
    return TASK_FORMAT_TEMPLATES[fmt]["knowledge"](row)


def process_row(i, row, temperature=0.3, max_tokens=512, dry_run=False):
    if dry_run:
        rationale = "[DRY RUN] explanation...\nFinal Answer: ..."
        knowledge = "Knowledge Statement: [DRY RUN] ..."
    else:
        rationale = call_openai(build_rationale_prompt(row), temperature, max_tokens)
        knowledge = call_openai(build_knowledge_prompt(row), temperature, max_tokens)
    return i, rationale, knowledge
