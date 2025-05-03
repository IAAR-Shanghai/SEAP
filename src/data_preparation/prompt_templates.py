# src/data_preparation/prompt_templates.py
"""
Prompt templates used to generate
  • rationale  (step-by-step, short)
  • knowledge (concise, transferable fact/summary)

Keys:
    TASK_FORMAT_TEMPLATES[task_format]["rationale" | "knowledge"]
Each value is a lambda that receives one row (dict-like) and returns a prompt str.
"""

TASK_FORMAT_TEMPLATES = {
    # -----------------------------------------------------------
    # 1) Multiple-choice reasoning & knowledge
    # -----------------------------------------------------------
    "multiple_choice": {
        "rationale": lambda row: (
            # Role & goal
            "You are an expert reasoning assistant. Analyse the following multiple-choice "
            "question, rule out wrong options, and reach the answer.\n"
            # Content
            f"\nQuestion:\n{row['question']}\n"
            # Output format & brevity
            "\n--- FORMAT ---\n"
            "Explanation (≤ 3 sentences):\n<your concise reasoning>\n"
            f"Final Answer: {row['label']}) {row['gold']}"
        ),

        "knowledge": lambda row: (
            "You are a concise tutor. Provide the core fact(s) or principle(s) that make the "
            "correct option evident. Keep it reusable for similar problems.\n"
            f"\nQuestion (for context):\n{row['question']}\n"
            f"Extra context:\n{row.get('corpus', '').strip()}\n"
            "\n--- FORMAT ---\n"
            "Knowledge Statement (≤ 50 words):\n<key concept + 1-line application>"
        ),
    },

    # -----------------------------------------------------------
    # 2) Generative tasks  (math explanations OR code synthesis)
    # -----------------------------------------------------------
    "generative": {
        "rationale": lambda row: (
            "You are a problem-solving assistant. Briefly outline your approach, then give the "
            "compact solution.\n"
            f"\nTask:\n{row['question']}\n"
            "\n--- FORMAT ---\n"
            "Explanation (≤ 4 sentences):\n<idea / algorithm / key steps>\n"
            "Solution:\n<final numeric answer OR code>"
        ),

        "knowledge": lambda row: (
            "You are a senior instructor. Summarise the principle or algorithm that solves the "
            "task, then illustrate it with a minimal working example. Keep it short.\n"
            f"\nTask:\n{row['question']}\n"
            f"Reference snippet:\n{row.get('corpus', '').strip()}\n"
            "\n--- FORMAT ---\n"
            "Knowledge Statement (≤ 60 words):\n<concept / formula / pattern>\n"
            "Demo (optional, ≤ 10 lines):\n<illustrative code or worked step>"
        ),
    },

    # -----------------------------------------------------------
    # 3) Language-modeling style passages
    # -----------------------------------------------------------
    "language_modeling": {
        "rationale": lambda row: (
            "You are a language analyst. Provide a VERY brief note on the passage’s style, "
            "topic, or structure. No answer is required.\n"
            f"\nPassage:\n{row['question']}\n"
            "\n--- FORMAT ---\n"
            "Explanation (≤ 2 sentences):\n<pattern/theme/topic>"
        ),

        "knowledge": lambda row: (
            "You are a linguistic summariser. Capture the key thematic or stylistic features of "
            "the text so it could be reused as calibration knowledge. Be succinct.\n"
            f"\nText:\n{row['corpus']}\n"
            "\n--- FORMAT ---\n"
            "Knowledge Statement (≤ 40 words):\n<summary of topic, genre, style>"
        ),
    },
}
