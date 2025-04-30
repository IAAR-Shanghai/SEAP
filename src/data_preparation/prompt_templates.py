# src/data_preparation/prompt_templates.py

"""
定义每种任务类型的 prompt 模板，包括 rationale 和 knowledge。
这些模板将在生成推理解释与知识陈述时调用，可按需自定义。
"""

TASK_FORMAT_TEMPLATES = {
    "multiple_choice": {
        "rationale": lambda row: (
            "You are a reasoning assistant. Carefully analyze the following multiple-choice question "
            "by considering each option and eliminating incorrect ones step by step.\n\n"
            f"Question:\n{row['question']}\n\n"
            "Explanation:\n<your reasoning here>\n\n"
            f"Final Answer: {row['label']}) {row['gold']}"
        ),
        "knowledge": lambda row: (
            "You are a knowledgeable tutor. Based on the question and the options provided, summarize the factual knowledge or principle required to answer correctly. "
            "Highlight any key definitions, facts, or reasoning heuristics involved.\n\n"
            f"Question:\n{row['question']}\n\n"
            f"Context (if any):\n{row['corpus']}\n\n"
            "Knowledge Statement:\n<concept + explanation + application>"
        )
    },

    "generative": {
        "rationale": lambda row: (
            "You are a programming assistant. Think step by step to solve the following task. "
            "Explain the key requirements, outline your logic, and finally write the implementation.\n\n"
            f"Prompt:\n{row['question']}\n\n"
            "Explanation:\n<your planning and thought process>\n\n"
            "Final Code:\n<your solution code>"
        ),
        "knowledge": lambda row: (
            "You are a software engineer writing a complete, instructional explanation of the code implementation. "
            "Include the problem's intent, applicable concepts (e.g., recursion, loops, edge cases), and then the final code.\n\n"
            f"Prompt:\n{row['question']}\n\n"
            f"Reference (from training):\n{row['corpus']}\n\n"
            "Knowledge Statement:\n<core concepts + implementation steps + clean code>"
        )
    },

    "language_modeling": {
        "rationale": lambda row: (
            "You are a language modeling assistant. Reflect briefly on the structure, topic, or tone of the given passage. "
            "No specific answer is needed, just a structural analysis.\n\n"
            f"Passage:\n{row['question']}\n\n"
            "Explanation:\n<text genre, style, or linguistic patterns>"
        ),
        "knowledge": lambda row: (
            "You are a language analyst. Provide a concise knowledge summary of the following passage, focusing on its linguistic or thematic features.\n\n"
            f"Text:\n{row['corpus']}\n\n"
            "Knowledge Statement:\n<linguistic or thematic insight>"
        )
    }
}
