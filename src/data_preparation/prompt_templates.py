TASK_FORMAT_TEMPLATES = {
    "multiple_choice": {
        "rationale": lambda row: (
            "You are a reasoning assistant. Briefly explain why the correct answer is right by ruling out incorrect choices.\n\n"
            "Example:\n"
            "Question:\nWhich material conducts electricity?\n"
            "Answer:\nCopper\n"
            "Rationale:\nCopper is a metal with free electrons that allow electric current to flow. Other materials like wood or rubber are insulators.\n\n"
            f"Question:\n{row['question']}\n"
            f"Answer:\n{row['gold']}\n"
            "Rationale:\n<short reasoning for answer>"
        ),

        "knowledge": lambda row: (
            "You are a scientific annotator. Based on the following multiple-choice question and its correct answer, "
            "write a short, reusable knowledge statement that supports the correct choice.\n\n"
            "Example:\n"
            "Question:\nWhich material conducts electricity?\n"
            "Answer:\nCopper\n"
            "Knowledge Statement:\nCopper is a metal and a good conductor of electricity due to its free electrons.\n\n"
            f"Question:\n{row['question']}\n"
            f"Answer:\n{row['gold']}\n"
            "Knowledge Statement:\n<core fact or concept that explains the answer>"
        )
    },

    "generative": {
        "rationale": lambda row: (
            "You are a math or programming assistant. Explain your reasoning briefly before giving the final result.\n\n"
            "Example (math):\n"
            "Problem:\nTom has 3 apples and buys 5 more. How many apples does he have?\n"
            "Rationale:\nTom starts with 3 apples and gets 5 more. 3 + 5 = 8.\n\n"
            "Example (code):\n"
            "Problem:\nWrite a function that returns the square of a number.\n"
            "Rationale:\nTo square a number, multiply it by itself.\n\n"
            f"Problem:\n{row['question']}\n"
            "Rationale:\n<short explanation before final answer>"
        ),

        "knowledge": lambda row: (
            "You are a knowledgeable assistant.\n"
            "If the problem is mathematical, describe the solution idea and final steps as a reusable explanation.\n"
            "If the problem involves coding, explain the function's purpose and summarize the algorithm with annotated code.\n\n"
            "Example (math):\n"
            "Problem:\nTom has 3 apples and buys 5 more. How many apples does he have?\n"
            "Solution:\n3 + 5 = 8\n"
            "Knowledge Statement:\nThis is a simple addition problem. Add initial quantity and new quantity to get the total.\n\n"
            "Example (code):\n"
            "Problem:\nWrite a function that returns the square of a number.\n"
            "Solution:\ndef square(n): return n * n\n"
            "Knowledge Statement:\nThe function multiplies the number by itself using the `*` operator to compute the square.\n\n"
            f"Problem:\n{row['question']}\n"
            f"Solution:\n{row['corpus']}\n"
            "Knowledge Statement:\n<step-by-step explanation + math or code logic>"
        )
    },

    "language_modeling": {
        "rationale": lambda row: (
            "You are a text analyst. Briefly describe the theme or tone of the passage. Do not interpret it in detail.\n\n"
            "Example:\n"
            "Passage:\nThe rain fell in sheets as the crowd gathered in silence.\n"
            "Rationale:\nThe passage is descriptive and moody, with a somber tone.\n\n"
            f"Passage:\n{row['question']}\n"
            "Rationale:\n<brief stylistic comment>"
        ),

        "knowledge": lambda row: (
            "The following is a raw passage sampled from text corpora. "
            "Please rewrite it in a normalized form: keep the meaning and tone, but remove formatting issues, dangling phrases, or incomplete sentences.\n\n"
            "Example:\n"
            "Text:\nWe're knee deep in preparations now for our family travel adventures... These ar\n"
            "Knowledge Statement:\nWe're preparing for family travel, and plan to keep kids calm with games and apps.\n\n"
            f"Text:\n{row['corpus']}\n"
            "Knowledge Statement:\n<clean, readable version with same intent>"
        )
    }
}
