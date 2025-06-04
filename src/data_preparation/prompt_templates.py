"""
Prompt templates for different task formats.

This module provides templates for generating prompts for various task types including
multiple choice questions, generative tasks, and language modeling. Each template
includes examples and instructions for generating rationales and knowledge statements.

Author: why
Date: 2024
"""

# Task format templates with examples and instructions
TASK_FORMAT_TEMPLATES = {
    "multiple_choice": {
        "rationale": lambda row: (
            "You are a reasoning assistant. Given a multiple-choice question and the correct answer, "
            "briefly explain why the answer is correct and why the other options are incorrect. "
            "For yes/no or true/false questions, justify the correct choice using evidence from the passage.\n\n"

            "Example 1:\n"
            "Question:\nWhich material conducts electricity?\n"
            "Options:\nA) Wood\nB) Copper\nC) Plastic\n"
            "Answer:\nB) Copper\n"
            "Rationale:\n"
            "A) Wood — Incorrect: It is an insulator.\n"
            "B) Copper — Correct: It has free electrons that allow electricity to flow.\n"
            "C) Plastic — Incorrect: It does not conduct electricity.\n\n"

            "Example 2:\n"
            "Question:\nWhich of these methods of aluminum disposal is best for the environment?\n"
            "Options:\nA) Burying it\nB) Burning it\nC) Recycling it\n"
            "Answer:\nC) Recycling it\n"
            "Rationale:\n"
            "A) Burying it — Incorrect: It wastes resources and can pollute soil.\n"
            "B) Burning it — Incorrect: It is energy-intensive and releases emissions.\n"
            "C) Recycling it — Correct: It conserves energy and reduces environmental impact.\n\n"

            "Example 3 (Judgment QA):\n"
            "Passage:\nThe movie premiered in 2018 and made over $1 billion worldwide.\n"
            "Question:\nHas the movie been released?\n"
            "Options:\nA) Yes\nB) No\n"
            "Answer:\nA) Yes\n"
            "Rationale:\nA) Yes — Correct: The passage states the movie premiered in 2018.\n"
            "B) No — Incorrect: It contradicts the given information.\n\n"

            f"Question:\n{row['question']}\n"
            f"Answer:\n{row['label']}) {row['gold']}\n"
            "Rationale:\n<analyze each option and explain why the correct one is best>"
        ),

        "knowledge": lambda row: (
            "You are a scientific annotator. The following text comes from a multiple-choice question "
            "and its correct answer. Your task is to convert this into a clear, standalone knowledge "
            "statement that expresses the factual or conceptual knowledge involved. If the text includes "
            "a question-answer pair, reformulate it into an informative, generalizable sentence. If it "
            "includes background description or reasoning steps, use them to enrich the explanation.\n\n"
            
            "Example 1 (Science QA):\n"
            "Corpus:\nWhich mixture contains ingredients that can be easily separated? fruit salad\n"
            "Knowledge Statement:\nA fruit salad is a heterogeneous mixture whose components, such as "
            "sliced fruit, can be separated by physical means without altering their properties.\n\n"

            "Example 2 (Judgment QA):\n"
            "Corpus:\nFilming took place from February to July 2017. The movie premiered in Madrid in "
            "May 2018 and was released in the United States on June 22, 2018. It grossed over $1.3 "
            "billion worldwide and became one of the highest-grossing films of the year.\n"
            "Question: Has the movie been released? Yes\n"
            "Knowledge Statement:\nThe movie completed production in 2017 and was publicly released in "
            "2018, with screenings in major cities and substantial box office performance. These facts "
            "indicate that the film is no longer upcoming but has been widely distributed and viewed.\n\n"

            f"Corpus:\n{row['corpus']}\n"
            "Knowledge Statement:\n<your explanatory reformulation here>"
        )
    },

    "generative": {
        "rationale": lambda row: (
            "You are a reasoning assistant. Given a problem and its solution, briefly explain the logic "
            "or key idea behind the answer.\n"
            "- For **math** problems, summarize the calculation steps and reasoning.\n"
            "- For **code** tasks, explain the goal and the main steps of the algorithm in plain English.\n\n"

            "Example 1 (Math):\n"
            "Problem:\nIn a class, 7 students like basketball, 5 like cricket, and 3 like both. "
            "How many like either sport?\n"
            "Answer:\n7 + 5 - 3 = 9\n"
            "Rationale:\nAdd both groups (7 + 5) and subtract the overlap (3) to avoid double-counting. "
            "Total = 9.\n\n"

            "Example 2 (Code):\n"
            "Problem:\nWrite a function to replace all spaces, commas, or periods with a colon.\n"
            "Answer:\ndef replace_specialchar(text): return re.sub('[ ,.]', ':', text)\n"
            "Rationale:\nUse a regular expression to match the target characters and replace them with "
            "a colon using `re.sub()`.\n\n"

            f"Problem:\n{row['question']}\n"
            f"Answer:\n{row['corpus']}\n"
            "Rationale:\n<brief explanation of math logic or code purpose>"
        ),

        "knowledge": lambda row: (
            "You are a knowledgeable assistant. Based on the following problem and solution, extract "
            "and present the reusable knowledge clearly.\n"
            "- If it's a **math** problem, write a short explanation of the method, then walk through "
            "the solution steps with clear logic.\n"
            "- If it's a **code** problem, provide a clean version of the function with inline comments "
            "explaining the logic.\n\n"

            "Example 1 (Math):\n"
            "Problem:\nIn a class, 7 students like basketball, 5 like cricket, and 3 like both. "
            "How many like either sport?\n"
            "Solution:\n7 + 5 - 3 = 9\n"
            "Knowledge Statement:\n"
            "To avoid double-counting students who like both sports, apply the inclusion-exclusion principle.\n"
            "- Total who like either = basketball + cricket − both\n"
            "- That is: 7 + 5 − 3 = 9 students.\n"
            "This principle is commonly used in set theory and overlapping group problems.\n\n"

            "Example 2 (Code):\n"
            "Problem:\nWrite a function to replace spaces, commas, or periods with a colon.\n"
            "Solution:\ndef replace_specialchar(text): return re.sub('[ ,.]', ':', text)\n"
            "Knowledge Statement:\n"
            "```python\n"
            "import re\n\n"
            "# This function replaces all spaces, commas, and periods with a colon.\n"
            "def replace_specialchar(text):\n"
            "    # The regular expression '[ ,.]' matches any space, comma, or period.\n"
            "    return re.sub('[ ,.]', ':', text)\n"
            "```\n"
            "This approach uses regular expressions and is helpful for simple text cleaning tasks.\n\n"

            f"Problem:\n{row['question']}\n"
            f"Solution:\n{row['corpus']}\n"
            "Knowledge Statement:\n<math explanation or annotated code follows>"
        )
    },

    "language_modeling": {
        "rationale": lambda row: (
            "You are a text analyst. Given a short passage, briefly describe its tone or writing style "
            "using neutral and simple terms. Avoid interpretation or summarization.\n\n"
            "Example:\n"
            "Passage:\nThe rain fell in sheets as the crowd gathered in silence.\n"
            "Rationale:\nDescriptive and somber tone.\n\n"
            "Example:\n"
            "Passage:\nKids love building things with cardboard boxes, especially when you give them "
            "markers and stickers.\n"
            "Rationale:\nCasual and instructional tone.\n\n"
            f"Passage:\n{row['question']}\n"
            "Rationale:\n<brief style/tone note>"
        ),

        "knowledge": lambda row: (
            "The following is a raw text passage from a web or document corpus. Rewrite it in clean, "
            "coherent English. Preserve the original meaning and tone, but fix broken phrases, trailing "
            "thoughts, and minor formatting issues.\n\n"
            "Example 1:\n"
            "Text:\nWe're knee deep in preparations now... 7 hour ferry rides, or just when we have work to do. These ar\n"
            "Knowledge Statement:\nWe're preparing for travel and need to keep the kids calm during long trips. "
            "Games and apps will help.\n\n"
            "Example 2:\n"
            "Text:\nAnother thing to think about when, like, planning meals is – uh – whether people got "
            "allergies and stuff.\n"
            "Knowledge Statement:\nWhen planning meals, it's important to consider guests' allergies and "
            "dietary restrictions.\n\n"
            f"Text:\n{row['corpus']}\n"
            "Knowledge Statement:\n<clean and fluent version with original meaning>"
        )
    }
}
