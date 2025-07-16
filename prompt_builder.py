import pandas as pd
import os
from typing import Dict, List


class PromptBuilder:
    def __init__(self, manual_prompt=None):
        self.manual_prompt = manual_prompt

    def get_prompt_variables(self, row: pd.Series) -> Dict[str, str]:
        """
        Extract and prepare prompt variables for a given row

        Args:
            row: DataFrame row containing the data

        Returns:
            Dictionary with template variables for the prompt
        """
        context = row["context"].strip()
        qtype = row["question_type"].lower()

        choices_raw = row["choices"]
        if choices_raw == "Yes/No":
            expertise_type = "reasoning about mathematical, logical, or computational scenarios"
            task_instruction = "Answer the yes/no question contained in the context."
            question = "Answer the yes/no question contained in the context."
            choices = ["Yes", "No"]
        else:
            expertise_type = "causal reasoning"
            choices = choices_raw.split("/")

            if qtype == "cause":
                task_instruction = "Based on the given situation, identify the most likely cause of the given event."
                question = "What is the most likely cause?"
            elif qtype == "effect":
                task_instruction = "Based on the given situation, identify the most likely effect of the given event."
                question = "What is the most likely effect?"
            else:
                task_instruction = "Based on the given situation, answer the following question based on the given context."
                question = "Choose the most appropriate answer."


        if self.manual_prompt:
            task_instruction = self.manual_prompt.strip()

        choices_text = "\n".join(f"- {c.strip()}" for c in choices)

        return {
            "expertise_type": expertise_type,
            "task_instruction": task_instruction,
            "context": context,
            "question": question,
            "choices": choices_text
        }

    # Koristeno za testiranje only
    def build_prompt(self, row: pd.Series) -> str:
        """
        Build a complete prompt string for backward compatibility

        Args:
            row: DataFrame row containing the data

        Returns:
            Complete prompt string
        """
        variables = self.get_prompt_variables(row)

        prompt = f"You are an expert in {variables['expertise_type']}.\n"
        prompt += f"{variables['task_instruction']}\n\n"
        prompt += f"Context:\n{variables['context']}\n\n"
        prompt += f"Question:\n{variables['question']}\n\n"
        prompt += f"Choices:\n{variables['choices']}\n"

        return prompt

#test
def test_andrea():
    file_path = input("Enter path to test CSV or JSONL (e.g., test/e_test.csv): ").strip()
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit(1)

    user_prompt = input("Enter your custom prompt (or press Enter to use automatic prompt generation): ").strip()
    manual_prompt = user_prompt if user_prompt else None

    if file_path.endswith(".jsonl"):
        df = pd.read_json(file_path, lines=True)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        print("Unsupported file format. Please use .csv or .jsonl")
        exit(1)
    builder = PromptBuilder(manual_prompt=manual_prompt)

    prompts = df.apply(builder.build_prompt, axis=1)

    txt_path = os.path.join(os.path.dirname(file_path), "prompts.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts, 1):
            f.write(f"--- Prompt {i} ---\n{prompt}\n\n")

    print(f"All prompts saved to: {txt_path}")

#test
def test_goch():
    file_path = input("Enter path to test CSV or JSONL (e.g., test/e_test.csv): ").strip()
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit(1)

    user_prompt = input("Enter your custom prompt (or press Enter to use automatic prompt generation): ").strip()
    manual_prompt = user_prompt if user_prompt else None

    if file_path.endswith(".jsonl"):
        df = pd.read_json(file_path, lines=True)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        print("Unsupported file format. Please use .csv or .jsonl")
        exit(1)

    df = df.head(5)
    builder = PromptBuilder(manual_prompt=manual_prompt)

    prompts = df.apply(builder.build_prompt, axis=1)

    for i, prompt in enumerate(prompts, 1):
        print(f"--- Prompt {i} ---\n{prompt}\n")


if __name__ == '__main__':
    # test_andrea()
    test_goch()