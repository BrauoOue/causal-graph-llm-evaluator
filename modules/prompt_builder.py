"""
Prompt Builder Module

This module provides functionality for generating structured prompts for LLM-based
causal reasoning tasks. It creates specialized prompts tailored to different
question types and data contexts.

Main Components:
- PromptBuilder: Core class that generates appropriate prompts based on input data
- Context-aware prompt templates for different reasoning scenarios
- Support for custom prompt overrides and specialized domain prompts

The module dynamically constructs prompts for different causal reasoning tasks,
including identifying causes from effects, effects from causes, and other causal
relationships across various domains (code, math, text).

Usage:
    builder = PromptBuilder(manual_prompt=custom_prompt)
    variables = builder.get_prompt_variables(data_row)
"""

import pandas as pd
import os
from typing import Dict, List
from modules.logger import get_logger

# Initialize logger
logger = get_logger(filename=__file__, console_color="cyan")


class PromptBuilder:
    def __init__(self, manual_prompt=None):
        self.manual_prompt = manual_prompt
        logger.debug(
            f"PromptBuilder initialized with manual_prompt: {'custom prompt provided' if manual_prompt else 'None'}")

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

        logger.debug(f"Processing row with question type: {qtype}")

        choices_raw = row["choices"]
        if choices_raw == "Yes/No":
            expertise_type = "reasoning about mathematical, logical, or computational scenarios"
            task_instruction = "Answer the yes/no question contained in the context."
            question = "Answer the yes/no question contained in the context."
            choices = ["Yes", "No"]
            logger.debug("Using Yes/No question template")
        else:
            expertise_type = "causal reasoning"
            choices = choices_raw.split("/")
            logger.debug(f"Found {len(choices)} multiple choice options")

            if qtype == "cause":
                task_instruction = "Based on the given situation, identify the most likely cause of the given event."
                question = "What is the most likely cause?"
                logger.debug("Using 'cause' question template")
            elif qtype == "effect":
                task_instruction = "Based on the given situation, identify the most likely effect of the given event."
                question = "What is the most likely effect?"
                logger.debug("Using 'effect' question template")
            else:
                task_instruction = "Based on the given situation, answer the following question based on the given context."
                question = "Choose the most appropriate answer."
                logger.debug(f"Using generic template for question type: {qtype}")

        if self.manual_prompt:
            logger.debug("Overriding task instruction with manual prompt")
            task_instruction = self.manual_prompt.strip()

        choices_text = "\n".join(f"- {c.strip()}" for c in choices)

        return {
            "expertise_type": expertise_type,
            "task_instruction": task_instruction,
            "context": context,
            "question": question,
            "choices": choices_text
        }

    def build_prompt(self, row: pd.Series) -> str:
        """
        Build a complete prompt string for backward compatibility

        Args:
            row: DataFrame row containing the data

        Returns:
            Complete prompt string
        """
        row_id = row.get("id", "unknown")
        logger.debug(f"Building prompt for row ID: {row_id}")

        variables = self.get_prompt_variables(row)

        # Track prompt length for potential token usage estimation
        context_length = len(variables['context'])
        logger.debug(f"Context length: {context_length} characters")

        prompt = f"You are an expert in {variables['expertise_type']}.\n"
        prompt += f"{variables['task_instruction']}\n\n"
        prompt += f"Context:\n{variables['context']}\n\n"
        prompt += f"Question:\n{variables['question']}\n\n"
        prompt += f"Choices:\n{variables['choices']}\n"

        total_length = len(prompt)
        logger.debug(f"Generated prompt with total length: {total_length} characters")

        return prompt


# test
def main():
    file_path = input("Enter path to test CSV or JSONL (e.g., test/e_test.csv): ").strip()
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        exit(1)

    user_prompt = input("Enter your custom prompt (or press Enter to use automatic prompt generation): ").strip()
    manual_prompt = user_prompt if user_prompt else None

    logger.info(f"Processing file: {file_path}")
    logger.info(f"Using {'custom prompt' if manual_prompt else 'auto-generated prompts'}")

    if file_path.endswith(".jsonl"):
        df = pd.read_json(file_path, lines=True)
        logger.info(f"Loaded JSONL file with {len(df)} entries")
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        logger.info(f"Loaded CSV file with {len(df)} entries")
    else:
        logger.error("Unsupported file format. Please use .csv or .jsonl")
        exit(1)

    builder = PromptBuilder(manual_prompt=manual_prompt)
    logger.info("Building prompts...")

    prompts = df.apply(builder.build_prompt, axis=1)
    logger.info(f"Generated {len(prompts)} prompts")

    txt_path = os.path.join(os.path.dirname(file_path), "prompts.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts, 1):
            f.write(f"--- Prompt {i} ---\n{prompt}\n\n")


if __name__ == '__main__':
    main()
