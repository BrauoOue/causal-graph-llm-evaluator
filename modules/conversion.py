"""
Dataset Conversion Module

This module provides functionality for standardizing diverse datasets into a unified format
for causal reasoning evaluation. It transforms various input formats into a consistent
structure that can be processed by the evaluation pipeline.

Main Components:
- DatasetMapping: Class that defines how to map source dataset columns to standardized format
- BuilderDataset: Class that handles the actual conversion process
- Standardization functions for different dataset types and structures

The module handles different types of causal reasoning data including code, math, text,
and general explanation tasks, ensuring consistent structure for downstream processing.

Usage:
    mapping = DatasetMapping(context="Premise", question="Question", question_type="Type", ...)
    result = BuilderDataset.convert(dataframe, mapping)
"""

from __future__ import annotations

import os.path

import pandas as pd
from modules.logger import get_logger

# Configure pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Initialize logger
logger = get_logger(filename=__file__)


class DatasetMapping:
    """
    A mapping class that defines how to transform dataset columns to a standardized format.

    This class holds the column mappings for different datasets to ensure consistent
    data structure across various input formats.
    """

    def __init__(self, context: str, question: str | None, question_type: str,
                 choices: list | None, label: str, explanation: str):
        """
        Initialize the dataset mapping configuration.

        Args:
            context (str): Column name containing the context/scenario
            question (str | None): Column name containing the question, None if not present
            question_type (str): Column name containing the question type
            choices (list | None): List of column names containing answer choices, None for Yes/No
            label (str): Column name containing the ground truth label
            explanation (str): Column name containing the explanation
        """
        logger.debug(f"Creating DatasetMapping with context='{context}', question='{question}', "
                    f"question_type='{question_type}', choices={choices}, label='{label}', "
                    f"explanation='{explanation}'")

        self.label = label
        self.choices = choices
        self.question_type = question_type
        self.question = question
        self.context = context
        self.explanation = explanation

    def get_columns(self) -> list[str]:
        """
        Get the standardized column names for the converted dataset.

        Returns:
            list[str]: List of standardized column names
        """
        return ["context", "question_type", "choices", "label", "explanation"]


def handle_question(q_type: str) -> str:
    """
    Standardize question type values to consistent format.

    This function converts various question type formats to standardized values:
    - Questions asking "from effect" become "cause"
    - Questions asking "from cause" become "effect"
    - Other question types remain unchanged

    Args:
        q_type (str): The original question type string

    Returns:
        str: The standardized question type
    """
    logger.debug(f"Processing question type: '{q_type}'")

    if 'from' in q_type.lower():
        if 'from effect' in q_type.lower():
            result = "cause"
        else:
            result = 'effect'
    else:
        result = q_type

    logger.debug(f"Question type '{q_type}' converted to '{result}'")
    return result


class BuilderDataset:
    """
    A utility class for converting datasets to a standardized format.

    This class provides static methods to transform various dataset formats
    into a consistent structure suitable for causal reasoning evaluation.
    """

    @staticmethod
    def convert(df: pd.DataFrame, mapping: DatasetMapping) -> pd.DataFrame:
        """
        Convert a dataset to standardized format using the provided mapping.

        This method:
        1. Creates a copy of the input dataframe
        2. Handles choices column (creates "Yes/No" if not present, joins if present)
        3. Renames columns according to mapping
        4. Combines context and question if question column exists
        5. Standardizes question types
        6. Returns only the required columns

        Args:
            df (pd.DataFrame): The input dataframe to convert
            mapping (DatasetMapping): The mapping configuration for column transformations

        Returns:
            pd.DataFrame: The converted dataframe with standardized columns
        """
        logger.info(f"Converting dataset with {len(df)} rows using mapping")
        logger.debug(f"Original columns: {list(df.columns)}")

        copy = df.copy()

        # Handle choices column
        if not mapping.choices:
            logger.debug("No choices mapping provided, setting to 'Yes/No'")
            copy["choices"] = "Yes/No"
        else:
            logger.debug(f"Processing choices from columns: {mapping.choices}")
            copy["choices"] = copy[mapping.choices].apply(lambda row: "/".join(row.astype(str)), axis=1)

        # Rename columns and handle question/context combination
        if mapping.question:
            logger.debug("Question column present, combining with context")
            copy.rename(columns={
                mapping.context: 'context',
                mapping.label: "label",
                mapping.explanation: "explanation",
                mapping.question: "question",
                mapping.question_type: "question_type"
            }, inplace=True)
            copy['context'] = copy['context'] + '\n\n' + copy['question']
        else:
            logger.debug("No question column, using context only")
            copy.rename(columns={
                mapping.context: 'context',
                mapping.label: "label",
                mapping.explanation: "explanation",
                mapping.question_type: "question_type"
            }, inplace=True)

        # Standardize question types
        logger.debug("Standardizing question types")
        copy["question_type"] = copy["question_type"].apply(handle_question)

        result = copy[mapping.get_columns()]
        logger.info(f"Dataset conversion completed. Output shape: {result.shape}")
        logger.debug(f"Final columns: {list(result.columns)}")

        return result


if __name__ == '__main__':
    logger.info("Starting dataset conversion script")

    # Define mappings for different datasets
    code_mapping = DatasetMapping(context="Code", question="Question", question_type="Question Type", choices=None,
                                  label="Ground Truth", explanation="Explanation")
    math_mapping = DatasetMapping(context="Mathematical Scenario", question="Question", question_type="Question Type",
                                  choices=None,
                                  label="Ground Truth", explanation="Explanation")

    text_mapping = DatasetMapping(context="Scenario and Question", question=None, question_type="Question Type",
                                  choices=None,
                                  label="Ground Truth", explanation="Explanation")

    e_mapping = DatasetMapping(context="premise", question=None, question_type="ask-for",
                               choices=["hypothesis1", "hypothesis2"],
                               label="label", explanation="conceptual_explanation")

    logger.info("Loading datasets")
    e_df = pd.read_json("../data/raw/e.jsonl", lines=True)
    code_df = pd.read_json("../data/raw/code.jsonl", lines=True)
    math_df = pd.read_json("../data/raw/math.jsonl", lines=True)
    text_df = pd.read_json("../data/raw/text.jsonl", lines=True)

    dfs = [code_df, math_df, text_df, e_df]
    mappings = [code_mapping, math_mapping, text_mapping, e_mapping]
    names = ["code", "math", "text", "e"]

    for df, mapping, name in zip(dfs, mappings, names):
        logger.info(f"Processing the: '{name}' dataset")
        result = BuilderDataset.convert(df, mapping)
        result.to_csv(f"./data/datasets/{name}_dataset.csv", index_label="id")
