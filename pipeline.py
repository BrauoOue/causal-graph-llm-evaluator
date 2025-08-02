"""
Causal Graph Evaluation Pipeline

This module implements the main evaluation pipeline for testing and benchmarking
causal reasoning capabilities of large language models. It orchestrates the complete
workflow from data loading to final evaluation.

Main components:
- Data loading and standardization from various dataset formats
- Causal reasoning prediction generation using LLMs
- Explanation evaluation of model predictions
- Result aggregation and metric calculation
- Error handling and logging throughout the process

The pipeline supports processing multiple datasets in sequence, with configurable
limits, custom prompts, and extensive logging of execution metrics.

Usage:
    # Run the full pipeline with default settings
    python pipeline.py

    # Run with custom settings (all options)
    python pipeline.py --limit 5 --use_manual_prompt --model_predictions gpt-4o --model_explanations gpt-4o \
        --max_tokens_predictions 2000 --max_tokens_explanations 5000 --log_level DEBUG
"""

import json
import os
import time
import traceback
import argparse
from typing import List, Dict, Any

import pandas as pd

from modules.agent import CausalReasoningAgent
from modules.conversion import DatasetMapping, BuilderDataset
from modules.evaluate import evaluate
from modules.explanation_evaluation_agent import ExplanationEvaluationAgent
from modules.logger import get_logger, set_log_level

# Initialize logger
logger = get_logger(filename=__file__, console_color="blue")


def save_metadata(metadata: Dict[str, Any], folder_path: str, dataset_name: str) -> None:
    """
    Save metadata information to a JSON file.

    Args:
        metadata: Dictionary containing metadata to save
        folder_path: Path to the folder where metadata will be saved
        dataset_name: Name of the dataset used for filename
    """
    try:
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, f"{dataset_name}.json")

        logger.debug(f"Saving metadata to {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata saved to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save metadata for {dataset_name}: {str(e)}")
        logger.debug(traceback.format_exc())


# Load datasets from a folder
def load_pd_files(folder: str) -> Dict[str, pd.DataFrame]:
    """
    Loads selected .jsonl files from the specified folder based on user choice.

    Args:
        folder (str): Directory containing .jsonl dataset files.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping filenames to loaded dataframes
    """
    datasets = {}

    try:
        logger.info(f"Loading datasets from folder: {folder}")
        files = os.listdir(folder)
        jsonl_files = [f for f in files if f.endswith(".jsonl")]

        if not jsonl_files:
            logger.warning(f"No .jsonl files found in {folder}")
            return datasets

        # Display available datasets with numbers
        print("Available datasets:")
        for i, filename in enumerate(jsonl_files, 1):
            print(f"[{i}] {filename}")

        # Prompt user to select datasets
        selection = input("\nSelect datasets by number (comma-separated) or type 'all': ").strip().lower()

        # Process user selection
        if selection == 'all':
            selected_files = jsonl_files
            logger.info("Processing all available datasets")
        else:
            try:
                # Parse selected indices
                selected_indices = [int(idx.strip()) for idx in selection.split(',') if idx.strip()]
                # Filter valid indices
                valid_indices = [idx for idx in selected_indices if 1 <= idx <= len(jsonl_files)]

                if not valid_indices:
                    logger.warning("No valid dataset numbers selected. Exiting.")
                    return datasets

                # Get selected filenames
                selected_files = [jsonl_files[idx - 1] for idx in valid_indices]
                logger.info(f"Selected datasets: {', '.join(selected_files)}")

            except ValueError:
                logger.error("Invalid input. Please enter comma-separated numbers or 'all'.")
                return datasets

        # Load selected datasets
        for filename in selected_files:
            try:
                path = os.path.join(folder, filename)
                logger.debug(f"Loading file: {path}")
                datasets[filename] = pd.read_json(path, lines=True)
                logger.info(f"Successfully loaded {filename} with {len(datasets[filename])} rows")
            except Exception as e:
                logger.error(f"Failed to load {filename}: {str(e)}")
                logger.debug(traceback.format_exc())
    except Exception as e:
        logger.error(f"Error accessing folder {folder}: {str(e)}")
        logger.debug(traceback.format_exc())

    return datasets


def standardize(data: Dict[str, pd.DataFrame], mapping_dict: Dict[str, DatasetMapping]) -> Dict[str, pd.DataFrame]:
    """
    Convert datasets to a standardized format using the provided mappings.

    Args:
        data: Dictionary of dataframes to standardize
        mapping_dict: Dictionary mapping dataset names to their column mappings

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of standardized dataframes
    """
    results = {}
    logger.info("Starting dataset standardization process")

    for dataframe_name in data:
        try:
            if dataframe_name not in mapping_dict:
                logger.warning(f"No mapping found for {dataframe_name}, skipping")
                continue

            logger.debug(f"Standardizing {dataframe_name} with {len(data[dataframe_name])} rows")
            mapping = mapping_dict[dataframe_name]
            result = BuilderDataset.convert(data[dataframe_name], mapping)
            result['id'] = result.index
            results[dataframe_name] = result
            logger.info(f"Standardized {dataframe_name}: {len(result)} rows processed")

        except Exception as e:
            logger.error(f"Failed to standardize {dataframe_name}: {str(e)}")
            logger.debug(traceback.format_exc())

    return results


def predict(
        datasets: Dict[str, pd.DataFrame],
        agent: CausalReasoningAgent,
        limit: int,
        use_custom_prompts: bool,
) -> List[Dict[str, Any]]:
    """
    Process datasets with causal reasoning agent to generate predictions.

    Args:
        datasets: Dictionary mapping dataset names to DataFrame objects
        agent: CausalReasoningAgent instance for generating predictions
        limit: Maximum number of rows to process from each dataset
        use_custom_prompts: Whether to prompt for manual prompt input

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing prediction results and metadata
                             for each dataset, including cost, execution time, and model info
    """
    result = []
    logger.info(f"Starting prediction process using {agent.model_name} model with limit {limit}")

    if not datasets:
        logger.warning("No datasets provided for prediction")
        return result

    for dataset_file_name in datasets:
        dataset_info = dict()
        dataset_name = dataset_file_name.split('.')[0]
        dataset_info["name"] = dataset_name

        logger.info(f"Processing dataset: '{dataset_name}' with {len(datasets[dataset_file_name])} rows")

        # Get manual prompt if desired
        if use_custom_prompts:
            logger.debug("Prompting for custom prompt input")
            logger.info(f"You are now processing the: '{dataset_name}' dataset.")
            manual_prompt = input(
                "Enter your custom prompt (or press Enter to use automatic prompt generation): ").strip()
            if manual_prompt:
                logger.info("Using custom prompt provided by user")
            else:
                logger.info("No custom prompt provided, using automatic generation")
                manual_prompt = None
        else:
            logger.info(f"You are now processing the: '{dataset_name}' dataset.")
            manual_prompt = None
            logger.info("No custom prompt provided, using automatic generation")

        # Track time and execute prediction
        time_start = time.time()
        try:
            logger.info(f"Starting prediction for {dataset_name}")
            cost = agent.predict_dataset_parallel(
                data=datasets[dataset_file_name],
                limit=limit,
                manual_prompt=manual_prompt,
                dataset_name=dataset_name
            )
            logger.info(f"Prediction for {dataset_name} completed successfully")

            time_end = time.time()
            duration = time_end - time_start

            dataset_info["cost"] = cost
            dataset_info["time"] = duration
            dataset_info["model"] = agent.model_name

            # Save prediction metadata to metadata folder
            save_metadata(
                dataset_info,
                "./output/metadata/predictions",
                dataset_name
            )

            logger.info(f"Dataset {dataset_name} processed in {duration:.2f} seconds, cost: {cost}")
            result.append(dataset_info)

        except Exception as e:
            logger.error(f"Error processing dataset {dataset_name}: {e}")
            logger.debug(traceback.format_exc())
            continue

    logger.info(f"Prediction process completed for {len(result)}/{len(datasets)} datasets")
    return result


def evaluate_explanation(dataset_names: List[str],
                         agent: ExplanationEvaluationAgent) -> List[Dict[str, Any]]:
    """
    Evaluate explanations for predictions across multiple datasets.

    This function loads prediction results for each dataset, evaluates the quality
    of explanations using the provided agent, and saves evaluation metadata.

    Args:
        dataset_names: List of dataset names (without file extensions) to evaluate
        agent: ExplanationEvaluationAgent instance to use for evaluation

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing evaluation results and metadata
                             for each dataset, including cost, execution time, and model info
    """
    evaluation_results = []
    logger.info(f"Starting explanation evaluation process using {agent.model_name} model")

    if not dataset_names:
        logger.warning("No dataset names provided for evaluation")
        return evaluation_results

    for dataset_file_name in dataset_names:
        dataset_eval_info = dict()
        dataset_name = dataset_file_name.split('.')[0]
        dataset_eval_info["name"] = dataset_name

        logger.info(f"Evaluating explanations for dataset: {dataset_name}")

        # Track time for evaluation
        time_start = time.time()

        try:
            # Load predictions
            logger.debug(f"Loading predictions for {dataset_name}")
            df = agent.load_predictions(dataset_name=dataset_name)

            if df is None or df.empty:
                logger.warning(f"No predictions found for {dataset_name}, skipping evaluation")
                continue

            logger.info(f"Loaded {len(df)} predictions for {dataset_name}")

            # Run evaluation
            logger.info(f"Starting evaluation for {dataset_name}")
            cost = agent.evaluate_dataset_parallel(
                df,
                save_results=True,
                dataset_name=dataset_name
            )
            logger.info(f"Evaluation for {dataset_name} completed successfully")

            time_end = time.time()
            duration = time_end - time_start

            dataset_eval_info["cost"] = cost
            dataset_eval_info["time"] = duration
            dataset_eval_info["model"] = agent.model_name

            # Save explanation metadata to metadata folder
            save_metadata(
                dataset_eval_info,
                "./output/metadata/explanations",
                dataset_name
            )

            logger.info(f"Dataset {dataset_name} evaluated in {duration:.2f} seconds, cost: {cost}")
            evaluation_results.append(dataset_eval_info)

        except FileNotFoundError as e:
            logger.error(f"Prediction file not found for {dataset_name}: {str(e)}")
            logger.debug(traceback.format_exc())
            continue
        except Exception as e:
            logger.error(f"Error evaluating explanations for dataset {dataset_name}: {str(e)}")
            logger.debug(traceback.format_exc())
            continue

    logger.info(f"Evaluation process completed for {len(evaluation_results)}/{len(dataset_names)} datasets")
    return evaluation_results


# Main function
def main():
    """
    Main execution function for the causal graph evaluation pipeline.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Causal Graph Evaluation Pipeline')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of rows to process from each dataset')
    parser.add_argument('--use_manual_prompt', action='store_true', help='Enable manual prompt input')
    parser.add_argument('--model_predictions', type=str, default='gpt-4o-mini',
                        help='Model name for predictions (default: gpt-4o-mini)')
    parser.add_argument('--model_explanations', type=str, default='gpt-4o-mini',
                        help='Model name for explanation evaluation (default: gpt-4o-mini)')
    parser.add_argument('--max_tokens_predictions', type=int, default=2000,
                        help='Maximum tokens for prediction agent responses (default: 2000)')
    parser.add_argument('--max_tokens_explanations', type=int, default=5000,
                        help='Maximum tokens for explanation evaluation agent responses (default: 5000)')
    parser.add_argument('--log_level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')

    args = parser.parse_args()

    # Configure logging level
    set_log_level(logger, args.log_level)
    logger.info(f"Log level set to {args.log_level}")

    try:
        logger.info("Starting causal graph evaluation pipeline")
        raw_data_folder = "./data/raw"

        # Log the configuration
        logger.info(f"Configuration: limit={args.limit}, use_manual_prompt={args.use_manual_prompt}, "
                    f"model_predictions={args.model_predictions}, model_explanations={args.model_explanations}, "
                    f"max_tokens_main_agent={args.max_tokens_predictions}, " f"max_tokens_explanations_agent={args.max_tokens_explanations}")

        # Step 1: Load datasets
        logger.info("Loading datasets from raw data folder")
        data = load_pd_files(raw_data_folder)

        if not data:
            logger.error("No datasets were loaded. Exiting pipeline.")
            return

        # Step 2: Initialize agent
        try:
            logger.info(f"Initializing CausalReasoningAgent with model {args.model_predictions}")
            agent = CausalReasoningAgent(
                model_name=args.model_predictions,
                max_tokens=args.max_tokens_predictions
            )
            logger.info(
                f"Successfully initialized agent using {agent.model_name} model with {args.max_tokens_predictions} max tokens")
        except Exception as e:
            logger.error(f"Failed to initialize CausalReasoningAgent: {str(e)}")
            logger.debug(traceback.format_exc())
            return

        # Step 3: Define mappings and standardize data
        logger.info("Setting up dataset mappings")
        mapping_dict = {
            "code.jsonl": DatasetMapping(context="Code", question="Question", question_type="Question Type",
                                         choices=None,
                                         label="Ground Truth", explanation="Explanation"),
            "math.jsonl": DatasetMapping(context="Mathematical Scenario", question="Question",
                                         question_type="Question Type",
                                         choices=None, label="Ground Truth", explanation="Explanation"),
            "text.jsonl": DatasetMapping(context="Scenario and Question", question=None, question_type="Question Type",
                                         choices=None, label="Ground Truth", explanation="Explanation"),
            "e.jsonl": DatasetMapping(context="premise", question=None, question_type="ask-for",
                                      choices=["hypothesis1", "hypothesis2"], label="label",
                                      explanation="conceptual_explanation"),
        }

        logger.info("Converting data to standardized datasets")
        datasets = standardize(data, mapping_dict)

        if not datasets:
            logger.error("Failed to standardize any datasets. Exiting pipeline.")
            return

        # Step 4: Make predictions
        logger.info("Starting prediction process")
        predictions_results = predict(datasets, agent, limit=args.limit, use_custom_prompts=args.use_manual_prompt)

        if not predictions_results:
            logger.warning("No prediction results were generated. Skipping evaluation.")
            return

        # Step 5: Initialize explanation evaluation agent
        try:
            logger.info(f"Initializing ExplanationEvaluationAgent with model {args.model_explanations}")
            explanation_agent = ExplanationEvaluationAgent(
                model_name=args.model_explanations,
                max_tokens=args.max_tokens_explanations
            )
            logger.info(
                f"Successfully initialized explanation agent using {explanation_agent.model_name} model with {args.max_tokens_explanations} max tokens")
        except Exception as e:
            logger.error(f"Failed to initialize ExplanationEvaluationAgent: {str(e)}")
            logger.debug(traceback.format_exc())
            return

        # Step 6: Evaluate explanations
        logger.info("Starting explanation evaluation")
        explanation_results = evaluate_explanation(
            dataset_names=list(datasets.keys()),
            agent=explanation_agent
        )

        if not explanation_results:
            logger.warning("No explanation evaluation results were generated.")
            return

        # Step 7: Generate final evaluation
        logger.info("Generating final evaluation")
        for predictions_result, explanation_result in zip(predictions_results, explanation_results):
            try:
                evaluate(predictions_result, explanation_result)
                logger.info(f"Successfully evaluated {predictions_result['name']} dataset")
            except Exception as e:
                logger.error(f"Error during evaluation of {predictions_result['name']}: {str(e)}")
                logger.debug(traceback.format_exc())

        logger.info("Pipeline execution completed successfully")

    except Exception as e:
        logger.error(f"Unhandled exception in main pipeline: {str(e)}")
        logger.debug(traceback.format_exc())
        logger.critical("Pipeline execution failed")


if __name__ == "__main__":
    main()
