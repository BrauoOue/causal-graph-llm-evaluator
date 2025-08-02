"""
Evaluation module for causal reasoning and explanation predictions.

This module provides functions to evaluate prediction and explanation results,
calculate metrics, and save evaluation results.
"""
import json
import os
import traceback
from typing import Dict, Any, List, Tuple

from logger import get_logger

# Initialize logger
logger = get_logger(filename=__file__,console_color="cyan")


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load and parse a JSON file safely with error handling.

    Args:
        file_path: Path to the JSON file to load

    Returns:
        List of dictionaries containing the parsed JSON data

    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        logger.debug(f"Loading JSON from: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.debug(f"Successfully loaded {len(data)} records from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error loading {file_path}: {str(e)}")
        logger.debug(traceback.format_exc())
        raise


def load(predictions_folder: str, dataset_name: str) -> List[Dict[str, Any]]:
    """
    Load prediction or evaluation results for a specific dataset.

    Args:
        predictions_folder: Folder containing the JSON results
        dataset_name: Name of the dataset to load

    Returns:
        List of dictionaries containing the loaded data
    """
    file_path = os.path.join(predictions_folder, f"{dataset_name}.json")
    try:
        return load_json_file(file_path)
    except FileNotFoundError:
        logger.error(f"Results file not found for dataset '{dataset_name}' in {predictions_folder}")
        return []


def save_results(result: Dict[str, Any], results_folder: str, dataset_name: str) -> bool:
    """
    Save evaluation results to a JSON file.

    Args:
        result: Dictionary containing evaluation metrics
        results_folder: Folder to save the results in
        dataset_name: Name of the dataset used to create the filename

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Ensure the results directory exists
        os.makedirs(results_folder, exist_ok=True)

        file_path = os.path.join(results_folder, f"{dataset_name}.json")
        logger.debug(f"Saving evaluation results to {file_path}")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.info(f"Evaluation results saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save evaluation results for {dataset_name}: {str(e)}")
        logger.debug(traceback.format_exc())
        return False


def calculate_metrics(
    predictions: List[Dict[str, Any]],
    explanations: List[Dict[str, Any]]
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Calculate evaluation metrics from prediction and explanation results.

    Args:
        predictions: List of prediction dictionaries
        explanations: List of explanation evaluation dictionaries

    Returns:
        Tuple containing:
        - Dictionary of calculated metrics (ratios)
        - Dictionary of count metrics (integers)
    """
    # Initialize counters
    valid_choice_count = 0
    correct_predictions_count = 0
    correct_explanations_count = 0
    total_confidence = 0

    # Ensure we have data to evaluate
    total_predictions = len(predictions)
    total_explanations = len(explanations)

    if not total_predictions:
        logger.warning("No prediction data available for evaluation")
        return {}, {"total_predictions": 0, "total_explanations": total_explanations}

    # Count metrics from each prediction and explanation
    for idx, (prediction, explanation) in enumerate(zip(predictions, explanations)):
        try:
            # Prediction metrics
            is_correct_prediction = prediction.get("is_correct", False)
            is_valid = prediction.get("is_valid_choice", False)

            correct_predictions_count += int(is_correct_prediction)
            valid_choice_count += int(is_valid)

            # Explanation metrics
            is_correct_explanation = explanation.get("correct_explanation", False)
            confidence = explanation.get("confidence", 0.0)

            correct_explanations_count += int(is_correct_explanation)
            total_confidence += confidence
        except Exception as e:
            logger.error(f"Error processing item at index {idx}: {str(e)}")
            logger.debug(f"Prediction: {prediction}")
            logger.debug(f"Explanation: {explanation}")

    # Calculate ratios and metrics
    counts = {
        "total_predictions": total_predictions,
        "correct_predictions": correct_predictions_count,
        "valid_choice_predictions": valid_choice_count,
        "total_explanations": total_explanations,
        "correct_explanations": correct_explanations_count,
    }

    # Avoid division by zero
    metrics = {}
    if total_predictions > 0:
        metrics["prediction_accuracy"] = correct_predictions_count / total_predictions
        metrics["valid_ratio"] = valid_choice_count / total_predictions

    if total_explanations > 0:
        metrics["explanation_accuracy"] = correct_explanations_count / total_explanations
        metrics["average_confidence"] = total_confidence / total_explanations

    return metrics, counts


def evaluate(
    prediction_result: Dict[str, Any],
    explanation_result: Dict[str, Any],
    save_result: bool = True,
    predictions_folder: str = "./output/predictions",
    explanations_folder: str = "./output/explanations",
    results_folder: str = "./output/results"
) -> Dict[str, Any]:
    """
    Evaluate prediction and explanation results for a dataset.

    Args:
        prediction_result: Dictionary containing prediction metadata (model, time, etc.)
        explanation_result: Dictionary containing explanation metadata
        save_result: Whether to save the evaluation results to a file
        predictions_folder: Folder containing prediction JSON files
        explanations_folder: Folder containing explanation evaluation JSON files
        results_folder: Folder to save evaluation results

    Returns:
        Dictionary containing all evaluation metrics
    """
    try:
        # Extract dataset name from the results
        dataset_name = prediction_result.get("name")
        if not dataset_name:
            logger.error("Missing dataset name in prediction_result")
            return {}

        logger.info(f"Starting evaluation for dataset: {dataset_name}")

        # Load the prediction and explanation data
        predictions = load(predictions_folder, dataset_name)
        explanations = load(explanations_folder, dataset_name)

        # Check if we have matching data to evaluate
        if not predictions or not explanations:
            logger.error(f"Missing data for {dataset_name} evaluation. "
                         f"Predictions: {len(predictions)}, Explanations: {len(explanations)}")
            return {}

        if len(predictions) != len(explanations):
            logger.warning(f"Mismatch in number of predictions ({len(predictions)}) and "
                          f"explanations ({len(explanations)}) for {dataset_name}")

        # Extract metadata from results
        prediction_model = prediction_result.get("model", "unknown")
        prediction_time = prediction_result.get("time", 0)
        prediction_cost = prediction_result.get("cost", 0)

        explanation_model = explanation_result.get("model", "unknown")
        explanation_time = explanation_result.get("time", 0)
        explanation_cost = explanation_result.get("cost", 0)

        # Calculate metrics
        metrics, counts = calculate_metrics(predictions, explanations)

        # Create the evaluation result dictionary
        evaluation_result = {
            "dataset": dataset_name,

            # Prediction data
            "predictions_model": prediction_model,
            "total_predictions": counts.get("total_predictions", 0),
            "correct_predictions": counts.get("correct_predictions", 0),
            "valid_choice_predictions": counts.get("valid_choice_predictions", 0),
            "prediction_accuracy": metrics.get("prediction_accuracy", 0),
            "predictions_execution_time": prediction_time,
            "predictions_total_cost": prediction_cost,

            # Explanation data
            "explanations_model": explanation_model,
            "total_explanations": counts.get("total_explanations", 0),
            "correct_explanations": counts.get("correct_explanations", 0),
            "explanations_accuracy": metrics.get("explanation_accuracy", 0),
            "average_explanation_confidence": metrics.get("average_confidence", 0),
            "explanations_execution_time": explanation_time,
            "explanations_total_cost": explanation_cost,
        }

        # Log the evaluation results
        logger.info(f"Evaluation Results for {dataset_name}:")
        logger.info(f"Prediction Model: {prediction_model}")
        logger.info(f"Prediction Accuracy: {metrics.get('prediction_accuracy', 0):.2%}")
        logger.info(f"Valid Choices: {metrics.get('valid_ratio', 0):.2%}")
        logger.info(f"Explanation Accuracy: {metrics.get('explanation_accuracy', 0):.2%}")
        logger.info(f"Average Confidence: {metrics.get('average_confidence', 0):.4f}")
        logger.info(f"Total Cost: ${prediction_cost + explanation_cost:.4f}")
        logger.info(f"Total Time: {prediction_time + explanation_time:.2f}s")

        # Save results if requested
        if save_result:
            save_results(evaluation_result, results_folder, dataset_name)

        return evaluation_result

    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        logger.debug(traceback.format_exc())
        return {}


def evaluate_all_datasets(
    results_folder: str = "./output/results",
    predictions_folder: str = "./output/predictions",
    explanations_folder: str = "./output/explanations",
    metadata_folder: str = "./output/metadata"
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all datasets found in the predictions folder.

    Args:
        results_folder: Folder to save evaluation results
        predictions_folder: Folder containing prediction results
        explanations_folder: Folder containing explanation results
        metadata_folder: Folder containing metadata for predictions and explanations

    Returns:
        Dictionary mapping dataset names to evaluation results
    """
    all_results = {}

    try:
        # Find all prediction files
        if not os.path.exists(predictions_folder):
            logger.error(f"Predictions folder not found: {predictions_folder}")
            return all_results

        prediction_files = [
            f[:-5] for f in os.listdir(predictions_folder)
            if f.endswith('.json')
        ]

        logger.info(f"Found {len(prediction_files)} datasets to evaluate")

        # Process each dataset
        for dataset_name in prediction_files:
            try:
                logger.info(f"Processing dataset: {dataset_name}")

                # Try to load metadata first from the metadata folder
                metadata_predictions_folder = os.path.join(metadata_folder, "predictions")
                metadata_explanations_folder = os.path.join(metadata_folder, "explanations")

                # Load prediction metadata if available
                prediction_metadata_path = os.path.join(metadata_predictions_folder, f"{dataset_name}.json")
                if os.path.exists(prediction_metadata_path):
                    logger.info(f"Loading prediction metadata for {dataset_name}")
                    try:
                        with open(prediction_metadata_path, 'r', encoding='utf-8') as f:
                            prediction_result = json.load(f)
                        logger.info(f"Successfully loaded prediction metadata for {dataset_name}")
                    except Exception as e:
                        logger.error(f"Error loading prediction metadata: {str(e)}")
                        prediction_result = {"name": dataset_name, "model": "unknown", "time": 0, "cost": 0}
                else:
                    logger.warning(f"No prediction metadata found for {dataset_name}, using defaults")
                    prediction_result = {"name": dataset_name, "model": "unknown", "time": 0, "cost": 0}

                # Load explanation metadata if available
                explanation_metadata_path = os.path.join(metadata_explanations_folder, f"{dataset_name}.json")
                if os.path.exists(explanation_metadata_path):
                    logger.info(f"Loading explanation metadata for {dataset_name}")
                    try:
                        with open(explanation_metadata_path, 'r', encoding='utf-8') as f:
                            explanation_result = json.load(f)
                        logger.info(f"Successfully loaded explanation metadata for {dataset_name}")
                    except Exception as e:
                        logger.error(f"Error loading explanation metadata: {str(e)}")
                        explanation_result = {"name": dataset_name, "model": "unknown", "time": 0, "cost": 0}
                else:
                    logger.warning(f"No explanation metadata found for {dataset_name}, using defaults")
                    explanation_result = {"name": dataset_name, "model": "unknown", "time": 0, "cost": 0}

                # Ensure dataset name is set correctly
                prediction_result["name"] = dataset_name
                explanation_result["name"] = dataset_name

                # Run evaluation with the metadata
                result = evaluate(
                    prediction_result,
                    explanation_result,
                    save_result=False,
                    predictions_folder=predictions_folder,
                    explanations_folder=explanations_folder,
                    results_folder=results_folder
                )

                if result:
                    all_results[dataset_name] = result

            except Exception as e:
                logger.error(f"Error evaluating dataset {dataset_name}: {str(e)}")
                logger.debug(traceback.format_exc())

        return all_results

    except Exception as e:
        logger.error(f"Error in evaluate_all_datasets: {str(e)}")
        logger.debug(traceback.format_exc())
        return all_results


if __name__ == "__main__":
    # When run directly, evaluate all datasets
    logger.info("Starting standalone evaluation of all datasets")
    results = evaluate_all_datasets()
    logger.info(f"Evaluated {len(results)} datasets successfully")
