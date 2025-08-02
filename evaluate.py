import json
import os
from typing import Dict, Any, List


def load(predictions_folder: str, dataset_name: str) -> List[Dict[str, Any]]:
    file_path = os.path.join(predictions_folder, f"{dataset_name}.json")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(result: Dict[str, float], results_folder: str, dataset_name: str) -> None:
    file_path = os.path.join(results_folder, f"{dataset_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


def evaluate(prediction_result: Dict[str, Any],
             explanation_result: Dict[str, Any],
             save_result=True,
             predictions_folder="./output/predictions",
             explanations_folder="./output/explanations",
             results_folder="./output/results"
             ):
    dataset_name = prediction_result["name"]

    predictions = load(predictions_folder, dataset_name)
    explanations = load(explanations_folder, dataset_name)

    prediction_model = prediction_result["model"]
    prediction_time = prediction_result["time"]
    prediction_cost = prediction_result["cost"]

    explanation_model = explanation_result["model"]
    explanation_time = explanation_result["time"]
    explanation_cost = explanation_result["cost"]

    valid_choice_count = 0
    correct_predictions_count = 0

    correct_explanations_count = 0
    average_confidence = 0

    for prediction, explanation in zip(predictions, explanations):
        is_correct_prediction = prediction.get("is_correct", False)
        is_valid = prediction.get("is_valid_choice", False)

        correct_predictions_count += int(is_correct_prediction)
        valid_choice_count += int(is_valid)

        is_correct_explanation = explanation.get("correct_explanation")
        confidence = explanation.get("confidence")

        correct_explanations_count += int(is_correct_explanation)
        average_confidence += confidence

    total_predictions = len(predictions)
    prediction_accuracy = correct_predictions_count / total_predictions
    valid_ratio = valid_choice_count / total_predictions

    total_explanations = len(explanations)
    explanation_accuracy = correct_explanations_count / total_explanations
    average_confidence = average_confidence / total_explanations

    evaluation_result = {
        "dataset": dataset_name,

        "predictions_model": prediction_model,
        "total_predictions": total_predictions,
        "correct_predictions": correct_predictions_count,
        "valid_choice_predictions": valid_choice_count,
        "prediction_accuracy": prediction_accuracy,

        "predictions_execution_time": prediction_time,
        "predictions_total_cost": prediction_cost,

        "explanations_model": explanation_model,
        "total_explanations": total_explanations,
        "correct_explanations": correct_explanations_count,
        "explanations_accuracy": explanation_accuracy,
        "average_explanation_confidence": average_confidence,

        "explanations_execution_time": explanation_time,
        "explanations_total_cost": explanation_cost,
    }

    print(f"\nEvaluation Results:")
    print(f"Accuracy (is_correct):\t{prediction_accuracy:.2f}")
    print(f"Valid Choices (% valid):\t{valid_ratio:.2f}")
    print(f"Execution Time :\t{prediction_time:.2f}")
    print(f" Total Predictions:\t{total_predictions}")

    if save_result:
        save_results(evaluation_result, results_folder, dataset_name)
