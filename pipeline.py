import pandas as pd
import os
import time

from evaluate import evaluate
from typing import List, Dict, Any, Optional

from agent import CausalReasoningAgent
from conversion import Mapping, BuilderDataset
from explanation_evaluation_agent import ExplanationEvaluationAgent


# Load datasets from a folder
def load_pd_files(folder: str) -> Dict[str, pd.DataFrame]:
    """
    Loads all .jsonl files from the specified folder.

    Args:
        folder (str): Directory containing .jsonl dataset files.

    Returns:

    """
    datasets = {}
    for filename in os.listdir(folder):
        if filename.endswith(".jsonl"):
            path = os.path.join(folder, filename)
            datasets[filename] = pd.read_json(path, lines=True)

    return datasets


def standardize(data: Dict[str, pd.DataFrame], mapping_dict: Dict[str, Mapping]) -> Dict[str, pd.DataFrame]:
    results = {}
    for dataframe_name in data:
        mapping = mapping_dict[dataframe_name]
        result = BuilderDataset.convert(data[dataframe_name], mapping)
        result['id'] = result.index
        results[dataframe_name] = result.head(100)

    return results


def predict(
        datasets: Dict[str, pd.DataFrame],
        agent: CausalReasoningAgent,
        manual_prompt: str
):
    """
    Process datasets with the specified template type.

    Args:
        datasets (Dict[str, List[Dict[str, Any]]]): Loaded datasets.
        agent (AbstractAgent): The agent to process data.
        manual_prompt (str): The manual prompt to use with "manual" mode.

    Returns:
        List[Dict[str, Any]]: Results including predictions.
    """
    result = []

    for dataset_file_name in datasets:
        dataset_info = dict()
        dataset_info["name"] = dataset_file_name
        dataset_name = dataset_file_name.split('.')[0]

        time_start = time.time()

        cost = agent.predict_dataset_parallel(data=datasets[dataset_file_name],
                                              manual_prompt=manual_prompt,
                                              dataset_name=dataset_name)

        time_end = time.time()

        dataset_info["cost"] = cost
        dataset_info["time"] = time_end - time_start
        dataset_info["model"] = agent.model_name

        result.append(dataset_info)

    return result


def evaluate_explanation(dataset_names: List[str],
                         agent: ExplanationEvaluationAgent):
    evaluation_results = []

    for dataset_file_name in dataset_names:
        dataset_eval_info = dict()
        dataset_eval_info["name"] = dataset_file_name
        dataset_name = dataset_file_name.split('.')[0]

        time_start = time.time()

        df = agent.load_predictions(dataset_name)

        cost = agent.evaluate_dataset_parallel(df,
                                               save_results=True,
                                               dataset_name=dataset_name),

        time_end = time.time()

        dataset_eval_info["cost"] = cost
        dataset_eval_info["time"] = time_end - time_start
        dataset_eval_info["model"] = agent.model_name

        evaluation_results.append(dataset_eval_info)

    return evaluation_results


# Main function
def main():
    """
    Main execution function.
    """
    raw_data_folder = "./data/raw"

    print("Loading datasets...")
    data = load_pd_files(raw_data_folder)

    agent = CausalReasoningAgent(max_tokens=10000)

    manual_prompt = input("Enter your custom prompt (or press Enter to use automatic prompt generation): ").strip()
    if not manual_prompt:
        manual_prompt = None

    print("Converting data to standardized datasets...")

    mapping_dict = {
        "code.jsonl": Mapping(context="Code", question="Question", question_type="Question Type", choices=None,
                              label="Ground Truth", explanation="Explanation"),
        "math.jsonl": Mapping(context="Mathematical Scenario", question="Question", question_type="Question Type",
                              choices=None, label="Ground Truth", explanation="Explanation"),
        "text.jsonl": Mapping(context="Scenario and Question", question=None, question_type="Question Type",
                              choices=None, label="Ground Truth", explanation="Explanation"),
        "e.jsonl": Mapping(context="premise", question=None, question_type="ask-for",
                           choices=["hypothesis1", "hypothesis2"], label="label", explanation="conceptual_explanation"),
    }
    datasets = standardize(data, mapping_dict)

    print("Making Predictions...")
    predictions_results = predict(datasets, agent, manual_prompt=manual_prompt)

    explanation_agent = ExplanationEvaluationAgent(max_tokens=1000)

    explanation_eval_results = evaluate_explanation(dataset_names=list(datasets.keys()),
                                                    agent=explanation_agent                                                    )

    for predictions_result, explanation_result in zip(predictions_results, explanation_eval_results):
        evaluate(predictions_result, explanation_result)


if __name__ == "__main__":
    main()
