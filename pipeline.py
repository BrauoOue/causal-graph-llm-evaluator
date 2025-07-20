import json
import os
import time
from typing import List, Dict, Any, Optional

import pandas as pd

from agent import CausalReasoningAgent
from conversion import Mapping, BuilderDataset


# Load datasets from a folder
def load_pd_files(folder: str) -> dict[str,pd.DataFrame]:
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


def process_datasets(
        my_dic: dict[str,pd.DataFrame],
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
    results = []

    for dataframe_name in my_dic:
        dic_data = dict()
        dic_data["name"] = dataframe_name
        split___ = dataframe_name.split('.')[0]
        time_start = time.time()
        cost = agent.predict_dataset_parallel(my_dic[dataframe_name], manual_prompt=manual_prompt, output_file = f"output/{split___}.json")
        time_end = time.time()
        dic_data["cost"] = cost
        dic_data["time"] = time_end - time_start
        results.append(dic_data)

    return results


def conversion(dataframes:dict[str,pd.DataFrame])->dict[str,pd.DataFrame]:
    code_mapping = Mapping(context="Code", question="Question", question_type="Question Type", choices=None,
                           label="Ground Truth", explanation="Explanation")
    math_mapping = Mapping(context="Mathematical Scenario", question="Question", question_type="Question Type",
                           choices=None,
                           label="Ground Truth", explanation="Explanation")

    text_mapping = Mapping(context="Scenario and Question", question=None, question_type="Question Type",
                           choices=None,
                           label="Ground Truth", explanation="Explanation")

    e_mapping = Mapping(context="premise", question=None, question_type="ask-for",
                        choices=["hypothesis1", "hypothesis2"],
                        label="label", explanation="conceptual_explanation")

    mapping_mapping = {
        "code.jsonl" : code_mapping,
        "math.jsonl" : math_mapping,
        "text.jsonl" : text_mapping,
        "e.jsonl" : e_mapping,
    }

    results = {}
    for dataframe_name in dataframes:
        mapping = mapping_mapping[dataframe_name]
        result = BuilderDataset.convert(dataframes[dataframe_name], mapping)
        result['id'] = result.index
        results[dataframe_name] = result.head(100)
        
    return results

# Main function
def main():
    """
    Main execution function.

    Args:
        template_type (str): Specify the template type ("auto" or "manual").
    """
    dataset_folder = "data/"

    print("Loading datasets...")
    datasets = load_pd_files(dataset_folder)

    agent = CausalReasoningAgent(max_tokens=10000)

    manual_prompt = input("Enter your custom prompt (or press Enter to use automatic prompt generation): ").strip()
    if not manual_prompt:
        manual_prompt = None

    print("Standardizing datasets...")
    converted_datasets = conversion(datasets)

    import evaluate,time

    print("Processing datasets...")
    results = process_datasets(converted_datasets, agent, manual_prompt=manual_prompt)




if __name__ == "__main__":
    main()
