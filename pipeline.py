import json
import os
from typing import List, Dict, Any
from agent import CausalReasoningAgent


class DummyAgent:
    def predict(self, input_data: Dict[str, Any], manual_prompt: str) -> Dict[str, str]:
        return {
            "input_data": input_data,
            "manual_prompt": manual_prompt,
            "predicted_explanation": "Dummy explanation"
        }


def load_jsonl_files(folder: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads all .jsonl files in the specified folder.

    Args:
        folder (str): The path to the folder containing .jsonl files.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Dictionary mapping filenames to their list of entries.
    """
    datasets = {}
    for filename in os.listdir(folder):
        if filename.endswith(".jsonl"):
            path = os.path.join(folder, filename)
            with open(path, 'r', encoding='utf-8') as file:
                datasets[filename] = [json.loads(line) for line in file]
    return datasets


def process_datasets(datasets: Dict[str, List[Dict[str, Any]]], agent: DummyAgent, manual_prompt: str):  # Change DummyAgent to CRA
    """
    Processes all datasets using the specified Agent instance.

    Args:
        datasets (Dict[str, List[Dict[str, Any]]]): The loaded datasets.
        agent (Agent): The agent instance used for predictions.
        manual_prompt (str): The manual prompt to pass for predictions.

    Returns:
        List[Dict[str, Any]]: A list of prediction results.
    """
    results = []
    for filename, entries in datasets.items():
        print(f"Processing dataset: {filename}")
        for entry in entries:
            prediction = agent.predict(input_data=entry, manual_prompt=manual_prompt)
            results.append({
                "filename": filename,
                "input_data": entry,
                "prediction": prediction
            })
    return results


def save_results(results: List[Dict[str, Any]], output_file: str):
    """
    Saves the results to a .jsonl file.

    Args:
        results (List[Dict[str, Any]]): The results to save.
        output_file (str): The output file path.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        for result in results:
            file.write(json.dumps(result) + '\n')
    print(f"Results saved to {output_file}")


def main():
    # Folder containing the datasets
    dataset_folder = "data/"

    # Manual prompt to be used
    manual_prompt = "Analyze the explanation and match it with the question."

    # Load datasets
    print("Loading datasets...")
    datasets = load_jsonl_files(dataset_folder)

    # Initialize the Agent
    agent = DummyAgent()

    # Process datasets
    print("Processing datasets...")
    results = process_datasets(datasets, agent, manual_prompt)

    # Save results to output.jsonl
    output_file = "output.jsonl"
    save_results(results, output_file)


if __name__ == "__main__":
    main()
