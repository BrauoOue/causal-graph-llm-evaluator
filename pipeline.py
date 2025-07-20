import json
import os
from typing import List, Dict, Any, Optional


# Abstract Agent class for extensibility
class AbstractAgent:
    def predict(self, input_data: Dict[str, Any], template: str) -> Dict[str, str]:
        raise NotImplementedError("Subclasses must implement the predict method.")


class DummyAgent(AbstractAgent):
    def predict(self, input_data: Dict[str, Any], template: str) -> Dict[str, str]:
        return {
            "input_data": input_data,
            "template": template,
            "predicted_explanation": "Dummy explanation"
        }


# Load datasets from a folder
def load_jsonl_files(folder: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads all .jsonl files from the specified folder.

    Args:
        folder (str): Directory containing .jsonl dataset files.

    Returns:
        Dict[str, List[Dict[str, Any]]]: A dictionary mapping filenames to their data entries.
    """
    datasets = {}
    for filename in os.listdir(folder):
        if filename.endswith(".jsonl"):
            path = os.path.join(folder, filename)
            with open(path, 'r', encoding='utf-8') as file:
                datasets[filename] = [json.loads(line) for line in file]
    return datasets


# Process datasets using the agent and a specific template
def process_datasets(
        datasets: Dict[str, List[Dict[str, Any]]],
        agent: AbstractAgent,
        manual_prompt: str,
        template_type: str = "auto"
) -> List[Dict[str, Any]]:
    """
    Process datasets with the specified template type.

    Args:
        datasets (Dict[str, List[Dict[str, Any]]]): Loaded datasets.
        agent (AbstractAgent): The agent to process data.
        manual_prompt (str): The manual prompt to use with "manual" mode.
        template_type (str): Template type ("auto" or "manual").

    Returns:
        List[Dict[str, Any]]: Results including predictions.
    """
    results = []
    for filename, entries in datasets.items():
        print(f"Processing file: {filename}, Template Type: {template_type}")
        for entry in entries:
            if template_type == "manual":
                template = manual_prompt
            elif template_type == "auto":
                template = generate_auto_prompt(entry)  # Generate a dynamic template
            else:
                raise ValueError(f"Invalid template_type: {template_type}")

            prediction = agent.predict(input_data=entry, template=template)
            results.append({
                "filename": filename,
                "input_data": entry,
                "prediction": prediction
            })
    return results


def generate_auto_prompt(data_entry: Dict[str, Any]) -> str:
    """
    Auto-generate a prompt for agents based on input data.

    Args:
        data_entry (Dict[str, Any]): A single data entry from the dataset.

    Returns:
        str: The auto-generated template.
    """
    # Example logic to generate auto-prompt (can be customized)
    return f"Provide an explanation for the question: {data_entry.get('question', 'N/A')}."


# Save processed results to file
def save_results(results: List[Dict[str, Any]], output_file: str):
    """
    Save results to a .jsonl file.

    Args:
        results (List[Dict[str, Any]]): Processed dataset results.
        output_file (str): Output file path.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        for result in results:
            file.write(json.dumps(result) + '\n')
    print(f"Results saved to {output_file}")


# Main function
def main(template_type: str = "auto"):
    """
    Main execution function.

    Args:
        template_type (str): Specify the template type ("auto" or "manual").
    """
    # Configuration
    dataset_folder = "data/"
    output_file = "output.jsonl"
    manual_prompt = "Analyze the explanation and match it with the question."

    # Load datasets
    print("Loading datasets...")
    datasets = load_jsonl_files(dataset_folder)

    # Initialize an agent (replace with a real LLM agent when needed)
    agent = DummyAgent()

    # Process datasets
    print("Processing datasets...")
    results = process_datasets(datasets, agent, manual_prompt, template_type)

    # Save results
    save_results(results, output_file)


if __name__ == "__main__":
    # Specify template type here (can be "auto" or "manual")
    main(template_type="auto")
