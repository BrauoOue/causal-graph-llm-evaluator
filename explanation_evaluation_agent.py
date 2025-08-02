import os
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Union

import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()


class ExplanationEvaluationResponse(BaseModel):
    """Structured response model for explanation evaluation"""
    correct_explanation: bool = Field(description="Whether the explanations are semantically similar and both correct")
    confidence: float = Field(description="Confidence level (0-1)", ge=0, le=1)


class ExplanationEvaluationAgent:
    """
    Agent for evaluating the similarity and correctness of causal reasoning explanations
    """

    def __init__(self,
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.1,
                 max_tokens: int = 500):
        """
        Initialize the explanation evaluation agent

        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens for response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.getenv("OPENAI_API_KEY")
        )

        self.output_parser = PydanticOutputParser(pydantic_object=ExplanationEvaluationResponse)
        self._load_system_prompt()
        self._create_prompt_template()

        self.evaluations: List[Dict[str, Any]] = []
        self.total_cost: float = 0.0

    def _load_system_prompt(self):
        """Load system prompt from file"""
        prompt_file = "./prompts/explanation/starting_instructions.txt"
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read().strip()
        except FileNotFoundError:
            self.system_prompt = """You are an expert evaluator of causal reasoning explanations. 
            Assess the similarity and correctness of two explanations about causal relationships."""

    def _create_prompt_template(self):
        """Create the chat prompt template with system and human messages"""

        system_template = f"""{self.system_prompt}

You must respond in the exact JSON format specified in the format instructions.
Be precise, analytical, and provide clear reasoning for your assessment."""

        human_template = """Please evaluate the following two explanations for their semantic similarity and correctness:

Predicted Explanation:
{predicted_explanation}

Correct Explanation:
{correct_explanation}

{format_instructions}

Please provide your evaluation in the exact JSON format specified above."""

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        self.chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])

    def _parse_response(self, response_text: str) -> ExplanationEvaluationResponse:
        """
        Parse the LLM response into structured format

        Args:
            response_text: Raw response from LLM

        Returns:
            Parsed response object
        """
        try:
            return self.output_parser.parse(response_text)
        except Exception as e:
            try:
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    data = json.loads(json_str)
                    return ExplanationEvaluationResponse(**data)
            except:
                pass

            return ExplanationEvaluationResponse(
                correct_explanation=False,
                confidence=0.0
            )

    def evaluate_single(self, row: pd.Series, index: int) -> Dict[str, Any]:
        """
        Evaluate a single explanation entry from DataFrame row

        Args:
            row: pandas Series containing prediction data
            index: Index of the row in the DataFrame

        Returns:
            Dictionary containing evaluation results
        """
        if not row.get("is_correct", False):
            return  {
                "id": index,
                "correct_explanation": False,
                "confidence": 1.0,
                "predicted_explanation": row.get("explanation", ""),
                "reference_explanation": row.get("correct_explanation", ""),
                "cost": 0.0
            }

        prompt_vars = {
            "predicted_explanation": row.get("explanation", ""),
            "correct_explanation": row.get("correct_explanation", ""),
            "context": row.get("context", "No additional context provided"),
            "format_instructions": self.output_parser.get_format_instructions()
        }

        messages = self.chat_prompt.format_messages(**prompt_vars)

        with get_openai_callback() as cb:
            response = self.llm.invoke(messages)
            self.total_cost += cb.total_cost

        parsed_response = self._parse_response(response.content)

        result = {
            "id": row.get("id", -1),
            "correct_explanation": parsed_response.correct_explanation,
            "confidence": parsed_response.confidence,
            "predicted_explanation": row.get("explanation", ""),
            "reference_explanation": row.get("correct_explanation", ""),
            "cost": cb.total_cost if 'cb' in locals() else 0.0
        }

        return result

    def _evaluate_single_with_retry(self, row_data: tuple, max_retries: int = 3) -> Dict[str, Any]:
        """
        Evaluate a single entry with retry logic for rate limiting

        Args:
            row_data: Tuple of (index, row) from DataFrame iteration
            max_retries: Maximum number of retries for rate limiting

        Returns:
            Dictionary containing evaluation results
        """
        idx, row = row_data

        for attempt in range(max_retries):
            try:
                result = self.evaluate_single(row, idx)
                return result
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    # Exponential backoff for rate limiting
                    wait_time = (2 ** attempt) * 1
                    print(f"Rate limit hit for entry {idx}, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Error processing entry {idx}: {str(e)}")
                    return {
                        "index": idx,
                        "error": str(e),
                        "correct_explanation": False,
                        "confidence": 0.0
                    }

        return {
            "index": idx,
            "error": "Max retries exceeded",
            "correct_explanation": False,
            "confidence": 0.0
        }

    def evaluate_dataset(self,
                         df: pd.DataFrame,
                         limit: Optional[int] = None,
                         save_results: bool = True,
                         output_file: str = "./output/explanation.json") -> List[Dict[str, Any]]:
        """
        Evaluate explanations for an entire dataset

        Args:
            df: pandas DataFrame containing prediction data
            limit: Optional limit on number of entries to process
            save_results: Whether to save results to file
            output_file: Output file name

        Returns:
            List of evaluation results
        """
        if limit:
            df = df.head(limit)

        results = []
        print(f"Evaluating explanations for {len(df)} entries...")

        for idx, row in df.iterrows():
            try:
                result = self.evaluate_single(row, idx)
                results.append(result)

                if (len(results)) % 10 == 0:
                    print(f"Processed {len(results)}/{len(df)} entries")

            except Exception as e:
                print(f"Error processing entry {idx}: {str(e)}")
                results.append({
                    "index": idx,
                    "error": str(e),
                    "correct_explanation": False,
                    "confidence": 0.0
                })

        self.evaluations = results

        if save_results:
            self.save_results(output_file)

        return results

    def evaluate_dataset_parallel(self,
                                  df: pd.DataFrame,
                                  limit: Optional[int] = None,
                                  batch_size: int = 10,
                                  max_workers: int = 5,
                                  save_results: bool = True,
                                  output_folder:str= "./output/explanations",
                                  dataset_name: str = "explanation_parallel") -> float:
        """
        Evaluate explanations for an entire dataset using parallel processing

        Args:
            df: pandas DataFrame containing prediction data
            limit: Optional limit on number of entries to process
            batch_size: Number of entries to process in each batch
            max_workers: Maximum number of concurrent threads
            save_results: Whether to save results to file
            dataset_name: Output file name

        Returns:
            List of evaluation results
        """
        if limit:
            df = df.head(limit)

        print(f"Evaluating explanations for {len(df)} entries with {max_workers} workers...")
        print(f"Processing in batches of {batch_size}")

        results = []

        for batch_start in range(0, len(df), batch_size):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]

            print(f"Processing batch {batch_start // batch_size + 1}/{(len(df) + batch_size - 1) // batch_size} "
                  f"(entries {batch_start + 1}-{batch_end})")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_entry = {
                    executor.submit(self._evaluate_single_with_retry, (idx, row)): idx
                    for idx, row in batch_df.iterrows()
                }

                batch_results = []
                for future in as_completed(future_to_entry):
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as e:
                        entry_idx = future_to_entry[future]
                        print(f"Error in thread for entry {entry_idx}: {str(e)}")
                        batch_results.append({
                            "index": entry_idx,
                            "error": str(e),
                            "correct_explanation": False,
                            "confidence": 0.0
                        })

                batch_results.sort(key=lambda x: x.get("index", 0))
                results.extend(batch_results)

            print(f"Completed batch {batch_start // batch_size + 1}, "
                  f"total processed: {len(results)}/{len(df)}")

            if batch_end < len(df):
                time.sleep(0.5)

        self.evaluations = results


        if save_results:
            path = os.path.join(output_folder,f"{dataset_name}.json")
            self.save_results(path)

        print(f"Parallel processing completed! Total cost: ${self.total_cost:.4f}")
        return self.total_cost

    def save_results(self, filename: str):
        """Save evaluation results to JSON file"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.evaluations, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")

    @staticmethod
    def load_predictions(predictions_folder="./output/predictions",dataset_name: str="predictions") -> pd.DataFrame:
        """Load predictions from JSON file and return as DataFrame"""
        filename = os.path.join(predictions_folder, f"{dataset_name}.json")
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)

    @staticmethod
    def load_predictions_from_csv(file_path: str) -> pd.DataFrame:
        """Load predictions from CSV file"""
        return pd.read_csv(file_path)


def main():
    """
    Main function to run the explanation evaluation agent
    """
    MODEL_NAME = "gpt-4o-mini"
    TEMPERATURE = 0.1
    MAX_TOKENS = 500

    # Load predictions data
    predictions_file = "./output/predictions.json"
    if not os.path.exists(predictions_file):
        print(f"Predictions file not found: {predictions_file}")
        return

    df = ExplanationEvaluationAgent.load_predictions(predictions_file)
    print(f"Loaded {len(df)} prediction entries")
    print(f"DataFrame columns: {list(df.columns)}")

    # Get user preferences (or set defaults for testing)
    limit = input("Enter number of entries to process (or press Enter for all): ").strip()
    limit = int(limit) if limit.isdigit() else None

    parallel_choice = input("Use parallel processing? (y/n): ").strip().lower()

    agent = ExplanationEvaluationAgent(
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS
    )

    if parallel_choice == 'y':
        batch_size = input("Enter batch size (default 10): ").strip()
        batch_size = int(batch_size) if batch_size.isdigit() else 10

        max_workers = input("Enter max workers (default 5): ").strip()
        max_workers = int(max_workers) if max_workers.isdigit() else 5

        results = agent.evaluate_dataset_parallel(
            df,
            limit=limit,
            batch_size=batch_size,
            max_workers=max_workers
        )
    else:
        results = agent.evaluate_dataset(df, limit=limit)

    # Print summary
    correct_count = sum(1 for r in results if r.get("correct_explanation", False))
    total_count = len(results)
    avg_confidence = sum(r.get("confidence", 0) for r in results) / total_count if total_count > 0 else 0

    print(f"\nEvaluation Summary:")
    print(f"Total entries: {total_count}")
    print(f"Correct explanations: {correct_count}")
    print(f"Accuracy: {correct_count/total_count:.2%}" if total_count > 0 else "N/A")
    print(f"Average confidence: {avg_confidence:.3f}")
    print(f"Total cost: ${agent.total_cost:.4f}")


if __name__ == "__main__":
    main()
