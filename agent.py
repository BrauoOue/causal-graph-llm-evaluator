import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from evaluate import evaluate

import pandas as pd
import json
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.prompts import  ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from prompt_builder import PromptBuilder

load_dotenv()


class PredictionResponse(BaseModel):
    """Structured response model for causal reasoning predictions"""
    chosen_answer: str = Field(description="The selected option from the given choices")
    explanation: str = Field(description="Short explanation of the reasoning (max 50 words).")
    confidence: float = Field(description="Confidence level (0-1)", ge=0, le=1, default=0.5)


class CausalReasoningAgent:
    """
    Agent for performing causal reasoning predictions using LangChain and LLMs
    """

    def __init__(self,
                 model_name: str = "gpt-4o-mini",
                 temperature: float = 0.1,
                 max_tokens: int = 500,
                 manual_prompt: Optional[str] = None):
        """
        Initialize the causal reasoning agent

        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens for response
            manual_prompt: Optional manual prompt override
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

        self.prompt_builder = PromptBuilder(manual_prompt=manual_prompt)
        self.output_parser = PydanticOutputParser(pydantic_object=PredictionResponse)

        self._create_prompt_template()

        self.predictions: List[Dict[str, Any]] = []
        self.total_cost: float = 0.0

    def _create_prompt_template(self):
        """Create the chat prompt template with system and human messages"""

        system_template = """You are an expert in {expertise_type}.
{task_instruction}

You must respond in the exact JSON format specified in the format instructions.
Be precise, analytical, and provide clear reasoning for your choices."""

        human_template = """Context:
{context}

Question:
{question}

Choices:
{choices}

{format_instructions}

Please provide your response in the exact JSON format specified above."""

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        self.chat_prompt = ChatPromptTemplate.from_messages([
            system_message_prompt,
            human_message_prompt
        ])

    def _parse_response(self, response_text: str) -> PredictionResponse:
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
                    return PredictionResponse(**data)
            except:
                pass

            return PredictionResponse(
                chosen_answer="Unable to parse response",
                explanation=f"Parsing error: {str(e)}. Raw response: {response_text}",
                confidence=0.0
            )

    def predict_single(self, row: pd.Series) -> Dict[str, Any]:
        """
        Make a prediction for a single row

        Args:
            row: DataFrame row to predict

        Returns:
            Dictionary containing prediction results
        """
        prompt_vars = self.prompt_builder.get_prompt_variables(row)

        prompt_vars["format_instructions"] = self.output_parser.get_format_instructions()

        messages = self.chat_prompt.format_messages(**prompt_vars)

        with get_openai_callback() as cb:
            response = self.llm.invoke(messages)
            self.total_cost += cb.total_cost

        parsed_response = self._parse_response(response.content)

        choices = row["choices"].split("/") if row["choices"] != "Yes/No" else ["Yes", "No"]

        is_valid_choice = any(choice.strip().lower() in parsed_response.chosen_answer.lower()
                              for choice in choices)

        try:
            # Try to use label as index (int)
            label_index = int(row["label"])
            correct_label = choices[label_index]
        except (ValueError, IndexError):
            # Use label as value (str)
            label_value = str(row["label"]).strip()
            correct_label = label_value if label_value in choices else choices[0]

        is_correct = correct_label.strip().lower() in parsed_response.chosen_answer.lower()

        # Moze treba da se izbrisat nekoi sto ne se potrebni za evaluacija kako: question_type, context, cost
        result = {
            "id": row.get("id", -1),
            "predicted_answer": parsed_response.chosen_answer,
            "correct_answer": correct_label,
            "is_correct": is_correct,
            "is_valid_choice": is_valid_choice,
            "explanation": parsed_response.explanation,
            "correct_explanation": row.get("explanation", "unknown"),
            "confidence": parsed_response.confidence,
            "question_type": row.get("question_type", "unknown"),
            "context": row["context"][:100] + "..." if len(row["context"]) > 100 else row["context"],
            "cost": cb.total_cost if 'cb' in locals() else 0.0
        }

        return result

    def _predict_single_with_retry(self, row_data: tuple, max_retries: int = 3) -> Dict[str, Any]:
        """
        Make a prediction for a single row with retry logic for rate limiting

        Args:
            row_data: Tuple of (index, row) from DataFrame iteration
            max_retries: Maximum number of retries for rate limiting

        Returns:
            Dictionary containing prediction results
        """
        idx, row = row_data

        for attempt in range(max_retries):
            try:
                result = self.predict_single(row)
                result["index"] = idx
                return result
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                    # Exponential backoff for rate limiting
                    wait_time = (2 ** attempt) * 1
                    print(f"Rate limit hit for row {idx}, waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"Error processing row {idx}: {str(e)}")
                    return {
                        "index": idx,
                        "error": str(e),
                        "question_type": row.get("question_type", "unknown")
                    }

        return {
            "index": idx,
            "error": "Max retries exceeded",
            "question_type": row.get("question_type", "unknown")
        }

    def predict_dataset(self,
                        data: pd.DataFrame,
                        save_results: bool = True,
                        output_file: str = "./output/predictions.json") -> List[Dict[str, Any]]:
        """
        Make predictions for an entire dataset

        Args:
            data: DataFrame containing the dataset
            save_results: Whether to save results to file
            output_file: Output file name

        Returns:
            List of prediction results
        """
        results = []

        print(f"Making predictions for {len(data)} rows...")

        for idx, row in data.iterrows():
            try:
                result = self.predict_single(row)
                results.append(result)

                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(data)} rows")

            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                results.append({
                    "index": row.get("index", idx),
                    "error": str(e),
                    "question_type": row.get("question_type", "unknown")
                })

        self.predictions = results

        if save_results:
            self.save_results(output_file)

        return results

    def predict_dataset_parallel(self,
                                 data: pd.DataFrame,
                                 batch_size: int = 10,
                                 max_workers: int = 5,
                                 save_results: bool = True,
                                 output_file: str = "predictions_parallel.json") -> List[Dict[str, Any]]:
        """
        Make predictions for an entire dataset using parallel processing

        Args:
            data: DataFrame containing the dataset
            batch_size: Number of rows to process in each batch
            max_workers: Maximum number of concurrent threads
            save_results: Whether to save results to file
            output_file: Output file name

        Returns:
            List of prediction results
        """
        print(f"Making predictions for {len(data)} rows with {max_workers} workers...")
        print(f"Processing in batches of {batch_size}")

        results = []

        for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start + batch_size, len(data))
            batch_data = data.iloc[batch_start:batch_end]

            print(f"Processing batch {batch_start // batch_size + 1}/{(len(data) + batch_size - 1) // batch_size} "
                  f"(rows {batch_start + 1}-{batch_end})")


            with ThreadPoolExecutor(max_workers=max_workers) as executor:

                future_to_row = {
                    executor.submit(self._predict_single_with_retry, (idx, row)): idx
                    for idx, row in batch_data.iterrows()
                }


                batch_results = []
                for future in as_completed(future_to_row):
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as e:
                        row_idx = future_to_row[future]
                        print(f"Error in thread for row {row_idx}: {str(e)}")
                        batch_results.append({
                            "index": row_idx,
                            "error": str(e),
                            "question_type": "unknown"
                        })

                batch_results.sort(key=lambda x: x.get("index", 0))
                results.extend(batch_results)

            print(f"Completed batch {batch_start // batch_size + 1}, "
                  f"total processed: {len(results)}/{len(data)}")

            if batch_end < len(data):
                time.sleep(0.5)

        self.predictions = results

        if save_results:
            self.save_results(output_file)

        print(f"Parallel processing completed! Total cost: ${self.total_cost:.4f}")
        return results

    def save_results(self, filename: str):
        """Save prediction results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.predictions, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")


def main():
    """
    Main function to run the causal reasoning agent
    """

    MODEL_NAME = "gpt-4o-mini"
    TEMPERATURE = 0.1
    MAX_TOKENS = 10000

    file_path = input("Enter path to test CSV or JSONL (e.g., test/e_test.csv): ").strip()

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    if file_path.endswith(".jsonl"):
        data = pd.read_json(file_path, lines=True)
    elif file_path.endswith(".csv"):
        data = pd.read_csv(file_path)
    else:
        print("Unsupported file format. Please use .csv or .jsonl")
        return

    custom_prompt = input("Enter custom prompt (or press Enter for default): ").strip()
    manual_prompt = custom_prompt if custom_prompt else None

    limit = input("Enter number of rows to process (or press Enter for all): ").strip()
    if limit.isdigit():
        data = data.head(int(limit))

    parallel_choice = input("Use parallel processing? (y/n): ").strip().lower()

    agent = CausalReasoningAgent(
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        manual_prompt=manual_prompt
    )

    if parallel_choice == 'y':
        batch_size = input("Enter batch size (default 10): ").strip()
        batch_size = int(batch_size) if batch_size.isdigit() else 10

        max_workers = input("Enter max workers (default 5): ").strip()
        max_workers = int(max_workers) if max_workers.isdigit() else 5

        results = agent.predict_dataset_parallel(
            data,
            batch_size=batch_size,
            max_workers=max_workers
        )
    else:
        results = agent.predict_dataset(data)

    return parallel_choice


if __name__ == "__main__":
    parallel_choice = main()
    evaluate(parallel_choice)