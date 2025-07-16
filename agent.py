import os
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
    explanation: str = Field(description="Short explanation of the reasoning (max 10 words).")
    confidence: float = Field(description="Confidence level (0-1)", ge=0, le=1, default=0.5)


class CausalReasoningAgent:
    """
    Agent for performing causal reasoning predictions using LangChain and LLMs
    """

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
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

        correct_label = choices[row["label"]]
        is_correct = correct_label.strip().lower() in parsed_response.chosen_answer.lower()

        # Moze treba da se izbrisat nekoi sto ne se potrebni za evaluacija kako: question_type, context, cost
        result = {
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

    def predict_dataset(self,
                        data: pd.DataFrame,
                        save_results: bool = True,
                        output_file: str = "predictions.json") -> List[Dict[str, Any]]:
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

    def save_results(self, filename: str):
        """Save prediction results to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.predictions, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")


def main():
    """
    Main function to run the causal reasoning agent
    """

    MODEL_NAME = "gpt-3.5-turbo"
    TEMPERATURE = 0.1
    MAX_TOKENS = 500

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

    agent = CausalReasoningAgent(
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        manual_prompt=manual_prompt
    )

    #Ovie results ili dokolku se zacuvuva fo fajl podocna kje se koristat za presmetuvanje accuracy i evaluacija
    results = agent.predict_dataset(data)


if __name__ == "__main__":
    main()