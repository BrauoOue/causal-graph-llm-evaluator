import pandas as pd
import os

class PromptBuilder:
    def __init__(self, manual_prompt=None):
        self.manual_prompt = manual_prompt

    def build_prompt(self, row):
        context = row["context"].strip()
        qtype = row["question_type"].lower()
        label = row["label"]
        explanation = row.get("explanation", "")

        if qtype == "cause":
            task_desc = "identify the most likely cause of the given event."
            question = "What is the most likely cause?"
        elif qtype == "effect":
            task_desc = "identify the most likely effect of the given event."
            question = "What is the most likely effect?"
        else:
            task_desc = "answer the following question based on the given context."
            question = "Choose the most appropriate answer."

        choices_raw = row["choices"]
        if choices_raw == "Yes/No":
            choices = ["Yes", "No"]
            prompt_header = "You are an expert in reasoning about mathematical, logical, or computational scenarios."
            question = "Answer the yes/no question contained in the context."
        else:
            choices = choices_raw.split("/")
            prompt_header = "You are an expert in causal reasoning."

        prompt = f"{prompt_header}\n"
        if self.manual_prompt:
            prompt += f"{self.manual_prompt.strip()}\n\n"
        else:
            prompt += f"Based on the following situation, {task_desc}\n\n"

        prompt += f"Context:\n{context}\n\n"
        prompt += f"Question:\n{question}\n\n"
        prompt += "Choices:\n" + "\n".join(f"- {c.strip()}" for c in choices)

        return prompt


if __name__ == '__main__':
    file_path = input("Enter path to test CSV or JSONL (e.g., test/e_test.csv): ").strip()
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        exit(1)

    user_prompt = input("Enter your custom prompt (or press Enter to use automatic prompt generation): ").strip()
    manual_prompt = user_prompt if user_prompt else None

    if file_path.endswith(".jsonl"):
        df = pd.read_json(file_path, lines=True)
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        print("Unsupported file format. Please use .csv or .jsonl")
        exit(1)
    builder = PromptBuilder(manual_prompt=manual_prompt)

    prompts = df.apply(builder.build_prompt, axis=1)

    txt_path = os.path.join(os.path.dirname(file_path), "prompts.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(prompts, 1):
            f.write(f"--- Prompt {i} ---\n{prompt}\n\n")

    print(f"All prompts saved to: {txt_path}")
