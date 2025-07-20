from __future__ import annotations

import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class Mapping:

    def __init__(self, context, question, question_type, choices: list | None, label, explanation):
        self.label = label
        self.choices = choices
        self.question_type = question_type
        self.question = question
        self.context = context
        self.explanation = explanation

    def get_columns(self):
        return ["context", "question_type", "choices", "label", "explanation"]


def handle_question(q_type: str):
    if 'from' in q_type.lower():
        if 'from effect' in q_type.lower():
            return "cause"
        return 'effect'

    return q_type


class BuilderDataset:

    @staticmethod
    def convert(df, mapping: Mapping):
        copy = df.copy()
        print(mapping.get_columns())

        if not mapping.choices:
            copy["choices"] = "Yes/No"
        else:
            copy["choices"] = copy[mapping.choices].apply(lambda row: "/".join(row.astype(str)), axis=1)

        if mapping.question:
            copy.rename(columns={
                mapping.context: 'context',
                mapping.label: "label",
                mapping.explanation: "explanation",
                mapping.question: "question",
                mapping.question_type: "question_type"
            }, inplace=True)
            copy['context'] = copy['context'] + '\n\n' + copy['question']
        else:
            copy.rename(columns={
                mapping.context: 'context',
                mapping.label: "label",
                mapping.explanation: "explanation",
                mapping.question_type: "question_type"
            }, inplace=True)

        copy["question_type"] = copy["question_type"].apply(handle_question)

        return copy[mapping.get_columns()]


if __name__ == '__main__':

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

    e_df = pd.read_json("./data/e.jsonl", lines=True)
    code_df = pd.read_json("./data/code.jsonl", lines=True)
    math_df = pd.read_json("./data/math.jsonl", lines=True)
    text_df = pd.read_json("./data/text.jsonl", lines=True)

    dfs = [code_df, math_df, text_df, e_df]
    mappings = [code_mapping, math_mapping, text_mapping, e_mapping]
    names = ["code", "math", "text", "e"]

    for df, mapping, name in zip(dfs, mappings, names):
        result = BuilderDataset.convert(df, mapping)
        result.to_csv(f"./test/{name}_test.csv")
