import json
import csv
import os
from difflib import SequenceMatcher

def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def exact_match(a, b):
    return a.strip().lower() == b.strip().lower()

def fuzzy_match(a, b):
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()

def evaluate(parallel_choice):
    path = "output/predictions_parallel.json" if parallel_choice.lower() == "y" else "output/predictions.json"
    predictions = load_predictions(path)

    exact_matches = 0
    fuzzy_scores = []
    valid_choice_count = 0
    correct_count = 0
    response_times = []

    csv_rows = []

    for item in predictions:
        pred_ans = item.get("predicted_answer", "")
        true_ans = item.get("correct_answer", "")
        is_correct = item.get("is_correct", False)
        is_valid = item.get("is_valid_choice", False)
        response_time = item.get("response_time", None)

        exact = exact_match(pred_ans, true_ans)
        fuzzy = fuzzy_match(pred_ans, true_ans)

        correct_count += int(is_correct)
        valid_choice_count += int(is_valid)
        if exact:
            exact_matches += 1

        if response_time is not None:
            try:
                response_time = float(response_time)
                response_times.append(response_time)
            except:
                pass

        csv_rows.append({
            **item,
            "exact_match": exact,
            "fuzzy_score": fuzzy,
        })

    total = len(predictions)
    accuracy = correct_count / total
    exact_acc = exact_matches / total
    avg_fuzzy = sum(fuzzy_scores) / total if fuzzy_scores else 0.0
    valid_ratio = valid_choice_count / total
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0

    print(f"\n  Evaluation Results:")
    print(f"  Accuracy (is_correct):       {accuracy:.2f}")
    print(f"  Exact Match Accuracy:        {exact_acc:.2f}")
    print(f"  Avg. Fuzzy Match Score:      {avg_fuzzy:.2f}")
    print(f"  Valid Choices (% valid):     {valid_ratio:.2f}")
    print(f"  Avg. Response Time (s):      {avg_response_time:.2f}")
    print(f"  Total Predictions:           {total}")

if __name__ == "__main__":
    choice = input("Use parallel predictions? (y/n): ")
    evaluate(choice)
