import pandas as pd
from pathlib import Path

from src.ai_assistant.gemini_router import route_with_gemini
from src.ai_assistant.assistant import answer_question

EVAL_PATH = Path(__file__).resolve().parent / "eval_questions.csv"


def run_evaluation():
    df = pd.read_csv(EVAL_PATH)

    total = len(df)
    correct = 0
    results = []

    for _, row in df.iterrows():
        question = row["question"]
        expected = row["expected_intent"]

        route = route_with_gemini(question)
        predicted = route["intent"]

        try:
            answer = answer_question(question)
            response_generated = bool(answer and isinstance(answer, str))
        except Exception:
            response_generated = False

        is_correct = predicted == expected and response_generated

        if is_correct:
            correct += 1

        results.append({
            "question": question,
            "expected_intent": expected,
            "predicted_intent": predicted,
            "response_generated": response_generated,
            "result": "PASS" if is_correct else "FAIL"
        })

    accuracy = correct / total * 100

    print("\nAI Assistant Evaluation")
    print("-" * 35)
    print(f"Questions tested     : {total}")
    print(f"Correct responses    : {correct}")
    print(f"Accuracy             : {accuracy:.2f}%")
    print("-" * 35)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    run_evaluation()