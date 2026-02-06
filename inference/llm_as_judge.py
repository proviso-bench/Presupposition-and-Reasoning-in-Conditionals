"""
LLM-as-a-Judge inference script for Conditional Probability Benchmark.
Uses Claude Sonnet 4.5 to evaluate model responses against checklist questions.
"""

import json
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Judge model configuration
JUDGE_MODEL = "claude-sonnet-4-5-20250929"

# Checklist categories mapping
CHECKLIST_CATEGORIES = {
    "with_context": [
        "accuracy_withC_filtered.json",
        "coherence_withC_filtered.json",
        "context_withC_filtered.json",
        "pragmatic_withC_filtered.json",
        "presupposition_withC_filtered.json"
    ],
    "without_context": [
        "accuracy_withoutC_filtered.json",
        "coherence_withoutC_filtered.json",
        "pragmatic_withoutC_filtered.json",
        "presupposition_withoutC_filtered.json"
    ]
}


def load_prompt_template(context_type: str) -> str:
    prompt_dir = Path(__file__).parent.parent / "prompt"

    if context_type == "with_context":
        prompt_file = prompt_dir / "llm_as_judge_with_context.txt"
    else:
        prompt_file = prompt_dir / "llm_as_judge_without_context.txt"

    with open(prompt_file, "r") as f:
        return f.read()


def load_checklist_questions(context_type: str) -> dict:
    checklist_dir = Path(__file__).parent.parent / "checklist" / "Filtered"

    if context_type == "with_context":
        checklist_path = checklist_dir / "with_context"
    else:
        checklist_path = checklist_dir / "without_context"

    all_questions = {}

    for filename in CHECKLIST_CATEGORIES[context_type]:
        filepath = checklist_path / filename
        if filepath.exists():
            with open(filepath, "r") as f:
                data = json.load(f)
                # Extract category name from filename (e.g., "accuracy" from "accuracy_withC_filtered.json")
                category = filename.replace("_withC_filtered.json", "").replace("_withoutC_filtered.json", "")
                # data is already a dict: {"Subcategory": ["question1", "question2", ...]}
                all_questions[category] = data

    return all_questions


def load_model_responses(source_type: str, model_name: str, context_type: str) -> list:
    if source_type == "closedsource":
        output_dir = Path(__file__).parent / "closedsource" / "outputs"
    else:
        output_dir = Path(__file__).parent / "opensource" / "outputs"

    # Construct filename pattern
    filename = f"{model_name}_{context_type}_likert.json"
    filepath = output_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Model output file not found: {filepath}")

    with open(filepath, "r") as f:
        return json.load(f)


def get_anthropic_client():
    """Initialize Anthropic client."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    return anthropic.Anthropic(api_key=api_key)


def run_judge_inference(client, prompt: str) -> str:
    """Run inference with Claude Sonnet 4.5 as judge."""
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text.strip()


def format_prompt_with_context(template: str, item: dict, checklist_question: str) -> str:
    """Format prompt for with_context evaluation."""
    return template.format(
        title=item.get("title", ""),
        background=item.get("background", ""),
        statement_1=item.get("statement_1", ""),
        statement_2=item.get("statement_2", ""),
        response=item.get("response", ""),
        checklist_question=checklist_question
    )


def format_prompt_without_context(template: str, item: dict, checklist_question: str) -> str:
    """Format prompt for without_context evaluation."""
    return template.format(
        title=item.get("title", ""),
        statement_1=item.get("statement_1", ""),
        statement_2=item.get("statement_2", ""),
        response=item.get("response", ""),
        checklist_question=checklist_question
    )


def parse_judge_response(response: str) -> bool:
    """Parse judge response to boolean."""
    response_lower = response.lower().strip()
    if "true" in response_lower:
        return True
    elif "false" in response_lower:
        return False
    else:
        # Default to False if unclear
        print(f"Warning: Unclear judge response '{response}', defaulting to False")
        return False


def get_output_file(source_type: str, model_name: str, context_type: str) -> Path:
    """Get output file path for judge results."""
    output_dir = Path(__file__).parent / "judge_outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir / f"judge_{source_type}_{model_name}_{context_type}.json"


def load_existing_results(output_file: Path) -> dict:
    """Load existing results from file."""
    if output_file.exists():
        try:
            with open(output_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, KeyError):
            return {}
    return {}


def save_results(results: dict, output_file: Path):
    """Save results to JSON file."""
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_evaluation(source_type: str, model_name: str, context_type: str):
    """Run LLM-as-a-judge evaluation."""
    client = get_anthropic_client()

    # Load data
    template = load_prompt_template(context_type)
    checklist_questions = load_checklist_questions(context_type)
    model_responses = load_model_responses(source_type, model_name, context_type)

    # Get output file and load existing results
    output_file = get_output_file(source_type, model_name, context_type)
    existing_results = load_existing_results(output_file)

    # Initialize results structure
    results = existing_results if existing_results else {
        "metadata": {
            "source_type": source_type,
            "model_name": model_name,
            "context_type": context_type,
            "judge_model": JUDGE_MODEL
        },
        "evaluations": []
    }

    # Create a set of already processed items (id, category, question)
    processed_ids = set()
    for eval_item in results.get("evaluations", []):
        # Handle both old format (question_category) and new format (subcategory)
        processed_ids.add((eval_item["id"], eval_item["category"], eval_item["question"]))

    # Count total evaluations needed
    # Structure: {"category": {"subcategory": ["q1", "q2", ...]}}
    total_questions = sum(
        len(questions)
        for category_data in checklist_questions.values()
        for questions in category_data.values()
    )
    total_evaluations = len(model_responses) * total_questions

    print(f"Running LLM-as-a-judge evaluation")
    print(f"Source: {source_type}, Model: {model_name}, Context: {context_type}")
    print(f"Judge model: {JUDGE_MODEL}")
    print(f"Total model responses: {len(model_responses)}")
    print(f"Total checklist questions: {total_questions}")
    print(f"Total evaluations needed: {total_evaluations}")
    print(f"Already processed: {len(processed_ids)}")
    print("-" * 50)

    current_eval = 0

    for item in model_responses:
        item_id = item["id"]

        # Skip if response is None or has error
        if item.get("response") is None or item.get("error"):
            print(f"Skipping item {item_id} - no response or error")
            continue

        # Iterate through: category (e.g., "accuracy") -> subcategory (e.g., "Presupposition Trigger Identification") -> questions
        for category, subcategories in checklist_questions.items():
            for subcategory, questions in subcategories.items():
                for question in questions:
                    current_eval += 1

                    # Skip if already processed
                    eval_key = (item_id, category, question)
                    if eval_key in processed_ids:
                        continue

                    # Format prompt based on context type
                    if context_type == "with_context":
                        prompt = format_prompt_with_context(template, item, question)
                    else:
                        prompt = format_prompt_without_context(template, item, question)

                    try:
                        judge_response = run_judge_inference(client, prompt)
                        result = parse_judge_response(judge_response)

                        eval_result = {
                            "id": item_id,
                            "title": item.get("title", ""),
                            "category": category,
                            "subcategory": subcategory,
                            "question": question,
                            "judge_response": judge_response,
                            "result": result
                        }

                        results["evaluations"].append(eval_result)
                        processed_ids.add(eval_key)

                        # Save incrementally
                        save_results(results, output_file)

                        print(f"[{current_eval}/{total_evaluations}] ID: {item_id}, {category}/{subcategory} - {result}")

                    except Exception as e:
                        print(f"Error evaluating item {item_id}, {category}/{subcategory}: {e}")
                        eval_result = {
                            "id": item_id,
                            "title": item.get("title", ""),
                            "category": category,
                            "subcategory": subcategory,
                            "question": question,
                            "judge_response": None,
                            "result": None,
                            "error": str(e)
                        }
                        results["evaluations"].append(eval_result)
                        save_results(results, output_file)

    # Calculate summary statistics
    true_count = sum(1 for e in results["evaluations"] if e.get("result") is True)
    false_count = sum(1 for e in results["evaluations"] if e.get("result") is False)
    error_count = sum(1 for e in results["evaluations"] if e.get("error"))

    results["summary"] = {
        "total_evaluations": len(results["evaluations"]),
        "true_count": true_count,
        "false_count": false_count,
        "error_count": error_count,
        "true_rate": true_count / len(results["evaluations"]) if results["evaluations"] else 0
    }

    save_results(results, output_file)
    print(f"\nResults saved to: {output_file}")
    print(f"Summary: {true_count} True, {false_count} False, {error_count} Errors")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run LLM-as-a-judge evaluation")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        choices=["closedsource", "opensource"],
        help="Source type of model outputs"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., gpt5, gemini, llama, qwen)"
    )
    parser.add_argument(
        "--context",
        type=str,
        required=True,
        choices=["with_context", "without_context"],
        help="Context type to evaluate"
    )

    args = parser.parse_args()

    run_evaluation(args.source, args.model, args.context)


if __name__ == "__main__":
    main()
