"""
Open-source model inference script for Conditional Probability Benchmark.
Models: Qwen/Qwen3-VL-30B-A3B-Thinking, meta-llama/Llama-3.1-8B-Instruct
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables BEFORE importing HuggingFace libraries
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Set cache directories
CACHE_DIR = "/cluster/scratch/yongyu/cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_DIR}/datasets"

import json
import argparse
from datetime import datetime
from huggingface_hub import InferenceClient

# Model configurations
MODELS = {
    "qwen": "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "llama": "meta-llama/Llama-3.1-8B-Instruct"
}

def load_prompt_template(prompt_type: str) -> str:
    """Load prompt template from file."""
    prompt_dir = Path(__file__).parent.parent.parent / "prompt"
    if prompt_type == "with_context":
        prompt_file = prompt_dir / "prompt_with_context.txt"
    else:
        prompt_file = prompt_dir / "prompt_without_context.txt"

    with open(prompt_file, "r") as f:
        return f.read()

def load_data() -> list:
    """Load problem set data."""
    data_path = Path(__file__).parent.parent.parent / "data" / "problem_set.json"
    with open(data_path, "r") as f:
        return json.load(f)

def format_prompt(template: str, item: dict, prompt_type: str) -> str:
    """Format prompt with item data."""
    if prompt_type == "with_context":
        # Generate a context based on the title
        context = f"You are evaluating a scenario about {item['title']}."
        return template.format(
            context=context,
            statement_1=item["statement_1"],
            statement_2=item["statement_2"]
        )
    else:
        return template.format(
            statement_1=item["statement_1"],
            statement_2=item["statement_2"]
        )

def run_inference(model_name: str, prompt_type: str, num_runs: int = 2):
    """Run inference on all items."""
    # Initialize client
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")

    client = InferenceClient(api_key=api_key)

    # Load data and prompt
    data = load_data()
    template = load_prompt_template(prompt_type)

    # Get model ID
    model_id = MODELS.get(model_name)
    if not model_id:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    results = []
    total_items = len(data) * num_runs

    print(f"Running inference with {model_id}")
    print(f"Prompt type: {prompt_type}")
    print(f"Total generations: {total_items}")
    print("-" * 50)

    for run_idx in range(num_runs):
        for item_idx, item in enumerate(data):
            prompt = format_prompt(template, item, prompt_type)

            try:
                # Create chat completion
                messages = [{"role": "user", "content": prompt}]

                response = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.7
                )

                response_text = response.choices[0].message.content

                result = {
                    "id": item["id"],
                    "run": run_idx + 1,
                    "title": item["title"],
                    "probability": item["probability"],
                    "statement_1": item["statement_1"],
                    "statement_2": item["statement_2"],
                    "prompt_type": prompt_type,
                    "model": model_id,
                    "response": response_text,
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)

                current = run_idx * len(data) + item_idx + 1
                print(f"[{current}/{total_items}] ID: {item['id']}, Run: {run_idx + 1}")

            except Exception as e:
                print(f"Error on item {item['id']}, run {run_idx + 1}: {e}")
                result = {
                    "id": item["id"],
                    "run": run_idx + 1,
                    "title": item["title"],
                    "probability": item["probability"],
                    "statement_1": item["statement_1"],
                    "statement_2": item["statement_2"],
                    "prompt_type": prompt_type,
                    "model": model_id,
                    "response": None,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                results.append(result)

    return results

def save_results(results: list, model_name: str, prompt_type: str):
    """Save results to JSON file."""
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{model_name}_{prompt_type}_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Run open-source model inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODELS.keys()),
        help="Model to use (qwen or llama)"
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        required=True,
        choices=["with_context", "without_context"],
        help="Prompt type to use"
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=2,
        help="Number of runs per item (default: 2)"
    )

    args = parser.parse_args()

    results = run_inference(args.model, args.prompt_type, args.num_runs)
    save_results(results, args.model, args.prompt_type)

    print(f"\nTotal results: {len(results)}")

if __name__ == "__main__":
    main()
