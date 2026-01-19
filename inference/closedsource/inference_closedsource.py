"""
Closed-source model inference script for Conditional Probability Benchmark.
Models: gpt-5, gemini-2.5-flash
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

# Load environment variables
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Model configurations
MODELS = {
    "gpt5": {"provider": "openai", "model_id": "gpt-5"},
    "gemini": {"provider": "google", "model_id": "gemini-2.5-flash"}
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

def get_openai_client():
    """Initialize OpenAI client."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)

def get_gemini_client():
    """Initialize Gemini client."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)
    return genai

def run_openai_inference(client, model_id: str, prompt: str) -> str:
    """Run inference with OpenAI model."""
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.7
    )
    return response.choices[0].message.content

def run_gemini_inference(client, model_id: str, prompt: str) -> str:
    """Run inference with Gemini model."""
    model = client.GenerativeModel(model_id)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=1024,
            temperature=0.7
        )
    )
    return response.text

def run_inference(model_name: str, prompt_type: str, num_runs: int = 2):
    """Run inference on all items."""
    model_config = MODELS.get(model_name)
    if not model_config:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    provider = model_config["provider"]
    model_id = model_config["model_id"]

    # Initialize client based on provider
    if provider == "openai":
        client = get_openai_client()
        inference_fn = lambda prompt: run_openai_inference(client, model_id, prompt)
    elif provider == "google":
        client = get_gemini_client()
        inference_fn = lambda prompt: run_gemini_inference(client, model_id, prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Load data and prompt
    data = load_data()
    template = load_prompt_template(prompt_type)

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
                response_text = inference_fn(prompt)

                result = {
                    "id": item["id"],
                    "run": run_idx + 1,
                    "title": item["title"],
                    "probability": item["probability"],
                    "statement_1": item["statement_1"],
                    "statement_2": item["statement_2"],
                    "prompt_type": prompt_type,
                    "model": model_id,
                    "provider": provider,
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
                    "provider": provider,
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
    parser = argparse.ArgumentParser(description="Run closed-source model inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODELS.keys()),
        help="Model to use (gpt5 or gemini)"
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
