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

def load_prompt_template(prompt_type: str, output_type: str) -> str:
    """Load prompt template from file."""
    prompt_dir = Path(__file__).parent.parent.parent / "prompt"
    
    # Construct filename based on prompt_type and output_type
    if prompt_type == "with_context":
        if output_type == "classification":
            prompt_file = prompt_dir / "prompt_with_context_classification.txt"
        else:  # likert
            prompt_file = prompt_dir / "prompt_with_context_likert.txt"
    else:  # without_context
        if output_type == "classification":
            prompt_file = prompt_dir / "prompt_without_context_classification.txt"
        else:  # likert
            prompt_file = prompt_dir / "prompt_without_context_likert.txt"

    with open(prompt_file, "r") as f:
        return f.read()

def load_data() -> list:
    """Load problem set data."""
    data_path = Path(__file__).parent.parent.parent / "problem_set.json"
    with open(data_path, "r") as f:
        return json.load(f)

def format_prompt(template: str, item: dict, prompt_type: str) -> str:
    """Format prompt with item data."""
    if prompt_type == "with_context":
        # Use the background field from the item
        background = item.get("background", f"You are evaluating a scenario about {item['title']}.")
        return template.format(
            background=background,
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
        # max_tokens=1024,
        # temperature=0.7
    )
    return response.choices[0].message.content

def run_gemini_inference(client, model_id: str, prompt: str) -> str:
    """Run inference with Gemini model."""
    model = client.GenerativeModel(model_id)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            # max_output_tokens=1024,
            # temperature=0.7
        )
    )
    return response.text

def run_inference(model_name: str, prompt_type: str, output_type: str):
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

    # Load data and prompt
    data = load_data()
    template = load_prompt_template(prompt_type, output_type)

    # Load existing results to resume if interrupted
    output_file = get_output_file(model_name, prompt_type, output_type)
    existing_results = load_existing_results(output_file)
    
    # Start with existing results converted to list (sorted by id)
    all_results = [existing_results[id] for id in sorted(existing_results.keys())] if existing_results else []
    results_dict = existing_results.copy()  # Keep dict for quick lookups
    
    total_items = len(data)

    print(f"Running inference with {model_id}")
    print(f"Prompt type: {prompt_type}")
    print(f"Output type: {output_type}")
    print(f"Total generations: {total_items}")
    if existing_results:
        print(f"Found {len(existing_results)} existing results - will update incrementally")
    print("-" * 50)

    for item_idx, item in enumerate(data):
        # Skip if already processed
        if item["id"] in results_dict:
            print(f"[{item_idx + 1}/{total_items}] ID: {item['id']} - Already processed, skipping")
            continue
            
        prompt = format_prompt(template, item, prompt_type)

        try:
            response_text = inference_fn(prompt)

            result = {
                "id": item["id"],
                "title": item["title"],
                "probability": item["probability"],
                "statement_1": item["statement_1"],
                "statement_2": item["statement_2"],
                "prompt_type": prompt_type,
                "output_type": output_type,
                "model": model_id,
                "response": response_text
            }
            # Add background if using with_context prompt type
            if prompt_type == "with_context" and "background" in item:
                result["background"] = item["background"]
            
            # Add to results
            results_dict[item["id"]] = result
            all_results = [results_dict[id] for id in sorted(results_dict.keys())]
            
            # Save incrementally after each item
            save_results_incremental(all_results, model_name, prompt_type, output_type)

            current = item_idx + 1
            print(f"[{current}/{total_items}] ID: {item['id']} - Saved")

        except Exception as e:
            print(f"Error on item {item['id']}: {e}")
            result = {
                "id": item["id"],
                "title": item["title"],
                "probability": item["probability"],
                "statement_1": item["statement_1"],
                "statement_2": item["statement_2"],
                "prompt_type": prompt_type,
                "output_type": output_type,
                "model": model_id,
                "response": None,
                "error": str(e)
            }
            # Add background if using with_context prompt type
            if prompt_type == "with_context" and "background" in item:
                result["background"] = item["background"]
            
            # Add to results even on error
            results_dict[item["id"]] = result
            all_results = [results_dict[id] for id in sorted(results_dict.keys())]
            
            # Save incrementally even on error
            save_results_incremental(all_results, model_name, prompt_type, output_type)

    return all_results

def get_output_file(model_name: str, prompt_type: str, output_type: str) -> Path:
    """Get output file path."""
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir / f"{model_name}_{prompt_type}_{output_type}.json"

def load_existing_results(output_file: Path) -> dict:
    """Load existing results from file, returning a dict keyed by id."""
    if output_file.exists():
        try:
            with open(output_file, "r") as f:
                existing = json.load(f)
                return {item["id"]: item for item in existing}
        except (json.JSONDecodeError, KeyError):
            return {}
    return {}

def save_results_incremental(all_results: list, model_name: str, prompt_type: str, output_type: str):
    """Save results to JSON file incrementally."""
    output_file = get_output_file(model_name, prompt_type, output_type)
    
    # Convert list to dict for easier updates
    results_dict = {item["id"]: item for item in all_results}
    
    # Load existing results and merge (in case file was modified externally)
    existing = load_existing_results(output_file)
    existing.update(results_dict)
    
    # Convert back to sorted list
    final_results = [existing[id] for id in sorted(existing.keys())]
    
    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    return output_file

def save_results(results: list, model_name: str, prompt_type: str, output_type: str):
    """Save results to JSON file."""
    output_file = get_output_file(model_name, prompt_type, output_type)
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

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
        help="Prompt type to use (with_context or without_context)"
    )
    parser.add_argument(
        "--output-type",
        type=str,
        required=True,
        choices=["classification", "likert"],
        help="Output type to use (classification or likert)"
    )

    args = parser.parse_args()

    results = run_inference(args.model, args.prompt_type, args.output_type)
    save_results(results, args.model, args.prompt_type, args.output_type)

    print(f"\nTotal results: {len(results)}")

if __name__ == "__main__":
    main()
