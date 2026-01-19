"""
Open-source model inference script for Conditional Probability Benchmark.
Models: Qwen/Qwen3-VL-30B-A3B-Thinking, meta-llama/Llama-3.1-8B-Instruct
Uses local transformers for inference.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables BEFORE importing HuggingFace libraries
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Set cache directories
CACHE_DIR = "/cluster/scratch/yongyu/cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_DIR}/datasets"

import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, AutoModel

# Get HuggingFace token
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_API_KEY")

# Model configurations
MODELS = {
    "qwen": {
        "model_id": "Qwen/Qwen3-VL-30B-A3B-Thinking",
        "type": "vision_language"
    },
    "llama": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "type": "causal_lm"
    }
}


class QwenVLInference:
    """Inference class for Qwen Vision-Language models."""

    def __init__(self, model_name: str, device_map: str = "auto"):
        self.model_name = model_name
        self.device_map = device_map
        self.model = None
        self.processor = None

    def load_model(self):
        """Load model and processor."""
        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor

        token_kwargs = {"token": HUGGINGFACE_TOKEN} if HUGGINGFACE_TOKEN else {}

        print(f"Loading Qwen VL model: {self.model_name}")
        print(f"Cache directory: {CACHE_DIR}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            **token_kwargs
        )

        # Load model with trust_remote_code for custom architectures
        self.model = AutoModel.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **token_kwargs
        )

        print(f"Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict:
        """Generate response for a prompt."""

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Format as chat messages (text-only)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True
        )
        inputs = inputs.to(self.model.device)

        # Generation parameters
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True if temperature > 0 else False,
            "top_p": kwargs.get("top_p", 0.9),
        }

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        # Decode response
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return {
            "response": response,
            "model": self.model_name,
            "tokens_generated": len(generated_ids_trimmed[0])
        }


class LlamaInference:
    """Inference class for Llama and standard causal LM models."""

    def __init__(self, model_name: str, device_map: str = "auto"):
        self.model_name = model_name
        self.device_map = device_map
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model and tokenizer."""
        token_kwargs = {"token": HUGGINGFACE_TOKEN} if HUGGINGFACE_TOKEN else {}

        print(f"Loading model: {self.model_name}")
        print(f"Cache directory: {CACHE_DIR}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            **token_kwargs
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            **token_kwargs
        )

        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Model loaded successfully on device: {self.model.device}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict:
        """Generate response for a prompt."""

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Format as chat messages
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            input_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False
            )
        else:
            input_text = prompt

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        # Generation parameters
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True if temperature > 0 else False,
            "top_p": kwargs.get("top_p", 0.9),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generate_kwargs)

        # Decode response (only the new tokens)
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        return {
            "response": response,
            "model": self.model_name,
            "tokens_generated": len(outputs[0]) - inputs["input_ids"].shape[-1]
        }


def get_inference_class(model_config: dict):
    """Get the appropriate inference class based on model type."""
    model_type = model_config.get("type", "causal_lm")
    model_id = model_config["model_id"]

    if model_type == "vision_language":
        return QwenVLInference(model_id)
    else:
        return LlamaInference(model_id)


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

    # Get model config
    model_config = MODELS.get(model_name)
    if not model_config:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    model_id = model_config["model_id"]

    # Initialize and load model
    inference = get_inference_class(model_config)
    inference.load_model()

    # Load data and prompt template
    data = load_data()
    template = load_prompt_template(prompt_type)

    results = []
    total_items = len(data) * num_runs

    print(f"\nRunning inference with {model_id}")
    print(f"Prompt type: {prompt_type}")
    print(f"Total generations: {total_items}")
    print("-" * 50)

    for run_idx in range(num_runs):
        for item_idx, item in enumerate(data):
            prompt = format_prompt(template, item, prompt_type)

            try:
                output = inference.generate(prompt, max_new_tokens=1024, temperature=0.7)
                response_text = output["response"]

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
                    "tokens_generated": output.get("tokens_generated"),
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
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="Device map for model loading (default: auto)"
    )

    args = parser.parse_args()

    results = run_inference(args.model, args.prompt_type, args.num_runs)
    save_results(results, args.model, args.prompt_type)

    print(f"\nTotal results: {len(results)}")


if __name__ == "__main__":
    main()
