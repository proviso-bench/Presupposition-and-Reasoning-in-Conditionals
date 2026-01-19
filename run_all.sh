#!/bin/bash
# Run all inference experiments for Conditional Probability Benchmark

echo "=========================================="
echo "Conditional Probability Benchmark Inference"
echo "=========================================="

# Set cache directories
export HF_HOME=/cluster/scratch/yongyu/cache
export TRANSFORMERS_CACHE=/cluster/scratch/yongyu/cache
export HF_DATASETS_CACHE=/cluster/scratch/yongyu/cache/datasets

echo "Cache directory: $HF_HOME"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Running experiments..."
echo ""

# Open-source models
echo "--- Open-source Models ---"

echo "[1/8] Qwen - with context"
python inference/opensource/inference_opensource.py --model qwen --prompt-type with_context

echo "[2/8] Qwen - without context"
python inference/opensource/inference_opensource.py --model qwen --prompt-type without_context

echo "[3/8] Llama - with context"
python inference/opensource/inference_opensource.py --model llama --prompt-type with_context

echo "[4/8] Llama - without context"
python inference/opensource/inference_opensource.py --model llama --prompt-type without_context

# Closed-source models
echo ""
echo "--- Closed-source Models ---"

echo "[5/8] GPT-5 - with context"
python inference/closedsource/inference_closedsource.py --model gpt5 --prompt-type with_context

echo "[6/8] GPT-5 - without context"
python inference/closedsource/inference_closedsource.py --model gpt5 --prompt-type without_context

echo "[7/8] Gemini - with context"
python inference/closedsource/inference_closedsource.py --model gemini --prompt-type with_context

echo "[8/8] Gemini - without context"
python inference/closedsource/inference_closedsource.py --model gemini --prompt-type without_context

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved in:"
echo "  - inference/opensource/outputs/"
echo "  - inference/closedsource/outputs/"
echo "=========================================="
