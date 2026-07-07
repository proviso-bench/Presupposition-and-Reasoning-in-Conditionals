# Conditional Probability Benchmark

Benchmark and evaluation code for assessing how well language models reason about the likelihood of a target statement given a conditional and optional context (e.g., “If Jack is a scuba diver, he will bring his wetsuit” → “Jack has a wetsuit”).

---

## Overview

The benchmark presents items from `problem_set.json`: each item has a conditional (statement_1), a target (statement_2), a gold probability band (high / mid / low), and optional background. Models are prompted to rate how likely the target is (Likert 1–7 or classification). Scripts support both API-based (GPT, Gemini) and local (Qwen, Llama) inference, plus LLM-as-judge evaluation over checklist criteria. Aggregation scripts produce summary tables and CSVs for analysis.

---

## Repository Structure

```
.
├── problem_set.json          # Benchmark items (id, title, probability, statement_1/2, background)
├── requirements.txt
├── run_all.sh                # Runs all 8 inference configs (4 models × 2 prompt types)
├── extract_likert_ratings.py # Parses model outputs → Likert tables (CSV)
├── .env                      # API keys (OPENAI_API_KEY, GOOGLE_API_KEY, HUGGINGFACE_API_KEY); not committed
├── prompt/                   # Templates for inference and judging
│   ├── prompt_with_context_likert.txt
│   ├── prompt_without_context_likert.txt
│   ├── prompt_with_context_classification.txt
│   ├── prompt_without_context_classification.txt
│   ├── llm_as_judge_with_context.txt
│   ├── llm_as_judge_without_context.txt
│   └── Checklist_gen/        # filteration, elaboration, diversification prompts
├── checklist/Filtered/       # Per-category checklist questions (accuracy, coherence, pragmatic, presupposition, context)
│   ├── with_context/
│   └── without_context/
└── inference/
    ├── closedsource/inference_closedsource.py  # GPT-5, Gemini (API)
    ├── opensource/inference_opensource.py      # Qwen, Llama (local)
    ├── llm_as_judge.py                         # Judge outputs with checklist
    ├── judge_outputs/                          # Judge JSON + summary_table.csv
    └── outputs/                                # Model outputs (per model, prompt_type)
        ├── closedsource/outputs/
        └── opensource/outputs/
```

---

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root and set.

---

## Data

- **problem_set.json**  
  One JSON array of items. Each item: `id`, `title`, `probability` (high|mid|low), `statement_1`, `statement_2`, `background`.

- **checklist/Filtered/**  
  Filtered checklist questions per condition (with/without context) and category (accuracy, coherence, pragmatic, presupposition; with_context also has context). Used by `llm_as_judge.py`.

---

## Running Inference

**All configs (open- and closed-source):**

```bash
bash run_all.sh
```

**Single runs:**

```bash
# Closed-source (GPT-5, Gemini)
python inference/closedsource/inference_closedsource.py --model gpt5 --prompt-type with_context
python inference/closedsource/inference_closedsource.py --model gemini --prompt-type without_context

# Open-source (Qwen, Llama)
python inference/opensource/inference_opensource.py --model qwen --prompt-type with_context
python inference/opensource/inference_opensource.py --model llama --prompt-type without_context
```

Outputs are written under `inference/closedsource/outputs/` and `inference/opensource/outputs/` as `{model}_{prompt_type}_likert.json` (or classification, depending on prompt).

---

## Evaluation

**Likert aggregation (for paper tables):**

```bash
python extract_likert_ratings.py
```

Reads all `*_likert.json` under `inference/.../outputs/`, extracts ratings, and writes CSVs under `inference/`, including `likert_final_aggregated_table.csv`.

**LLM-as-judge (checklist):**

```bash
python inference/llm_as_judge.py --source closedsource --model gpt5 --prompt-type with_context
```

Produces judge verdicts and summary under `inference/judge_outputs/`.


---

## Citation

```bibtex
@inproceedings{azin-etal-2026-presupposition,
    title = "Presupposition and Reasoning in Conditionals: A Theory-Based Study of Humans and {LLM}s",
    author = "Azin, Tara  and Yu, Yongan  and Singh, Raj  and Jouravlev, Olessia",
    editor = "Bonial, Claire  and Berzak, Yevgeni",
    booktitle = "Proceedings of the 30th Conference on Computational Natural Language Learning",
    year = "2026",
    address = "San Diego, California, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.conll-main.26/",
    doi = "10.18653/v1/2026.conll-main.26",
    pages = "452--470",
}
```
