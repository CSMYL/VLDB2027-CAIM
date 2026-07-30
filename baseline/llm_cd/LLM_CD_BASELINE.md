# LLM-CD Baseline

This baseline adapts **Causal Discovery through Synergizing Large Language Model
and Data-Driven Reasoning** to the CAIM tabular data pipeline.

## What Is Implemented

The code follows the original LLM-CD flow as closely as possible without
patching files inside `causal-learn`:

1. Load CAIM-format tabular data via `baseline/baseline_datasets.py`.
2. Restore the full table by putting the target value back into `x[target_idx]`.
3. Run PC from `causal-learn` as the data-driven causal discovery step.
4. Optionally ask an LLM to judge near-threshold conditional-independence cases.
5. Optionally ask an LLM to refine directed edges with `KEEP / FLIP / REMOVE`.
6. Optionally ask an LLM to orient undirected edges.
7. Break cycles to obtain a DAG.
8. Take direct parents of the target node.
9. Train a parent-only predictor and report test metrics.

The original paper injects LLM calls directly inside PC skeleton discovery. This
implementation approximates that step by checking near-threshold marginal CI
cases after PC, because it avoids modifying the installed `causal-learn`
package. Edge refinement/orientation follows the paper's prompt style directly.

## Environment

Install the baseline dependencies:

```bash
.venv/bin/python -m pip install -r requirements-baseline.txt
```

Use a project-local Matplotlib cache on this Mac:

```bash
mkdir -p .cache/matplotlib
```

## Dry Run Without LLM

```bash
MPLCONFIGDIR=$PWD/.cache/matplotlib \
.venv/bin/python baseline/test_llm_cd_baseline.py \
  --dataset synthetic \
  --sample_size 200 \
  --predictor rf \
  --output .cache/llm_cd/synthetic_dry_run.json
```

## Run With DeepSeek

Do not hard-code API keys. Set them only in the shell:

```bash
export LLMCD_API_KEY="..."
export LLMCD_BASE_URL="https://api.deepseek.com"
export LLMCD_MODEL="deepseek-v4-flash"

MPLCONFIGDIR=$PWD/.cache/matplotlib \
.venv/bin/python baseline/test_llm_cd_baseline.py \
  --dataset synthetic \
  --sample_size 200 \
  --use_llm \
  --max_llm_pairs 10 \
  --predictor rf \
  --output .cache/llm_cd/synthetic_llm.json
```

LLM responses are cached in `.cache/llm_cd/cache.json` by default.

## Data Notes

The CAIM paper evaluates on eight real-world datasets: `adult`,
`cardio`, `creditcard`, `diamonds`, `elevator`, `housesale`, `crime`, and `meps`. The local
workspace has newer dataset loaders than the published CAIM repository, so this
baseline follows the local loaders as the source of truth.

Feature descriptions are available from headers when possible. Headerless data
uses standard public dataset names when known, otherwise generic names. This is
expected to make LLM-CD less informative on datasets whose column semantics are
hidden or anonymized, such as PCA features in CreditCard.
