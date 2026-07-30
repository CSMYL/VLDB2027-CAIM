# LLM-CD Baseline

This directory implements the paper baseline **LLM-CD**:

> Causal Discovery through Synergizing Large Language Model and Data-Driven Reasoning

It aims to faithfully reproduce the original paper's pipeline of "data-driven causal discovery + LLM causal judgment + training predictors using only target parent nodes", while adapting to this project's CAIM data format and offline server environments.

Core pipeline:

1. Use the PC algorithm to discover an initial causal graph from tabular data.
2. Use LLM to judge uncertain conditional independence relations, edge orientations, and cycle handling.
3. After obtaining the final graph, extract only the direct parent nodes of the target variable.
4. Train an MLP using these parents and evaluate on the test set.

Since production servers may be offline, the pipeline is split into three stages: stages 1 and 3 can run offline; only stage 2 (LLM API calls) requires internet.

## 5-Minute Quick Start

Set up the environment from the project root. Mac local has been verified with `.venv`; new machines can install:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-baseline.txt
```

Run a demo without API:

```bash
baseline/llmcd/scripts/run_e2e.sh baseline/llmcd/configs/synthetic_demo.json
```

Output at:

```text
baseline/llmcd/runs/synthetic_demo/
```

You should see three files:

- `pc_graph.json`: initial PC graph
- `final_graph.json`: final graph after LLM or fallback rule correction
- `metrics.json`: test metrics after training MLP with target parent nodes only

## Production Dataset Configs

8 production configs are ready:

| Dataset | Config | Target |
| --- | --- | --- |
| Adult | `baseline/llmcd/configs/adult.json` | `income`, column 14 |
| Cardio | `baseline/llmcd/configs/cardio.json` | `cardio`, column 11 |
| CreditCard | `baseline/llmcd/configs/creditcard.json` | `Class`, column 30 |
| Diamonds | `baseline/llmcd/configs/diamonds.json` | `price`, column 9 |
| Elevator | `baseline/llmcd/configs/elevator.json` | `vibration`, column 7 |
| Housesale | `baseline/llmcd/configs/housesale.json` | `Price`, column 39 |
| Crime | `baseline/llmcd/configs/crime.json` | `ViolentCrimesPerPop`, column 122 |
| MEPS | `baseline/llmcd/configs/meps.json` | `healthcare_utilization`, column 138 |

These configs use the project's existing loaders rather than directly re-reading raw CSVs, maintaining consistency with CAIM main experiments in preprocessing, column order, and target positions.

Verified local shapes:

```text
adult       (48842, 15)   target 14 income
cardio      (70000, 12)   target 11 cardio
creditcard  (284806, 31)  target 30 Class
diamonds    (53940, 10)   target 9  price
elevator    (109563, 8)   target 7  vibration
housesale   (30138, 40)   target 39 Price
crime       (1994, 123)   target 122 ViolentCrimesPerPop
meps        (48982, 139)  target 138 healthcare_utilization
```

Production configs default to MLP training:

```json
"training": {
  "predictor": "mlp"
}
```

## One-Command End-to-End

Without LLM, using fallback rules only:

```bash
baseline/llmcd/scripts/run_e2e.sh baseline/llmcd/configs/adult.json
```

With LLM:

```bash
export LLMCD_API_KEY="your-api-key"
export LLMCD_BASE_URL="https://api.deepseek.com"
export LLMCD_MODEL="deepseek-v4-flash"
export USE_LLM=1

baseline/llmcd/scripts/run_e2e.sh baseline/llmcd/configs/adult.json
```

Note: never write real API keys into json, py, or sh files. Only pass via environment variables.

## Recommended Offline Server Workflow

If the server has no internet, run in three steps.

First, print the step-by-step commands:

```bash
baseline/llmcd/scripts/step_by_step_commands.sh baseline/llmcd/configs/adult.json
```

### Step 1: Run PC on offline server

```bash
.venv/bin/python baseline/llmcd/discover_pc_graph.py \
  --config baseline/llmcd/configs/adult.json \
  --run_dir baseline/llmcd/runs/adult_llmcd
```

Output:

```text
baseline/llmcd/runs/adult_llmcd/pc_graph.json
```

Copy these two files to an internet-connected machine:

- `baseline/llmcd/configs/adult.json`
- `baseline/llmcd/runs/adult_llmcd/pc_graph.json`

### Step 2: Call LLM on internet machine

```bash
export LLMCD_API_KEY="your-api-key"
export LLMCD_BASE_URL="https://api.deepseek.com"
export LLMCD_MODEL="deepseek-v4-flash"

.venv/bin/python baseline/llmcd/judge_llm_graph.py \
  --config baseline/llmcd/configs/adult.json \
  --pc_graph baseline/llmcd/runs/adult_llmcd/pc_graph.json \
  --run_dir baseline/llmcd/runs/adult_llmcd \
  --use_llm
```

Output:

```text
baseline/llmcd/runs/adult_llmcd/final_graph.json
```

Copy `final_graph.json` back to the offline server.

### Step 3: Train and test on offline server

```bash
.venv/bin/python baseline/llmcd/train_eval_from_graph.py \
  --config baseline/llmcd/configs/adult.json \
  --graph baseline/llmcd/runs/adult_llmcd/final_graph.json \
  --run_dir baseline/llmcd/runs/adult_llmcd
```

Output:

```text
baseline/llmcd/runs/adult_llmcd/metrics.json
```

## File Structure

```text
baseline/llmcd/
  configs/
    adult.json
    cardio.json
    creditcard.json
    diamonds.json
    elevator.json
    housesale.json
    synthetic_demo.json
    template.json
  scripts/
    run_e2e.sh
    step_by_step_commands.sh
  runs/
    .gitkeep
    <run_name>/
      pc_graph.json
      final_graph.json
      metrics.json
  common.py
  discover_pc_graph.py
  judge_llm_graph.py
  train_eval_from_graph.py
  .env.example
```

Project root also requires:

- `baseline/llm_cd_baseline.py`: shared data loader, PC helpers, parent node predictor evaluation.
- `baseline/llm_cd_prompts.py`: LLM-CD prompts and API call logic.
- `requirements-baseline.txt`: baseline dependencies.

## Stage Inputs and Outputs

### Stage 1: Data to PC Graph

```bash
.venv/bin/python baseline/llmcd/discover_pc_graph.py \
  --config <config.json> \
  --run_dir <run_dir>
```

Output `pc_graph.json`, key fields:

- `pc_directed_edges`: directed edges found by PC
- `pc_undirected_edges`: undirected edges retained by PC
- `uncertain_ci_pairs`: CI pairs near the significance threshold
- `feature_names`: variable names
- `feature_descriptions`: variable descriptions for LLM

### Stage 2: PC Graph to Final Graph

With LLM:

```bash
.venv/bin/python baseline/llmcd/judge_llm_graph.py \
  --config <config.json> \
  --pc_graph <run_dir>/pc_graph.json \
  --run_dir <run_dir> \
  --use_llm
```

Offline fallback:

```bash
.venv/bin/python baseline/llmcd/judge_llm_graph.py \
  --config <config.json> \
  --pc_graph <run_dir>/pc_graph.json \
  --run_dir <run_dir>
```

Output `final_graph.json`, key fields:

- `graph`: final adjacency matrix
- `parents`: direct parent indices of target node
- `parent_names`: direct parent names of target node
- `ci_decisions`: LLM decisions on CI pairs
- `edge_decisions`: LLM decisions on edge orientation, retention, deletion

### Stage 3: Final Graph to Train/Test

```bash
.venv/bin/python baseline/llmcd/train_eval_from_graph.py \
  --config <config.json> \
  --graph <run_dir>/final_graph.json \
  --run_dir <run_dir>
```

Output `metrics.json`, where `metrics.test` contains the final test results.

## Configuration

Copy the template:

```bash
cp baseline/llmcd/configs/template.json baseline/llmcd/configs/my_dataset.json
```

If using the project's existing CAIM loader, only modify:

```json
{
  "run_name": "adult_llmcd",
  "dataset": "adult"
}
```

Available dataset names:

- `adult`
- `cardio`
- `creditcard`
- `diamonds`
- `elevator`
- `housesale`
- `crime`
- `meps`
- `synthetic`

For custom CSV, fill in `data`:

```json
{
  "run_name": "my_table_llmcd",
  "dataset": "my_table",
  "data": {
    "table_path": "raw_data/my_table.csv",
    "header": "infer",
    "target_column": "label",
    "target_idx": null,
    "categorical_indices": [1, 3],
    "continuous_indices": [0, 2, 4],
    "standardize_continuous": true,
    "drop_columns": []
  }
}
```

Fields:
- `table_path`: CSV path. If empty, uses project existing loader.
- `header`: `"infer"` for header present; `null` for no header.
- `target_column`: target column name. Recommended when header exists.
- `target_idx`: target column index (0-based).
- `categorical_indices`: categorical variable column indices.
- `continuous_indices`: continuous variable column indices.
- `standardize_continuous`: whether to standardize continuous variables.
- `drop_columns`: column names to exclude.

LLM-CD relies on variable semantics; write `features` carefully:

```json
{
  "features": [
    {
      "index": 0,
      "name": "age",
      "description": "The person's age."
    }
  ]
}
```

`index` must match the data matrix column index.

## Dataset Semantic Notes

- `adult`: reads processed numeric matrices; config descriptions use Adult Census original field semantics.
- `cardio`: CAIM preprocessing discretized several continuous medical variables. Config descriptions use original medical meanings, but PC and training use processed values.
- `creditcard`: `V1` through `V28` are anonymous PCA features with no known business meaning. LLM advantages are naturally limited on such data.
- `diamonds`: `price` is moved to the last column as regression target.
- `elevator`: `vibration` is moved to the last column as regression target; `x1` through `x5` are anonymous or engineering sensor features.
- `housesale`: 6 city CSVs merged and shuffled with seed 42, `Price` is the last column regression target.
- `crime`: 122 numerical features from socio-economic and law enforcement data; target is `ViolentCrimesPerPop` (regression).
- `meps`: 4 continuous + 134 categorical features from healthcare survey data; target is healthcare utilization (regression).

## PC Parameters

```json
"pc": {
  "alpha": 0.05,
  "independence_test": "auto",
  "sample_size": 2000,
  "ci_threshold": 0.001,
  "max_uncertain_pairs": 100,
  "jitter_scale": 0.000001
}
```

Common adjustments:
- PC too slow: reduce `sample_size`.
- High-dim too slow: set `sample_size` to `500` for pipeline validation first.
- Singular matrix error: code auto-retries PC with tiny `jitter_scale`.
- Reduce LLM calls: lower `max_uncertain_pairs`.

`creditcard`, `housesale`, `crime`, and `meps` already have `sample_size` set to `500` due to higher dimensionality. Training still uses full data.

## Predictor Parameters

```json
"training": {
  "predictor": "mlp"
}
```

Options:
- `mlp`: MLPClassifier or MLPRegressor (default).
- `rf`: RandomForest, works for both classification and regression.
- `logistic`: LogisticRegression for classification.
- `linear`: LinearRegression for regression.

For experimental consistency, keep the same predictor setting across all datasets.

## Mac Local Verification

Locally verified:
- JSON syntax check for all 8 production configs
- Shape, target index, task type check for all 8 configs and loaders
- PC graph generation for all 8 datasets
- Fallback final graph generation for all 8 datasets
- MLP training and metrics output for all 8 datasets

Fallback parent node examples:

```text
adult       -> marital_status, sex, capital_gain, capital_loss
cardio      -> age_group, systolic_pressure_bin, cholesterol
creditcard  -> V11, V14, V25
diamonds    -> color, clarity
elevator    -> x4
housesale   -> Location, No_of_Bedrooms, Resale
crime       -> PctKids2Par, PctIlleg, NumStreet
meps        -> AGE, PCS42, MCS42
```

These are smoke-test results from the no-API fallback, not equivalent to formal LLM-CD results. Production experiments should add `--use_llm` in stage 2.

## FAQ

### API not working

Check:

```bash
echo "$LLMCD_BASE_URL"
echo "$LLMCD_MODEL"
test -n "$LLMCD_API_KEY" && echo "api key is set"
```

Do not print the key or write it to files.

### Server has no internet

Only run stages 1 and 3 on the offline server. Run stage 2 on an internet-connected machine and copy `final_graph.json` back.

### PC is slow

First reduce `pc.sample_size` in config. This only affects the causal graph discovery stage; final MLP training still uses full data.

### No column names or anonymous columns

Works, but LLM judgment will be weaker. `creditcard` PCA features are an example of this case.

## Differences from Original Paper

This implementation does not modify `causal-learn` internals. Instead, LLM intervention happens in an independent stage after PC graph construction. This makes it easier to transfer intermediate files between offline and online machines and simplifies debugging; the trade-off is that LLM intervention during the skeleton phase is not fully embedded in the PC search process.
