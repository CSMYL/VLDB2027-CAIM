# How to Add a New Dataset to TabGNN

This document explains how to add a new dataset to the TabGNN training framework.

## Overview

Adding a new dataset requires:
1. Prepare the data file (CSV format)
2. Create `ds_info.json` (task type and feature info)
3. Create `db_info_fz.json` (graph database structure)
4. Register the dataset in `tabular_ds_info.json`
5. Test that the dataset runs correctly

---

## Detailed Steps

### 1. Prepare Data File

#### 1.1 Data Format Requirements

- **CSV format**: Last column must be the label column, named `TARGET`
- **Feature columns**: All columns except the last are features
- **Column names**:
  - If there are column names (header), first row is the header
  - If no header, defaults to `feature0, feature1, ..., featureN-1, TARGET`

#### 1.2 Copy Data File

```bash
cp /path/to/your/dataset.csv data/test_data/your_dataset.csv
```

### 2. Create `ds_info.json`

Create `your_dataset.ds_info.json` in `data/test_data/`.

#### 2.1 Binary Classification Example

```json
{
  "task": "binary classification",
  "columns": [
    {
      "name": "feature0",
      "type": "NUMERIC"
    },
    {
      "name": "feature1",
      "type": "CATEGORICAL",
      "cardinality": 5
    },
    ...
    {
      "name": "TARGET",
      "type": "CATEGORICAL",
      "cardinality": 2
    }
  ]
}
```

#### 2.2 Regression Example

```json
{
  "task": "regression",
  "columns": [
    {
      "name": "feature0",
      "type": "NUMERIC"
    },
    ...
    {
      "name": "TARGET",
      "type": "NUMERIC"
    }
  ]
}
```

#### 2.3 Feature Type Notes

- **NUMERIC**: Continuous numeric features (integer or float)
- **CATEGORICAL**: Categorical features
  - Must specify `cardinality` (number of categories)
  - e.g., `"cardinality": 5` means the feature has 5 distinct values

#### 2.4 Quick Generation Script

```python
import pandas as pd
import json

df = pd.read_csv('data/test_data/your_dataset.csv', header=None)
num_cols = len(df.columns)

task_type = "regression"  # or "binary classification"

columns = []
for i in range(num_cols - 1):
    unique_count = df.iloc[:, i].nunique()
    if unique_count < 100:
        columns.append({
            "name": f"feature{i}",
            "type": "CATEGORICAL",
            "cardinality": int(unique_count)
        })
    else:
        columns.append({
            "name": f"feature{i}",
            "type": "NUMERIC"
        })

if task_type == "regression":
    columns.append({
        "name": "TARGET",
        "type": "NUMERIC"
    })
else:
    target_unique = df.iloc[:, -1].nunique()
    columns.append({
        "name": "TARGET",
        "type": "CATEGORICAL",
        "cardinality": int(target_unique)
    })

ds_info = {
    "task": task_type,
    "columns": columns
}

with open('data/test_data/your_dataset.ds_info.json', 'w') as f:
    json.dump(ds_info, f, indent=2)
```

### 3. Create `db_info_fz.json`

Create `your_dataset.db_info_fz.json` in `data/your_dataset/`.

#### 3.1 Binary Classification Example

```json
{
 "task": {
  "type": "classification",
  "n_classes": 2,
  "n_train": 8000,
  "n_test": 2000,
  "train_class_counts": [6000, 2000]
 },
 "node_type_to_int": {
  "Main_table": 0
 },
 "edge_type_to_int": {
  "SELF": 0,
  "SIMILARITY_EDGE": 1
 },
 "node_types_and_features": {
  "Main_table": {
   "INDEX_ID": {"type": "SCALAR"},
   "feature0": {"type": "SCALAR"},
   "feature1": {"type": "CATEGORICAL", "cardinality": 5},
   "TARGET": {"type": "CATEGORICAL", "cardinality": 2}
  }
 },
 "label_feature": "Main_table.TARGET"
}
```

#### 3.2 Regression Example

```json
{
 "task": {
  "type": "regression",
  "n_classes": 1,
  "n_train": 8000,
  "n_test": 2000,
  "train_class_counts": []
 },
 "node_type_to_int": {"Main_table": 0},
 "edge_type_to_int": {"SELF": 0, "SIMILARITY_EDGE": 1},
 "node_types_and_features": {
  "Main_table": {
   "INDEX_ID": {"type": "SCALAR"},
   "feature0": {"type": "SCALAR"},
   "TARGET": {"type": "SCALAR"}
  }
 },
 "label_feature": "Main_table.TARGET"
}
```

#### 3.3 Auto-Generation Script

```python
import pandas as pd
import json
import numpy as np
import os

df = pd.read_csv('data/test_data/your_dataset.csv', header=None)
n_total = len(df)
train_split = 0.8
n_train = int(n_total * train_split)
n_test = n_total - n_train

with open('data/test_data/your_dataset.ds_info.json', 'r') as f:
    ds_info = json.load(f)

if ds_info['task'] == 'regression':
    train_class_counts = []
    task_type_db = "regression"
    n_classes = 1
else:
    train_targets = df.iloc[:n_train, -1]
    unique_targets = sorted(train_targets.unique())
    train_class_counts = [int((train_targets == t).sum()) for t in unique_targets]
    task_type_db = "classification"
    n_classes = len(unique_targets)

db_info = {
    "task": {
        "type": task_type_db,
        "n_classes": n_classes,
        "n_train": n_train,
        "n_test": n_test,
        "train_class_counts": train_class_counts
    },
    "node_type_to_int": {"Main_table": 0},
    "edge_type_to_int": {"SELF": 0, "SIMILARITY_EDGE": 1},
    "node_types_and_features": {
        "Main_table": {"INDEX_ID": {"type": "SCALAR"}}
    },
    "label_feature": "Main_table.TARGET"
}

for col in ds_info['columns']:
    if col['name'] != 'TARGET':
        if col['type'] == 'NUMERIC':
            db_info["node_types_and_features"]["Main_table"][col['name']] = {"type": "SCALAR"}
        else:
            db_info["node_types_and_features"]["Main_table"][col['name']] = {
                "type": "CATEGORICAL", "cardinality": col['cardinality']
            }

target_col = ds_info['columns'][-1]
if target_col['type'] == 'NUMERIC':
    db_info["node_types_and_features"]["Main_table"]["TARGET"] = {"type": "SCALAR"}
else:
    db_info["node_types_and_features"]["Main_table"]["TARGET"] = {
        "type": "CATEGORICAL", "cardinality": target_col['cardinality']
    }

os.makedirs('data/your_dataset', exist_ok=True)
with open('data/your_dataset/your_dataset.db_info_fz.json', 'w') as f:
    json.dump(db_info, f, indent=1)
```

### 4. Register in `tabular_ds_info.json`

Edit `data/tabular_ds_info.json`:

```json
{
  ...
  "your_dataset": {
    "processed": {
      "task": "binary classification",
      "local_path": "test_data/your_dataset.csv",
      "ds_info": "data/test_data/your_dataset.ds_info.json"
    }
  }
}
```

### 5. Handle CSV File Format

If the CSV has a header, add special handling in `TabularDataset.py`:

```python
elif dataset_name in ['elevator']:
    self.raw_data = pd.read_csv(raw_data_path, header=0, names=col_names)
elif dataset_name in ['housesale', 'creditcard']:
    self.raw_data = pd.read_csv(raw_data_path, header=0)
    if len(self.raw_data.columns) == len(col_names):
        self.raw_data.columns = col_names
```

### 6. Test the Dataset

#### 6.1 Create Small Test Dataset

```bash
head -1000 data/test_data/your_dataset.csv > data/test_data/your_dataset_test_small.csv
```

#### 6.2 Test Run

```bash
./run.sh --dataset your_dataset --epochs 1 --batch_size 32 --device cuda
```

---

## Complete Example: Adding `example`

### Step 1: Prepare Data

```bash
cp /path/to/example.csv data/test_data/example.csv
```

### Step 2: Create `example.ds_info.json`

```json
{
  "task": "binary classification",
  "columns": [
    {"name": "feature0", "type": "NUMERIC"},
    {"name": "feature1", "type": "CATEGORICAL", "cardinality": 3},
    {"name": "TARGET", "type": "CATEGORICAL", "cardinality": 2}
  ]
}
```

### Step 3: Create `example.db_info_fz.json`

```json
{
 "task": {"type": "classification", "n_classes": 2, "n_train": 8000, "n_test": 2000, "train_class_counts": [5000, 3000]},
 "node_type_to_int": {"Main_table": 0},
 "edge_type_to_int": {"SELF": 0, "SIMILARITY_EDGE": 1},
 "node_types_and_features": {
  "Main_table": {
   "INDEX_ID": {"type": "SCALAR"},
   "feature0": {"type": "SCALAR"},
   "feature1": {"type": "CATEGORICAL", "cardinality": 3},
   "TARGET": {"type": "CATEGORICAL", "cardinality": 2}
  }
 },
 "label_feature": "Main_table.TARGET"
}
```

### Step 4: Register

```json
{
  ...
  "example": {
    "processed": {
      "task": "binary classification",
      "local_path": "test_data/example.csv",
      "ds_info": "data/test_data/example.ds_info.json"
    }
  }
}
```

### Step 5: Test

```bash
./run.sh --dataset example --epochs 1 --batch_size 32 --device cuda
```

---

## Notes

1. **Data split**: Default 80/20 train/test split
2. **Connect keys**: If no categorical features, similarity-based connections (`__similarity__`) are used automatically
3. **Task types**:
   - Binary classification: AUC metric
   - Regression: RMSE metric
4. **Feature encoding**:
   - NUMERIC features use `ScalarRobustScalerEnc`
   - CATEGORICAL features use `ScalarQuantileOrdinalEnc`

---

## Reference: Existing Datasets

1. **cardio** - binary classification, all categorical
2. **creditcard** - binary classification, all continuous
3. **diamonds** - regression, all continuous
4. **elevator** - regression, all continuous (uses elevator_cleaned.csv)
5. **housesale** - regression, mostly categorical
6. **adult** - binary classification, all continuous

Reference these datasets' config files when creating new dataset configurations.
