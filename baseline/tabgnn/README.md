# TabGNN - GNN for Tabular Data Training

This project implements a GNN-based training framework for tabular data classification and regression tasks, using the Multiplex Graph design from the TabGNN paper.

## Overview

This project supports:
- Automatic conversion of CSV tabular data into Multiplex Graph structures
- Training with Graph Convolutional Networks (GCN)
- Classification tasks (AUC metrics) and regression tasks (RMSE metrics)
- Connect keys based on categorical features or KNN-based similarity

## Supported Datasets

| Dataset | Task Type | Metric | Feature Type | Size |
|--------|---------|---------|---------|-----------|
| adult | binary classification | AUC | all continuous | 48,842 |
| cardio | binary classification | AUC | all categorical | 70,000 |
| creditcard | binary classification | AUC | all continuous | 284,806 |
| diamonds | regression | RMSE | all continuous | 53,940 |
| elevator | regression | RMSE | all continuous | 109,563 |
| housesale | regression | RMSE | mostly categorical | 30,138 |

## Environment Setup

### 1. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate tabgnn
```

**Note**: The current environment uses `tabgnn_clean` (Python 3.8) with GPU support configured.

### 2. Install PyTorch CUDA (Linux GPU)

**Current config** (CUDA 12.1):
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
```

### 3. Verify Installation

```python
import torch
import dgl
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"DGL version: {dgl.__version__}")
```

## Data Format

### CSV File Format

- Last column must be the label column (named `TARGET`), other columns are features
- Features can be NUMERIC or CATEGORICAL
- CSV files may have a header row or no header
- Default 80/20 train/test split

### Dataset File Structure

Each dataset requires:

```
data/
├── test_data/
│   ├── your_dataset.csv
│   └── your_dataset.ds_info.json
├── your_dataset/
│   └── your_dataset.db_info_fz.json
└── tabular_ds_info.json
```

## Adding New Datasets

See [README_ADD_DATASET.md](README_ADD_DATASET.md) for detailed instructions.

**Quick steps**:
1. Prepare CSV data file (last column is TARGET)
2. Create `ds_info.json` (task type and feature definitions)
3. Create `db_info_fz.json` (graph database structure)
4. Register in `tabular_ds_info.json`
5. Test run

### ds_info.json Configuration

Each dataset needs a `.ds_info.json` config file in `data/test_data/`.

**Example: adult.ds_info.json**
```json
{
  "task": "binary classification",
  "columns": [
    {"name": "feature0", "type": "NUMERIC"},
    {"name": "feature1", "type": "NUMERIC"},
    {"name": "TARGET", "type": "CATEGORICAL", "cardinality": 2}
  ]
}
```

**Fields:**
- `task`: task type, supports `"binary classification"`, `"multiclass classification"`, `"regression"`
- `columns`: array of column definitions
  - `name`: column name (last must be `TARGET`)
  - `type`: `"NUMERIC"` or `"CATEGORICAL"`
  - `cardinality`: required for categorical columns

### Dataset Registration

Register in `data/tabular_ds_info.json`:

```json
{
  "dataset_name": {
    "processed": {
      "task": "binary classification",
      "local_path": "test_data/dataset.csv",
      "ds_info": "data/test_data/dataset.ds_info.json"
    }
  }
}
```

## Usage

### Basic Usage

Modify hyperparameters in `run.sh` (lines 18-26), then run:

```bash
./run.sh
```

Or override via command line:

```bash
./run.sh --dataset cardio --epochs 20 --batch_size 64 --device cuda
```

Direct python usage:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tabgnn_clean
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/targets/x86_64-linux/lib:/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"
python train.py --dataset adult --epochs 10 --batch_size 32 --device cuda
```

### Available Datasets

- `adult` - binary classification
- `cardio` - binary classification
- `creditcard` - binary classification
- `diamonds` - regression
- `elevator` - regression
- `housesale` - regression

### Training Parameters

- `--dataset`: dataset name (required)
- `--epochs`: number of epochs (default: 10)
- `--batch_size`: batch size (default: 32)
- `--hidden_dim`: hidden dimension (default: 64)
- `--n_layers`: number of GNN layers (default: 3)
- `--dropout`: dropout rate (default: 0.2)
- `--lr`: learning rate (default: 0.001)
- `--train_split`: train split ratio (default: 0.8)
- `--device`: device (default: cuda, or cpu)

### Full Parameter Reference

```bash
python train.py \
  --dataset adult \              # dataset name (must be registered in tabular_ds_info.json)
  --epochs 10 \                  # number of epochs (default 10)
  --batch_size 32 \              # batch size (default 32)
  --train_split 0.8 \            # train split ratio (default 0.8)
  --hidden_dim 64 \              # hidden dimension (default 64)
  --n_layers 3 \                 # number of GNN layers (default 3)
  --fcout_layers 64 32 \         # output MLP layer sizes (default [64, 32])
  --dropout 0.2 \                # dropout rate (default 0.2)
  --lr 0.001 \                   # learning rate (default 0.001)
  --connect_keys col1 col2 \     # connect keys (categorical feature names), None for auto-detect
  --max_neighbors 10 \           # max neighbors per connect key (default 10)
  --device cpu                   # device (default cpu, or cuda)
```

## Example Runs

### Example 1: Adult dataset (purely numeric features)

```bash
python train.py --dataset adult --epochs 10 --batch_size 32 --hidden_dim 64
```

This will:
- Auto-detect no categorical features, use KNN similarity as connect keys
- Train for 10 epochs
- Output training time, test time, and AUC per epoch

### Example 2: Specify connect keys

```bash
python train.py --dataset your_dataset --connect_keys category1 category2 --epochs 20
```

This will:
- Use specified categorical features as connect keys for multiplex graph construction
- Train for 20 epochs

## Output

Each epoch outputs:
- **Training info**: train loss, train accuracy (classification), training time
- **Test info**: test loss, test metric (AUC or RMSE), test time
- **Best marker**: `*` indicates best test metric so far

**Example output:**
```
Epoch 1/10: Train Loss: 0.4089, Train Acc: 0.8118 (246.70s) | Test Loss: 0.4165, Test AUC: 0.8868 (19.96s) *
Epoch 2/10: Train Loss: 0.3521, Train Acc: 0.8456 (245.12s) | Test Loss: 0.4012, Test AUC: 0.8923 (19.85s) *
...
```

## Evaluation Metrics

- **Classification**: AUC (Area Under ROC Curve)
- **Regression**: RMSE (Root Mean Squared Error)

## Project Structure

```
TabGNN/
├── train.py
├── environment.yml
├── data/
│   ├── CSVToGraphAdapter.py
│   ├── TabularDataset.py
│   ├── tabular_ds_info.json
│   └── test_data/
│       ├── adult.csv
│       └── adult.ds_info.json
├── models/
│   └── GNN/
│       ├── GCN.py
│       └── GNNModelBase.py
└── utils.py
```

## Notes

1. Ensure the last column in CSV is the `TARGET` label column
2. Large datasets building KNN graphs may use significant memory; adjust `--max_neighbors` accordingly
3. On Linux GPU, replace PyTorch with CUDA version
4. If data has categorical features, specify `--connect_keys`; purely numeric data auto-uses KNN similarity
5. New datasets must be registered in `data/tabular_ds_info.json`

## FAQ

**Q: How to run on Linux GPU?**
A: Install PyTorch CUDA version per "Environment Setup" section, then use `--device cuda`.

**Q: How to add a new dataset?**
A: See [README_ADD_DATASET.md](README_ADD_DATASET.md). Quick steps: 1) prepare CSV and ds_info.json; 2) create db_info_fz.json; 3) register in `data/tabular_ds_info.json`; 4) test run.

**Q: How to choose connect keys?**
A: Prefer categorical features as connect keys; if none, KNN similarity is used automatically.

## License

This project is based on the original TabGNN codebase.
