# CAIM: Towards Efficient and Interpretable Structured Data Analytics via Causal-Aware Hierarchical Modeling

We propose **C**ausal-**A**ware H**i**erarchical **M**odeling (CAIM), a framework that bridges the gap between high-capacity neural network models and causal analysis for structured data analytics. CAIM consists of causal-attentive representation, causal dependency extraction, and hierarchy-aware prediction modules.

The causal-attentive representation module designs a learnable causal matrix to guide the column-wise representation learning. The causal dependency extraction module incorporates a self-supervised relation learning task, regularized by causal structural constraints (i.e., acyclicity and sparsity), to accurately extract causal relationships from complex inter-column interactions. Finally, each column’s representations are aggregated via the causal hierarchy-aware weights to perform prediction.

A lazy-update mechanism is further proposed to reduce the proportion of costly causal dependency extraction for improved efficiency without sacrificing overall performance.

## Framework

![Framework](./README.assets/over1.png)

## Project Structure

```
Causal_attention/
├── models/ #Model files
│   ├── causal_attention_msk_model.py #Main model
│   ├── causal_attention_msk_model_add_mask_design.py #Additive mask design
│   ├── causal_attention_msk_model_inner_softmax_mask_design.py #Softmax mask design
│   ├── causal_attention_msk_model_parallel_recon_design.py #Parallel reconstruction design
│   ├── causal_attention_msk_model_parents_predictor_design.py #Flattening design
│   ├── causal_attention_msk_model_wo_mask_attention.py #Without causal mask
│   ├── causal_attention_msk_model_wo_pred_weight.py #Without pred weight
│   ├── embedding_layers.py #Embedding layers
│   └── mask_generators.py #Causal Mask generators
├── datasets/ #pytorch dataset classes
│   ├── creditcard_dataset.py
│   ├── cardio_dataset.py
│   ├── adult_dataset.py
│   ├── diamonds_dataset.py
│   ├── elevator_dataset.py
│   ├── housesale_dataset.py
│   ├── crime_dataset.py
│   ├── meps_dataset.py
│   ├── tabular_datasets.py
│   ├── ood_shift_datasets.py
│   ├── numerical_dag_dataset.py #Numerical DAG dataset for synthetic experiments
│   └── synthetic_dataset.py
├── utils/ #Training and evaluation functions
│   ├── train_msk_utils.py
│   ├── train_msk_utils_wo_dag_loss.py
│   ├── train_msk_utils_wo_prediction_reg.py
│   ├── train_msk_utils_wo_reconstruction_loss.py
│   ├── train_msk_utils_wo_sparse_loss.py
│   └── generate_numerical_dag_data.py #Synthetic DAG data generator
├── baseline/ #Baseline test code
│   ├── baseline_datasets.py
│   ├── train_baseline_utils.py
│   ├── xgboost/ #XGBoost baseline
│   │   └── run_xgboost.py
│   ├── automl/ #AutoML baselines (AutoGluon and AutoSklearn)
│   │   ├── test_automl_classification.py
│   │   ├── test_automl_regression.py
│   │   ├── autog_cls.py
│   │   ├── autog_reg.py
│   │   ├── autos_cls.py
│   │   └── autos_reg.py
│   ├── tabm/ #TabM baseline
│   │   └── test_tabm_baseline.py
│   ├── test_ft_transformer_baseline.py #FT-Transformer
│   ├── test_tabtransformer_baseline.py #Tab-Transformer
│   ├── saint/
│   │   ├── data_adapter.py
│   │   └── train_with_custom_data.py #SAINT test script
│   ├── orion_bix/ #Orion-BiX baseline
│   │   └── test_orion_bix_baseline.py
│   ├── att_reg/ #ATT-Reg baseline
│   │   ├── train_attreg.py
│   │   ├── data_loader.py
│   │   ├── preprocess_data.py
│   │   ├── models/
│   │   └── utils/
│   ├── tabgnn/ #TabGNN baseline
│   │   ├── train.py
│   │   ├── preprocess_data.py
│   │   ├── models/
│   │   ├── experiments/
│   │   └── data/
│   ├── castle/
│   │   ├── CASTLE.py
│   │   ├── CASTLE_CF.py
│   │   ├── main.py #regression tasks
│   │   ├── main_cf.py #classification tasks
│   │   └── test_castle_causal_learning.py #causal graph learning evaluation
│   ├── test_linear_baseline.py #LogCause
│   ├── llm_cd/ #LLM-CD baseline
│   │   ├── test_llm_cd_baseline.py
│   │   ├── llm_cd_baseline.py
│   │   ├── llm_cd_prompts.py
│   │   ├── baseline_datasets.py
│   │   └── llmcd/
│   └── notears/ #NOTEARS baseline
│       ├── test_notears_synthetic.py
│       ├── test_logcause_synthetic.py
│       └── test_end_to_end.py
├── tests/ #CAIM model main function
│   ├── test_causal_attention_msk_model.py
│   ├── test_causal_graph_learning.py #Causal graph learning evaluation on synthetic data
│   └── test_causal_attention_msk_shift.py #Distribution shift experiments
├── raw_data/ #Processed datasets (see Datasets section below)
│   ├── creditcard.csv
│   ├── cardio.csv
│   ├── adult.csv
│   ├── diamonds.csv
│   ├── elevator.csv
│   ├── housesale.csv
│   ├── crime.csv
│   ├── meps.csv
│   ├── diamonds_ood_experiments/
│   ├── generate_ood_splits.py
│   ├── numerical_dag_data_5vars.csv #Synthetic DAG data with 5 variables
│   ├── numerical_dag_data_10vars.csv #Synthetic DAG data with 10 variables
│   ├── numerical_dag_adj_5vars.npy #True adjacency matrix for 5-variable DAG
│   └── numerical_dag_adj_10vars.npy #True adjacency matrix for 10-variable DAG
└── README.md
```

## Requirements

- pytorch=2.2.2
  - pytorch-cuda=12.1
  - python=3.9.19
  - tab-transformer-pytorch==0.4.2 (for FT-Transformer / Tab-Transformer baselines)
  - tensorflow=1.15.0 (for CASTLE baseline)
  - tabm + rtdl_num_embeddings (for TabM baseline)
  - orion-bix (for Orion-BiX baseline)
  - causal-learn (for LLM-CD and NOTEARS baselines)
  - openai (for LLM-CD baseline)
  - dgl (for TabGNN baseline)

## Datasets

All datasets are publicly available on Kaggle or the UCI Machine Learning. Data needs to be processed into CSV format, with some datasets requiring special processing. The specific download addresses and special processing for each dataset are as follows:

#### CreditCard Dataset

- **Source**: [creditcard.csv](https://www.kaggle.com/datasets/arockiaselciaa/creditcardcsv)
- **Task**: Binary classification (fraudulent transactions detection)
- **Preprocessing**:
  - None

#### Cardiovascular (Cardio) Dataset

- **Source**: [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
- **Task**: Binary classification (cardiovascular disease prediction)
- **Preprocessing**:
  - Four features: height, weight, ap_hi, ap_lo are discretized into categorical variables with intervals of 10

#### Adult Dataset

- **Source**: [Adult - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult)
- **Task**: Binary classification (income prediction)
- **Preprocessing**:
  - None

#### Diamonds Dataset

- **Source**: [Diamonds](https://www.kaggle.com/datasets/shivam2503/diamonds)
- **Task**: Regression (diamond price prediction)
- **Preprocessing**:
  - The price column is moved to the last column as the prediction target

#### Elevator Dataset

- **Source**: [Elevator Predictive Maintenance Dataset](https://www.kaggle.com/datasets/shivamb/elevator-predictive-maintenance-dataset)
- **Task**: Regression (elevator vibration prediction)
- **Preprocessing**:
  - The vibration column is moved to the last column as the prediction target

#### Housesale Dataset

- **Source**: [Housing Prices in Metropolitan Areas of India](https://www.kaggle.com/datasets/ruchi798/housing-prices-in-metropolitan-areas-of-india)
- **Task**: Regression (house price prediction)
- **Preprocessing**:
  - All 6 CSV files in the dataset are concatenated together, and the data is shuffled with random seed 42

#### Crime Dataset

- **Source**: [Crime and Communities - UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/211/communities+and+crime)
- **Task**: Regression (violent crime rate prediction)
- **Preprocessing**:
  - 119 continuous features (standardized) + 3 categorical features (one-hot encoded)

#### MEPS Dataset

- **Source**: [Medical Expenditure Panel Survey](https://meps.ahrq.gov/)
- **Task**: Regression (healthcare utilization prediction)
- **Preprocessing**:
  - 4 continuous features (AGE, PCS42, MCS42, K6SUM42) + 134 categorical features (one-hot encoded)

## Data Preparation

Before running experiments, please prepare the datasets as follows:

1. **Download and process real-world datasets**: According to the instructions in the Datasets section, download the 8 datasets (CreditCard, Cardio, Adult, Diamonds, Elevator, Housesale, Crime, MEPS) and process them into CSV format. Place the processed CSV files in the `raw_data/` directory.

2. **Generate synthetic datasets**: Run the synthetic data generator to create the numerical DAG datasets. Modify `num_vars` (5 or 10) and `mechanism` ("linear" or "nonlinear") in the `Config` class to switch between settings.

   - Linear 5-node: `num_vars = 5`, `mechanism = "linear"`
   - Linear 10-node: `num_vars = 10`, `mechanism = "linear"`
   - Nonlinear 10-node: `num_vars = 10`, `mechanism = "nonlinear"`

   ```bash
   python utils/generate_numerical_dag_data.py
   ```

   Each run outputs 6 files to `raw_data/` (e.g., for 5 vars):
   - `raw_data/numerical_dag_data_5vars.csv`
   - `raw_data/numerical_dag_adj_5vars.npy`
   - `raw_data/numerical_dag_weights_5vars.npy`
   - `raw_data/numerical_dag_functions_5vars.txt`
   - `raw_data/numerical_dag_env_params_5vars.txt`
   - `raw_data/numerical_dag_config_5vars.txt`


## Training models

* **our model:**
  * cd tests
  * python test_causal_attention_msk_model.py --dataset creditcard --num_heads 4 --num_layers 2 --gpu_id 0 --d_model 32 --learning_rate 0.001

* **baselines:**
  * **XGBoost**
    * cd baseline/xgboost
    * python run_xgboost.py --dataset Diamonds --n_folds 5 --random_state 42 --test_size 0.2
  * **AutoML**
    * cd baseline/automl
    * python test_automl_classification.py --dataset creditcard --time_limit 480
    * python test_automl_regression.py --dataset diamonds --time_limit 480
  * **TabM**
    * cd baseline/tabm
    * python test_tabm_baseline.py --dataset creditcard --lr 2e-3 --epochs 30
  * **FT-Transformer**
    * cd baseline
    * python test_ft_transformer_baseline.py --dataset creditcard --d_token 64 --n_heads 8 --n_blocks 2 --lr 0.001
  * **Tab-Transformer**
    * cd baseline
    * python test_tabtransformer_baseline.py --dataset creditcard --d_token 64 --n_heads 8 --n_blocks 6 --lr 0.001
  * **SAINT**
    * cd baseline/saint
    * python train_with_custom_data.py --dataset creditcard --embedding_size 64 --transformer_depth 4 --attention_heads 4 --lr 0.0001
  * **Orion-BiX**
    * cd baseline/orion_bix
    * python test_orion_bix_baseline.py --dataset creditcard --n_estimators 32
    * For regression datasets: python test_orion_bix_baseline.py --dataset diamonds --n_bins 10
  * **ATT-Reg**
    * cd baseline/att_reg
    * python preprocess_data.py
    * python train_attreg.py --dataset adult --data_dir ./data/ --nfeat 14 --nfield 14
  * **TabGNN**
    * cd baseline/tabgnn
    * python preprocess_data.py
    * python train.py --dataset adult --epochs 10 --batch_size 32
  * **CASTLE**
    * cd baseline/castle
    * python main_cf.py --csv ../../raw_data/creditcard.csv --n_folds 5 --reg_lambda 1.0 --reg_beta 5.0 --extension creditcard
    * python main.py --csv ../../raw_data/diamonds.csv --n_folds 5 --reg_lambda 1.0 --reg_beta 5.0 --extension diamonds
  * **LogCause**
    * cd baseline
    * python test_linear_baseline.py --dataset creditcard --lr 0.0001
  * **LLM-CD**
    * cd baseline/llm_cd
    * Without LLM: python test_llm_cd_baseline.py --dataset creditcard --sample_size 200 --predictor rf
    * With LLM: set LLMCD_API_KEY, LLMCD_BASE_URL, LLMCD_MODEL env vars, add --use_llm

## Running Experiments

To run all experiments at once, use the provided shell script:

```bash
bash run.sh
```


This script will sequentially run:

1. CAIM Model main experiments
2. Baseline experiments (XGBoost, AutoML, TabM, FT-Transformer, Tab-Transformer, SAINT, Orion-BiX, ATT-Reg, TabGNN, CASTLE, LogCause, LLM-CD)
3. Synthetic data experiments (CAIM, CASTLE, NOTEARS, LogCause)
4. End-to-end learning experiment (NOTEARS mask vs CAIM)
5. Distribution shift experiments (feature-level shift and Simpson's Paradox on Diamonds)
6. Design choices experiments
7. Ablation study experiments

Individual experiments can also be run using the detailed commands listed below.

## Synthetic Data Experiments

Experiments on synthetic DAG data to evaluate causal graph learning performance:

* **CAIM on synthetic**
  * python tests/test_causal_graph_learning.py --model causal_attention --dataset numerical_5vars
  * python tests/test_causal_graph_learning.py --model causal_attention --dataset numerical_10vars
* **CASTLE on synthetic**
  * python tests/test_causal_graph_learning.py --model castle --dataset numerical_5vars --castle_reg_lambda 0.01 --castle_reg_beta 0.1 --castle_max_steps 50
  * python tests/test_causal_graph_learning.py --model castle --dataset numerical_10vars --castle_reg_lambda 0.01 --castle_reg_beta 0.1 --castle_max_steps 50
* **NOTEARS on synthetic**
  * cd baseline/notears
  * python test_notears_synthetic.py --dataset numerical_5vars
  * python test_notears_synthetic.py --dataset numerical_10vars
  * For nonlinear data: python test_notears_synthetic.py --dataset numerical_10vars --nonlinear

* **LogCause on synthetic**
  * cd baseline/notears
  * python test_logcause_synthetic.py --dataset numerical_5vars
  * python test_logcause_synthetic.py --dataset numerical_10vars
  * For nonlinear data: python test_logcause_synthetic.py --dataset numerical_10vars --nonlinear

## End-to-end Learning Experiment

Compare two-stage (NOTEARS causal mask + CAIM predictor) vs end-to-end CAIM:

* **NOTEARS as mask vs CAIM**
  * cd baseline/notears
  * python test_end_to_end.py --dataset adult
  * python test_end_to_end.py --dataset diamonds

## Distribution Shift Experiments

OOD robustness evaluation on Diamonds dataset, following the paper with two representative settings: a feature-level shift and a spurious-correlation reversal (Simpson's Paradox):

* **Generate OOD splits**:
  ```bash
  cd raw_data
  python generate_ood_splits.py
  ```

* **Feature-level shift** (Color feature: D/E/F in train, G/H/I/J in test):
  ```bash
  cd tests
  python test_causal_attention_msk_shift.py --dataset shift1
  ```

* **Simpson's Paradox shift** (spurious correlation reversal between Carat and Clarity):
  ```bash
  python test_causal_attention_msk_shift.py --dataset shift2
  ```

* **Additional shift variant**:
  ```bash
  python test_causal_attention_msk_shift.py --dataset shift3
  ```

* **Compare CAIM vs CAIM w/o Causal**:
  
  * python tests/test_causal_attention_msk_shift.py --dataset shift2 --model_source models.causal_attention_msk_model_wo_mask_attention

## Design Choices

Experiments evaluating different design choices in CAIM, conducted on Adult and Diamonds datasets following the paper (substitute `--dataset` and `--prefix` with `adult` or `diamonds`):

* **Add mask** (additive masking vs multiplicative)
  * python tests/test_causal_attention_msk_model.py --prefix adult --dataset adult --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_add_mask_design
  * python tests/test_causal_attention_msk_model.py --prefix diamonds --dataset diamonds --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_add_mask_design
* **Softmax mask** (softmax-internal masking vs multiplicative)
  * python tests/test_causal_attention_msk_model.py --prefix adult --dataset adult --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_inner_softmax_mask_design
  * python tests/test_causal_attention_msk_model.py --prefix diamonds --dataset diamonds --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_inner_softmax_mask_design
* **Flattening** (concatenation vs hierarchy-aware aggregation)
  * python tests/test_causal_attention_msk_model.py --prefix adult --dataset adult --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_parents_predictor_design
  * python tests/test_causal_attention_msk_model.py --prefix diamonds --dataset diamonds --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_parents_predictor_design
* **Mean Pooling** (equal weights vs learnable prediction weights)
  * python tests/test_causal_attention_msk_model.py --prefix adult --dataset adult --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_wo_pred_weight
  * python tests/test_causal_attention_msk_model.py --prefix diamonds --dataset diamonds --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_wo_pred_weight
* **Parallel Reconstruction** (parallel vs MSK-based reconstruction)
  * python tests/test_causal_attention_msk_model.py --prefix adult --dataset adult --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_parallel_recon_design
  * python tests/test_causal_attention_msk_model.py --prefix diamonds --dataset diamonds --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_parallel_recon_design

## Ablation Study

Experiments removing individual components to evaluate their contributions, conducted on Adult and Diamonds datasets following the paper (substitute `--dataset` and `--prefix` with `adult` or `diamonds`):

* **w/o pred reg**
  * python tests/test_causal_attention_msk_model.py --prefix adult --dataset adult --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_prediction_reg
  * python tests/test_causal_attention_msk_model.py --prefix diamonds --dataset diamonds --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_prediction_reg
* **w/o DAG loss**
  * python tests/test_causal_attention_msk_model.py --prefix adult --dataset adult --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_dag_loss
  * python tests/test_causal_attention_msk_model.py --prefix diamonds --dataset diamonds --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_dag_loss
* **w/o sparse loss**
  * python tests/test_causal_attention_msk_model.py --prefix adult --dataset adult --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_sparse_loss
  * python tests/test_causal_attention_msk_model.py --prefix diamonds --dataset diamonds --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_sparse_loss
* **w/o reconstruction loss**
  * python tests/test_causal_attention_msk_model.py --prefix adult --dataset adult --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_reconstruction_loss
  * python tests/test_causal_attention_msk_model.py --prefix diamonds --dataset diamonds --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_reconstruction_loss
* **w/o pred weight**
  * python tests/test_causal_attention_msk_model.py --prefix adult --dataset adult --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_wo_pred_weight
  * python tests/test_causal_attention_msk_model.py --prefix diamonds --dataset diamonds --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_wo_pred_weight
* **w/o causal mask**
  * python tests/test_causal_attention_msk_model.py --prefix adult --dataset adult --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_wo_mask_attention
  * python tests/test_causal_attention_msk_model.py --prefix diamonds --dataset diamonds --num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001 --model_source models.causal_attention_msk_model_wo_mask_attention
