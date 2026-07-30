#!/bin/bash

# CAIM: Comprehensive Experiment Runner
# This script runs all experiments for the CAIM model and baselines
# Make sure all dependencies are installed and data is prepared before running

# Exit on any error
set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================="
echo "CAIM Experiment Runner"
echo "Starting experiments..."
echo "========================================="
echo ""
# Note: For standard deviation calculation, we run each experiment 5 times with random seeds 42, 123, 456, 789, 1024


# ============================================
# 1. CAIM Model - Main Experiments
# ============================================
echo "========================================="
echo "Section 1: CAIM Model - Main Experiments"
echo "========================================="

cd tests

# CAIM hyperparameters from paper:
# - Hidden size: 32 (d_model)
# - Batch size: 128
# - Learning rate: 1e-3
# - Number of layers: 2
# - Number of heads: hidden_size / 8 = 32 / 8 = 4
# - Epochs: 20
# - Dropout: 0.1
# - Weight decay: 1e-5

datasets=("adult" "cardio" "creditcard" "diamonds" "elevator" "housesale")
for dataset in "${datasets[@]}"; do
    echo "Running CAIM on $dataset dataset..."
    python test_causal_attention_msk_model.py --dataset $dataset --d_model 32 --num_heads 4 --num_layers 2 --batch_size 128 --learning_rate 0.001 --num_epochs 20 --dropout 0.1 --weight_decay 1e-5 --gpu_id 0
    echo ""
done

cd ..

# ============================================
# 2. FT-Transformer Baseline (All Datasets)
# ============================================
echo "========================================="
echo "Section 2: FT-Transformer Baseline"
echo "========================================="

cd baseline

# FT-Transformer hyperparameters from paper:
# - Hidden size: 64 (d_token)
# - Batch size: 128
# - Learning rate: 1e-3
# - Number of layers: 2
# - Number of heads: hidden_size / 8 = 64 / 8 = 8
# - Epochs: 30

for dataset in "${datasets[@]}"; do
    echo "Running FT-Transformer on $dataset dataset..."
    python test_ft_transformer_baseline.py --dataset $dataset --d_token 64 --n_heads 8 --n_blocks 2 --batch_size 128 --lr 0.001 --epochs 30 --gpu_id 0
    echo ""
done

cd ..

echo ""

# ============================================
# 3. Tab-Transformer Baseline (All Datasets)
# ============================================
echo "========================================="
echo "Section 3: Tab-Transformer Baseline"
echo "========================================="

cd baseline

# Tab-Transformer hyperparameters from paper:
# - Hidden size: 32 (d_token)
# - Batch size: 128
# - Learning rate: 1e-3
# - Number of layers: 6
# - Number of heads: hidden_size / 8 = 32 / 8 = 4
# - Epochs: 30

for dataset in "${datasets[@]}"; do
    echo "Running Tab-Transformer on $dataset dataset..."
    python test_tabtransformer_baseline.py --dataset $dataset --d_token 32 --n_heads 4 --n_blocks 6 --batch_size 128 --lr 0.001 --epochs 30 --gpu_id 0
    echo ""
done

cd ..

echo ""

# ============================================
# 4. SAINT Baseline (All Datasets)
# ============================================
echo "========================================="
echo "Section 4: SAINT Baseline"
echo "========================================="

cd baseline/saint

# SAINT hyperparameters from paper:
# - Hidden size: 64 (embedding_size)
# - Batch size: 128
# - Learning rate: 1e-4
# - Number of layers: 4 (transformer_depth)
# - Number of heads: 4 (attention_heads)
# - Epochs: 50

for dataset in "${datasets[@]}"; do
    echo "Running SAINT on $dataset dataset..."
    python train_with_custom_data.py --dataset $dataset --embedding_size 64 --batchsize 128 --transformer_depth 4 --attention_heads 4 --lr 0.0001 --epochs 50
    echo ""
done

cd ../..
echo ""

# ============================================
# 5. CASTLE Baseline (All Datasets)
# ============================================
echo "========================================="
echo "Section 5: CASTLE Baseline"
echo "========================================="

cd baseline/castle

# CASTLE hyperparameters from paper:
# - Hidden size: 32
# - Batch size: 64
# - Learning rate: 1e-4
# - Number of MLP layers: 2
# - Alpha: 1
# - Beta: 5
# - Epochs: 50
# - n_folds: 5

classification_datasets=("adult" "cardio" "creditcard")
regression_datasets=("diamonds" "elevator" "housesale")

for dataset in "${classification_datasets[@]}"; do
    echo "Running CASTLE for classification on $dataset dataset..."
    python main_cf.py --csv ../../raw_data/${dataset}.csv --n_folds 5 --reg_lambda 1.0 --reg_beta 5.0 --extension $dataset
    echo ""
done

for dataset in "${regression_datasets[@]}"; do
    echo "Running CASTLE for regression on $dataset dataset..."
    python main.py --csv ../../raw_data/${dataset}.csv --n_folds 5 --reg_lambda 1.0 --reg_beta 5.0 --extension $dataset
    echo ""
done

cd ../..
echo ""

# ============================================
# 6. LogCause Baseline (All Datasets)
# ============================================
echo "========================================="
echo "Section 6: LogCause Baseline"
echo "========================================="

cd baseline

# LogCause hyperparameters from paper:
# - Hidden size: 32
# - Batch size: 64
# - Learning rate: 1e-5
# - Epochs: 200

for dataset in "${datasets[@]}"; do
    echo "Running LogCause on $dataset dataset..."
    python test_linear_baseline.py --dataset $dataset --batch_size 64 --lr 0.00001 --epochs 200
    echo ""
done

cd ..
echo ""

# ============================================
# 7. TabM Baseline (All Datasets)
# ============================================
echo "========================================="
echo "Section 7: TabM Baseline"
echo "========================================="

cd baseline/tabm

# TabM hyperparameters from paper:
# - Hidden size: 512 (d_block)
# - Batch size: 256
# - Learning rate: 2e-3
# - Number of MLP blocks: 3
# - Ensemble size k: 32
# - Epochs: 30

for dataset in "${datasets[@]}"; do
    echo "Running TabM on $dataset dataset..."
    python test_tabm_baseline.py --dataset $dataset --k 32 --n_blocks 3 --d_block 512 --batch_size 256 --lr 2e-3 --epochs 30 --gpu_id 0
    echo ""
done

cd ../..
echo ""

# ============================================
# 8. Orion-BiX Baseline (All Datasets)
# ============================================
echo "========================================="
echo "Section 8: Orion-BiX Baseline"
echo "========================================="

cd baseline/orion_bix

# Orion-BiX hyperparameters from paper:
# - n_estimators: 32 (classification) / n_bins: 10 (regression)
# - Batch size: 8
# - Learning rate: 1e-4
# - Epochs: 100

classification_datasets_bix=("adult" "cardio" "creditcard")
regression_datasets_bix=("diamonds" "elevator" "housesale")

for dataset in "${classification_datasets_bix[@]}"; do
    echo "Running Orion-BiX on $dataset dataset (classification)..."
    python test_orion_bix_baseline.py --dataset $dataset --n_estimators 32 --batch_size 8 --gpu_id 0
    echo ""
done

for dataset in "${regression_datasets_bix[@]}"; do
    echo "Running Orion-BiX on $dataset dataset (regression)..."
    python test_orion_bix_baseline.py --dataset $dataset --n_bins 10 --batch_size 8 --gpu_id 0
    echo ""
done

cd ../..
echo ""

# ============================================
# 9. ATT-Reg Baseline (All Datasets)
# ============================================
echo "========================================="
echo "Section 9: ATT-Reg Baseline"
echo "========================================="

cd baseline/att_reg

# First preprocess data
echo "Preprocessing data for ATT-Reg..."
python preprocess_data.py
echo ""

# ATT-Reg hyperparameters:
# - Embedding size: 10
# - Hidden size: 600
# - Number of layers: 2
# - Batch size: 4096
# - Learning rate: 3e-3
# - Epochs: 100

for dataset in "${datasets[@]}"; do
    echo "Running ATT-Reg on $dataset dataset..."
    python train_attreg.py --dataset $dataset --data_dir ./data/ --nfeat 14 --nfield 14 --nemb 10 --h 600 --nlayer 2 --batch_size 4096 --lr 0.003 --epoch 100
    echo ""
done

cd ../..
echo ""

# ============================================
# 10. TabGNN Baseline (All Datasets)
# ============================================
echo "========================================="
echo "Section 10: TabGNN Baseline"
echo "========================================="

cd baseline/tabgnn

# First preprocess data
echo "Preprocessing data for TabGNN..."
python preprocess_data.py
echo ""

# TabGNN hyperparameters from paper:
# - Hidden size: 64
# - Number of GNN layers: 3
# - Batch size: 32
# - Learning rate: 1e-3
# - Epochs: 10

for dataset in "${datasets[@]}"; do
    echo "Running TabGNN on $dataset dataset..."
    python train.py --dataset $dataset --hidden_dim 64 --n_layers 3 --batch_size 32 --lr 0.001 --epochs 10
    echo ""
done

cd ../..
echo ""

# ============================================
# 11. LLM-CD Baseline (Selected Datasets)
# ============================================
echo "========================================="
echo "Section 11: LLM-CD Baseline"
echo "========================================="

cd baseline/llm_cd

# LLM-CD baseline (without LLM by default):
# - Sample size: 200
# - Predictor: rf (random forest)
# Set LLMCD_API_KEY, LLMCD_BASE_URL, LLMCD_MODEL env vars and add --use_llm to enable LLM

llmcd_datasets=("adult" "cardio" "creditcard" "diamonds" "elevator" "housesale" "crime" "meps")
for dataset in "${llmcd_datasets[@]}"; do
    echo "Running LLM-CD on $dataset dataset..."
    python test_llm_cd_baseline.py --dataset $dataset --sample_size 200 --predictor rf
    echo ""
done

cd ../..
echo ""

# ============================================
# 12. AutoML and XGBoost Baselines
# ============================================
echo "========================================="
echo "Section 12: AutoML and XGBoost Baselines"
echo "========================================="

cd baseline

# AutoML
echo "Running AutoML for classification on all classification datasets..."
cd automl
classification_datasets_for_automl=("adult" "cardio" "creditcard")
for dataset in "${classification_datasets_for_automl[@]}"; do
    echo "Running AutoML classification on $dataset dataset..."
    python test_automl_classification.py --dataset $dataset --time_limit 480
    echo ""
done
cd ..

echo "Running AutoML for regression on all regression datasets..."
cd automl
regression_datasets_for_automl=("diamonds" "elevator" "housesale")
for dataset in "${regression_datasets_for_automl[@]}"; do
    echo "Running AutoML regression on $dataset dataset..."
    python test_automl_regression.py --dataset $dataset --time_limit 480
    echo ""
done
cd ..

# XGBoost
echo "Running XGBoost baseline..."
cd xgboost
for dataset in "adult" "cardio" "creditcard" "diamonds" "elevator" "housesale" "crime" "meps"; do
    echo "Running XGBoost on $dataset dataset..."
    python run_xgboost.py --dataset $dataset --n_folds 5 --random_state 42 --test_size 0.2
    echo ""
done
cd ..

cd ..

echo ""

# ============================================
# 13. Synthetic Data Experiments
# ============================================
echo "========================================="
echo "Section 13: Synthetic Data Experiments"
echo "========================================="

cd tests

# CAIM on synthetic 5-variable data
echo "Running CAIM on numerical_5vars dataset..."
python test_causal_graph_learning.py --model causal_attention --dataset numerical_5vars
echo ""

# CAIM on synthetic 10-variable data
echo "Running CAIM on numerical_10vars dataset..."
python test_causal_graph_learning.py --model causal_attention --dataset numerical_10vars
echo ""

# CASTLE on synthetic 5-variable data
echo "Running CASTLE on numerical_5vars dataset..."
python test_causal_graph_learning.py --model castle --dataset numerical_5vars --castle_reg_lambda 0.01 --castle_reg_beta 0.1 --castle_max_steps 50
echo ""

# CASTLE on synthetic 10-variable data
echo "Running CASTLE on numerical_10vars dataset..."
python test_causal_graph_learning.py --model castle --dataset numerical_10vars --castle_reg_lambda 0.01 --castle_reg_beta 0.1 --castle_max_steps 50
echo ""

cd ..

# NOTEARS on synthetic
echo "Running NOTEARS on synthetic datasets..."
cd baseline/notears

echo "NOTEARS on numerical_5vars (linear)..."
python test_notears_synthetic.py --dataset numerical_5vars
echo ""

echo "NOTEARS on numerical_10vars (linear)..."
python test_notears_synthetic.py --dataset numerical_10vars
echo ""

echo "NOTEARS on numerical_10vars (nonlinear)..."
python test_notears_synthetic.py --dataset numerical_10vars --nonlinear
echo ""

cd ../..

# LogCause on synthetic
echo "Running LogCause on synthetic datasets..."
cd baseline/notears

echo "LogCause on numerical_5vars (linear)..."
python test_logcause_synthetic.py --dataset numerical_5vars
echo ""

echo "LogCause on numerical_10vars (linear)..."
python test_logcause_synthetic.py --dataset numerical_10vars
echo ""

echo "LogCause on numerical_10vars (nonlinear)..."
python test_logcause_synthetic.py --dataset numerical_10vars --nonlinear
echo ""

cd ../..

echo ""

# ============================================
# 14. End-to-end Learning Experiment
# ============================================
echo "========================================="
echo "Section 14: End-to-end Learning Experiment"
echo "========================================="

cd baseline/notears

echo "NOTEARS mask vs CAIM end-to-end on adult..."
python test_end_to_end.py --dataset adult
echo ""

echo "NOTEARS mask vs CAIM end-to-end on diamonds..."
python test_end_to_end.py --dataset diamonds
echo ""

cd ../..

echo ""

# ============================================
# 15. Distribution Shift Experiments
# ============================================
echo "========================================="
echo "Section 15: Distribution Shift Experiments"
echo "========================================="

# First generate OOD splits
echo "Generating OOD splits for Diamonds dataset..."
cd raw_data
python generate_ood_splits.py
cd ..
echo ""

# Run shift experiments
cd tests

echo "Running CAIM on shift1 (feature-level: Color)..."
python test_causal_attention_msk_shift.py --dataset shift1
echo ""

echo "Running CAIM on shift2 (Simpson's Paradox)..."
python test_causal_attention_msk_shift.py --dataset shift2
echo ""

echo "Running CAIM on shift3..."
python test_causal_attention_msk_shift.py --dataset shift3
echo ""

echo "Running CAIM w/o Causal on shift2 (comparison)..."
python test_causal_attention_msk_shift.py --dataset shift2 --model_source models.causal_attention_msk_model_wo_mask_attention
echo ""

cd ..

echo ""

# ============================================
# 16. Design Choices Experiments
# ============================================
echo "========================================="
echo "Section 16: Design Choices Experiments"
echo "========================================="

cd tests

# Design choice experiments use adult and diamonds datasets with standard hyperparameters
design_params="--num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001"

design_datasets=("adult" "diamonds")
for dataset in "${design_datasets[@]}"; do
    # Add mask design
    echo "Running Add Mask design on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $design_params --model_source models.causal_attention_msk_model_add_mask_design
    echo ""

    # Softmax mask design
    echo "Running Softmax Mask design on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $design_params --model_source models.causal_attention_msk_model_inner_softmax_mask_design
    echo ""

    # Flattening design
    echo "Running Flattening design on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $design_params --model_source models.causal_attention_msk_model_parents_predictor_design
    echo ""

    # Mean Pooling design
    echo "Running Mean Pooling design on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $design_params --model_source models.causal_attention_msk_model_wo_pred_weight
    echo ""

    # Parallel Reconstruction design
    echo "Running Parallel Reconstruction design on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $design_params --model_source models.causal_attention_msk_model_parallel_recon_design
    echo ""
done

cd ..
echo ""

# ============================================
# 17. Ablation Study Experiments
# ============================================
echo "========================================="
echo "Section 17: Ablation Study Experiments"
echo "========================================="

cd tests

# Ablation experiments use adult and diamonds datasets with standard hyperparameters
ablation_params="--num_heads 4 --num_layers 1 --gpu_id 0 --patience 50 --d_model 32 --learning_rate 0.0001"

ablation_datasets=("adult" "diamonds")
for dataset in "${ablation_datasets[@]}"; do
    # w/o pred reg
    echo "Running w/o pred reg on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ablation_params --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_prediction_reg
    echo ""

    # w/o DAG loss
    echo "Running w/o DAG loss on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ablation_params --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_dag_loss
    echo ""

    # w/o sparse loss
    echo "Running w/o sparse loss on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ablation_params --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_sparse_loss
    echo ""

    # w/o reconstruction loss
    echo "Running w/o reconstruction loss on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ablation_params --model_source models.causal_attention_msk_model --train_source utils.train_msk_utils_wo_reconstruction_loss
    echo ""

    # w/o pred weight
    echo "Running w/o pred weight on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ablation_params --model_source models.causal_attention_msk_model_wo_pred_weight
    echo ""

    # w/o causal mask
    echo "Running w/o causal mask on $dataset..."
    python test_causal_attention_msk_model.py --prefix $dataset --dataset $dataset $ablation_params --model_source models.causal_attention_msk_model_wo_mask_attention
    echo ""
done

cd ..

# ============================================
# All Experiments Completed
# ============================================
echo ""
echo "========================================="
echo "All experiments completed successfully!"
echo "========================================="
echo ""
echo "Results should be available in their respective output directories."
echo "Check the logs for detailed performance metrics."
echo ""
