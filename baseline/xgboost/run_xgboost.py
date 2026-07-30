import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

# ---------------------------------------------------------------------------
# Dataset registry — maps dataset name to (task_type, csv_filename)
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    'adult':       ('classification', 'adult.csv'),
    'cardio':      ('classification', 'cardio.csv'),
    'creditcard':  ('classification', 'creditcard.csv'),
    'diamonds':    ('regression',     'diamonds.csv'),
    'elevator':    ('regression',     'elevator.csv'),
    'housesale':   ('regression',     'housesale.csv'),
    'crime':       ('regression',     'crime.csv'),
    'meps':        ('regression',     'meps.csv'),
}


def load_single_data(csv_path, scale_target=True):
    """Load a CSV file, auto-detect header, return X (features) and y (target).

    The last column is treated as the target; all others are features.
    Categorical columns (object dtype) are ordinal-encoded.
    Numerical columns are MinMax-scaled after filling NaN with mode.
    If scale_target is True, the target is Standard-scaled (for regression).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    # Try reading with header; if the first row looks like data (all numeric),
    # fall back to header=None
    df = pd.read_csv(csv_path, header=None)

    # If the first row contains any non-numeric string, treat it as a header
    first_row = df.iloc[0].astype(str).tolist()
    has_header = any(not s.replace('.', '').replace('-', '').replace('e', '').replace('E', '').isdigit()
                     for s in first_row if s.lower() != 'nan')
    if has_header:
        df = pd.read_csv(csv_path, header=0)
    else:
        df = pd.read_csv(csv_path, header=None)

    # Last column is the target
    target_col = df.columns[-1]
    df.dropna(axis=0, subset=[target_col], inplace=True)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Encode categorical features
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        encoder = OrdinalEncoder(dtype=np.float64, handle_unknown='use_encoded_value', unknown_value=-1)
        X[cat_cols] = encoder.fit_transform(X[cat_cols])

    # Fill NaN and scale numerical features
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        for col in num_cols:
            mode_val = X[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else 0
            X[col].fillna(fill_val, inplace=True)
        X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    # Standardize target for regression; keep as-is for classification
    if scale_target:
        scaler = StandardScaler()
        y = pd.Series(scaler.fit_transform(y.values.reshape(-1, 1)).flatten())

    print(f"# data: {len(X)}, # feat: {X.shape[1]}")
    return X, y


def log_config(log_name):
    exp_dir = f"xgboost_{log_name}_{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"
    exp_log_dir = os.path.join("logs", exp_dir)
    os.makedirs(exp_log_dir, exist_ok=True)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=[
        logging.FileHandler(os.path.join(exp_log_dir, "log.txt")),
        logging.StreamHandler()
    ])
    return exp_log_dir


def main():
    parser = argparse.ArgumentParser(description='XGBoost baseline for structured data prediction')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Dataset name (e.g., adult, diamonds, creditcard)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Validation split ratio within each fold (default: 0.2)')
    args = parser.parse_args()

    task_type, csv_name = DATASET_CONFIGS[args.dataset]

    # Resolve path to the centralized raw_data/ directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    csv_path = os.path.join(project_root, 'raw_data', csv_name)

    exp_log_dir = log_config(args.dataset)

    logging.info(f"Dataset: {args.dataset}  Task: {task_type}  CSV: {csv_path}")
    logging.info(f"n_folds={args.n_folds}  random_state={args.random_state}  test_size={args.test_size}")

    scale_target = (task_type == 'regression')
    X, y = load_single_data(csv_path, scale_target=scale_target)

    # Choose CV splitter and model
    if task_type == 'classification':
        skf = StratifiedKFold(n_splits=args.n_folds, random_state=args.random_state, shuffle=True)
    else:
        skf = KFold(n_splits=args.n_folds, random_state=args.random_state, shuffle=True)

    score_list = []
    metric_name = "AUC" if task_type == 'classification' else "RMSE"

    for fold_id, (trn_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        train_data = X.iloc[trn_idx]
        train_label = y.iloc[trn_idx]
        X_test = X.iloc[val_idx]
        y_test = y.iloc[val_idx]

        X_train, X_val, y_train, y_val = train_test_split(
            train_data, train_label, test_size=args.test_size,
            random_state=args.random_state, shuffle=True
        )

        if task_type == 'classification':
            model = XGBClassifier(random_state=args.random_state, eval_metric='logloss')
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            p_pred = model.predict_proba(X_test)
            # For binary classification, use probability of positive class
            if p_pred.shape[1] == 2:
                p_pred = p_pred[:, 1]
            else:
                # Multi-class: use predict_proba directly with multi_class='ovo'
                pass
            try:
                # Determine if binary or multi-class
                n_classes = len(np.unique(y))
                if n_classes > 2:
                    score = roc_auc_score(y_test, p_pred, multi_class='ovo')
                else:
                    score = roc_auc_score(y_test, p_pred)
            except Exception:
                score = roc_auc_score(y_test, model.predict(X_test))
        else:
            model = XGBRegressor(random_state=args.random_state)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            p_pred = model.predict(X_test)
            score = np.sqrt(mean_squared_error(y_test, p_pred))

        score_list.append(score)
        logging.info(f"Fold_{fold_id} {metric_name}===>{args.dataset}==> {score:.4f}")

    mean_score = np.mean(score_list)
    std_score = np.std(score_list)
    logging.info(f"Mean {args.n_folds}-fold {metric_name}===>{args.dataset}==> {mean_score:.4f} ± {std_score:.4f}")

    result_df = pd.DataFrame(
        {"score": score_list, "mean": mean_score, "std": std_score},
        index=[f"fold_{i}" for i in range(1, args.n_folds + 1)]
    )
    res_path = os.path.join(exp_log_dir, "results.csv")
    result_df.to_csv(res_path, index=True)
    logging.info(f"Results saved to {res_path}")


if __name__ == '__main__':
    main()
