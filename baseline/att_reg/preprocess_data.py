import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DATASETS = {
    'adult':       {'csv': 'adult.csv',       'task': 'classification'},
    'cardio':      {'csv': 'cardio.csv',      'task': 'classification'},
    'creditcard':  {'csv': 'creditcard.csv',  'task': 'classification'},
    'diamonds':    {'csv': 'diamonds.csv',    'task': 'regression'},
    'elevator':    {'csv': 'elevator.csv',    'task': 'regression'},
    'housesale':   {'csv': 'housesale.csv',   'task': 'regression'},
    'crime':       {'csv': 'crime.csv',       'task': 'regression'},
    'meps':        {'csv': 'meps.csv',        'task': 'regression'},
}


def csv_to_libsvm(csv_path, output_dir, task_type):
    """Convert a CSV file to LibSVM format (train/val/test splits).

    Each output line: label 1:val 2:val 3:val ...
    Features are 1-indexed as per LibSVM convention.
    """
    df = pd.read_csv(csv_path)

    # Separate features and target (last column is target)
    target_col = df.columns[-1]
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    # Encode categorical columns
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(0).astype(float)

    # Standardize numerical features
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    # Encode target
    if task_type == 'classification':
        if y.dtype == 'object' or y.dtype == 'category':
            y = LabelEncoder().fit_transform(y.astype(str))
        y = y.astype(float)
    else:
        y = y.astype(float)

    n = len(df)
    indices = np.random.RandomState(42).permutation(n)
    test_n = int(0.2 * n)
    val_n = int(0.2 * (n - test_n))
    test_idx = indices[:test_n]
    val_idx = indices[test_n:test_n + val_n]
    train_idx = indices[test_n + val_n:]

    os.makedirs(output_dir, exist_ok=True)

    for split_name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        path = os.path.join(output_dir, f'{split_name}.libsvm')
        with open(path, 'w') as f:
            for i in idx:
                features = ' '.join(f'{j+1}:{X.iloc[i, j]:.6f}' for j in range(X.shape[1]))
                f.write(f'{y.iloc[i]:.6f} {features}\n')
        print(f'  {split_name}: {len(idx)} samples -> {path}')


def main():
    raw_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'raw_data'
    )
    output_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    print(f'Raw data dir: {raw_data_dir}')
    print(f'Output dir:   {output_base}')
    print()

    for name, info in DATASETS.items():
        csv_path = os.path.join(raw_data_dir, info['csv'])
        if not os.path.exists(csv_path):
            print(f'SKIP {name}: {csv_path} not found')
            continue
        output_dir = os.path.join(output_base, name)
        print(f'Converting {name} ({info["task"]})...')
        csv_to_libsvm(csv_path, output_dir, info['task'])

    print('\nDone. LibSVM files written to baseline/att_reg/data/')


if __name__ == '__main__':
    main()
