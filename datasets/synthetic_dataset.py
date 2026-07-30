import pandas as pd
from sklearn.preprocessing import StandardScaler
from datasets import ReconstructDataset
import numpy as np
import os


def _load_and_parse_synthetic_data(csv_path):
    """Load synthetic CSV and separate continuous vs categorical columns.

    Continuous columns are float-typed; categorical columns are integer-typed.
    Returns (df, cont_cols, cat_cols).
    """
    df = pd.read_csv(csv_path)
    cont_cols = []
    cat_cols = []
    for col in df.columns:
        if df[col].dtype in ('float64', 'float32'):
            cont_cols.append(col)
        else:
            cat_cols.append(col)
    return df, cont_cols, cat_cols


def load_synthetic_reconstruct_dataset(csv_path=None):
    """Load synthetic_data.csv and construct ReconstructDataset and v vector.

    Returns:
        dataset: ReconstructDataset instance
        v: np.ndarray of length F, 0 for continuous features, 1 for categorical features
        num_classes_dict: dict mapping categorical feature indices to their cardinality
    """
    if csv_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        csv_path = os.path.join(project_root, 'raw_data', 'synthetic_data.csv')
    df, cont_cols, cat_cols = _load_and_parse_synthetic_data(csv_path)

    scaler = StandardScaler()
    df[cont_cols] = scaler.fit_transform(df[cont_cols])
    df[cat_cols] = df[cat_cols].astype(int)

    dataset = ReconstructDataset(df, cont_columns=cont_cols, cat_columns=cat_cols)

    v = []
    for col in df.columns:
        v.append(0 if col in cont_cols else 1)
    v = np.array(v, dtype=np.int64)

    num_classes_dict = {}
    for i, col in enumerate(df.columns):
        if col in cat_cols:
            num_classes = df[col].nunique()
            num_classes_dict[i] = num_classes

    return dataset, v, num_classes_dict
