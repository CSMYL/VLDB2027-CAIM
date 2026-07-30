import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets.numerical_dag_dataset import load_numerical_dag_5vars_dataset, load_numerical_dag_10vars_dataset


class LinearModel(nn.Module):
    """Simple linear predictor — same architecture as LogCause."""
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def compute_causal_metrics(pred_adj, true_adj, threshold=0.1):
    """Compute TPR and FPR."""
    pred_binary = (np.abs(pred_adj) > threshold).astype(int)
    true_binary = true_adj.astype(int)
    np.fill_diagonal(pred_binary, 0)

    tp = np.sum((pred_binary == 1) & (true_binary == 1))
    fp = np.sum((pred_binary == 1) & (true_binary == 0))
    fn = np.sum((pred_binary == 0) & (true_binary == 1))
    tn = np.sum((pred_binary == 0) & (true_binary == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr, fpr


def extract_adj_from_weights(model, input_dim):
    """Extract adjacency matrix from linear model weights.
    Shape: (input_dim,) weights for each input feature → output.
    We wrap it into (input_dim+1, input_dim+1) matrix where the last row/col is the target.
    """
    weights = model.linear.weight.data.cpu().numpy().flatten()  # (input_dim,)
    # Build full adjacency: (F, F) where F = input_dim + 1 (includes target)
    F = input_dim + 1
    adj = np.zeros((F, F))
    # Column i → target (last row) has weight[i]
    adj[-1, :input_dim] = weights
    return adj


def main():
    parser = argparse.ArgumentParser(description='LogCause on Synthetic DAG Data')
    parser.add_argument('--dataset', type=str, default='numerical_5vars',
                        choices=['numerical_5vars', 'numerical_10vars'])
    parser.add_argument('--nonlinear', action='store_true',
                        help='Use nonlinear data')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_l1', type=float, default=5e-2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load synthetic data
    if args.dataset == 'numerical_5vars':
        dataset_obj, v, num_classes_dict, true_adj, env_labels = load_numerical_dag_5vars_dataset()
    else:
        dataset_obj, v, num_classes_dict, true_adj, env_labels = load_numerical_dag_10vars_dataset()

    logger.info(f'Dataset: {args.dataset}, true edges: {true_adj.sum()}')
    n_features = len(v)
    target_idx = n_features - 1
    input_dim = n_features - 1

    # Convert to numpy
    X_all, y_all = [], []
    for i in range(len(dataset_obj)):
        x, y = dataset_obj[i]
        X_all.append(x.numpy())
        y_all.append(y.numpy())
    X_all = np.array(X_all).astype(np.float32)
    y_all = np.array(y_all).astype(np.float32)

    # Split: 80/20 train/test
    n = len(X_all)
    idx = np.random.permutation(n)
    train_n = int(0.8 * n)
    train_idx, test_idx = idx[:train_n], idx[train_n:]

    # Remove target from input features
    X_train = np.concatenate([X_all[train_idx, :target_idx], X_all[train_idx, target_idx+1:]], axis=1)
    X_test = np.concatenate([X_all[test_idx, :target_idx], X_all[test_idx, target_idx+1:]], axis=1)
    y_train = y_all[train_idx, target_idx]
    y_test = y_all[test_idx, target_idx]

    # Convert to tensors
    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)
    X_test_t = torch.tensor(X_test, device=device)

    # Standardize targets
    y_mean, y_std = y_train_t.mean(), y_train_t.std()
    y_train_t = (y_train_t - y_mean) / (y_std + 1e-8)

    # Model
    model = LinearModel(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Train
    logger.info(f'Training LogCause for {args.epochs} epochs...')
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        # L1 regularization on weights (LogCause style)
        l1_loss = args.lambda_l1 * torch.sum(torch.abs(model.linear.weight))
        total_loss = loss + l1_loss
        total_loss.backward()
        optimizer.step()

    # Evaluate prediction (RMSE)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_t) * (y_std + 1e-8) + y_mean
        rmse = np.sqrt(np.mean((y_pred.cpu().numpy() - y_test) ** 2))

    # Evaluate causal discovery
    pred_adj = extract_adj_from_weights(model, input_dim)
    tpr, fpr = compute_causal_metrics(pred_adj, true_adj)

    logger.info(f'TPR: {tpr:.4f}, FPR: {fpr:.4f}, RMSE: {rmse:.4f}')


if __name__ == '__main__':
    main()
