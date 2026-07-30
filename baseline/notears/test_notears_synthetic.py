import os
import sys
import argparse
import logging
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets.numerical_dag_dataset import load_numerical_dag_5vars_dataset, load_numerical_dag_10vars_dataset


def compute_causal_metrics(pred_adj, true_adj):
    """Compute TPR and FPR for causal graph evaluation."""
    pred_binary = (np.abs(pred_adj) > 0.3).astype(int)
    true_binary = true_adj.astype(int)
    np.fill_diagonal(pred_binary, 0)

    tp = np.sum((pred_binary == 1) & (true_binary == 1))
    fp = np.sum((pred_binary == 1) & (true_binary == 0))
    fn = np.sum((pred_binary == 0) & (true_binary == 1))
    tn = np.sum((pred_binary == 0) & (true_binary == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return tpr, fpr


def run_notears(X, lambda1=0.1, loss_type='l2', max_iter=100):
    """Run NOTEARS on data matrix X (n_samples, d_features)."""
    try:
        from causallearn.search.ScoreBased.NOTEARS import notears_linear
        # causal-learn's NOTEARS returns an adjacency matrix
        W_est = notears_linear(X, lambda1=lambda1, loss_type=loss_type, max_iter=max_iter)
        return W_est
    except ImportError:
        logging.error("causal-learn not installed. Run: pip install causal-learn")
        return None
    except Exception as e:
        logging.error(f"NOTEARS failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='NOTEARS on Synthetic DAG Data')
    parser.add_argument('--dataset', type=str, default='numerical_5vars',
                        choices=['numerical_5vars', 'numerical_10vars'],
                        help='Synthetic dataset')
    parser.add_argument('--nonlinear', action='store_true',
                        help='Use nonlinear data (default: linear)')
    parser.add_argument('--lambda1', type=float, default=0.1,
                        help='L1 regularization strength')
    parser.add_argument('--max_iter', type=int, default=100,
                        help='Max iterations')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    np.random.seed(args.seed)

    # Load synthetic data
    if args.dataset == 'numerical_5vars':
        dataset_obj, v, num_classes_dict, true_adj, env_labels = load_numerical_dag_5vars_dataset()
    else:
        dataset_obj, v, num_classes_dict, true_adj, env_labels = load_numerical_dag_10vars_dataset()

    logger.info(f'Dataset: {args.dataset}')
    logger.info(f'True edges: {true_adj.sum()}, shape: {true_adj.shape}')

    # Extract data matrix X from dataset
    X_list, y_list = [], []
    for i in range(len(dataset_obj)):
        x, y = dataset_obj[i]
        X_list.append(x.numpy())
        y_list.append(y.numpy())
    X = np.array(X_list)

    # For nonlinear data: use full data without assuming linearity
    # (NOTEARS linear will still run but with degraded performance)
    if args.nonlinear:
        logger.info('Running NOTEARS on nonlinear data (expected lower TPR)')
    else:
        logger.info('Running NOTEARS on linear Gaussian data')

    # Run NOTEARS
    logger.info(f'Running NOTEARS (lambda1={args.lambda1}, max_iter={args.max_iter})...')
    W_est = run_notears(X, lambda1=args.lambda1, max_iter=args.max_iter)

    if W_est is None:
        logger.error('NOTEARS failed to run')
        return

    # Compute metrics
    tpr, fpr = compute_causal_metrics(W_est, true_adj)
    logger.info(f'TPR: {tpr:.4f}, FPR: {fpr:.4f}')

    # NOTEARS is pure causal discovery, no prediction
    logger.info('RMSE: -- (NOTEARS is causal discovery only, no prediction)')


if __name__ == '__main__':
    main()
