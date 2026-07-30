import os
import sys
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baseline.baseline_datasets import (
    load_adult_baseline_dataset,
    load_diamonds_baseline_dataset,
)


def run_notears_on_data(X, lambda1=0.1, max_iter=100):
    """Run NOTEARS on data matrix X and return binarized adjacency."""
    try:
        from causallearn.search.ScoreBased.NOTEARS import notears_linear
        W_est = notears_linear(X, lambda1=lambda1, loss_type='l2', max_iter=max_iter)
        # Binarize: threshold at 0.3
        adj_binary = (np.abs(W_est) > 0.3).astype(np.float32)
        np.fill_diagonal(adj_binary, 0)
        return adj_binary
    except ImportError:
        logging.error("causal-learn not installed. Run: pip install causal-learn")
        return None


def convert_dataset_to_numpy(dataset_obj):
    """Convert CAIM dataset to numpy matrix."""
    X_list = []
    for i in range(len(dataset_obj)):
        x, _ = dataset_obj[i]
        X_list.append(x.numpy())
    return np.array(X_list)


def create_fixed_mask_model(model_cls, v, num_classes_dict, fixed_mask, **kwargs):
    """Create a CAIM model variant that uses a fixed causal mask instead of learning one."""
    # Use the wo_pred_weight model as base (simpler predictor), or main model
    # We create a custom mask_generator that always returns the fixed mask
    class FixedMaskGenerator(torch.nn.Module):
        def __init__(self, mask):
            super().__init__()
            self.register_buffer('fixed_mask', torch.tensor(mask, dtype=torch.float32))

        def forward(self, mask_logits, batch_size):
            return self.fixed_mask.unsqueeze(0).expand(batch_size, -1, -1)

        def get_causal_mask(self, mask_logits):
            return self.fixed_mask

        def update_parameters(self):
            pass

        def update_parameters_for_early_stopping(self):
            pass

        def get_parameters(self):
            return {}

        def initialize_mask_logits(self, device='cpu'):
            return torch.zeros(1, 1, device=device)

    mask_gen = FixedMaskGenerator(fixed_mask).to(kwargs.get('device', 'cpu'))
    model = model_cls(
        v=v,
        num_classes_dict=num_classes_dict,
        d_model=kwargs.get('d_model', 64),
        num_heads=kwargs.get('num_heads', 4),
        num_layers=kwargs.get('num_layers', 1),
        dropout=kwargs.get('dropout', 0.1),
        share_embedding=True,
        mask_generator=mask_gen,
        target_idx=kwargs.get('target_idx', len(v) - 1),
    )
    return model


def main():
    parser = argparse.ArgumentParser(description='End-to-end Learning Experiment')
    parser.add_argument('--dataset', type=str, default='adult',
                        choices=['adult', 'diamonds'])
    parser.add_argument('--lambda1', type=float, default=0.1,
                        help='NOTEARS L1 regularization')
    parser.add_argument('--gpu_id', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id >= 0 and torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # Step 1: Load data
    if args.dataset == 'adult':
        dataset_obj, v, num_classes_dict, target_idx = load_adult_baseline_dataset()
        metric_name = 'AUC'
    else:
        dataset_obj, v, num_classes_dict, target_idx = load_diamonds_baseline_dataset()
        metric_name = 'RMSE'

    if not isinstance(v, torch.Tensor):
        v = torch.tensor(v)
    v = v.to(device)

    logger.info(f'Dataset: {args.dataset}, size: {len(dataset_obj)}, features: {len(v)}')

    # Step 2: Run NOTEARS to get causal graph
    X = convert_dataset_to_numpy(dataset_obj)
    logger.info(f'Running NOTEARS on {args.dataset} (data shape: {X.shape})...')
    fixed_mask = run_notears_on_data(X, lambda1=args.lambda1)

    if fixed_mask is None:
        logger.error('NOTEARS failed — cannot proceed with experiment')
        return

    n_edges = fixed_mask.sum()
    logger.info(f'NOTEARS discovered graph: {n_edges} edges (sparsity: {n_edges / (len(v) ** 2):.4f})')

    # Step 3: Run CAIM with fixed NOTEARS mask vs end-to-end CAIM
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from models.causal_attention_msk_model import CausalAttentionMskModel
    from models.mask_generators import SigmoidMaskGenerator
    from utils.train_msk_utils import train_msk_model

    # Split data
    total_size = len(dataset_obj)
    train_test_split_idx = int(0.8 * total_size)
    train_and_val = torch.utils.data.Subset(dataset_obj, range(0, train_test_split_idx))
    test_set = torch.utils.data.Subset(dataset_obj, range(train_test_split_idx, total_size))
    train_val_size = len(train_and_val)
    val_size = int(0.2 * train_val_size)
    train_size = train_val_size - val_size
    gen = torch.Generator().manual_seed(args.seed)
    train_set, val_set = torch.utils.data.random_split(
        train_and_val, [train_size, val_size], generator=gen
    )

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    logger.info(f'Train/Val/Test: {len(train_set)}/{len(val_set)}/{len(test_set)}')

    # ---- Two-stage: NOTEARS mask as fixed mask ----
    logger.info('\n=== Two-stage: NOTEARS mask + CAIM predictor ===')
    mask_gen_fixed = create_fixed_mask_model(
        CausalAttentionMskModel, v, num_classes_dict, fixed_mask,
        d_model=64, num_heads=4, num_layers=1, dropout=0.1,
        target_idx=target_idx, device=device
    ).to(device)

    # Actually, create_fixed_mask_model returns a full model. Let me just use the generator approach.
    class FixedMaskGen(torch.nn.Module):
        def __init__(self, mask):
            super().__init__()
            self.register_buffer('m', torch.tensor(mask, dtype=torch.float32))

        def forward(self, logits, bs):
            return self.m.unsqueeze(0).expand(bs, -1, -1)

        def get_causal_mask(self, logits):
            return self.m

        def update_parameters(self):
            pass

        def update_parameters_for_early_stopping(self):
            pass

        def get_parameters(self):
            return {}

        def initialize_mask_logits(self, device='cpu'):
            return torch.zeros(1, 1, device=device)

    mask_gen = FixedMaskGen(fixed_mask).to(device)
    model_notears = CausalAttentionMskModel(
        v=v, num_classes_dict=num_classes_dict, d_model=64,
        num_heads=4, num_layers=1, dropout=0.1, share_embedding=True,
        mask_generator=mask_gen, target_idx=target_idx,
    ).to(device)

    logger.info('Training CAIM with NOTEARS fixed mask...')
    model_notears = train_msk_model(
        model=model_notears, train_loader=train_loader, val_loader=val_loader,
        v=v, num_classes_dict=num_classes_dict, test_loader=test_loader,
        prediction_idx=target_idx, learning_rate=1e-4, weight_decay=1e-5,
        num_epochs=100, patience=50, alpha=1.0, beta=0.001,
        device=device, save_dir='checkpoints_notears_mask',
    )

    # ---- End-to-end: CAIM with learned mask ----
    logger.info('\n=== End-to-end: CAIM with learned mask ===')
    mask_gen_e2e = SigmoidMaskGenerator(
        num_features=len(v), initial_threshold=0.2,
        final_threshold=0.2, threshold_multiplier=1.1,
    ).to(device)
    model_e2e = CausalAttentionMskModel(
        v=v, num_classes_dict=num_classes_dict, d_model=64,
        num_heads=4, num_layers=1, dropout=0.1, share_embedding=True,
        mask_generator=mask_gen_e2e, target_idx=target_idx,
    ).to(device)

    logger.info('Training CAIM end-to-end...')
    model_e2e = train_msk_model(
        model=model_e2e, train_loader=train_loader, val_loader=val_loader,
        v=v, num_classes_dict=num_classes_dict, test_loader=test_loader,
        prediction_idx=target_idx, learning_rate=1e-4, weight_decay=1e-5,
        num_epochs=100, patience=50, alpha=1.0, beta=0.001,
        device=device, save_dir='checkpoints_e2e',
    )

    # ---- Final evaluation ----
    logger.info('\n=== Final Comparison ===')
    from utils.train_msk_utils import evaluate_test_set

    results_notears = evaluate_test_set(model_notears, test_loader, target_idx, device, v)
    results_e2e = evaluate_test_set(model_e2e, test_loader, target_idx, device, v)

    if metric_name == 'AUC':
        logger.info(f'NOTEARS (as causal mask): AUC = {results_notears["auc"]:.4f}')
        logger.info(f'CAIM (end-to-end):        AUC = {results_e2e["auc"]:.4f}')
    else:
        import math
        rmse_notears = math.sqrt(results_notears['mse'])
        rmse_e2e = math.sqrt(results_e2e['mse'])
        logger.info(f'NOTEARS (as causal mask): RMSE = {rmse_notears:.4f}')
        logger.info(f'CAIM (end-to-end):        RMSE = {rmse_e2e:.4f}')


if __name__ == '__main__':
    main()
