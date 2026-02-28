"""
Utility Functions for PicoCal Transformer

Helper functions for visualization, metrics, and data processing.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple


def compute_cluster_metrics(predictions: List[Dict], targets: List[Dict]) -> Dict:
    """
    Compute reconstruction metrics.

    Args:
        predictions: List of prediction dicts with keys 'energies', 'positions', 'times'
        targets: List of target dicts with same keys

    Returns:
        Dict of metrics
    """
    energy_errors = []
    position_errors = []
    time_errors = []

    for pred, tgt in zip(predictions, targets):
        # Match predictions to targets
        pred_e = pred['energies'][:, 0].cpu().numpy()
        tgt_e = tgt['energies'][:, 0].cpu().numpy()

        pred_pos = pred['positions'][:, :2].cpu().numpy()
        tgt_pos = tgt['positions'][:, :2].cpu().numpy()

        # Simple matching: find closest in position
        for i, (pe, pp) in enumerate(zip(pred_e, pred_pos)):
            if len(tgt_e) == 0:
                break

            # Find closest target
            distances = np.sqrt(np.sum((tgt_pos - pp)**2, axis=1))
            j = np.argmin(distances)

            if distances[j] < 100:  # Within 100 mm
                energy_errors.append(pe - tgt_e[j])
                position_errors.append(distances[j])

                # Time error
                if 'times' in pred and 'times' in tgt:
                    time_errors.append(
                        pred['times'][i, 0].item() - tgt['times'][j, 0].item()
                    )

    metrics = {
        'energy_bias': np.mean(energy_errors) if energy_errors else 0,
        'energy_resolution': np.std(energy_errors) if energy_errors else 0,
        'position_bias': np.mean(position_errors) if position_errors else 0,
        'position_resolution': np.std(position_errors) if position_errors else 0,
    }

    if time_errors:
        metrics['time_bias'] = np.mean(time_errors)
        metrics['time_resolution'] = np.std(time_errors)

    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, path):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, scheduler, path, device='cuda'):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint.get('metrics', {})


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_learning_rate(optimizer):
    """Get current learning rate from optimizer."""
    for param_group in optimizer.param_groups:
        return param_group['lr']


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def inverse_lr_schedule(optimizer, epoch, initial_lr, warmup_epochs, total_epochs):
    """Inverse learning rate schedule with warmup."""
    if epoch < warmup_epochs:
        lr = initial_lr * (epoch + 1) / warmup_epochs
    else:
        # Inverse decay
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = initial_lr * (1 - progress) / (1 + 0.1 * progress)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def log_transform(x, epsilon=1.0):
    """Log transform for energy/timing values."""
    return torch.log(x + epsilon)


def inverse_log_transform(y, epsilon=1.0):
    """Inverse log transform."""
    return torch.exp(y) - epsilon


# For compatibility with existing code
class TClusterAdapter:
    """
    Adapter to make transformer clusters compatible with existing TCluster code.
    """

    def __init__(self, transformer_cluster):
        self._cluster = transformer_cluster

    def getE(self):
        return self._cluster.getE()

    def getEF(self):
        return self._cluster.getEF()

    def getEB(self):
        return self._cluster.getEB()

    def getX(self):
        return self._cluster.getX()

    def getY(self):
        return self._cluster.getY()

    def getZ(self):
        return self._cluster.getZ()

    def getT(self):
        return self._cluster.getT()

    def getTF(self):
        return self._cluster.getTF()

    def getTB(self):
        return self._cluster.getTB()

    def getEt(self):
        return self._cluster.getEt()

    def getID(self):
        return self._cluster.getID()

    def getNph(self, section=2):
        """Approximate light yield from energy."""
        # This is a rough approximation
        return self._cluster.getE() / 0.2  # Assuming 0.2 MeV per photon

    def getCalibE(self):
        return self._cluster.getE()

    def getCalibX(self):
        return self._cluster.getX()

    def getCalibY(self):
        return self._cluster.getY()
