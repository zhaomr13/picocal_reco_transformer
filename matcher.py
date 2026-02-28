"""
Hungarian Matcher for Cluster Assignment

Implements optimal assignment between predicted clusters and ground truth clusters
using the Hungarian algorithm (linear sum assignment).

Based on DETR's matching strategy, adapted for calorimeter reconstruction.
"""

import torch
from torch import nn
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include no-object. Because of this,
    in general, there are more predictions than targets. We do a 1-to-1 matching
    of the best predictions, while the others are un-matched (and thus treated as
    non-objects/clusters).

    The matching cost considers:
    - Classification cost: Whether a cluster exists
    - Energy cost: L1 distance between predicted and target energies
    - Position cost: L2 distance between predicted and target positions
    - Time cost: L1 distance between predicted and target times
    """

    def __init__(self, cost_existence=1.0, cost_energy=1.0, cost_position=1.0, cost_time=0.5):
        """
        Args:
            cost_existence: Weight for classification (cluster existence) cost
            cost_energy: Weight for energy matching cost
            cost_position: Weight for position matching cost
            cost_time: Weight for time matching cost
        """
        super().__init__()
        self.cost_existence = cost_existence
        self.cost_energy = cost_energy
        self.cost_position = cost_position
        self.cost_time = cost_time

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Perform the matching between predictions and targets.

        Args:
            outputs: Dict with:
                - pred_logits: [batch_size, num_queries, 1] existence logits
                - pred_energies: [batch_size, num_queries, 3] energies (E, E_F, E_B)
                - pred_positions: [batch_size, num_queries, 3] positions (x, y, z)
                - pred_times: [batch_size, num_queries, 3] times (t, t_F, t_B)

            targets: List of dicts (one per batch element), each with:
                - labels: [num_target_clusters] tensor of 1s (clusters exist)
                - energies: [num_target_clusters, 3] target energies
                - positions: [num_target_clusters, 3] target positions
                - times: [num_target_clusters, 3] target times

        Returns:
            List of tuples (index_i, index_j) for each batch element:
                - index_i: indices of selected predictions
                - index_j: indices of corresponding selected targets
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # Flatten to compute cost matrix in batch
        # [batch_size * num_queries, ...]
        out_existence = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [B*Q, 1]
        out_energies = outputs["pred_energies"].flatten(0, 1)  # [B*Q, 3]
        out_positions = outputs["pred_positions"].flatten(0, 1)  # [B*Q, 3]
        out_times = outputs["pred_times"].flatten(0, 1)  # [B*Q, 3]

        # Concatenate target labels and properties
        tgt_existence = torch.cat([v["labels"] for v in targets])  # [total_targets]
        tgt_energies = torch.cat([v["energies"] for v in targets])  # [total_targets, 3]
        tgt_positions = torch.cat([v["positions"] for v in targets])  # [total_targets, 3]
        tgt_times = torch.cat([v["times"] for v in targets])  # [total_targets, 3]

        # Compute classification cost (existence)
        # Cost is negative probability of being a cluster (we want high probability)
        cost_existence = -out_existence  # [B*Q, 1] or [B*Q]
        if cost_existence.dim() > 1:
            cost_existence = cost_existence.squeeze(-1)  # [B*Q]

        # Compute energy cost (L1 distance)
        # Normalize by typical energy scale (e.g., 1000 MeV = 1 GeV)
        energy_scale = 1000.0
        cost_energy = torch.cdist(out_energies, tgt_energies, p=1) / energy_scale  # [B*Q, total_targets]

        # Compute position cost (L2 distance)
        # Normalize by typical calorimeter scale (~4000 mm)
        position_scale = 4000.0
        cost_position = torch.cdist(out_positions, tgt_positions, p=2) / position_scale  # [B*Q, total_targets]

        # Compute time cost (L1 distance)
        # Normalize by typical timing resolution (~1 ns)
        time_scale = 1.0
        cost_time = torch.cdist(out_times, tgt_times, p=1) / time_scale  # [B*Q, total_targets]

        # Final cost matrix
        C = (
            self.cost_existence * cost_existence.unsqueeze(1) +
            self.cost_energy * cost_energy +
            self.cost_position * cost_position +
            self.cost_time * cost_time
        )  # [B*Q, total_targets]

        C = C.view(bs, num_queries, -1).cpu()  # [B, Q, total_targets]

        # Split by batch element sizes
        sizes = [len(v["labels"]) for v in targets]

        # Apply Hungarian algorithm for each batch element
        indices = []
        start_idx = 0
        for i, size in enumerate(sizes):
            if size == 0:
                # No targets for this batch element
                indices.append((torch.tensor([], dtype=torch.int64),
                               torch.tensor([], dtype=torch.int64)))
                continue

            # Get cost matrix for this batch element
            c = C[i, :, start_idx:start_idx + size]  # [num_queries, num_targets]

            # Solve assignment problem
            row_ind, col_ind = linear_sum_assignment(c.numpy())

            indices.append((
                torch.as_tensor(row_ind, dtype=torch.int64),
                torch.as_tensor(col_ind, dtype=torch.int64)
            ))

            start_idx += size

        return indices


def build_matcher(args=None, **kwargs):
    """
    Build the Hungarian matcher.

    Args:
        args: Namespace with matcher hyperparameters
        **kwargs: Override specific parameters

    Returns:
        HungarianMatcher instance
    """
    default_args = {
        'cost_existence': 1.0,
        'cost_energy': 1.0,
        'cost_position': 1.0,
        'cost_time': 0.5,
    }

    if args is not None:
        for key in default_args:
            if hasattr(args, key):
                default_args[key] = getattr(args, key)

    default_args.update(kwargs)

    return HungarianMatcher(**default_args)
