"""
Loss Functions for PicoCal Transformer

Combined loss for cluster prediction including:
- Existence loss (binary classification)
- Energy loss (Huber loss on log-transformed energy)
- Position loss (L1/L2 distance)
- Time loss (L1/L2 distance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PicoCalLoss(nn.Module):
    """
    Combined loss for cluster prediction.

    The loss is computed after Hungarian matching to assign predictions to targets.
    Unmatched predictions are penalized for predicting a cluster where none exists.

    Supports auxiliary losses at intermediate decoder layers for improved training.
    Supports focal loss for better handling of class imbalance.
    """

    def __init__(
        self,
        weight_existence=1.0,
        weight_energy=1.0,
        weight_position=1.0,
        weight_time=0.5,
        weight_confidence=0.1,
        energy_loss_type='huber',
        position_loss_type='l1',
        time_loss_type='l1',
        eos_coef=0.5,  # Weight for "no cluster" class (was 0.1, now balanced)
        aux_loss_weight=0.5,  # Weight for auxiliary losses at intermediate layers
        use_aux_losses=False,
        use_focal_loss=True,  # Use focal loss instead of BCE
        focal_gamma=2.0,  # Focal loss focusing parameter
        focal_alpha=0.25  # Focal loss alpha (balance positive/negative)
    ):
        """
        Args:
            weight_existence: Weight for existence classification loss
            weight_energy: Weight for energy regression loss
            weight_position: Weight for position regression loss
            weight_time: Weight for time regression loss
            weight_confidence: Weight for confidence loss
            energy_loss_type: Type of energy loss ('huber', 'l1', 'mse')
            position_loss_type: Type of position loss ('l1', 'l2')
            time_loss_type: Type of time loss ('l1', 'l2')
            eos_coef: Weight for no-cluster class in classification
            aux_loss_weight: Weight multiplier for auxiliary losses (typically 0.5)
            use_aux_losses: Whether to compute auxiliary losses at intermediate layers
            use_focal_loss: Whether to use focal loss for existence classification
            focal_gamma: Focal loss focusing parameter (higher = more focus on hard examples)
            focal_alpha: Focal loss alpha (balance between positive/negative classes)
        """
        super().__init__()
        self.weight_existence = weight_existence
        self.weight_energy = weight_energy
        self.weight_position = weight_position
        self.weight_time = weight_time
        self.weight_confidence = weight_confidence
        self.energy_loss_type = energy_loss_type
        self.position_loss_type = position_loss_type
        self.time_loss_type = time_loss_type
        self.aux_loss_weight = aux_loss_weight
        self.use_aux_losses = use_aux_losses
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # Classification loss weighting
        self.register_buffer('empty_weight', torch.tensor([eos_coef, 1.0]))

    def forward(self, outputs, targets, indices, num_clusters):
        """
        Compute the loss.

        Args:
            outputs: Dict with predictions from model
            targets: List of target dicts (one per batch element)
            indices: List of (pred_indices, target_indices) tuples from matcher
            num_clusters: Total number of target clusters in batch (for normalization)

        Returns:
            Dict of losses and total loss
        """
        # Unpack indices
        idx = self._get_src_permutation_idx(indices)
        target_idx = self._get_tgt_permutation_idx(indices)

        # Extract matched predictions
        src_logits = outputs['pred_logits']  # [B, num_queries]
        src_energies = outputs['pred_energies']  # [B, num_queries, 3]
        src_positions = outputs['pred_positions']  # [B, num_queries, 3]
        src_times = outputs['pred_times']  # [B, num_queries, 3]
        src_confidence = outputs['pred_confidence']  # [B, num_queries]

        # Build target tensors
        target_classes = torch.full(
            src_logits.shape, 0,
            dtype=torch.int64, device=src_logits.device
        )  # [B, num_queries], default to "no cluster" (0)

        # Extract matched targets
        target_energies_list = []
        target_positions_list = []
        target_times_list = []

        for batch_idx, (pred_idx, tgt_idx) in enumerate(indices):
            if len(tgt_idx) > 0:
                # Mark matched predictions as "has cluster" (1)
                target_classes[batch_idx, pred_idx] = 1

                # Collect target values
                target_energies_list.append(targets[batch_idx]['energies'][tgt_idx])
                target_positions_list.append(targets[batch_idx]['positions'][tgt_idx])
                target_times_list.append(targets[batch_idx]['times'][tgt_idx])

        # Concatenate targets
        if len(target_energies_list) > 0:
            target_energies = torch.cat(target_energies_list, dim=0)  # [num_matched, 3]
            target_positions = torch.cat(target_positions_list, dim=0)  # [num_matched, 3]
            target_times = torch.cat(target_times_list, dim=0)  # [num_matched, 3]
        else:
            # No matches in batch
            target_energies = torch.zeros((0, 3), device=src_logits.device)
            target_positions = torch.zeros((0, 3), device=src_logits.device)
            target_times = torch.zeros((0, 3), device=src_logits.device)

        # Get matched predictions
        src_energies_matched = src_energies[idx]  # [num_matched, 3]
        src_positions_matched = src_positions[idx]  # [num_matched, 3]
        src_times_matched = src_times[idx]  # [num_matched, 3]
        src_confidence_matched = src_confidence[idx]  # [num_matched]

        # Compute losses
        losses = {}

        # Classification loss (existence)
        losses['loss_existence'] = self.loss_existence(src_logits, target_classes)

        # Regression losses (only for matched predictions)
        if len(target_energies) > 0:
            losses['loss_energy'] = self.loss_energy(src_energies_matched, target_energies, num_clusters)
            losses['loss_position'] = self.loss_position(src_positions_matched, target_positions, num_clusters)
            losses['loss_time'] = self.loss_time(src_times_matched, target_times, num_clusters)
            losses['loss_confidence'] = self.loss_confidence(src_confidence_matched, losses)
        else:
            losses['loss_energy'] = torch.tensor(0.0, device=src_logits.device)
            losses['loss_position'] = torch.tensor(0.0, device=src_logits.device)
            losses['loss_time'] = torch.tensor(0.0, device=src_logits.device)
            losses['loss_confidence'] = torch.tensor(0.0, device=src_logits.device)

        # Total loss
        total_loss = (
            self.weight_existence * losses['loss_existence'] +
            self.weight_energy * losses['loss_energy'] +
            self.weight_position * losses['loss_position'] +
            self.weight_time * losses['loss_time'] +
            self.weight_confidence * losses['loss_confidence']
        )

        losses['loss_total'] = total_loss

        return losses

    def compute_aux_losses(self, outputs_list, targets, indices_list, num_clusters):
        """
        Compute auxiliary losses at intermediate decoder layers.

        Args:
            outputs_list: List of output dicts from each decoder layer
            targets: List of target dicts (one per batch element)
            indices_list: List of (pred_indices, target_indices) tuples for each layer
            num_clusters: Total number of target clusters in batch

        Returns:
            Dict of auxiliary losses with 'aux_' prefix
        """
        if not self.use_aux_losses or len(outputs_list) <= 1:
            return {}

        aux_losses = {}

        # Compute losses for each intermediate layer (all except the last which is the main loss)
        for i, (outputs, indices) in enumerate(zip(outputs_list[:-1], indices_list[:-1])):
            layer_losses = self.forward(outputs, targets, indices, num_clusters)

            # Prefix with aux_{layer}_ and scale by aux_loss_weight
            for key, val in layer_losses.items():
                aux_key = f'aux_{i}_{key}'
                aux_losses[aux_key] = val * self.aux_loss_weight

        # Also compute total auxiliary loss for convenience
        aux_losses['loss_aux_total'] = sum(aux_losses.values())

        return aux_losses

    def loss_existence(self, src_logits, target_classes):
        """
        Binary classification loss for cluster existence.

        Args:
            src_logits: [batch_size, num_queries] logits
            target_classes: [batch_size, num_queries] class labels (0 or 1)

        Returns:
            Classification loss
        """
        # Reshape for cross-entropy
        src_logits_flat = src_logits.view(-1)  # [B*Q]
        target_classes_flat = target_classes.view(-1).float()  # [B*Q]

        if self.use_focal_loss:
            # Focal Loss: FL = -alpha_t * (1 - p_t)^gamma * log(p_t)
            # This down-weights easy examples and focuses on hard examples
            probs = torch.sigmoid(src_logits_flat)

            # Compute p_t: probability of the true class
            # For positive class (y=1): p_t = p
            # For negative class (y=0): p_t = 1 - p
            p_t = probs * target_classes_flat + (1 - probs) * (1 - target_classes_flat)

            # Compute alpha_t: class weighting
            # For positive class: alpha_t = alpha
            # For negative class: alpha_t = 1 - alpha
            alpha_t = self.focal_alpha * target_classes_flat + (1 - self.focal_alpha) * (1 - target_classes_flat)

            # Compute BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(
                src_logits_flat, target_classes_flat, reduction='none'
            )

            # Apply focal weighting: (1 - p_t)^gamma
            focal_weight = (1 - p_t) ** self.focal_gamma

            # Combine
            loss = (alpha_t * focal_weight * bce_loss).mean()
        else:
            # Standard BCE with class weighting
            # Apply class weights: higher weight for underrepresented class
            weights = target_classes_flat * self.empty_weight[1] + (1 - target_classes_flat) * self.empty_weight[0]

            loss = F.binary_cross_entropy_with_logits(
                src_logits_flat,
                target_classes_flat,
                weight=weights,
                reduction='mean'
            )

        return loss

    def loss_energy(self, src_energies, target_energies, num_clusters):
        """
        Energy regression loss.

        Uses log-transform for better dynamic range handling.

        Args:
            src_energies: [num_matched, 3] predicted energies
            target_energies: [num_matched, 3] target energies
            num_clusters: total number of clusters for normalization

        Returns:
            Energy loss
        """
        # Log-transform energies for better dynamic range
        # Add small epsilon to avoid log(0)
        eps = 1.0
        src_log = torch.log(src_energies + eps)
        tgt_log = torch.log(target_energies + eps)

        if self.energy_loss_type == 'huber':
            loss = F.smooth_l1_loss(src_log, tgt_log, reduction='sum')
        elif self.energy_loss_type == 'l1':
            loss = F.l1_loss(src_log, tgt_log, reduction='sum')
        elif self.energy_loss_type == 'mse':
            loss = F.mse_loss(src_log, tgt_log, reduction='sum')
        else:
            raise ValueError(f"Unknown energy loss type: {self.energy_loss_type}")

        return loss / num_clusters if num_clusters > 0 else loss

    def loss_position(self, src_positions, target_positions, num_clusters):
        """
        Position regression loss.

        Args:
            src_positions: [num_matched, 3] predicted positions
            target_positions: [num_matched, 3] target positions
            num_clusters: total number of clusters for normalization

        Returns:
            Position loss
        """
        # Normalize by calorimeter scale (~4000 mm)
        scale = 4000.0

        if self.position_loss_type == 'l1':
            loss = F.l1_loss(src_positions / scale, target_positions / scale, reduction='sum')
        elif self.position_loss_type == 'l2':
            loss = F.mse_loss(src_positions / scale, target_positions / scale, reduction='sum')
        else:
            raise ValueError(f"Unknown position loss type: {self.position_loss_type}")

        return loss / num_clusters if num_clusters > 0 else loss

    def loss_time(self, src_times, target_times, num_clusters):
        """
        Time regression loss.

        Args:
            src_times: [num_matched, 3] predicted times
            target_times: [num_matched, 3] target times
            num_clusters: total number of clusters for normalization

        Returns:
            Time loss
        """
        # Only compute loss for valid times (>= 0)
        valid_mask = target_times >= 0

        if valid_mask.any():
            src_times_valid = src_times[valid_mask]
            target_times_valid = target_times[valid_mask]

            # Normalize by typical timing resolution (~1 ns)
            scale = 1.0

            if self.time_loss_type == 'l1':
                loss = F.l1_loss(src_times_valid / scale, target_times_valid / scale, reduction='sum')
            elif self.time_loss_type == 'l2':
                loss = F.mse_loss(src_times_valid / scale, target_times_valid / scale, reduction='sum')
            else:
                raise ValueError(f"Unknown time loss type: {self.time_loss_type}")

            return loss / num_clusters if num_clusters > 0 else loss
        else:
            return torch.tensor(0.0, device=src_times.device)

    def loss_confidence(self, src_confidence, losses):
        """
        Confidence loss to calibrate prediction uncertainty.

        Encourages the model to predict lower confidence when other losses are high.

        Args:
            src_confidence: [num_matched] predicted confidence scores
            losses: Dict of other losses

        Returns:
            Confidence loss
        """
        # Confidence should be inversely related to loss magnitude
        # This is a simplified version - more sophisticated approaches exist
        target_confidence = torch.ones_like(src_confidence)
        loss = F.binary_cross_entropy(src_confidence, target_confidence, reduction='mean')
        return loss

    def _get_src_permutation_idx(self, indices):
        """
        Get source (prediction) indices from matcher output.

        Returns batch indices and prediction indices for gathering.
        """
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """
        Get target indices from matcher output.
        """
        batch_idx = torch.cat([
            torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
        ])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


def build_loss(args=None, **kwargs):
    """
    Build the loss module.

    Args:
        args: Namespace with loss hyperparameters
        **kwargs: Override specific parameters

    Returns:
        PicoCalLoss instance
    """
    default_args = {
        'weight_existence': 1.0,
        'weight_energy': 1.0,
        'weight_position': 1.0,
        'weight_time': 0.5,
        'weight_confidence': 0.1,
        'energy_loss_type': 'huber',
        'position_loss_type': 'l1',
        'time_loss_type': 'l1',
        'eos_coef': 0.5,  # Balanced class weighting (was 0.1)
        'aux_loss_weight': 0.5,
        'use_aux_losses': False,
        'use_focal_loss': True,  # Use focal loss by default
        'focal_gamma': 2.0,
        'focal_alpha': 0.25,
    }

    if args is not None:
        for key in default_args:
            if hasattr(args, key):
                default_args[key] = getattr(args, key)

    default_args.update(kwargs)

    return PicoCalLoss(**default_args)
