"""
Training Script for PicoCal Transformer

Training loop with:
- Learning rate warmup and cosine decay
- Validation during training
- Checkpointing best model
- Loss logging
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from picocal_reco_transformer.model import build_model
from picocal_reco_transformer.matcher import build_matcher
from picocal_reco_transformer.loss import build_loss
from picocal_reco_transformer.dataset import PicoCalDataset, SyntheticDataset, collate_fn


def get_args_parser():
    """Define command line arguments."""
    parser = argparse.ArgumentParser('PicoCal Transformer Training', add_help=False)

    # Model parameters
    parser.add_argument('--d_model', default=256, type=int, help='Transformer hidden dimension')
    parser.add_argument('--nhead', default=8, type=int, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', default=4, type=int, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', default=4, type=int, help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', default=1024, type=int, help='Feedforward dimension')
    parser.add_argument('--num_cluster_queries', default=20, type=int, help='Number of cluster queries')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')
    parser.add_argument('--position_encoding', default='sine', type=str, choices=['sine', 'learned'])

    # Matcher parameters
    parser.add_argument('--cost_existence', default=1.0, type=float, help='Cost weight for existence')
    parser.add_argument('--cost_energy', default=1.0, type=float, help='Cost weight for energy')
    parser.add_argument('--cost_position', default=1.0, type=float, help='Cost weight for position')
    parser.add_argument('--cost_time', default=0.5, type=float, help='Cost weight for time')

    # Loss parameters
    parser.add_argument('--weight_existence', default=1.0, type=float, help='Loss weight for existence')
    parser.add_argument('--weight_energy', default=1.0, type=float, help='Loss weight for energy')
    parser.add_argument('--weight_position', default=1.0, type=float, help='Loss weight for position')
    parser.add_argument('--weight_time', default=0.5, type=float, help='Loss weight for time')
    parser.add_argument('--eos_coef', default=0.5, type=float, help='Weight for no-cluster class (0.5=balanced, was 0.1)')
    parser.add_argument('--use_aux_losses', action='store_true', help='Use auxiliary losses at intermediate decoder layers')
    parser.add_argument('--aux_loss_weight', default=0.5, type=float, help='Weight for auxiliary losses')
    parser.add_argument('--use_focal_loss', action='store_true', help='Use focal loss for existence classification')
    parser.add_argument('--focal_gamma', default=2.0, type=float, help='Focal loss gamma parameter (higher = more focus on hard examples)')
    parser.add_argument('--focal_alpha', default=0.25, type=float, help='Focal loss alpha (balance pos/neg, 0.25 = down-weight easy negatives)')
    parser.add_argument('--use_cell_assignment', action='store_true', help='Use cell-to-cluster assignment loss for attention supervision')
    parser.add_argument('--cell_assignment_weight', default=0.1, type=float, help='Weight for cell assignment loss')
    parser.add_argument('--cell_assignment_distance_threshold', default=500.0, type=float, help='Max distance (mm) for assigning cell to cluster')
    parser.add_argument('--position_scale', default=4000.0, type=float, help='Scale for position loss normalization (mm)')

    # Training parameters
    parser.add_argument('--lr', default=5e-5, type=float, help='Learning rate (default: 5e-5 for fine convergence)')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='Warmup epochs')
    parser.add_argument('--clip_grad', default=1.0, type=float, help='Gradient clipping')

    # Data parameters
    parser.add_argument('--data_path', default='', type=str, help='Path to data files')
    parser.add_argument('--use_synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--max_events', default=None, type=int, help='Maximum events to load')
    parser.add_argument('--max_cells', default=500, type=int, help='Maximum cells per event')
    parser.add_argument('--num_workers', default=4, type=int, help='DataLoader workers')

    # Geometry
    parser.add_argument('--lumi_condition', default='Run5_2024_refined_spacal_pb', type=str,
                        help='Luminosity condition for geometry')
    parser.add_argument('--module_info', default='', type=str,
                        help='Path to ModuleInfo root file')

    # Output
    parser.add_argument('--output_dir', default='./output', type=str, help='Output directory')
    parser.add_argument('--save_freq', default=10, type=int, help='Save checkpoint every N epochs')
    parser.add_argument('--eval_freq', default=5, type=int, help='Evaluate every N epochs')

    # Resume
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')

    # Device
    parser.add_argument('--device', default='cuda', type=str, help='Device to use')

    return parser


def train_one_epoch(model, criterion, matcher, data_loader, optimizer, device, epoch, args, log_file=None):
    """Train for one epoch."""
    model.train()
    criterion.train()

    total_loss = 0
    loss_dict_total = {}

    start_time = time.time()

    for i, batch in enumerate(data_loader):
        # Move to device
        features = batch['features'].to(device)
        positions = batch['positions'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets']

        # Move targets to device
        for t in targets:
            for key in t:
                if isinstance(t[key], torch.Tensor):
                    t[key] = t[key].to(device)

        # Forward pass with auxiliary outputs and attention weights if enabled
        return_auxiliary = args.use_aux_losses
        return_attention = args.use_cell_assignment
        outputs = model(features, positions, mask, return_auxiliary=return_auxiliary, return_attention=return_attention)

        # Match predictions to targets (main outputs)
        indices = matcher(outputs, targets)

        # Compute main loss
        num_clusters = sum(len(t['labels']) for t in targets)
        num_clusters = torch.as_tensor([num_clusters], dtype=torch.float32, device=device)

        loss_dict = criterion(outputs, targets, indices, num_clusters)
        loss = loss_dict['loss_total']

        # Compute auxiliary losses if enabled
        if args.use_aux_losses and 'aux_outputs' in outputs:
            # Match and compute loss for each auxiliary output
            aux_indices_list = [matcher(aux_out, targets) for aux_out in outputs['aux_outputs']]
            outputs_list = outputs['aux_outputs'] + [outputs]  # All outputs including final
            indices_list = aux_indices_list + [indices]

            aux_losses = criterion.compute_aux_losses(outputs_list, targets, indices_list, num_clusters)

            # Add auxiliary losses to total loss
            if 'loss_aux_total' in aux_losses:
                loss = loss + aux_losses['loss_aux_total']

            # Merge auxiliary losses into loss_dict for logging
            loss_dict.update(aux_losses)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        for key, val in loss_dict.items():
            if key not in loss_dict_total:
                loss_dict_total[key] = 0
            loss_dict_total[key] += val.item()

        if (i + 1) % 10 == 0:
            print(f"  Batch {i+1}/{len(data_loader)}, Loss: {loss.item():.4f}")

    # Average losses
    avg_loss = total_loss / len(data_loader)
    for key in loss_dict_total:
        loss_dict_total[key] /= len(data_loader)

    epoch_time = time.time() - start_time

    print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
    print(f"  Loss breakdown: {', '.join([f'{k}: {v:.4f}' for k, v in loss_dict_total.items()])}")

    # Log to JSON
    if log_file is not None:
        log_entry = {
            'epoch': epoch,
            'phase': 'train',
            'loss_total': avg_loss,
            **{k: float(v) for k, v in loss_dict_total.items()},
            'time': epoch_time
        }
        log_file.write(json.dumps(log_entry) + '\n')
        log_file.flush()

    return avg_loss, loss_dict_total


@torch.no_grad()
def evaluate(model, criterion, matcher, data_loader, device, epoch=None, log_file=None, args=None):
    """Evaluate on validation set."""
    model.eval()
    criterion.eval()

    total_loss = 0
    loss_dict_total = {}

    for batch in data_loader:
        features = batch['features'].to(device)
        positions = batch['positions'].to(device)
        mask = batch['mask'].to(device)
        targets = batch['targets']

        for t in targets:
            for key in t:
                if isinstance(t[key], torch.Tensor):
                    t[key] = t[key].to(device)

        # Forward pass with auxiliary outputs and attention weights if enabled
        return_auxiliary = args.use_aux_losses if args else False
        return_attention = args.use_cell_assignment if args else False
        outputs = model(features, positions, mask, return_auxiliary=return_auxiliary, return_attention=return_attention)
        indices = matcher(outputs, targets)

        num_clusters = sum(len(t['labels']) for t in targets)
        num_clusters = torch.as_tensor([num_clusters], dtype=torch.float32, device=device)

        loss_dict = criterion(outputs, targets, indices, num_clusters)
        loss = loss_dict['loss_total']

        # Compute auxiliary losses if enabled (for monitoring)
        if return_auxiliary and 'aux_outputs' in outputs:
            aux_indices_list = [matcher(aux_out, targets) for aux_out in outputs['aux_outputs']]
            outputs_list = outputs['aux_outputs'] + [outputs]
            indices_list = aux_indices_list + [indices]

            aux_losses = criterion.compute_aux_losses(outputs_list, targets, indices_list, num_clusters)
            loss_dict.update(aux_losses)
            if 'loss_aux_total' in aux_losses:
                loss = loss + aux_losses['loss_aux_total']

        total_loss += loss.item()
        for key, val in loss_dict.items():
            if key not in loss_dict_total:
                loss_dict_total[key] = 0
            loss_dict_total[key] += val.item()

    avg_loss = total_loss / len(data_loader)
    for key in loss_dict_total:
        loss_dict_total[key] /= len(data_loader)

    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"  Loss breakdown: {', '.join([f'{k}: {v:.4f}' for k, v in loss_dict_total.items()])}")

    # Log to JSON
    if log_file is not None and epoch is not None:
        log_entry = {
            'epoch': epoch,
            'phase': 'val',
            'loss_total': avg_loss,
            **{k: float(v) for k, v in loss_dict_total.items()}
        }
        log_file.write(json.dumps(log_entry) + '\n')
        log_file.flush()

    return avg_loss, loss_dict_total


def main(args):
    """Main training function."""
    print("=" * 80)
    print("PicoCal Transformer Training")
    print("=" * 80)
    print(f"Arguments: {args}")
    print()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Load geometry if using real data
    geometry = None
    if not args.use_synthetic:
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reconstruction'))
            from modules.Geometry import TGeometry

            if args.module_info:
                module_info = args.module_info
            else:
                module_info = f"./modules/ModuleInfo_{args.lumi_condition}.root"

            print(f"Loading geometry from {module_info}")
            geometry = TGeometry(moduleinfo=module_info, LumiCondition=args.lumi_condition)
        except Exception as e:
            print(f"Warning: Could not load geometry: {e}")
            print("Falling back to synthetic data")
            args.use_synthetic = True

    # Create datasets
    if args.use_synthetic:
        print("Using synthetic dataset")
        train_dataset = SyntheticDataset(num_events=10000, max_cells=args.max_cells)
        val_dataset = SyntheticDataset(num_events=1000, max_cells=args.max_cells)
    else:
        print(f"Loading data from {args.data_path}")
        # For now, use synthetic even with data path (for testing)
        # In production, this would load real data
        train_dataset = SyntheticDataset(num_events=10000, max_cells=args.max_cells)
        val_dataset = SyntheticDataset(num_events=1000, max_cells=args.max_cells)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Build model
    print("Building model...")
    model = build_model(args)
    model = model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_params:,} trainable parameters")

    # Build matcher and criterion
    matcher = build_matcher(args)
    criterion = build_loss(args)
    criterion = criterion.to(device)

    # Optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr * 0.1,
        },
    ]
    optimizer = AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.lr * 0.01
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_epochs]
    )

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Open log file
    log_file_path = output_dir / 'training_log.json'
    log_file = open(log_file_path, 'w')
    print(f"Logging to {log_file_path}")

    # Training loop
    print()
    print("Starting training...")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        # Train
        train_loss, train_loss_dict = train_one_epoch(
            model, criterion, matcher, train_loader, optimizer, device, epoch, args, log_file
        )

        # Step scheduler
        scheduler.step()

        # Evaluate
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            val_loss, val_loss_dict = evaluate(model, criterion, matcher, val_loader, device, epoch, log_file, args)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'args': vars(args)
                }
                torch.save(checkpoint, output_dir / 'best_model.pth')
                print(f"Saved best model (val_loss: {val_loss:.4f})")

        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, output_dir / f'checkpoint_epoch{epoch + 1}.pth')

    # Close log file
    log_file.close()

    print()
    print("=" * 80)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PicoCal Transformer Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
