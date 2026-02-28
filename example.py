"""
Example usage of PicoCal Transformer reconstruction.

This script demonstrates how to use the transformer-based reconstruction
for calorimeter data, comparing it with traditional clustering.
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reconstruction'))

import torch
import numpy as np


def example_synthetic_reconstruction():
    """Example: Reconstruct synthetic data."""
    print("=" * 80)
    print("Example: Synthetic Data Reconstruction")
    print("=" * 80)

    from picocal_reco_transformer.model import build_model
    from picocal_reco_transformer.dataset import SyntheticDataset
    from picocal_reco_transformer.inference import TransformerReconstructor

    # Create model
    print("\n1. Creating model...")
    model = build_model(
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_cluster_queries=20
    )
    print(f"   Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create synthetic dataset
    print("\n2. Generating synthetic data...")
    dataset = SyntheticDataset(num_events=10, max_cells=200)
    sample = dataset[0]

    print(f"   Number of cells: {sample['num_cells']}")
    print(f"   Number of target clusters: {len(sample['targets']['labels'])}")

    # Create reconstructor (using random weights for demo)
    print("\n3. Running reconstruction...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")

    reconstructor = TransformerReconstructor(model, device=device, existence_threshold=0.3)

    # Prepare cells from dataset
    cells = []
    for i in range(sample['num_cells']):
        cells.append({
            'eF': sample['features'][i, 0].item(),
            'eB': sample['features'][i, 1].item(),
            'x': sample['features'][i, 2].item(),
            'y': sample['features'][i, 3].item(),
            'z': sample['features'][i, 4].item(),
            'tF': sample['features'][i, 5].item(),
            'tB': sample['features'][i, 6].item(),
            'dx': sample['features'][i, 7].item(),
            'dy': sample['features'][i, 8].item(),
            'region': int(sample['features'][i, 9].item())
        })

    # Run reconstruction
    clusters = reconstructor.reconstruct(cells)

    print(f"\n4. Results:")
    print(f"   Found {len(clusters)} clusters")
    for i, cluster in enumerate(clusters):
        print(f"   Cluster {i+1}: E={cluster.getE():.1f} MeV, "
              f"pos=({cluster.getX():.1f}, {cluster.getY():.1f}), "
              f"conf={cluster.confidence:.3f}")

    print("\n" + "=" * 80)


def example_training():
    """Example: Train on synthetic data."""
    print("=" * 80)
    print("Example: Training on Synthetic Data")
    print("=" * 80)

    from picocal_reco_transformer.model import build_model
    from picocal_reco_transformer.matcher import build_matcher
    from picocal_reco_transformer.loss import build_loss
    from picocal_reco_transformer.dataset import SyntheticDataset, collate_fn
    from torch.utils.data import DataLoader

    # Setup
    print("\n1. Setting up training...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Device: {device}")

    # Create model
    model = build_model(
        d_model=128,  # Smaller for demo
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_cluster_queries=10
    ).to(device)

    matcher = build_matcher()
    criterion = build_loss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create dataset
    print("\n2. Creating dataset...")
    dataset = SyntheticDataset(num_events=100, max_cells=200)
    dataloader = DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_fn
    )
    print(f"   Dataset size: {len(dataset)}")

    # Training loop
    print("\n3. Training for 3 epochs...")
    model.train()
    criterion.train()

    for epoch in range(3):
        total_loss = 0
        n_batches = 0

        for i, batch in enumerate(dataloader):
            if i >= 10:  # Just 10 batches for demo
                break

            features = batch['features'].to(device)
            positions = batch['positions'].to(device)
            mask = batch['mask'].to(device)
            targets = batch['targets']

            for t in targets:
                for key in t:
                    if isinstance(t[key], torch.Tensor):
                        t[key] = t[key].to(device)

            # Forward
            outputs = model(features, positions, mask)

            # Match and compute loss
            indices = matcher(outputs, targets)
            num_clusters = sum(len(t['labels']) for t in targets)
            num_clusters = torch.as_tensor([num_clusters], dtype=torch.float32, device=device)

            loss_dict = criterion(outputs, targets, indices, num_clusters)
            loss = loss_dict['loss_total']

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"   Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    print("\n   Training complete!")
    print("\n" + "=" * 80)


def example_model_architecture():
    """Example: Print model architecture."""
    print("=" * 80)
    print("Example: Model Architecture")
    print("=" * 80)

    from picocal_reco_transformer.model import build_model

    print("\n1. Creating model...")
    model = build_model(
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        num_cluster_queries=20
    )

    print("\n2. Model structure:")
    print("-" * 40)

    # Print module structure
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"{name:20s}: {n_params:>10,} parameters")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("-" * 40)
    print(f"{'Total':20s}: {total:>10,} parameters")
    print(f"{'Trainable':20s}: {trainable:>10,} parameters")

    print("\n3. Forward pass test:")
    batch_size = 2
    num_cells = 100

    # Random input
    features = torch.randn(batch_size, num_cells, 10)
    positions = torch.randn(batch_size, num_cells, 2) * 1000
    mask = torch.zeros(batch_size, num_cells, dtype=torch.bool)

    # Forward
    model.eval()
    with torch.no_grad():
        outputs = model(features, positions, mask)

    print(f"   Input:  features {features.shape}, positions {positions.shape}")
    print(f"   Output:")
    for key, val in outputs.items():
        print(f"      {key:15s}: {val.shape}")

    print("\n" + "=" * 80)


def example_comparison():
    """Example: Compare transformer with traditional clustering."""
    print("=" * 80)
    print("Example: Comparison with Traditional Clustering")
    print("=" * 80)

    from picocal_reco_transformer.dataset import SyntheticDataset

    print("\nThis example would compare transformer-based reconstruction")
    print("with traditional clustering on the same events.")
    print()
    print("To run this comparison with real data:")
    print("  python inference.py \\")
    print("    --checkpoint best_model.pth \\")
    print("    --input /path/to/data.root \\")
    print("    --output output.root \\")
    print("    --compare")
    print()

    # Show synthetic example
    dataset = SyntheticDataset(num_events=5)
    sample = dataset[0]

    print("Synthetic event statistics:")
    print(f"  Number of cells: {sample['num_cells']}")
    print(f"  True clusters: {len(sample['targets']['labels'])}")
    print(f"  True energies: {sample['targets']['energies'][:, 0].tolist()}")
    print(f"  True positions: {sample['targets']['positions'][:, :2].tolist()}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('PicoCal Transformer Examples')
    parser.add_argument('--example', type=str, default='all',
                        choices=['all', 'synthetic', 'training', 'architecture', 'comparison'],
                        help='Which example to run')
    args = parser.parse_args()

    examples = {
        'synthetic': example_synthetic_reconstruction,
        'training': example_training,
        'architecture': example_model_architecture,
        'comparison': example_comparison
    }

    if args.example == 'all':
        for name, func in examples.items():
            func()
            print()
    else:
        examples[args.example]()
