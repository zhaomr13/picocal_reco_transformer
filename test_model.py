"""
Quick test script to verify the transformer model works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch


def test_cell_encoder():
    """Test CellEncoder."""
    print("Testing CellEncoder...")
    from picocal_reco_transformer.cell_encoder import CellEncoder

    encoder = CellEncoder(d_model=256)

    # Create dummy input
    batch_size, num_cells = 2, 50
    features = torch.randn(batch_size, num_cells, 10)

    # Forward pass
    embeddings = encoder(features)

    assert embeddings.shape == (batch_size, num_cells, 256)
    print(f"  ✓ Output shape: {embeddings.shape}")


def test_position_encoding():
    """Test PositionEncoding."""
    print("Testing PositionEncoding...")
    from picocal_reco_transformer.position_encoding import PositionEncodingSine

    pos_enc = PositionEncodingSine(d_model=256)

    # Create dummy positions
    batch_size, num_cells = 2, 50
    positions = torch.randn(batch_size, num_cells, 2) * 1000

    # Forward pass
    pos_embed = pos_enc(positions)

    assert pos_embed.shape == (batch_size, num_cells, 256)
    print(f"  ✓ Output shape: {pos_embed.shape}")


def test_model():
    """Test full model."""
    print("Testing PicoCalTransformerModel...")
    from picocal_reco_transformer.model import build_model

    model = build_model(
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        num_cluster_queries=10
    )

    # Create dummy input
    batch_size, num_cells = 2, 50
    features = torch.randn(batch_size, num_cells, 10)
    positions = torch.randn(batch_size, num_cells, 2) * 1000
    mask = torch.zeros(batch_size, num_cells, dtype=torch.bool)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(features, positions, mask)

    assert outputs['pred_logits'].shape == (batch_size, 10)
    assert outputs['pred_energies'].shape == (batch_size, 10, 3)
    assert outputs['pred_positions'].shape == (batch_size, 10, 3)
    assert outputs['pred_times'].shape == (batch_size, 10, 3)
    assert outputs['pred_confidence'].shape == (batch_size, 10)

    print(f"  ✓ pred_logits shape: {outputs['pred_logits'].shape}")
    print(f"  ✓ pred_energies shape: {outputs['pred_energies'].shape}")
    print(f"  ✓ pred_positions shape: {outputs['pred_positions'].shape}")
    print(f"  ✓ pred_times shape: {outputs['pred_times'].shape}")
    print(f"  ✓ pred_confidence shape: {outputs['pred_confidence'].shape}")


def test_matcher():
    """Test HungarianMatcher."""
    print("Testing HungarianMatcher...")
    from picocal_reco_transformer.matcher import HungarianMatcher

    matcher = HungarianMatcher()

    # Create dummy outputs and targets
    batch_size, num_queries = 2, 10

    outputs = {
        'pred_logits': torch.randn(batch_size, num_queries),
        'pred_energies': torch.randn(batch_size, num_queries, 3).abs() * 1000,
        'pred_positions': torch.randn(batch_size, num_queries, 3) * 1000,
        'pred_times': torch.randn(batch_size, num_queries, 3).abs(),
        'pred_confidence': torch.rand(batch_size, num_queries)
    }

    targets = [
        {
            'labels': torch.ones(3),
            'energies': torch.rand(3, 3) * 5000,
            'positions': torch.randn(3, 3) * 1000,
            'times': torch.rand(3, 3) * 10
        },
        {
            'labels': torch.ones(2),
            'energies': torch.rand(2, 3) * 5000,
            'positions': torch.randn(2, 3) * 1000,
            'times': torch.rand(2, 3) * 10
        }
    ]

    # Forward pass
    indices = matcher(outputs, targets)

    assert len(indices) == batch_size
    print(f"  ✓ Matched {len(indices)} batch elements")


def test_loss():
    """Test PicoCalLoss."""
    print("Testing PicoCalLoss...")
    from picocal_reco_transformer.loss import PicoCalLoss
    from picocal_reco_transformer.matcher import HungarianMatcher

    criterion = PicoCalLoss()
    matcher = HungarianMatcher()

    # Create dummy data
    batch_size, num_queries = 2, 10

    outputs = {
        'pred_logits': torch.randn(batch_size, num_queries),
        'pred_energies': torch.randn(batch_size, num_queries, 3).abs() * 1000,
        'pred_positions': torch.randn(batch_size, num_queries, 3) * 1000,
        'pred_times': torch.randn(batch_size, num_queries, 3).abs(),
        'pred_confidence': torch.rand(batch_size, num_queries)
    }

    targets = [
        {
            'labels': torch.ones(3),
            'energies': torch.rand(3, 3) * 5000,
            'positions': torch.randn(3, 3) * 1000,
            'times': torch.rand(3, 3) * 10
        },
        {
            'labels': torch.ones(2),
            'energies': torch.rand(2, 3) * 5000,
            'positions': torch.randn(2, 3) * 1000,
            'times': torch.rand(2, 3) * 10
        }
    ]

    # Match and compute loss
    indices = matcher(outputs, targets)
    num_clusters = sum(len(t['labels']) for t in targets)
    num_clusters = torch.as_tensor([num_clusters], dtype=torch.float32)

    losses = criterion(outputs, targets, indices, num_clusters)

    assert 'loss_total' in losses
    assert 'loss_existence' in losses
    assert 'loss_energy' in losses
    print(f"  ✓ loss_total: {losses['loss_total'].item():.4f}")


def test_dataset():
    """Test SyntheticDataset."""
    print("Testing SyntheticDataset...")
    from picocal_reco_transformer.dataset import SyntheticDataset, collate_fn
    from torch.utils.data import DataLoader

    dataset = SyntheticDataset(num_events=10, max_cells=100)

    assert len(dataset) == 10

    sample = dataset[0]
    assert 'features' in sample
    assert 'positions' in sample
    assert 'targets' in sample

    print(f"  ✓ Dataset size: {len(dataset)}")
    print(f"  ✓ Sample features shape: {sample['features'].shape}")
    print(f"  ✓ Sample targets: {len(sample['targets']['labels'])} clusters")

    # Test DataLoader
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))

    assert batch['features'].shape[0] == 2
    print(f"  ✓ Batch features shape: {batch['features'].shape}")


def test_full_pipeline():
    """Test full training pipeline."""
    print("Testing full training pipeline...")
    from picocal_reco_transformer.model import build_model
    from picocal_reco_transformer.matcher import HungarianMatcher
    from picocal_reco_transformer.loss import build_loss
    from picocal_reco_transformer.dataset import SyntheticDataset, collate_fn
    from torch.utils.data import DataLoader

    # Setup
    model = build_model(d_model=64, nhead=2, num_encoder_layers=1, num_decoder_layers=1)
    matcher = HungarianMatcher()
    criterion = build_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Dataset
    dataset = SyntheticDataset(num_events=20, max_cells=50)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

    # Training step
    model.train()
    batch = next(iter(loader))

    features = batch['features']
    positions = batch['positions']
    mask = batch['mask']
    targets = batch['targets']

    outputs = model(features, positions, mask)
    indices = matcher(outputs, targets)

    num_clusters = sum(len(t['labels']) for t in targets)
    num_clusters = torch.as_tensor([num_clusters], dtype=torch.float32)

    losses = criterion(outputs, targets, indices, num_clusters)
    loss = losses['loss_total']

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"  ✓ Training loss: {loss.item():.4f}")
    print(f"  ✓ Gradients computed successfully")


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("Running PicoCal Transformer Tests")
    print("=" * 80)
    print()

    tests = [
        test_cell_encoder,
        test_position_encoding,
        test_model,
        test_matcher,
        test_loss,
        test_dataset,
        test_full_pipeline
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
            print()
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
            print()

    print("=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
