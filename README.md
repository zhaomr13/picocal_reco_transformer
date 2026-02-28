# PicoCal Transformer: DETR-inspired Calorimeter Reconstruction

Transformer-based reconstruction algorithm for the LHCb SPACAL calorimeter, inspired by DETR (DEtection TRansformer). This approach treats calorimeter reconstruction as a set prediction problem where clusters are directly predicted from the full set of active cells using transformer attention mechanisms.

## Overview

Traditional clustering algorithms rely on local seed finding and iterative clustering with hand-tuned parameters. This transformer-based approach offers several advantages:

- **Global attention**: Captures long-range dependencies between cells
- **Set-based prediction**: Naturally handles variable number of clusters
- **No hand-tuned clustering parameters**: No seed thresholds or cluster shapes needed
- **Learned energy sharing**: Learns optimal energy sharing between overlapping clusters
- **Unified framework**: Single model for position, energy, and timing reconstruction

## Architecture

```
Input Cells (sparse set of active cells)
    ↓
Cell Encoder (embeds cell features: energy, position, timing, geometry)
    ↓
Positional Encoding (spatial encoding from cell positions)
    ↓
Transformer Encoder (4-6 layers, global self-attention over cells)
    ↓
Transformer Decoder (4-6 layers, cross-attention with learned cluster queries)
    ↓
Prediction Heads (energy, position, timing, existence)
    ↓
Hungarian Matching (optimal assignment to ground truth)
```

### Key Components

1. **Cell Encoder** (`cell_encoder.py`): Embeds each cell's features (energy deposits, position, timing, geometry) into d-dimensional vectors.

2. **Positional Encoding** (`position_encoding.py`): Provides spatial awareness using sinusoidal or learned encodings based on cell (x,y) positions.

3. **Transformer Model** (`model.py`): Core architecture with encoder-decoder structure and prediction heads.

4. **Hungarian Matcher** (`matcher.py`): Optimal assignment of predictions to ground truth clusters.

5. **Loss Functions** (`loss.py`): Combined loss for existence, energy, position, and timing.

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- ROOT (for data loading)
- scipy (for Hungarian matching)

### Setup

```bash
# Source the LCG environment (provides ROOT and Python)
source /cvmfs/sft.cern.ch/lcg/views/LCG_105_LHCB_7/x86_64-el9-gcc12-opt/setup.sh

# The package is self-contained within the PicoCal repository
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."
```

## Quick Start

### 1. Training

Train on synthetic data (for testing):

```bash
python -m picocal_reco_transformer.train \
    --use_synthetic \
    --epochs 50 \
    --batch_size 8 \
    --output_dir ./output
```

Train on real data:

```bash
python -m picocal_reco_transformer.train \
    --data_path /path/to/OutTrigd_files/ \
    --lumi_condition Run5_2024_refined_spacal_pb \
    --epochs 100 \
    --batch_size 4 \
    --output_dir ./output
```

### 2. Inference

Run reconstruction on data:

```bash
python -m picocal_reco_transformer.inference \
    --checkpoint ./output/best_model.pth \
    --input /path/to/OutTrigd.root \
    --output ./transformer_clusters.root \
    --lumi_condition Run5_2024_refined_spacal_pb
```

Compare with traditional clustering:

```bash
python -m picocal_reco_transformer.inference \
    --checkpoint ./output/best_model.pth \
    --input /path/to/OutTrigd.root \
    --compare \
    --seeding 0
```

### 3. Examples

Run example scripts:

```bash
# Show model architecture
python picocal_reco_transformer/example.py --example architecture

# Test on synthetic data
python picocal_reco_transformer/example.py --example synthetic

# Quick training demo
python picocal_reco_transformer/example.py --example training
```

## Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 256 | Transformer hidden dimension |
| `nhead` | 8 | Number of attention heads |
| `num_encoder_layers` | 4 | Number of encoder layers |
| `num_decoder_layers` | 4 | Number of decoder layers |
| `dim_feedforward` | 1024 | Feedforward dimension |
| `num_cluster_queries` | 20 | Maximum clusters per event |
| `dropout` | 0.1 | Dropout rate |

## File Structure

```
picocal_reco_transformer/
├── __init__.py              # Package initialization
├── cell_encoder.py          # Cell feature embedding
├── position_encoding.py     # Spatial positional encoding
├── model.py                 # DETR-inspired transformer model
├── matcher.py               # Hungarian matching
├── loss.py                  # Combined training loss
├── dataset.py               # ROOT file data loading
├── train.py                 # Training script
├── inference.py             # Inference/reconstruction script
├── utils.py                 # Helper functions
├── example.py               # Example usage scripts
└── README.md                # This file
```

## Integration with Existing Code

The transformer model integrates with existing PicoCal reconstruction code:

- **TGeometry** (`reconstruction/modules/Geometry.py`): Loads calorimeter geometry
- **TCellReco** (`reconstruction/modules/CellReco.py`): Reads ROOT files and creates cells
- **TCluster** (`reconstruction/modules/Cluster.py`): Cluster format for output compatibility

The `TClusterTransformer` class in `inference.py` provides an interface compatible with existing `TCluster` code.

## Output Format

The inference script outputs a ROOT TTree with branches:

| Branch | Type | Description |
|--------|------|-------------|
| `evtNumber` | I | Event number |
| `nclusters` | I | Number of clusters in event |
| `x` | D | Cluster x position (mm) |
| `y` | D | Cluster y position (mm) |
| `z` | D | Cluster z position (mm) |
| `t` | D | Cluster time (ns) |
| `tF` | D | Front section time (ns) |
| `tB` | D | Back section time (ns) |
| `e` | D | Total energy (MeV) |
| `eF` | D | Front section energy (MeV) |
| `eB` | D | Back section energy (MeV) |
| `confidence` | D | Prediction confidence |
| `existence_prob` | D | Probability of cluster existence |

## Training Tips

1. **Learning rate**: Start with 1e-4, use warmup + cosine decay
2. **Batch size**: Use 4-8 depending on GPU memory
3. **Data**: Start with synthetic data to verify model works, then switch to real
4. **Loss weights**: Adjust `weight_existence` vs `weight_energy` based on task priority
5. **Gradient clipping**: Use `clip_grad=1.0` for stability

## Performance Considerations

- **Inference speed**: ~50-100 events/sec on GPU (V100)
- **Memory usage**: ~2-4 GB for batch_size=4
- **Max cells**: Default 500 cells per event (adjust based on occupancy)

## References

- Original DETR paper: "End-to-End Object Detection with Transformers" (Carion et al., ECCV 2020)
- Traditional PicoCal reconstruction: `reconstruction/reco_example.py`

## Troubleshooting

**ImportError for ROOT**: Make sure you've sourced the LCG setup script:
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_105_LHCB_7/x86_64-el9-gcc12-opt/setup.sh
```

**CUDA out of memory**: Reduce `batch_size` or `max_cells`:
```bash
python -m picocal_reco_transformer.train --batch_size 2 --max_cells 300
```

**No clusters found**: Lower `existence_threshold` in inference:
```bash
python -m picocal_reco_transformer.inference --existence_threshold 0.3
```

## Future Work

- [ ] Attention visualization for interpretability
- [ ] Multi-GPU training support
- [ ] Quantization for faster inference
- [ ] Integration with physics analysis frameworks
- [ ] Physics-constrained loss functions

## Contact

For questions or issues, please contact the PicoCal development team or open an issue on the repository.
