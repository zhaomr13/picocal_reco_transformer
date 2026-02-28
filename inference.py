"""
Inference Script for PicoCal Transformer

Reconstruction/inference using a trained transformer model.
Outputs clusters in a format compatible with existing code.

Usage:
    python inference.py --checkpoint best_model.pth --input /path/to/OutTrigd.root --output output.root
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from picocal_reco_transformer.model import build_model
from picocal_reco_transformer.dataset import PicoCalDataset, collate_fn


def get_args_parser():
    """Define command line arguments."""
    parser = argparse.ArgumentParser('PicoCal Transformer Inference', add_help=False)

    # Model parameters (should match training)
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to model checkpoint')
    parser.add_argument('--d_model', default=256, type=int, help='Transformer hidden dimension')
    parser.add_argument('--nhead', default=8, type=int, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', default=4, type=int, help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', default=4, type=int, help='Number of decoder layers')
    parser.add_argument('--dim_feedforward', default=1024, type=int, help='Feedforward dimension')
    parser.add_argument('--num_cluster_queries', default=20, type=int, help='Number of cluster queries')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')
    parser.add_argument('--position_encoding', default='sine', type=str, choices=['sine', 'learned'])

    # Input/Output
    parser.add_argument('--input', required=True, type=str, help='Input ROOT file or directory')
    parser.add_argument('--output', default='transformer_output.root', type=str, help='Output ROOT file')
    parser.add_argument('--max_events', default=None, type=int, help='Maximum events to process')
    parser.add_argument('--max_cells', default=500, type=int, help='Maximum cells per event')

    # Geometry
    parser.add_argument('--lumi_condition', default='Run5_2024_refined_spacal_pb', type=str,
                        help='Luminosity condition for geometry')
    parser.add_argument('--module_info', default='', type=str,
                        help='Path to ModuleInfo root file')

    # Inference parameters
    parser.add_argument('--existence_threshold', default=0.5, type=float,
                        help='Threshold for cluster existence (sigmoid)')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size for inference')
    parser.add_argument('--device', default='cuda', type=str, help='Device to use')

    # Comparison with traditional clustering
    parser.add_argument('--compare', action='store_true',
                        help='Compare with traditional clustering')
    parser.add_argument('--seeding', default=0, type=int,
                        help='Seeding mode for traditional clustering comparison')

    return parser


class TClusterTransformer:
    """
    Cluster class compatible with TCluster interface.

    Wraps transformer predictions in an interface similar to traditional TCluster.
    """

    def __init__(self, pred_dict, idx):
        """
        Args:
            pred_dict: Dictionary with predictions from model
            idx: Index of this cluster in predictions
        """
        self.id = idx

        # Energies
        self.e = pred_dict['pred_energies'][idx, 0].item()
        self.eF = pred_dict['pred_energies'][idx, 1].item()
        self.eB = pred_dict['pred_energies'][idx, 2].item()

        # Position
        self.x = pred_dict['pred_positions'][idx, 0].item()
        self.y = pred_dict['pred_positions'][idx, 1].item()
        self.z = pred_dict['pred_positions'][idx, 2].item()

        # Time
        self.t = pred_dict['pred_times'][idx, 0].item()
        self.tF = pred_dict['pred_times'][idx, 1].item()
        self.tB = pred_dict['pred_times'][idx, 2].item()

        # Confidence
        self.confidence = pred_dict['pred_confidence'][idx].item()
        self.existence_prob = torch.sigmoid(pred_dict['pred_logits'][idx]).item()

    def getE(self):
        return self.e

    def getEF(self):
        return self.eF

    def getEB(self):
        return self.eB

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getZ(self):
        return self.z

    def getT(self):
        return self.t

    def getTF(self):
        return self.tF

    def getTB(self):
        return self.tB

    def getEt(self):
        """Compute transverse energy."""
        import math
        xy2 = self.x**2 + self.y**2
        return self.e * math.sqrt(xy2) / math.sqrt(xy2 + self.z**2)

    def getID(self):
        return self.id

    def __str__(self):
        return (f"TClusterTransformer(id={self.id}, E={self.e:.2f} MeV, "
                f"pos=({self.x:.1f}, {self.y:.1f}, {self.z:.1f}), "
                f"t={self.t:.2f} ns, conf={self.confidence:.3f})")


class TransformerReconstructor:
    """
    High-level interface for reconstruction using the transformer model.
    """

    def __init__(self, model, device='cuda', existence_threshold=0.5):
        """
        Args:
            model: Trained PicoCalTransformerModel
            device: Device to run on
            existence_threshold: Threshold for cluster existence
        """
        self.model = model
        self.device = device
        self.existence_threshold = existence_threshold
        self.model.eval()

    @torch.no_grad()
    def reconstruct(self, cells, max_cells=500):
        """
        Reconstruct clusters from a list of cells.

        Args:
            cells: List of TCell objects or cell feature dicts
            max_cells: Maximum number of cells to process

        Returns:
            List of TClusterTransformer objects
        """
        # Extract features
        features, positions, mask = self._prepare_input(cells, max_cells)

        # Move to device
        features = features.to(self.device)
        positions = positions.to(self.device)
        mask = mask.to(self.device)

        # Forward pass
        outputs = self.model(features, positions, mask)

        # Extract predictions above threshold
        clusters = self._extract_clusters(outputs)

        return clusters

    def _prepare_input(self, cells, max_cells):
        """Prepare input tensors from cells."""
        n_cells = min(len(cells), max_cells)

        features = torch.zeros(1, max_cells, 10)
        positions = torch.zeros(1, max_cells, 2)
        mask = torch.ones(1, max_cells, dtype=torch.bool)

        for i, cell in enumerate(cells[:n_cells]):
            if hasattr(cell, 'getEF'):
                # TCell object
                eF = max(cell.getEF(), 0.0)
                eB = max(cell.getEB(), 0.0)
                x = cell.getX()
                y = cell.getY()
                z = 12620.0
                tF = max(cell.getTF(), 0.0) if cell.getTF() > 0 else 0.0
                tB = max(cell.getTB(), 0.0) if cell.getTB() > 0 else 0.0
                dx = cell.getDx()
                dy = cell.getDy()
                module = cell.getModule()
                region = module.getRegion() if module else 1
            else:
                # Dict
                eF = max(cell.get('eF', 0), 0.0)
                eB = max(cell.get('eB', 0), 0.0)
                x = cell.get('x', 0)
                y = cell.get('y', 0)
                z = cell.get('z', 12620.0)
                tF = max(cell.get('tF', 0), 0.0)
                tB = max(cell.get('tB', 0), 0.0)
                dx = cell.get('dx', 30.0)
                dy = cell.get('dy', 30.0)
                region = cell.get('region', 1)

            features[0, i] = torch.tensor([eF, eB, x, y, z, tF, tB, dx, dy, region])
            positions[0, i] = torch.tensor([x, y])
            mask[0, i] = False

        return features, positions, mask

    def _extract_clusters(self, outputs):
        """Extract clusters from model outputs."""
        clusters = []

        # Get existence probabilities
        existence_probs = torch.sigmoid(outputs['pred_logits'][0])  # [num_queries]

        for i in range(len(existence_probs)):
            if existence_probs[i] > self.existence_threshold:
                cluster = TClusterTransformer(outputs, i)
                cluster.existence_prob = existence_probs[i].item()
                clusters.append(cluster)

        # Sort by energy (descending)
        clusters.sort(key=lambda c: c.e, reverse=True)

        return clusters


def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get args from checkpoint
    args_dict = checkpoint.get('args', {})

    # Create args object
    class Args:
        pass
    args = Args()
    for key, val in args_dict.items():
        setattr(args, key, val)

    # Build model
    model = build_model(args)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")

    return model, args


def process_file(input_path, output_path, model, geometry, args):
    """Process a ROOT file and save results."""
    import ROOT

    # Create output file
    fout = ROOT.TFile(output_path, "RECREATE")
    tree = ROOT.TTree("clusters", "Transformer reconstructed clusters")

    # Setup branches
    from array import array
    nclusters = array('i', [0])
    evtNumber = array('i', [0])
    cx = array('d', [0.])
    cy = array('d', [0.])
    cz = array('d', [0.])
    ct = array('d', [0.])
    ctF = array('d', [0.])
    ctB = array('d', [0.])
    ly = array('d', [0.])
    e = array('d', [0.])
    eF = array('d', [0.])
    eB = array('d', [0.])
    confidence = array('d', [0.])
    existence_prob = array('d', [0.])

    tree.Branch("evtNumber", evtNumber, "evtNumber/I")
    tree.Branch("nclusters", nclusters, "nclusters/I")
    tree.Branch("x", cx, "x/D")
    tree.Branch("y", cy, "y/D")
    tree.Branch("z", cz, "z/D")
    tree.Branch("t", ct, "t/D")
    tree.Branch("tF", ctF, "tF/D")
    tree.Branch("tB", ctB, "tB/D")
    tree.Branch("e", e, "e/D")
    tree.Branch("eF", eF, "eF/D")
    tree.Branch("eB", eB, "eB/D")
    tree.Branch("confidence", confidence, "confidence/D")
    tree.Branch("existence_prob", existence_prob, "existence_prob/D")

    # Load input
    fin = ROOT.TFile.Open(input_path)
    if not fin or fin.IsZombie():
        print(f"Error: Could not open {input_path}")
        return

    tin = fin.Get("tree")
    if not tin:
        print(f"Error: Could not find tree in {input_path}")
        return

    n_events = tin.GetEntries()
    if args.max_events:
        n_events = min(n_events, args.max_events)

    print(f"Processing {n_events} events from {input_path}")

    # Create reconstructor
    reconstructor = TransformerReconstructor(
        model,
        device=args.device,
        existence_threshold=args.existence_threshold
    )

    # Import TCellReco for reading
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reconstruction'))
    from modules.CellReco import TCellReco
    from modules.Calorimeter import TCalorimeter

    TCellReco.global_minimum_energy = 0.0
    TCellReco.global_minimum_seed_energy = 50.0

    start_time = time.time()

    for i_event in range(n_events):
        if (i_event + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = (i_event + 1) / elapsed
            print(f"  Event {i_event + 1}/{n_events} ({rate:.1f} events/sec)")

        tin.GetEntry(i_event)
        evtNumber[0] = i_event

        # Read cells
        cell_reco = TCellReco(tin, i_event, geometry=geometry)
        cells = cell_reco.getHitCells()

        if len(cells) == 0:
            nclusters[0] = 0
            tree.Fill()
            continue

        # Reconstruct with transformer
        clusters = reconstructor.reconstruct(cells, max_cells=args.max_cells)
        nclusters[0] = len(clusters)

        # Fill tree
        for cluster in clusters:
            cx[0] = cluster.getX()
            cy[0] = cluster.getY()
            cz[0] = cluster.getZ()
            ct[0] = cluster.getT()
            ctF[0] = cluster.getTF()
            ctB[0] = cluster.getTB()
            e[0] = cluster.getE()
            eF[0] = cluster.getEF()
            eB[0] = cluster.getEB()
            confidence[0] = cluster.confidence
            existence_prob[0] = cluster.existence_prob

            tree.Fill()

    fin.Close()

    # Write output
    fout.cd()
    tree.Write()
    fout.Close()

    elapsed = time.time() - start_time
    print(f"Processed {n_events} events in {elapsed:.1f}s ({n_events/elapsed:.1f} events/sec)")
    print(f"Output saved to {output_path}")


def compare_with_traditional(input_path, model, geometry, args, n_events=100):
    """Compare transformer reconstruction with traditional clustering."""
    import ROOT
    import math

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reconstruction'))
    from modules.CellReco import TCellReco
    from modules.Calorimeter import TCalorimeter

    TCellReco.global_minimum_energy = 0.0
    TCellReco.global_minimum_seed_energy = 50.0

    # Load input
    fin = ROOT.TFile.Open(input_path)
    tin = fin.Get("tree")

    reconstructor = TransformerReconstructor(
        model,
        device=args.device,
        existence_threshold=args.existence_threshold
    )

    n_events = min(n_events, tin.GetEntries())

    # Accumulate statistics
    transformer_counts = []
    traditional_counts = []
    energy_diffs = []
    position_diffs = []

    print(f"\nComparing reconstruction on {n_events} events...")

    for i_event in range(n_events):
        tin.GetEntry(i_event)

        # Read cells
        cell_reco = TCellReco(tin, i_event, geometry=geometry)
        cells = cell_reco.getHitCells()

        if len(cells) == 0:
            continue

        # Transformer reconstruction
        t_clusters = reconstructor.reconstruct(cells, max_cells=args.max_cells)

        # Traditional reconstruction
        calo = TCalorimeter(tin, i_event, seeding=args.seeding, geometry=geometry)
        tr_clusters = calo.getClusters(2)

        transformer_counts.append(len(t_clusters))
        traditional_counts.append(len(tr_clusters))

        # Match and compare
        for t_cl in t_clusters:
            for tr_cl in tr_clusters:
                dx = t_cl.getX() - tr_cl.getX()
                dy = t_cl.getY() - tr_cl.getY()
                dist = math.sqrt(dx**2 + dy**2)

                if dist < 100:  # Match within 100 mm
                    energy_diffs.append(t_cl.getE() - tr_cl.getE())
                    position_diffs.append(dist)
                    break

    fin.Close()

    # Print comparison
    print("\nComparison Results:")
    print("-" * 40)
    print(f"Transformer clusters per event: {np.mean(transformer_counts):.2f} +/- {np.std(transformer_counts):.2f}")
    print(f"Traditional clusters per event: {np.mean(traditional_counts):.2f} +/- {np.std(traditional_counts):.2f}")

    if energy_diffs:
        print(f"Energy difference (T - Tr): {np.mean(energy_diffs):.2f} +/- {np.std(energy_diffs):.2f} MeV")
        print(f"Position difference: {np.mean(position_diffs):.2f} +/- {np.std(position_diffs):.2f} mm")


def main(args):
    """Main inference function."""
    print("=" * 80)
    print("PicoCal Transformer Inference")
    print("=" * 80)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model, model_args = load_model(args.checkpoint, device)

    # Load geometry
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
        print(f"Error loading geometry: {e}")
        return

    # Compare mode
    if args.compare:
        compare_with_traditional(args.input, model, geometry, args)
        return

    # Process file
    process_file(args.input, args.output, model, geometry, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PicoCal Transformer Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
