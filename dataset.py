"""
Dataset Module for PicoCal Transformer

Handles loading ROOT files with calorimeter data and preparing training batches.
Integrates with existing TCellReco and TGeometry classes.

Supports:
- Loading from OutTrigd_*.root files
- Loading ground truth from flux files
- Creating training targets from true photon positions
- Batch collation with padding for variable numbers of cells
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import sys
import os

# Add reconstruction modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'reconstruction'))


class PicoCalDataset(Dataset):
    """
    Dataset for calorimeter reconstruction training.

    Loads events from ROOT files and creates:
    - Input: List of active cells with features
    - Target: List of true clusters (from flux files or reconstructed)
    """

    def __init__(
        self,
        root_file_list,
        geometry,
        flux_file_list=None,
        max_events=None,
        min_cell_energy=1e-6,
        min_seed_energy=50.0,
        max_cells=500,
        use_reconstructed_clusters=False,
        seeding=0
    ):
        """
        Args:
            root_file_list: List of paths to OutTrigd_*.root files or directory
            geometry: TGeometry instance with calorimeter geometry
            flux_file_list: Optional list of flux files with ground truth
            max_events: Maximum number of events to load (None for all)
            min_cell_energy: Minimum cell energy threshold
            min_seed_energy: Minimum seed energy threshold
            max_cells: Maximum cells per event (for padding)
            use_reconstructed_clusters: If True, use traditional clustering for targets
            seeding: Seeding mode for traditional clustering (if used)
        """
        self.geometry = geometry
        self.max_cells = max_cells
        self.use_reconstructed_clusters = use_reconstructed_clusters
        self.seeding = seeding

        # Import here to avoid issues if ROOT is not available during import
        try:
            from modules.CellReco import TCellReco
            from modules.Calorimeter import TCalorimeter
            self.TCellReco = TCellReco
            self.TCalorimeter = TCalorimeter

            # Set global thresholds
            TCellReco.global_minimum_energy = min_cell_energy
            TCellReco.global_minimum_seed_energy = min_seed_energy
        except ImportError as e:
            print(f"Warning: Could not import reconstruction modules: {e}")
            self.TCellReco = None
            self.TCalorimeter = None

        # Build file list
        self.file_list = self._build_file_list(root_file_list)

        if flux_file_list is not None:
            self.flux_file_list = self._build_file_list(flux_file_list)
            assert len(self.flux_file_list) == len(self.file_list), \
                "Number of flux files must match number of data files"
        else:
            self.flux_file_list = None

        # Build event index
        self.event_index = self._build_event_index(max_events)

        print(f"Dataset initialized with {len(self.event_index)} events")

    def _build_file_list(self, file_list):
        """Build list of files from directory or list."""
        if isinstance(file_list, str):
            if os.path.isdir(file_list):
                # List files matching OutTrigd_*.root pattern
                import glob
                files = sorted(glob.glob(os.path.join(file_list, "OutTrigd_*.root")))
                return files
            else:
                return [file_list]
        return list(file_list)

    def _build_event_index(self, max_events):
        """Build index of (file_idx, event_idx) tuples."""
        event_index = []

        try:
            import ROOT

            for file_idx, filepath in enumerate(self.file_list):
                try:
                    f = ROOT.TFile.Open(filepath)
                    if not f or f.IsZombie():
                        continue

                    tree = f.Get("tree")
                    if not tree:
                        f.Close()
                        continue

                    n_entries = tree.GetEntries()

                    for event_idx in range(n_entries):
                        event_index.append((file_idx, event_idx))

                        if max_events and len(event_index) >= max_events:
                            f.Close()
                            return event_index

                    f.Close()

                except Exception as e:
                    print(f"Error reading file {filepath}: {e}")
                    continue

        except ImportError:
            print("Warning: ROOT not available, creating dummy index")
            # For testing without ROOT
            event_index = [(0, i) for i in range(100)]

        return event_index

    def __len__(self):
        return len(self.event_index)

    def __getitem__(self, idx):
        """
        Get a single event.

        Returns:
            Dictionary with:
            - features: [num_cells, 10] cell features
            - positions: [num_cells, 2] cell (x, y) positions
            - cell_ids: [num_cells] cell IDs
            - targets: Dict with target cluster properties
        """
        file_idx, event_idx = self.event_index[idx]
        filepath = self.file_list[file_idx]

        # Load event data
        cells, clusters = self._load_event(filepath, event_idx)

        # Extract features from cells
        features, positions, cell_ids = self._extract_cell_features(cells)

        # Build targets
        if clusters:
            targets = self._build_targets(clusters)
        else:
            # Empty targets
            targets = {
                'labels': torch.zeros(0, dtype=torch.long),  # Existence
                'energies': torch.zeros((0, 3)),  # E, E_F, E_B
                'positions': torch.zeros((0, 3)),  # x, y, z
                'times': torch.zeros((0, 3)),  # t, t_F, t_B
            }

        return {
            'features': features,
            'positions': positions,
            'cell_ids': cell_ids,
            'targets': targets,
            'num_cells': len(cells)
        }

    def _load_event(self, filepath, event_idx):
        """Load cells and clusters from a single event."""
        cells = []
        clusters = []

        if self.TCellReco is None:
            return cells, clusters

        try:
            import ROOT

            f = ROOT.TFile.Open(filepath)
            if not f or f.IsZombie():
                return cells, clusters

            tree = f.Get("tree")
            if not tree:
                f.Close()
                return cells, clusters

            # Load event
            tree.GetEntry(event_idx)

            # Create cell reco
            cell_reco = self.TCellReco(tree, event_idx, geometry=self.geometry)
            cells = cell_reco.getHitCells()

            # Get clusters (either from flux files or traditional clustering)
            if self.use_reconstructed_clusters:
                # Use traditional clustering for targets
                calo = self.TCalorimeter(
                    tree, event_idx,
                    seeding=self.seeding,
                    geometry=self.geometry
                )
                clusters = calo.getClusters(2)

            f.Close()

        except Exception as e:
            print(f"Error loading event {event_idx} from {filepath}: {e}")

        return cells, clusters

    def _extract_cell_features(self, cells):
        """Extract features from list of TCell objects."""
        if not cells:
            return (
                torch.zeros((0, 10)),
                torch.zeros((0, 2)),
                torch.zeros(0, dtype=torch.long)
            )

        features = []
        positions = []
        cell_ids = []

        for cell in cells[:self.max_cells]:
            # Energy features
            eF = max(cell.getEF(), 0.0)
            eB = max(cell.getEB(), 0.0)

            # Position features
            x = cell.getX()
            y = cell.getY()
            z = 12620.0  # Calorimeter surface z

            # Time features
            tF = max(cell.getTF(), 0.0) if cell.getTF() > 0 else 0.0
            tB = max(cell.getTB(), 0.0) if cell.getTB() > 0 else 0.0

            # Geometry features
            dx = cell.getDx()
            dy = cell.getDy()

            # Module info
            module = cell.getModule()
            region = module.getRegion() if module else 1

            features.append([eF, eB, x, y, z, tF, tB, dx, dy, region])
            positions.append([x, y])
            cell_ids.append(cell.getID())

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(positions, dtype=torch.float32),
            torch.tensor(cell_ids, dtype=torch.long)
        )

    def _build_targets(self, clusters):
        """Build target tensors from list of TCluster objects."""
        labels = []
        energies = []
        positions = []
        times = []

        for cluster in clusters:
            # All clusters in targets are "existing"
            labels.append(1)

            # Energy (E, E_F, E_B)
            energies.append([
                cluster.getE(),
                cluster.getEF(),
                cluster.getEB()
            ])

            # Position (x, y, z)
            positions.append([
                cluster.getX(),
                cluster.getY(),
                cluster.getZ()
            ])

            # Time (t, t_F, t_B)
            times.append([
                cluster.getT(),
                cluster.getTF(),
                cluster.getTB()
            ])

        return {
            'labels': torch.tensor(labels, dtype=torch.long),
            'energies': torch.tensor(energies, dtype=torch.float32),
            'positions': torch.tensor(positions, dtype=torch.float32),
            'times': torch.tensor(times, dtype=torch.float32),
        }

    def get_event_display(self, idx):
        """
        Get data for event display.

        Returns:
            Dict with cell positions, energies, and cluster positions
        """
        file_idx, event_idx = self.event_index[idx]
        filepath = self.file_list[file_idx]

        cells, clusters = self._load_event(filepath, event_idx)

        cell_data = []
        for cell in cells:
            cell_data.append({
                'x': cell.getX(),
                'y': cell.getY(),
                'eF': cell.getEF(),
                'eB': cell.getEB(),
                'e': cell.getE(),
                'id': cell.getID()
            })

        cluster_data = []
        for cluster in clusters:
            cluster_data.append({
                'x': cluster.getX(),
                'y': cluster.getY(),
                'e': cluster.getE(),
                'eF': cluster.getEF(),
                'eB': cluster.getEB(),
                'seed_id': cluster.getID()
            })

        return {
            'cells': cell_data,
            'clusters': cluster_data
        }


def collate_fn(batch):
    """
    Collate function for DataLoader.

    Handles variable number of cells per event by padding.
    """
    # Find maximum number of cells in batch
    max_cells = max(item['num_cells'] for item in batch)
    max_cells = min(max_cells, 500)  # Cap at 500

    batch_size = len(batch)

    # Initialize padded tensors
    features = torch.zeros(batch_size, max_cells, 10)
    positions = torch.zeros(batch_size, max_cells, 2)
    cell_ids = torch.zeros(batch_size, max_cells, dtype=torch.long)
    mask = torch.ones(batch_size, max_cells, dtype=torch.bool)  # True = padded

    # Targets
    target_list = []

    for i, item in enumerate(batch):
        n_cells = min(item['num_cells'], max_cells)

        if n_cells > 0:
            features[i, :n_cells] = item['features'][:n_cells]
            positions[i, :n_cells] = item['positions'][:n_cells]
            cell_ids[i, :n_cells] = item['cell_ids'][:n_cells]
            mask[i, :n_cells] = False  # Not padded

        target_list.append(item['targets'])

    return {
        'features': features,
        'positions': positions,
        'cell_ids': cell_ids,
        'mask': mask,
        'targets': target_list
    }


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing without ROOT files.

    Generates simple cluster patterns for testing the model.
    """

    def __init__(self, num_events=1000, max_cells=500, num_clusters_range=(1, 5)):
        self.num_events = num_events
        self.max_cells = max_cells
        self.num_clusters_range = num_clusters_range

    def __len__(self):
        return self.num_events

    def __getitem__(self, idx):
        # Random number of clusters
        num_clusters = np.random.randint(*self.num_clusters_range)

        # Generate cluster centers
        cluster_centers = []
        for _ in range(num_clusters):
            center_x = np.random.uniform(-3000, 3000)
            center_y = np.random.uniform(-2000, 2000)
            energy = np.random.uniform(100, 5000)
            cluster_centers.append({
                'x': center_x,
                'y': center_y,
                'energy': energy,
                'eF': energy * 0.4,
                'eB': energy * 0.6,
                't': np.random.uniform(0, 10),
                'tF': np.random.uniform(0, 10),
                'tB': np.random.uniform(0, 10),
                'z': 12620.0
            })

        # Generate cells around cluster centers
        cells = []
        for cluster in cluster_centers:
            # Number of cells per cluster
            n_cells = np.random.randint(5, 20)

            for _ in range(n_cells):
                # Random position around cluster center
                dx = np.random.normal(0, 30)
                dy = np.random.normal(0, 30)

                x = cluster['x'] + dx
                y = cluster['y'] + dy

                # Energy based on distance from center
                dist = np.sqrt(dx**2 + dy**2)
                energy_frac = np.exp(-dist / 50) * np.random.uniform(0.5, 1.0)

                eF = cluster['eF'] * energy_frac / n_cells
                eB = cluster['eB'] * energy_frac / n_cells

                cells.append({
                    'eF': eF,
                    'eB': eB,
                    'x': x,
                    'y': y,
                    'z': cluster['z'],
                    'tF': cluster['tF'] + np.random.normal(0, 0.1),
                    'tB': cluster['tB'] + np.random.normal(0, 0.1),
                    'dx': 30.0,
                    'dy': 30.0,
                    'region': 1,
                    'id': len(cells)
                })

        # Extract features
        features = torch.tensor([
            [c['eF'], c['eB'], c['x'], c['y'], c['z'],
             c['tF'], c['tB'], c['dx'], c['dy'], c['region']]
            for c in cells[:self.max_cells]
        ], dtype=torch.float32)

        positions = torch.tensor([
            [c['x'], c['y']] for c in cells[:self.max_cells]
        ], dtype=torch.float32)

        cell_ids = torch.tensor([c['id'] for c in cells[:self.max_cells]], dtype=torch.long)

        # Build targets
        labels = torch.ones(num_clusters, dtype=torch.long)
        energies = torch.tensor([
            [c['energy'], c['eF'], c['eB']] for c in cluster_centers
        ], dtype=torch.float32)
        positions_t = torch.tensor([
            [c['x'], c['y'], c['z']] for c in cluster_centers
        ], dtype=torch.float32)
        times = torch.tensor([
            [c['t'], c['tF'], c['tB']] for c in cluster_centers
        ], dtype=torch.float32)

        return {
            'features': features,
            'positions': positions,
            'cell_ids': cell_ids,
            'targets': {
                'labels': labels,
                'energies': energies,
                'positions': positions_t,
                'times': times
            },
            'num_cells': len(features)
        }
