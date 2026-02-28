"""
Cell Encoder Module

Encodes individual calorimeter cells into feature embeddings for the transformer.
Each active cell becomes a token with features derived from its energy deposits,
position, timing, and geometry information.
"""

import torch
import torch.nn as nn
import math


class CellEncoder(nn.Module):
    """
    Encodes calorimeter cell information into d-dimensional embeddings.

    Input features per cell:
    - Energy deposits: eF, eB (Front/Back section energies in MeV)
    - Position: x, y, z (cell center position in mm)
    - Time: tF, tB (Front/Back timing in ns)
    - Geometry: cell size dx, dy (mm), module type, region
    - Optional: module ID encoding

    The energy and time features are log-transformed for better dynamic range.
    """

    def __init__(self, d_model=256, max_module_id=30, use_module_embedding=True):
        """
        Args:
            d_model: Transformer hidden dimension
            max_module_id: Maximum number of modules for embedding
            use_module_embedding: Whether to use learnable module embeddings
        """
        super().__init__()
        self.d_model = d_model
        self.use_module_embedding = use_module_embedding

        # Input feature dimensions
        # eF, eB, x, y, z, tF, tB, dx, dy, region, [module_embedding]
        n_continuous_features = 10  # Without module embedding

        # Feature normalization/embedding layers
        # Use log-transform for energy and time to handle wide dynamic range
        self.energy_log_shift = 1.0  # Add small offset before log

        # Continuous feature projection
        self.continuous_proj = nn.Sequential(
            nn.Linear(n_continuous_features, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )

        # Module type embedding (for different module types 1-7)
        self.module_type_embed = nn.Embedding(8, d_model // 4)

        # Optional: Module ID embedding for geometric awareness
        if use_module_embedding:
            self.module_id_embed = nn.Embedding(max_module_id * 1000, d_model // 4)

        # Layer norm and dropout for regularization
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, cell_features, module_ids=None, module_types=None):
        """
        Encode cell features into embeddings.

        Args:
            cell_features: Tensor of shape [batch_size, num_cells, n_features]
                Features in order: [eF, eB, x, y, z, tF, tB, dx, dy, region]
            module_ids: Optional tensor of module IDs [batch_size, num_cells]
            module_types: Optional tensor of module types [batch_size, num_cells]

        Returns:
            Cell embeddings of shape [batch_size, num_cells, d_model]
        """
        batch_size, num_cells, _ = cell_features.shape

        # Extract features
        eF = cell_features[..., 0:1]
        eB = cell_features[..., 1:2]
        x = cell_features[..., 2:3]
        y = cell_features[..., 3:4]
        z = cell_features[..., 4:5]
        tF = cell_features[..., 5:6]
        tB = cell_features[..., 6:7]
        dx = cell_features[..., 7:8]
        dy = cell_features[..., 8:9]
        region = cell_features[..., 9:10]

        # Log-transform energy and time for better dynamic range
        eF_log = torch.log(eF + self.energy_log_shift)
        eB_log = torch.log(eB + self.energy_log_shift)
        tF_log = torch.log(tF.abs() + self.energy_log_shift) * torch.sign(tF)
        tB_log = torch.log(tB.abs() + self.energy_log_shift) * torch.sign(tB)

        # Normalize position by calorimeter scale (approximate)
        x_norm = x / 5000.0  # ~5m scale
        y_norm = y / 5000.0
        z_norm = z / 15000.0  # ~15m scale

        # Normalize cell size
        dx_norm = dx / 120.0  # Normalize by typical cell size
        dy_norm = dy / 120.0

        # Combine continuous features
        continuous_feats = torch.cat([
            eF_log, eB_log,
            x_norm, y_norm, z_norm,
            tF_log, tB_log,
            dx_norm, dy_norm,
            region
        ], dim=-1)

        # Project continuous features
        cont_embedding = self.continuous_proj(continuous_feats)

        # Add module type embedding if provided
        if module_types is not None:
            type_embed = self.module_type_embed(module_types.long())
            cont_embedding = cont_embedding + type_embed

        # Add module ID embedding if requested
        if self.use_module_embedding and module_ids is not None:
            # Clamp module IDs to valid range
            module_ids_clamped = torch.clamp(module_ids.long(), 0, self.module_id_embed.num_embeddings - 1)
            id_embed = self.module_id_embed(module_ids_clamped)
            cont_embedding = cont_embedding + id_embed

        # Apply normalization and dropout
        embeddings = self.norm(cont_embedding)
        embeddings = self.dropout(embeddings)

        return embeddings

    def get_cell_features_from_cells(self, cells, geometry=None):
        """
        Convert list of TCell objects to feature tensors.

        Args:
            cells: List of TCell objects
            geometry: Optional TGeometry for additional info

        Returns:
            Tuple of (features, module_ids, module_types) tensors
        """
        features = []
        module_ids = []
        module_types = []

        for cell in cells:
            # Get cell properties
            eF = cell.getEF()
            eB = cell.getEB()
            x = cell.getX()
            y = cell.getY()
            z = 12620.0  # Default calorimeter surface z
            tF = cell.getTF() if cell.getTF() > 0 else 0.0
            tB = cell.getTB() if cell.getTB() > 0 else 0.0
            dx = cell.getDx()
            dy = cell.getDy()

            # Get module info
            module = cell.getModule()
            module_id = module.getID() if module else 0
            module_type = module.getType() if module else 1
            region = module.getRegion() if module else 1

            features.append([eF, eB, x, y, z, tF, tB, dx, dy, region])
            module_ids.append(module_id * 1000 + cell.getID() % 1000)
            module_types.append(module_type)

        return (
            torch.tensor(features, dtype=torch.float32),
            torch.tensor(module_ids, dtype=torch.long),
            torch.tensor(module_types, dtype=torch.long)
        )


class CellFeatureExtractor:
    """
    Utility class to extract features from TCell objects for batch processing.
    Handles padding and masking for variable number of cells per event.
    """

    def __init__(self, max_cells=500):
        self.max_cells = max_cells

    def extract_batch(self, cell_lists):
        """
        Extract features from a batch of cell lists.

        Args:
            cell_lists: List of lists, where each inner list contains TCell objects for one event

        Returns:
            Dictionary with:
            - features: [batch_size, max_cells, n_features] tensor
            - mask: [batch_size, max_cells] bool tensor (True for padded cells)
            - module_ids: [batch_size, max_cells] tensor
            - module_types: [batch_size, max_cells] tensor
        """
        batch_size = len(cell_lists)

        # Initialize tensors
        features = torch.zeros(batch_size, self.max_cells, 10)
        mask = torch.ones(batch_size, self.max_cells, dtype=torch.bool)
        module_ids = torch.zeros(batch_size, self.max_cells, dtype=torch.long)
        module_types = torch.zeros(batch_size, self.max_cells, dtype=torch.long)

        for i, cells in enumerate(cell_lists):
            n_cells = min(len(cells), self.max_cells)

            for j, cell in enumerate(cells[:n_cells]):
                # Extract features
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

                features[i, j] = torch.tensor([eF, eB, x, y, z, tF, tB, dx, dy, region])
                mask[i, j] = False  # Not padded
                module_ids[i, j] = cell.getID()
                module_types[i, j] = module.getType() if module else 1

        return {
            'features': features,
            'mask': mask,
            'module_ids': module_ids,
            'module_types': module_types,
            'num_cells': [min(len(c), self.max_cells) for c in cell_lists]
        }
