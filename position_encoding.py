"""
Positional Encoding Module

Provides positional encodings for cell positions to give the transformer
spatial awareness. Supports both sinusoidal and learned positional encodings.
"""

import math
import torch
from torch import nn


class PositionEncodingSine(nn.Module):
    """
    Sinusoidal positional encoding based on cell (x, y) coordinates.

    Uses the standard Transformer positional encoding scheme with different
    frequencies for different dimensions, adapted for continuous spatial coordinates.
    """

    def __init__(self, d_model=256, temperature=10000, normalize=True, scale=2*math.pi):
        """
        Args:
            d_model: Transformer hidden dimension
            temperature: Temperature for frequency scaling
            normalize: Whether to normalize coordinates to [0, scale]
            scale: Scale factor for normalized coordinates
        """
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale

        # Separate encodings for x and y (each gets d_model/2 dimensions)
        self.num_pos_feats = d_model // 2

    def forward(self, positions):
        """
        Args:
            positions: Tensor of shape [batch_size, num_cells, 2] containing (x, y) coordinates

        Returns:
            Positional encodings of shape [batch_size, num_cells, d_model]
        """
        batch_size, num_cells, _ = positions.shape

        # Extract x and y coordinates
        x = positions[..., 0]  # [batch_size, num_cells]
        y = positions[..., 1]  # [batch_size, num_cells]

        # Normalize coordinates if requested
        if self.normalize:
            # Normalize to [0, scale] based on actual calorimeter dimensions
            # X: -3000 to 3000 (6000mm span), Y: -2000 to 2000 (4000mm span)
            # This ensures equal encoding utilization for both axes
            x = (x + 3000) / 6000 * self.scale
            y = (y + 2000) / 4000 * self.scale
            # Clamp to valid range
            x = torch.clamp(x, 0, self.scale)
            y = torch.clamp(y, 0, self.scale)

        # Compute frequency bands
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=positions.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Compute positional encodings
        pos_x = x[..., None] / dim_t  # [batch_size, num_cells, num_pos_feats]
        pos_y = y[..., None] / dim_t  # [batch_size, num_cells, num_pos_feats]

        # Apply sin/cos alternating
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)

        # Concatenate x and y encodings
        pos = torch.cat([pos_x, pos_y], dim=-1)  # [batch_size, num_cells, d_model]

        return pos


class PositionEncodingLearned(nn.Module):
    """
    Learned positional encoding using embedding tables for x and y coordinates.

    Discretizes the calorimeter space into bins and learns embeddings for each bin.
    """

    def __init__(self, d_model=256, num_bins=50, x_range=(-4000, 4000), y_range=(-4000, 4000)):
        """
        Args:
            d_model: Transformer hidden dimension
            num_bins: Number of bins for each dimension
            x_range: Tuple of (min_x, max_x) in mm
            y_range: Tuple of (min_y, max_y) in mm
        """
        super().__init__()
        self.d_model = d_model
        self.num_bins = num_bins
        self.x_range = x_range
        self.y_range = y_range

        # Learned embeddings for x and y bins
        self.x_embed = nn.Embedding(num_bins, d_model // 2)
        self.y_embed = nn.Embedding(num_bins, d_model // 2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.x_embed.weight)
        nn.init.uniform_(self.y_embed.weight)

    def _discretize(self, coords, range_min, range_max):
        """Convert continuous coordinates to bin indices."""
        # Normalize to [0, 1]
        normalized = (coords - range_min) / (range_max - range_min)
        # Clamp to valid range
        normalized = torch.clamp(normalized, 0, 1)
        # Convert to bin indices
        bin_indices = (normalized * (self.num_bins - 1)).long()
        return bin_indices

    def forward(self, positions):
        """
        Args:
            positions: Tensor of shape [batch_size, num_cells, 2] containing (x, y) coordinates

        Returns:
            Positional encodings of shape [batch_size, num_cells, d_model]
        """
        # Discretize positions
        x_bins = self._discretize(positions[..., 0], self.x_range[0], self.x_range[1])
        y_bins = self._discretize(positions[..., 1], self.y_range[0], self.y_range[1])

        # Look up embeddings
        x_emb = self.x_embed(x_bins)  # [batch_size, num_cells, d_model//2]
        y_emb = self.y_embed(y_bins)  # [batch_size, num_cells, d_model//2]

        # Concatenate
        pos = torch.cat([x_emb, y_emb], dim=-1)  # [batch_size, num_cells, d_model]

        return pos


class PositionEncodingRadial(nn.Module):
    """
    Radial positional encoding based on distance from beam axis.

    Useful for calorimeters where shower properties depend on radial position.
    """

    def __init__(self, d_model=256, r_max=6000, num_angle_bins=36):
        """
        Args:
            d_model: Transformer hidden dimension
            r_max: Maximum radius in mm
            num_angle_bins: Number of angular bins for phi encoding
        """
        super().__init__()
        self.d_model = d_model
        self.r_max = r_max

        # Radial distance embedding
        self.r_embed = nn.Embedding(50, d_model // 2)

        # Angular embedding
        self.phi_embed = nn.Embedding(num_angle_bins, d_model // 2)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.r_embed.weight)
        nn.init.uniform_(self.phi_embed.weight)

    def forward(self, positions):
        """
        Args:
            positions: Tensor of shape [batch_size, num_cells, 2] containing (x, y) coordinates

        Returns:
            Positional encodings of shape [batch_size, num_cells, d_model]
        """
        x = positions[..., 0]
        y = positions[..., 1]

        # Compute radial distance and angle
        r = torch.sqrt(x**2 + y**2)
        phi = torch.atan2(y, x)

        # Discretize radius
        r_bins = (r / self.r_max * 49).long()
        r_bins = torch.clamp(r_bins, 0, 49)

        # Discretize angle
        phi_bins = ((phi + math.pi) / (2 * math.pi) * (self.phi_embed.num_embeddings - 1)).long()
        phi_bins = torch.clamp(phi_bins, 0, self.phi_embed.num_embeddings - 1)

        # Look up embeddings
        r_emb = self.r_embed(r_bins)
        phi_emb = self.phi_embed(phi_bins)

        # Concatenate
        pos = torch.cat([r_emb, phi_emb], dim=-1)

        return pos


class CombinedPositionEncoding(nn.Module):
    """
    Combines multiple positional encodings (e.g., Cartesian + Radial).
    """

    def __init__(self, d_model=256, encodings=['sine', 'radial']):
        super().__init__()
        self.d_model = d_model
        self.encodings = nn.ModuleList()

        d_per_encoding = d_model // len(encodings)

        for enc_type in encodings:
            if enc_type == 'sine':
                self.encodings.append(PositionEncodingSine(d_per_encoding))
            elif enc_type == 'learned':
                self.encodings.append(PositionEncodingLearned(d_per_encoding))
            elif enc_type == 'radial':
                self.encodings.append(PositionEncodingRadial(d_per_encoding))
            else:
                raise ValueError(f"Unknown encoding type: {enc_type}")

        # Project to full dimension if needed
        if len(encodings) * d_per_encoding != d_model:
            self.proj = nn.Linear(len(encodings) * d_per_encoding, d_model)
        else:
            self.proj = None

    def forward(self, positions):
        """
        Args:
            positions: Tensor of shape [batch_size, num_cells, 2] containing (x, y) coordinates

        Returns:
            Positional encodings of shape [batch_size, num_cells, d_model]
        """
        encodings = [enc(positions) for enc in self.encodings]
        combined = torch.cat(encodings, dim=-1)

        if self.proj is not None:
            combined = self.proj(combined)

        return combined
