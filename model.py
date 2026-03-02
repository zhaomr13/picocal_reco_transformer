"""
Transformer Model for PicoCal Reconstruction

DETR-inspired architecture for calorimeter cluster prediction.
Uses a transformer encoder to process all cells and a decoder with
learned cluster queries to predict clusters directly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import copy


class PicoCalTransformerModel(nn.Module):
    """
    DETR-inspired transformer model for calorimeter reconstruction.

    Architecture:
    - Cell encoder: Embeds cell features into d_model dimensions
    - Transformer encoder: Processes all cells with global self-attention
    - Transformer decoder: Uses learned cluster queries to attend to cells
    - Prediction heads: Output cluster predictions (energy, position, timing, existence)
    """

    def __init__(
        self,
        cell_encoder,
        position_encoding,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        num_cluster_queries=20,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=True
    ):
        """
        Args:
            cell_encoder: Module to encode cell features into embeddings
            position_encoding: Module to compute positional encodings from cell positions
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimension of feedforward network
            num_cluster_queries: Maximum number of clusters to predict per event
            dropout: Dropout probability
            activation: Activation function ('relu' or 'gelu')
            normalize_before: Whether to use pre-normalization
            return_intermediate_dec: Return intermediate decoder outputs for auxiliary losses
        """
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_cluster_queries = num_cluster_queries

        # Cell feature encoder and position encoding
        self.cell_encoder = cell_encoder
        self.position_encoding = position_encoding

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # Transformer decoder
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm,
            return_intermediate=return_intermediate_dec
        )

        # Learnable cluster queries (object queries in DETR)
        self.query_embed = nn.Embedding(num_cluster_queries, d_model)

        # Query anchor positions - learnable "home" positions for each query
        # Initialize spread across the calorimeter (X: -3000 to 3000, Y: -2000 to 2000)
        self.query_anchors = nn.Parameter(
            torch.randn(num_cluster_queries, 2) * 1500.0
        )

        # Prediction heads
        self.cluster_head = ClusterPredictionHead(d_model)

        self._reset_parameters()

        # Initialize position head to predict small offsets from anchors
        # This must be done after _reset_parameters to override xavier init
        self._init_position_head_for_anchors()

    def _reset_parameters(self):
        # Don't apply xavier to query_anchors - keep their random spread
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'query_anchors' not in name:
                nn.init.xavier_uniform_(p)

    def _init_position_head_for_anchors(self):
        """
        Initialize position head to predict small offsets from query anchors.
        This helps queries start near their anchor positions and specialize.
        """
        # Get the last layer of position head
        position_head_last = self.cluster_head.position_head[-1]

        # Initialize weights to near zero (small offsets)
        position_head_last.weight.data.fill_(0.0)
        position_head_last.bias.data.fill_(0.0)

        # Set z bias to calorimeter surface (z ≈ 12620 mm)
        position_head_last.bias.data[2] = 12620.0

        # Re-initialize query anchors with proper spread (xavier overwrites them)
        nn.init.normal_(self.query_anchors, mean=0.0, std=1500.0)

        print(f"Initialized position head for anchor-based prediction")
        print(f"Query anchors initialized with std: {self.query_anchors.data.std():.2f} mm")

    def forward(self, cell_features, cell_positions, mask=None, return_auxiliary=False):
        """
        Forward pass.

        Args:
            cell_features: [batch_size, num_cells, n_features] tensor of cell features
            cell_positions: [batch_size, num_cells, 2] tensor of (x, y) positions
            mask: Optional [batch_size, num_cells] bool tensor (True for padded cells)
            return_auxiliary: If True, return predictions from all decoder layers

        Returns:
            Dictionary with:
            - pred_logits: [batch_size, num_queries, 1] cluster existence logits
            - pred_energies: [batch_size, num_queries, 3] (E, E_F, E_B) energies
            - pred_positions: [batch_size, num_queries, 3] (x, y, z) positions
            - pred_times: [batch_size, num_queries, 3] (t, t_F, t_B) times
            - pred_confidence: [batch_size, num_queries] confidence scores
            - aux_outputs: List of prediction dicts from intermediate layers (if return_auxiliary=True)
        """
        batch_size, num_cells, _ = cell_features.shape

        # Encode cell features
        cell_embeddings = self.cell_encoder(cell_features)  # [B, N, d_model]

        # Add positional encodings
        pos_embed = self.position_encoding(cell_positions)  # [B, N, d_model]

        # Transpose for transformer: [N, B, d_model]
        src = cell_embeddings.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 0, 2)

        # Create key padding mask if provided
        if mask is not None:
            src_key_padding_mask = mask  # [B, N]
        else:
            src_key_padding_mask = None

        # Transformer encoder
        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask, pos=pos_embed)

        # Prepare decoder queries
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)  # [num_queries, B, d_model]
        tgt = torch.zeros_like(query_embed)  # [num_queries, B, d_model]

        # Transformer decoder
        hs = self.decoder(
            tgt, memory,
            memory_key_padding_mask=src_key_padding_mask,
            pos=pos_embed,
            query_pos=query_embed
        )  # [num_layers, num_queries, B, d_model] or [1, num_queries, B, d_model]

        # Take decoder outputs
        if self.decoder.return_intermediate:
            # hs has shape [num_layers, num_queries, B, d_model]
            # Convert to list of [batch_size, num_queries, d_model]
            output_embeddings_list = [h.transpose(0, 1) for h in hs]
        else:
            output_embeddings_list = [hs.transpose(0, 1)]

        # Apply prediction heads to all layers
        all_predictions = [
            self.cluster_head(emb, self.query_anchors)
            for emb in output_embeddings_list
        ]

        # The final prediction is from the last layer
        predictions = all_predictions[-1]

        # Add auxiliary outputs if requested
        if return_auxiliary and len(all_predictions) > 1:
            predictions['aux_outputs'] = all_predictions[:-1]

        return predictions


class ClusterPredictionHead(nn.Module):
    """
    Prediction heads for cluster properties with query anchors.

    For each cluster query, predicts:
    - Existence: Binary classification (cluster exists or not)
    - Energy: Total energy and Front/Back components
    - Position: Offset from query anchor (delta_x, delta_y, z)
    - Time: Combined and Front/Back timing
    - Confidence: Prediction confidence score

    Query anchors provide "home" positions for each query, preventing collapse.
    """

    def __init__(self, d_model=256, hidden_dim=256):
        super().__init__()

        # Shared feature processing
        self.shared_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Existence head (binary classification)
        self.existence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Energy head (log-transformed for better dynamic range)
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # E_total, E_F, E_B
        )

        # Position head - predicts OFFSET from anchor, not absolute position
        # This allows each query to specialize to its "home" region
        self.position_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # delta_x, delta_y, z
        )

        # Time head
        self.time_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)  # t, t_F, t_B
        )

        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, features, query_anchors=None):
        """
        Args:
            features: [batch_size, num_queries, d_model]
            query_anchors: [num_queries, 2] learnable anchor positions (x, y)

        Returns:
            Dictionary with predictions
        """
        # Shared features
        shared = self.shared_mlp(features)  # [B, num_queries, hidden_dim]

        # Predictions
        pred_logits = self.existence_head(shared).squeeze(-1)  # [B, num_queries]
        pred_energies = self.energy_head(shared)  # [B, num_queries, 3]
        pred_position_offsets = self.position_head(shared)  # [B, num_queries, 3]
        pred_times = self.time_head(shared)  # [B, num_queries, 3]
        pred_confidence = self.confidence_head(shared).squeeze(-1)  # [B, num_queries]

        # Apply log-transform to energies (predict log(E + 1) to handle wide dynamic range)
        pred_energies = F.softplus(pred_energies)  # Ensure positive energies

        # Compute final positions: anchor + offset
        if query_anchors is not None:
            # query_anchors: [num_queries, 2] -> [1, num_queries, 2]
            anchors_xy = query_anchors.unsqueeze(0)  # [1, num_queries, 2]

            # Add offset to anchor for x, y; use predicted z directly
            # pred_position_offsets: [B, num_queries, 3] = (dx, dy, z)
            pred_positions_x = anchors_xy[..., 0] + pred_position_offsets[..., 0]
            pred_positions_y = anchors_xy[..., 1] + pred_position_offsets[..., 1]
            pred_positions_z = pred_position_offsets[..., 2]

            # Stack to get final positions [B, num_queries, 3]
            pred_positions = torch.stack([
                pred_positions_x,
                pred_positions_y,
                pred_positions_z
            ], dim=-1)
        else:
            # Fallback to direct prediction (for backward compatibility)
            pred_positions = pred_position_offsets

        return {
            'pred_logits': pred_logits,  # [B, num_queries]
            'pred_energies': pred_energies,  # [B, num_queries, 3]
            'pred_positions': pred_positions,  # [B, num_queries, 3]
            'pred_times': pred_times,  # [B, num_queries, 3]
            'pred_confidence': pred_confidence  # [B, num_queries]
        }


class TransformerEncoder(nn.Module):
    """Transformer encoder with multiple layers."""

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src

        for layer in self.layers:
            output = layer(
                output, src_mask=mask,
                src_key_padding_mask=src_key_padding_mask, pos=pos
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    """Transformer decoder with multiple layers."""

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output, memory,
                tgt_mask=tgt_mask, memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos, query_pos=query_pos
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output) if self.norm else output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with self-attention and feedforward."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        # Self-attention with positional encodings
        q = k = self.with_pos_embed(src, pos)
        src2, _ = self.self_attn(
            q, k, value=src, attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):
    """Single transformer decoder layer with self-attention, cross-attention, and feedforward."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        # Self-attention on queries
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, _ = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross-attention to encoder memory
        tgt2, _ = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Feedforward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


def _get_clones(module, N):
    """Create N deep copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """Return activation function by name."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"Activation should be relu/gelu, not {activation}")


def build_model(args=None, **kwargs):
    """
    Build the PicoCal transformer model with default or provided arguments.

    Args:
        args: Namespace with model hyperparameters
        **kwargs: Override specific parameters

    Returns:
        PicoCalTransformerModel instance
    """
    # Default hyperparameters
    default_args = {
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'dim_feedforward': 1024,
        'num_cluster_queries': 20,
        'dropout': 0.1,
        'activation': 'relu',
        'normalize_before': False,
        'return_intermediate_dec': True,
        'max_cells': 500,
        'max_module_id': 30,
        'use_module_embedding': True,
        'position_encoding_type': 'sine',
    }

    if args is not None:
        # Override defaults with provided args
        for key in default_args:
            if hasattr(args, key):
                default_args[key] = getattr(args, key)

    # Apply explicit kwargs
    default_args.update(kwargs)

    # Import here to avoid circular dependency
    from .cell_encoder import CellEncoder
    from .position_encoding import PositionEncodingSine, PositionEncodingLearned

    # Create encoder and position encoding
    cell_encoder = CellEncoder(
        d_model=default_args['d_model'],
        max_module_id=default_args['max_module_id'],
        use_module_embedding=default_args['use_module_embedding']
    )

    if default_args['position_encoding_type'] == 'sine':
        position_encoding = PositionEncodingSine(d_model=default_args['d_model'])
    else:
        position_encoding = PositionEncodingLearned(d_model=default_args['d_model'])

    # Create model
    model = PicoCalTransformerModel(
        cell_encoder=cell_encoder,
        position_encoding=position_encoding,
        d_model=default_args['d_model'],
        nhead=default_args['nhead'],
        num_encoder_layers=default_args['num_encoder_layers'],
        num_decoder_layers=default_args['num_decoder_layers'],
        dim_feedforward=default_args['dim_feedforward'],
        num_cluster_queries=default_args['num_cluster_queries'],
        dropout=default_args['dropout'],
        activation=default_args['activation'],
        normalize_before=default_args['normalize_before'],
        return_intermediate_dec=default_args['return_intermediate_dec']
    )

    return model
