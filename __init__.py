"""
Transformer-based reconstruction for PicoCal (DETR-inspired)

This package implements a transformer-based reconstruction algorithm for the LHCb SPACAL
calorimeter, treating reconstruction as a set prediction problem where clusters are
directly predicted from the full set of active cells using transformer attention.
"""

__version__ = "0.1.0"

from .cell_encoder import CellEncoder
from .position_encoding import PositionEncodingSine, PositionEncodingLearned
from .model import PicoCalTransformerModel, ClusterPredictionHead
from .matcher import HungarianMatcher
from .loss import PicoCalLoss
from .dataset import PicoCalDataset, collate_fn

__all__ = [
    'CellEncoder',
    'PositionEncodingSine',
    'PositionEncodingLearned',
    'PicoCalTransformerModel',
    'ClusterPredictionHead',
    'HungarianMatcher',
    'PicoCalLoss',
    'PicoCalDataset',
    'collate_fn',
]
