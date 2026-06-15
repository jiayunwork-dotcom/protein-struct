from .base import (
    StructurePrediction,
    PredictionResult,
    STRUCTURE_STATES,
    STRUCTURE_NAMES,
    STRUCTURE_COLORS,
)
from .gor import GOR4Predictor
from .nn import NNPredictor
from .lstm import LSTMPredictor

__all__ = [
    "StructurePrediction",
    "PredictionResult",
    "GOR4Predictor",
    "NNPredictor",
    "LSTMPredictor",
    "STRUCTURE_STATES",
    "STRUCTURE_NAMES",
    "STRUCTURE_COLORS",
]
