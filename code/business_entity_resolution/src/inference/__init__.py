"""Inference layer."""
from .predictor import EntityResolutionPredictor
from .dl_refiner import DLRefiner

__all__ = ["EntityResolutionPredictor", "DLRefiner"]
