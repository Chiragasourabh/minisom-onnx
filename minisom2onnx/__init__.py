"""
minisom2onnx

A Python library for converting MiniSom models to ONNX format.
"""

from .minisom2onnx import to_onnx
from .version import __version__

__all__ = ["to_onnx", "__version__"]
