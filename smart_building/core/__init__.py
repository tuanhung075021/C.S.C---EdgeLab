"""
Core package initialization for EcoZone AI pipeline.
Exposes VisionStream, PerspectiveTransformer, get_zone_id, YOLOv8OpenVINODetector, and ColorDetector.
"""

from .vision_stream import VisionStream
from .transform import PerspectiveTransformer, get_zone_id
from .detector import YOLOv8OpenVINODetector
from .color_detector import ColorDetector

__all__ = [
    "VisionStream",
    "PerspectiveTransformer",
    "get_zone_id",
    "YOLOv8OpenVINODetector",
    "ColorDetector",
]

