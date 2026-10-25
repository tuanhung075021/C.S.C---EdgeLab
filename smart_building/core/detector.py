"""
OpenVINO YOLOv8 Person Detector Module for EcoZone AI.
Includes multi-threaded inference, class 0 (person) filtering, bottom-center zone mapping,
and a robust mock detection fallback for simulation mode or missing models.
"""

import os
import logging
import threading
from typing import Tuple, List, Optional
import numpy as np
import cv2

try:
    import config
    MODEL_XML = getattr(config, "MODEL_XML", "models/yolov8n_openvino_model/yolov8n.xml")
    CONF_THRESHOLD = getattr(config, "CONFIDENCE_THRESHOLD", 0.35)
    VALID_CLASS_IDS = getattr(config, "VALID_CLASS_IDS", [0])
    IS_SIMULATION = getattr(config, "IS_SIMULATION", False)
except ImportError:
    try:
        from .. import config  # type: ignore
        MODEL_XML = getattr(config, "MODEL_XML", "models/yolov8n_openvino_model/yolov8n.xml")
        CONF_THRESHOLD = getattr(config, "CONFIDENCE_THRESHOLD", 0.35)
        VALID_CLASS_IDS = getattr(config, "VALID_CLASS_IDS", [0])
        IS_SIMULATION = getattr(config, "IS_SIMULATION", False)
    except ImportError:
        MODEL_XML = "models/yolov8n_openvino_model/yolov8n.xml"
        CONF_THRESHOLD = 0.35
        VALID_CLASS_IDS = [0]
        IS_SIMULATION = False

from core.transform import get_zone_id

logger = logging.getLogger(__name__)

OPENVINO_AVAILABLE = False
try:
    from openvino import Core
    OPENVINO_AVAILABLE = True
except ImportError:
    try:
        from openvino.runtime import Core
        OPENVINO_AVAILABLE = True
    except ImportError:
        OPENVINO_AVAILABLE = False


class YOLOv8OpenVINODetector:
    """
    YOLOv8 OpenVINO Person Detector with thread safety and simulation/mock fallback.
    """

    def __init__(
        self,
        model_xml: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        valid_class_ids: Optional[list[int]] = None,
        is_simulation: Optional[bool] = None,
    ) -> None:
        self.model_xml = model_xml if model_xml is not None else MODEL_XML
        self.conf_threshold = conf_threshold if conf_threshold is not None else CONF_THRESHOLD
        self.valid_class_ids = valid_class_ids if valid_class_ids is not None else VALID_CLASS_IDS
        self.is_simulation = is_simulation if is_simulation is not None else IS_SIMULATION

        self._lock = threading.Lock()
        self.use_real_model = False
        self.compiled_model = None
        self.input_layer = None
        self.output_layer = None

        if OPENVINO_AVAILABLE and not self.is_simulation and os.path.exists(self.model_xml):
            try:
                core = Core()
                model = core.read_model(model=self.model_xml)
                self.compiled_model = core.compile_model(model=model, device_name="CPU")
                self.input_layer = self.compiled_model.input(0)
                self.output_layer = self.compiled_model.output(0)
                self.use_real_model = True
                logger.info("OpenVINO YOLOv8 Detector initialized successfully from %s", self.model_xml)
            except Exception as e:
                logger.warning("Failed to compile OpenVINO model %s: %s. Using mock detection fallback.", self.model_xml, e)
                self.use_real_model = False
        else:
            logger.info("OpenVINO IR model unavailable at %s or simulation mode active. Using mock detection fallback.", self.model_xml)
            self.use_real_model = False

    def detect(self, frame: np.ndarray, transformer=None) -> Tuple[List[int], int]:
        """
        Thread-safe detection and zone mapping.
        Accepts an optional PerspectiveTransformer; if provided, detection runs on the raw
        frame (better quality) and bounding box anchor points are mapped through the
        perspective matrix for accurate zone assignment.

        Returns:
            (occupied_zones, total_people_count):
            occupied_zones: list of 6 integers (0 or 1)
            total_people_count: int
        """
        with self._lock:
            if frame is None or frame.size == 0:
                return [0] * 6, 0

            if self.use_real_model and self.compiled_model is not None:
                return self._detect_openvino(frame, transformer=transformer)
            else:
                return self._detect_mock(frame)

    def _detect_openvino(self, frame: np.ndarray, transformer=None) -> Tuple[List[int], int]:
        orig_h, orig_w = frame.shape[:2]

        # Resize to 640x640 network input
        resized = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0).astype(np.float32) / 255.0

        results = self.compiled_model([input_tensor])[self.output_layer]
        predictions = results[0].T  # Shape: (8400, 84)

        # Get max score across all allowed object classes
        class_scores = predictions[:, 4:]
        scores = np.max(class_scores[:, self.valid_class_ids], axis=1)
        mask = scores >= self.conf_threshold

        valid_preds = predictions[mask]
        valid_scores = scores[mask]

        if len(valid_preds) == 0:
            return [0] * 6, 0

        boxes = []
        scale_x = orig_w / 640.0
        scale_y = orig_h / 640.0

        for pred in valid_preds:
            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
            x1 = (cx - w / 2.0) * scale_x
            y1 = (cy - h / 2.0) * scale_y
            bw = w * scale_x
            bh = h * scale_y
            boxes.append([int(x1), int(y1), int(bw), int(bh)])

        indices = cv2.dnn.NMSBoxes(boxes, valid_scores.tolist(), self.conf_threshold, 0.45)

        occupied_zones = [0] * 6
        if len(indices) == 0:
            return occupied_zones, 0

        if isinstance(indices, tuple):
            indices = indices[0]
        else:
            indices = indices.flatten()

        total_people = len(indices)

        for idx in indices:
            pred = valid_preds[idx]
            cx_net, cy_net, h_net = pred[0], pred[1], pred[3]
            x_center_raw = cx_net * scale_x
            y_max_raw = (cy_net + h_net / 2.0) * scale_y

            # If transformer available: map anchor point through perspective to get
            # Bird's-Eye coordinates, then compute zone from those coords.
            if transformer is not None:
                try:
                    pt_warped = transformer.transform_point(x_center_raw, y_max_raw)
                    x_center, y_max = pt_warped[0], pt_warped[1]
                except Exception:
                    x_center, y_max = x_center_raw, y_max_raw
            else:
                x_center, y_max = x_center_raw, y_max_raw

            zone_id = get_zone_id(x_center, y_max)
            occupied_zones[zone_id] = 1

        return occupied_zones, total_people

    def _detect_mock(self, frame: np.ndarray) -> Tuple[List[int], int]:
        """
        Mock detection engine for simulation mode or missing model files.
        Detects green synthetic target shapes in frames (from synthetic stream) or returns standard mock results.
        """
        occupied_zones = [0] * 6
        orig_h, orig_w = frame.shape[:2]

        # Convert to HSV for robust color detection under different lighting
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # STRICTOR Green hue range (55-85), High Saturation (150+), High Value (100+)
        # This prevents yellowish walls or dim lighting from being detected as green.
        lower_green = np.array([55, 150, 100], dtype=np.uint8)
        upper_green = np.array([85, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_green, upper_green)


        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]

        total_people = 0
        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_center = x + w / 2.0
            y_max = y + h

            zone_id = get_zone_id(x_center, y_max)
            occupied_zones[zone_id] = 1
            total_people += 1

        return occupied_zones, total_people
