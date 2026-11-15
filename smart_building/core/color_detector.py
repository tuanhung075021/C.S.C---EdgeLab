"""
Color Detection Module for EcoZone AI Smart Building System.
Detects yellow proxy objects (such as yellow sticker cups) in camera / warped frames,
cleans morphological noise, extracts contours, computes bottom-center anchors,
optionally applies perspective transform, and maps detections to 6 zones.
"""

import logging
import threading
from typing import List, Optional, Sequence, Tuple, Union
import cv2
import numpy as np

try:
    import config
    COLOR_YELLOW_HSV_LOWER = getattr(config, "COLOR_YELLOW_HSV_LOWER", [20, 100, 100])
    COLOR_YELLOW_HSV_UPPER = getattr(config, "COLOR_YELLOW_HSV_UPPER", [35, 255, 255])
    COLOR_MIN_CONTOUR_AREA = getattr(config, "COLOR_MIN_CONTOUR_AREA", 500)
    WARP_WIDTH = getattr(config, "WARP_WIDTH", 600)
    WARP_HEIGHT = getattr(config, "WARP_HEIGHT", 400)
except ImportError:
    try:
        from .. import config  # type: ignore
        COLOR_YELLOW_HSV_LOWER = getattr(config, "COLOR_YELLOW_HSV_LOWER", [20, 100, 100])
        COLOR_YELLOW_HSV_UPPER = getattr(config, "COLOR_YELLOW_HSV_UPPER", [35, 255, 255])
        COLOR_MIN_CONTOUR_AREA = getattr(config, "COLOR_MIN_CONTOUR_AREA", 500)
        WARP_WIDTH = getattr(config, "WARP_WIDTH", 600)
        WARP_HEIGHT = getattr(config, "WARP_HEIGHT", 400)
    except ImportError:
        COLOR_YELLOW_HSV_LOWER = [20, 100, 100]
        COLOR_YELLOW_HSV_UPPER = [35, 255, 255]
        COLOR_MIN_CONTOUR_AREA = 500
        WARP_WIDTH = 600
        WARP_HEIGHT = 400

from core.transform import get_zone_id

logger = logging.getLogger(__name__)


class ColorDetector:
    """
    HSV Color Detector for identifying yellow markers/proxies in 30x40cm physical model.
    Provides thread-safe detection and binary mask inspection.
    """

    def __init__(
        self,
        lower_hsv: Optional[Union[np.ndarray, Sequence[int]]] = None,
        upper_hsv: Optional[Union[np.ndarray, Sequence[int]]] = None,
        min_contour_area: Optional[float] = None,
        morph_kernel_size: Tuple[int, int] = (5, 5),
        enable_glare_recovery: bool = True,
    ) -> None:
        """
        Initialize ColorDetector with HSV bounds, minimum contour area, and structuring element.

        Args:
            lower_hsv: Lower HSV threshold bounds [H, S, V] (default [20, 100, 100])
            upper_hsv: Upper HSV threshold bounds [H, S, V] (default [35, 255, 255])
            min_contour_area: Minimum contour pixel area to retain (default 500.0)
            morph_kernel_size: Kernel size tuple for morphological opening/closing
            enable_glare_recovery: Recover specular-glared highlights adjacent to valid detections
        """
        raw_lower = lower_hsv if lower_hsv is not None else COLOR_YELLOW_HSV_LOWER
        raw_upper = upper_hsv if upper_hsv is not None else COLOR_YELLOW_HSV_UPPER

        self.lower_hsv = np.array(raw_lower, dtype=np.uint8)
        self.upper_hsv = np.array(raw_upper, dtype=np.uint8)

        raw_area = min_contour_area if min_contour_area is not None else COLOR_MIN_CONTOUR_AREA
        self.min_contour_area = float(raw_area)

        self.morph_kernel_size = morph_kernel_size
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
        self._seed_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.enable_glare_recovery = bool(enable_glare_recovery)
        self._glare_halo_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        self._last_mask = None

        self._lock = threading.RLock()
        logger.info(
            "ColorDetector initialized with HSV [%s - %s], min_area=%.1f, kernel=%s",
            self.lower_hsv.tolist(),
            self.upper_hsv.tolist(),
            self.min_contour_area,
            self.morph_kernel_size,
        )

    def update_config(
        self,
        lower_hsv: Optional[Union[np.ndarray, Sequence[int]]] = None,
        upper_hsv: Optional[Union[np.ndarray, Sequence[int]]] = None,
        min_contour_area: Optional[float] = None,
        enable_glare_recovery: Optional[bool] = None,
    ) -> dict:
        """Dynamically update HSV bounds, minimum contour area, and glare recovery setting."""
        with self._lock:
            if lower_hsv is not None:
                self.lower_hsv = np.array(lower_hsv, dtype=np.uint8)
            if upper_hsv is not None:
                self.upper_hsv = np.array(upper_hsv, dtype=np.uint8)
            if min_contour_area is not None:
                self.min_contour_area = max(10.0, float(min_contour_area))
            if enable_glare_recovery is not None:
                self.enable_glare_recovery = bool(enable_glare_recovery)
            return {
                "lower_hsv": self.lower_hsv.tolist(),
                "upper_hsv": self.upper_hsv.tolist(),
                "min_contour_area": float(self.min_contour_area),
                "enable_glare_recovery": bool(self.enable_glare_recovery),
            }

    def get_config(self) -> dict:
        """Return current detector configuration as JSON-serializable dict."""
        with self._lock:
            return {
                "lower_hsv": self.lower_hsv.tolist(),
                "upper_hsv": self.upper_hsv.tolist(),
                "min_contour_area": float(self.min_contour_area),
                "enable_glare_recovery": bool(self.enable_glare_recovery),
            }

    def _compute_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Internal helper to segment yellow pixels, recover LED specular glare via 2-step
        geodesic dilation from 3x3-filtered yellow seeds, and perform morphological cleanup.
        """
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
            or frame.ndim != 3
            or frame.shape[2] != 3
            or not np.issubdtype(frame.dtype, np.number)
        ):
            return np.zeros((WARP_HEIGHT, WARP_WIDTH), dtype=np.uint8)

        if frame.dtype != np.uint8:
            frame = np.clip(
                np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0),
                0,
                255,
            ).astype(np.uint8)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        base_mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        # 3x3 Morphology OPEN first preserves thin 3-4px yellow cup rims while rejecting isolated noise
        seed_mask = cv2.morphologyEx(base_mask, cv2.MORPH_OPEN, self._seed_kernel)

        # 2-step Geodesic Dilation constrained to washed-out specular glare candidates
        if self.enable_glare_recovery and int(self.lower_hsv[1]) > 20 and np.any(seed_mask):
            h_low = max(0, int(self.lower_hsv[0]) - 5)
            h_high = min(179, int(self.upper_hsv[0]) + 5)
            glare_lower = np.array([h_low, 15, 210], dtype=np.uint8)
            glare_upper = np.array([h_high, int(self.lower_hsv[1]) - 1, 255], dtype=np.uint8)
            glare_candidates = cv2.inRange(hsv, glare_lower, glare_upper)

            recovered = seed_mask
            for _ in range(2):
                halo = cv2.dilate(recovered, self._glare_halo_kernel)
                recovered = cv2.bitwise_or(recovered, cv2.bitwise_and(glare_candidates, halo))
            mask = recovered
        else:
            mask = seed_mask

        # 5x5 Morphology OPEN then CLOSE to reject remaining noise and seal interior gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        self._last_mask = mask.copy()
        return mask

    def get_last_mask(self):
        """Return a thread-safe copy of the most recently computed binary mask, or None."""
        with self._lock:
            return None if self._last_mask is None else self._last_mask.copy()

    def get_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate and return the binary segmented mask for debug/inspection.

        Args:
            frame: Input BGR image frame (numpy ndarray)

        Returns:
            Binary mask (numpy ndarray, uint8) where yellow pixels are 255 and rest are 0.
        """
        with self._lock:
            return self._compute_mask(frame)

    def detect(self, frame: np.ndarray, transformer=None) -> Tuple[List[int], int]:
        """
        Detect yellow objects, extract contours, filter noise by area, and map to 6 zones.

        Args:
            frame: Input BGR frame (numpy ndarray, typically 600x400 warped or raw)
            transformer: Optional PerspectiveTransformer for mapping raw frame coordinates

        Returns:
            (occupied_zones, total_people):
                occupied_zones: List of 6 integers (0 or 1) indicating zone occupancy
                total_people: Total count of valid detected objects
        """
        with self._lock:
            if (
                frame is None
                or not isinstance(frame, np.ndarray)
                or frame.size == 0
                or frame.ndim != 3
                or frame.shape[2] != 3
                or not np.issubdtype(frame.dtype, np.number)
            ):
                return [0] * 6, 0

            mask = self._compute_mask(frame)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            occupied_zones = [0] * 6
            total_people = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_contour_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                x_center = float(x + w / 2.0)
                y_max = float(y + h)

                if transformer is not None:
                    try:
                        pt_warped = transformer.transform_point(x_center, y_max)
                        x_center, y_max = float(pt_warped[0]), float(pt_warped[1])
                        if not (0.0 <= x_center <= float(WARP_WIDTH) and 0.0 <= y_max <= float(WARP_HEIGHT)):
                            continue
                    except Exception as e:
                        logger.warning(
                            "Perspective transform failed for point (%.1f, %.1f): %s",
                            x_center,
                            y_max,
                            e,
                        )

                zone_id = get_zone_id(x_center, y_max)
                if 0 <= zone_id < 6:
                    occupied_zones[zone_id] = 1
                total_people += 1

            return occupied_zones, total_people
