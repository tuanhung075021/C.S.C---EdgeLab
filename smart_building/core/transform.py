"""
Perspective Transform & Spatial Zone Mapper Module for EcoZone AI.
Provides Homography warping to 600x400 top-down view and zone ID lookup.
"""

import math
import logging
from typing import Optional, Tuple
import numpy as np
import cv2

try:
    import config
    WARP_WIDTH = getattr(config, "WARP_WIDTH", 600)
    WARP_HEIGHT = getattr(config, "WARP_HEIGHT", 400)
    GRID_ROWS = getattr(config, "GRID_ROWS", 3)
    GRID_COLS = getattr(config, "GRID_COLS", 2)
    CELL_WIDTH = getattr(config, "CELL_WIDTH", 300)
    CELL_HEIGHT = getattr(config, "CELL_HEIGHT", 133)
    MATRIX_PATH = getattr(config, "PERSPECTIVE_MATRIX_PATH", "perspective_matrix.npy")
except ImportError:
    try:
        from .. import config  # type: ignore
        WARP_WIDTH = getattr(config, "WARP_WIDTH", 600)
        WARP_HEIGHT = getattr(config, "WARP_HEIGHT", 400)
        GRID_ROWS = getattr(config, "GRID_ROWS", 3)
        GRID_COLS = getattr(config, "GRID_COLS", 2)
        CELL_WIDTH = getattr(config, "CELL_WIDTH", 300)
        CELL_HEIGHT = getattr(config, "CELL_HEIGHT", 133)
        MATRIX_PATH = getattr(config, "PERSPECTIVE_MATRIX_PATH", "perspective_matrix.npy")
    except ImportError:
        WARP_WIDTH = 600
        WARP_HEIGHT = 400
        GRID_ROWS = 3
        GRID_COLS = 2
        CELL_WIDTH = 300
        CELL_HEIGHT = 133
        MATRIX_PATH = "perspective_matrix.npy"

logger = logging.getLogger(__name__)


def get_zone_id(
    x_center: float,
    y_max: float,
    grid_rows: int = GRID_ROWS,
    grid_cols: int = GRID_COLS,
    cell_width: int = CELL_WIDTH,
    cell_height: int = CELL_HEIGHT,
) -> int:
    """
    Calculate the occupied zone ID (0..5) for a given bounding box anchor point (x_center, y_max).
    Maps pixel coordinates in 600x400 warped canvas to a 3-row x 2-column grid.

    Grid Layout:
        Zone 0: Row 0, Col 0 (Top-Left / Row0-Left)       [0..300, 0..133]
        Zone 1: Row 0, Col 1 (Top-Right / Row0-Right)     [300..600, 0..133]
        Zone 2: Row 1, Col 0 (Middle-Left / Row1-Left)    [0..300, 133..266]
        Zone 3: Row 1, Col 1 (Middle-Right / Row1-Right)  [300..600, 133..266]
        Zone 4: Row 2, Col 0 (Bottom-Left / Row2-Left)    [0..300, 266..400]
        Zone 5: Row 2, Col 1 (Bottom-Right / Row2-Right)  [300..600, 266..400]

    Formula:
        col = floor(x_center / cell_width), clamped to [0, grid_cols - 1] (i.e. [0, 1])
        row = floor(y_max / cell_height), clamped to [0, grid_rows - 1] (i.e. [0, 2])
        zone_id = row * grid_cols + col, clamped to [0, grid_rows * grid_cols - 1] (i.e. [0, 5])
    """
    if x_center is None or y_max is None:
        return 0

    if cell_width <= 0 or cell_height <= 0 or grid_rows <= 0 or grid_cols <= 0:
        return 0

    try:
        x = float(x_center)
        y = float(y_max)
    except (ValueError, TypeError):
        return 0

    if math.isnan(x) or math.isnan(y):
        return 0

    # Determine column with infinite handling
    if math.isinf(x):
        col = 0 if x < 0 else grid_cols - 1
    else:
        col = int(math.floor(x / float(cell_width)))

    # Determine row with infinite handling
    if math.isinf(y):
        row = 0 if y < 0 else grid_rows - 1
    else:
        row = int(math.floor(y / float(cell_height)))

    # Clamp col and row within grid bounds
    col_clamped = max(0, min(grid_cols - 1, col))
    row_clamped = max(0, min(grid_rows - 1, row))

    zone_id = row_clamped * grid_cols + col_clamped
    max_zone = grid_rows * grid_cols - 1
    return max(0, min(max_zone, zone_id))


class PerspectiveTransformer:
    """
    Handles loading the 3x3 Homography perspective matrix and warping input frames to 600x400.
    Falls back to 3x3 identity matrix if matrix file is missing or invalid.
    """

    def __init__(self, matrix_path: Optional[str] = None) -> None:
        self.matrix_path = matrix_path if matrix_path is not None else MATRIX_PATH
        self.output_size = (WARP_WIDTH, WARP_HEIGHT)
        self.M = self._load_matrix(self.matrix_path)

    @property
    def matrix(self) -> np.ndarray:
        """Alias for self.M providing access to the 3x3 transform matrix."""
        return self.M

    @matrix.setter
    def matrix(self, val: np.ndarray) -> None:
        self.M = val

    def _load_matrix(self, path: str) -> np.ndarray:
        try:
            M = np.load(path)
            if isinstance(M, np.ndarray) and M.shape == (3, 3):
                logger.info("Successfully loaded perspective matrix from %s", path)
                return M.astype(np.float32)
            else:
                logger.warning("Invalid matrix shape in %s. Using identity matrix fallback.", path)
        except Exception as e:
            logger.warning("Could not load perspective matrix from %s (%s). Using identity matrix fallback.", path, e)

        return np.eye(3, dtype=np.float32)

    def warp(self, frame: np.ndarray) -> np.ndarray:
        """
        Warps incoming BGR frame to 600x400 perspective canvas.
        Returns 600x400 BGR numpy array.
        """
        if frame is None or frame.size == 0:
            return np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8)

        return cv2.warpPerspective(frame, self.M, self.output_size)

    def transform_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transforms a single 2D coordinate pair (x, y) through the 3x3 homography matrix.
        Returns (warped_x, warped_y).
        """
        try:
            px = float(x)
            py = float(y)
        except (ValueError, TypeError):
            return (0.0, 0.0)

        vec = np.array([px, py, 1.0], dtype=np.float32)
        transformed = self.M @ vec
        w = transformed[2]
        if abs(w) > 1e-7:
            return (float(transformed[0] / w), float(transformed[1] / w))
        return (float(transformed[0]), float(transformed[1]))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transforms an array of 2D points with shape (N, 2) through the homography matrix.
        Returns warped points with shape (N, 2).
        """
        if points is None or len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        warped = cv2.perspectiveTransform(pts, self.M)
        return warped.reshape(-1, 2)
