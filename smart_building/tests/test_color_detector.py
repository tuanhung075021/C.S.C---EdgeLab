"""
Unit and Integration Tests for ColorDetector Module (Milestone 2)
EcoZone AI Smart Energy Management System
"""

import os
import sys
import threading
import numpy as np
import cv2
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core import ColorDetector
from core.transform import PerspectiveTransformer, get_zone_id


class TestColorDetector:
    """Test suite covering ColorDetector functionality, noise filtering, transformation, and edge cases."""

    def test_init_defaults(self):
        """Verify ColorDetector initializes with expected default bounds and area."""
        detector = ColorDetector()
        assert np.array_equal(detector.lower_hsv, [20, 100, 100])
        assert np.array_equal(detector.upper_hsv, [35, 255, 255])
        assert detector.min_contour_area == 500.0

    def test_init_custom_parameters(self):
        """Verify ColorDetector accepts custom HSV bounds and min_contour_area."""
        detector = ColorDetector(
            lower_hsv=[15, 80, 80],
            upper_hsv=[40, 255, 255],
            min_contour_area=300.0,
            morph_kernel_size=(3, 3)
        )
        assert np.array_equal(detector.lower_hsv, [15, 80, 80])
        assert np.array_equal(detector.upper_hsv, [40, 255, 255])
        assert detector.min_contour_area == 300.0

    def test_detect_none_and_empty_frame(self):
        """Verify None or empty frame returns zeros and 0 count without exception."""
        detector = ColorDetector()
        zones, count = detector.detect(None)
        assert zones == [0, 0, 0, 0, 0, 0]
        assert count == 0

        empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
        zones2, count2 = detector.detect(empty_frame)
        assert zones2 == [0, 0, 0, 0, 0, 0]
        assert count2 == 0

    def test_detect_blank_grey_frame(self):
        """Verify blank grey frame (no yellow) returns all 0 zones and total_people = 0."""
        detector = ColorDetector()
        frame = np.full((400, 600, 3), 128, dtype=np.uint8)
        zones, count = detector.detect(frame)
        assert zones == [0, 0, 0, 0, 0, 0]
        assert count == 0

    def test_detect_single_yellow_rectangle_zone_0(self):
        """
        Verify synthetic yellow rectangle in left column top row (Zone 0):
        x=100, y=50, w=50, h=50 (area = 2500 >= 500)
        x_center = 125, y_max = 100 -> Zone 0
        """
        detector = ColorDetector()
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        # BGR for Yellow is (0, 255, 255)
        cv2.rectangle(frame, (100, 50), (150, 100), (0, 255, 255), -1)

        zones, count = detector.detect(frame)
        assert zones == [1, 0, 0, 0, 0, 0]
        assert count == 1

    def test_detect_noise_contour_below_min_area(self):
        """
        Verify synthetic yellow rectangle with area 400px^2 (<500px^2) is discarded as noise:
        w=20, h=20 -> area = 400
        """
        detector = ColorDetector(min_contour_area=500.0)
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 50), (120, 70), (0, 255, 255), -1)

        zones, count = detector.detect(frame)
        assert zones == [0, 0, 0, 0, 0, 0]
        assert count == 0

    def test_detect_multiple_yellow_rectangles_across_zones(self):
        """
        Verify multiple yellow rectangles in different zones:
        - Zone 0 (Row 0, Col 0): x=50, y=30, w=40, h=40 (x_center=70, y_max=70)
        - Zone 3 (Row 1, Col 1): x=350, y=150, w=40, h=40 (x_center=370, y_max=190)
        - Zone 4 (Row 2, Col 0): x=100, y=280, w=40, h=40 (x_center=120, y_max=320)
        - Zone 5 (Row 2, Col 1): x=450, y=300, w=40, h=40 (x_center=470, y_max=340)
        Total 4 objects, zones [1, 0, 0, 1, 1, 1]
        """
        detector = ColorDetector(min_contour_area=500.0)
        frame = np.zeros((400, 600, 3), dtype=np.uint8)

        # 40x40 = 1600 px^2 each
        cv2.rectangle(frame, (50, 30), (90, 70), (0, 255, 255), -1)
        cv2.rectangle(frame, (350, 150), (390, 190), (0, 255, 255), -1)
        cv2.rectangle(frame, (100, 280), (140, 320), (0, 255, 255), -1)
        cv2.rectangle(frame, (450, 300), (490, 340), (0, 255, 255), -1)

        zones, count = detector.detect(frame)
        assert zones == [1, 0, 0, 1, 1, 1]
        assert count == 4

    def test_detect_with_perspective_transformer(self):
        """
        Verify detection when PerspectiveTransformer is passed.
        Draw yellow shape in raw frame coordinates and verify transformed zone mapping.
        """
        # Identity transformer first
        transformer = PerspectiveTransformer()
        detector = ColorDetector()
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(frame, (400, 50), (460, 110), (0, 255, 255), -1)

        zones, count = detector.detect(frame, transformer=transformer)
        assert zones == [0, 1, 0, 0, 0, 0]
        assert count == 1

    def test_get_mask(self):
        """Verify get_mask returns binary mask with 255 on yellow pixels and 0 elsewhere."""
        detector = ColorDetector()
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 50), (200, 150), (0, 255, 255), -1)

        mask = detector.get_mask(frame)
        assert mask.shape == (400, 600)
        assert mask.dtype == np.uint8
        # Center of rectangle should be 255 (yellow)
        assert mask[100, 150] == 255
        # Background pixel should be 0
        assert mask[10, 10] == 0

    def test_get_mask_none_frame(self):
        """Verify get_mask on None frame returns zero array."""
        detector = ColorDetector()
        mask = detector.get_mask(None)
        assert mask.shape == (400, 600)
        assert np.all(mask == 0)

    def test_thread_safety_concurrent_detect(self):
        """Verify concurrent detect calls across multiple threads produce consistent results."""
        detector = ColorDetector()
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(frame, (50, 30), (120, 100), (0, 255, 255), -1)  # Zone 0

        errors = []

        def worker():
            for _ in range(50):
                try:
                    zones, count = detector.detect(frame)
                    assert zones == [1, 0, 0, 0, 0, 0]
                    assert count == 1
                    mask = detector.get_mask(frame)
                    assert mask.shape == (400, 600)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
