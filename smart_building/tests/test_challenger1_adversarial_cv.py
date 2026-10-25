"""
Challenger 1: Empirical Adversarial Computer Vision Stress Test Suite
Targeting: core/color_detector.py, core/detector.py, and core/transform.py

Coverage:
1. Extreme Image Inputs: all-white, all-black, pure noise, gradient shifts, NaN/Inf pixels, 0-dim / 1-pixel frames.
2. Exact Contour Area Thresholding: 494/499.5 px^2 (rejected) vs 500/504 px^2 (accepted).
3. Coordinate Boundary Clamping: x=0, 299, 300, 600, y=0, 132, 133, 265, 266, 400, negative & infinite coordinates.
4. Multi-Cup Stress: up to 10+ yellow cups placed simultaneously across all zones.
5. Morphological Filtering Resilience: noisy salt-and-pepper artifacts, thin lines, hollow rings.
6. OpenVINO Detector & Perspective Transformation Pipeline Integration.
7. Multi-threaded concurrency under heavy vision workload.
"""

import math
import os
import sys
import threading
import queue
import pytest
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from core.color_detector import ColorDetector
from core.detector import YOLOv8OpenVINODetector
from core.transform import get_zone_id, PerspectiveTransformer


# =============================================================================
# SUITE 1: EXTREME IMAGE INPUTS ADVERSARIAL STRESS
# =============================================================================
class TestAdversarialExtremeImageInputs:
    """Stress tests ColorDetector and YOLOv8OpenVINODetector with extreme/pathological image inputs."""

    def test_cv_stress_01_all_white_frame(self):
        """Pure white frame (255, 255, 255) must yield zero detections."""
        detector = ColorDetector()
        white_frame = np.full((400, 600, 3), 255, dtype=np.uint8)
        occupied, count = detector.detect(white_frame)
        assert occupied == [0, 0, 0, 0, 0, 0]
        assert count == 0
        mask = detector.get_mask(white_frame)
        assert np.count_nonzero(mask) == 0

    def test_cv_stress_02_all_black_frame(self):
        """Pure black frame (0, 0, 0) must yield zero detections."""
        detector = ColorDetector()
        black_frame = np.zeros((400, 600, 3), dtype=np.uint8)
        occupied, count = detector.detect(black_frame)
        assert occupied == [0, 0, 0, 0, 0, 0]
        assert count == 0
        mask = detector.get_mask(black_frame)
        assert np.count_nonzero(mask) == 0

    def test_cv_stress_03_pure_uniform_random_noise(self):
        """Random uniform RGB noise frames must not trigger false positive detections."""
        detector = ColorDetector()
        rng = np.random.default_rng(seed=101)
        for trial in range(10):
            noise_frame = rng.integers(0, 256, size=(400, 600, 3), dtype=np.uint8)
            occupied, count = detector.detect(noise_frame)
            assert occupied == [0, 0, 0, 0, 0, 0], f"False positive in trial {trial}"
            assert count == 0

    def test_cv_stress_04_gaussian_noise_frame(self):
        """High-variance Gaussian noise frames must produce zero detections."""
        detector = ColorDetector()
        rng = np.random.default_rng(seed=202)
        for mean, std in [(128, 40), (200, 30), (50, 20)]:
            noise = rng.normal(mean, std, size=(400, 600, 3))
            noise_frame = np.clip(noise, 0, 255).astype(np.uint8)
            occupied, count = detector.detect(noise_frame)
            assert occupied == [0, 0, 0, 0, 0, 0]
            assert count == 0

    def test_cv_stress_05_linear_and_radial_color_gradients(self):
        """Non-yellow color gradients (grayscale, blue-magenta, green-cyan) must produce zero detections."""
        detector = ColorDetector()

        # 1. Grayscale linear brightness gradient (0 to 255)
        gray_grad = np.tile(np.linspace(0, 255, 600, dtype=np.uint8), (400, 1))
        gray_frame = cv2.cvtColor(gray_grad, cv2.COLOR_GRAY2BGR)
        occ_gray, cnt_gray = detector.detect(gray_frame)
        assert occ_gray == [0, 0, 0, 0, 0, 0]
        assert cnt_gray == 0

        # 2. Blue-to-Magenta gradient (B=255, R ramping 0..255, G=0)
        bm_frame = np.zeros((400, 600, 3), dtype=np.uint8)
        bm_frame[:, :, 0] = 255
        for c in range(600):
            bm_frame[:, c, 2] = int((c / 600.0) * 255)
        occ_bm, cnt_bm = detector.detect(bm_frame)
        assert occ_bm == [0, 0, 0, 0, 0, 0]
        assert cnt_bm == 0

        # 3. Localized yellow cup embedded inside gradient background
        # Yellow cup @ (150, 66) -> Zone 0
        cv2.rectangle(gray_frame, (135, 50), (165, 80), (0, 255, 255), -1)
        occ_cup, cnt_cup = detector.detect(gray_frame)
        assert occ_cup == [1, 0, 0, 0, 0, 0]
        assert cnt_cup == 1

    def test_cv_stress_06_nan_and_inf_float_frames(self):
        """Float32 frames with NaN, +Inf, -Inf must be handled gracefully without crashing."""
        detector = ColorDetector()
        # All NaN
        nan_frame = np.full((400, 600, 3), np.nan, dtype=np.float32)
        occ, cnt = detector.detect(nan_frame)
        assert occ == [0, 0, 0, 0, 0, 0]
        assert cnt == 0

        # All +Inf
        inf_frame = np.full((400, 600, 3), np.inf, dtype=np.float32)
        occ, cnt = detector.detect(inf_frame)
        assert occ == [0, 0, 0, 0, 0, 0]
        assert cnt == 0

        # Mixed NaN / Inf / Floats
        mixed_frame = np.ones((400, 600, 3), dtype=np.float32) * 0.5
        mixed_frame[100:150, 100:150] = np.nan
        mixed_frame[200:250, 200:250] = np.inf
        occ, cnt = detector.detect(mixed_frame)
        assert occ == [0, 0, 0, 0, 0, 0]
        assert cnt == 0

    def test_cv_stress_07_zero_dim_and_empty_frames(self):
        """Zero-dimension and empty numpy arrays must return default ([0]*6, 0)."""
        detector = ColorDetector()
        assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == ([0, 0, 0, 0, 0, 0], 0)
        assert detector.detect(np.zeros((0, 600, 3), dtype=np.uint8)) == ([0, 0, 0, 0, 0, 0], 0)
        assert detector.detect(np.zeros((400, 0, 3), dtype=np.uint8)) == ([0, 0, 0, 0, 0, 0], 0)
        assert detector.detect(np.empty((0,), dtype=np.uint8)) == ([0, 0, 0, 0, 0, 0], 0)

    def test_cv_stress_08_single_pixel_frames(self):
        """Single pixel frames (1x1) must not trigger detection."""
        detector = ColorDetector()
        # Yellow 1x1 pixel
        yellow_1px = np.full((1, 1, 3), (0, 255, 255), dtype=np.uint8)
        assert detector.detect(yellow_1px) == ([0, 0, 0, 0, 0, 0], 0)
        # Black 1x1 pixel
        black_1px = np.zeros((1, 1, 3), dtype=np.uint8)
        assert detector.detect(black_1px) == ([0, 0, 0, 0, 0, 0], 0)

    def test_cv_stress_09_extreme_aspect_ratios_and_sizes(self):
        """Extreme aspect ratios (1x600, 600x1, 4000x3000) must execute without crash."""
        detector = ColorDetector()
        # 1x600 line
        line_h = np.zeros((1, 600, 3), dtype=np.uint8)
        assert detector.detect(line_h) == ([0, 0, 0, 0, 0, 0], 0)

        # 600x1 line
        line_v = np.zeros((600, 1, 3), dtype=np.uint8)
        assert detector.detect(line_v) == ([0, 0, 0, 0, 0, 0], 0)

        # 2x2 micro frame
        micro = np.zeros((2, 2, 3), dtype=np.uint8)
        assert detector.detect(micro) == ([0, 0, 0, 0, 0, 0], 0)

    def test_cv_stress_10_non_array_inputs(self):
        """Non-numpy or None inputs must return default zero output safely."""
        detector = ColorDetector()
        assert detector.detect(None) == ([0, 0, 0, 0, 0, 0], 0)
        assert detector.detect("invalid_string") == ([0, 0, 0, 0, 0, 0], 0)
        assert detector.detect([1, 2, 3]) == ([0, 0, 0, 0, 0, 0], 0)
        assert detector.detect({}) == ([0, 0, 0, 0, 0, 0], 0)


# =============================================================================
# SUITE 2: EXACT CONTOUR AREA THRESHOLDING ADVERSARIAL STRESS
# =============================================================================
class TestAdversarialContourAreaThresholding:
    """Stress tests strict enforcement of the >= 500 px^2 contour area filtering."""

    def test_cv_stress_11_exact_contour_area_494_rejected_vs_500_accepted(self):
        """
        Verify contour area boundary:
        - 26x19 yellow rectangle (cv2.contourArea = 494.0 px^2) is REJECTED (< 500).
        - 25x20 yellow rectangle (cv2.contourArea = 500.0 px^2) is ACCEPTED (>= 500).
        - 21x24 yellow rectangle (cv2.contourArea = 504.0 px^2) is ACCEPTED (>= 500).
        """
        detector = ColorDetector(min_contour_area=500.0)

        # 1. 26x19 rectangle -> area 494 px^2 in Zone 0
        frame_494 = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(frame_494, (100, 50), (126, 69), (0, 255, 255), -1)
        occ_494, cnt_494 = detector.detect(frame_494)
        assert occ_494 == [0, 0, 0, 0, 0, 0], f"Expected rejected, got {occ_494}"
        assert cnt_494 == 0

        # 2. 25x20 rectangle -> area 500 px^2 in Zone 0
        frame_500 = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(frame_500, (100, 50), (125, 70), (0, 255, 255), -1)
        occ_500, cnt_500 = detector.detect(frame_500)
        assert occ_500 == [1, 0, 0, 0, 0, 0], f"Expected Zone 0 occupied, got {occ_500}"
        assert cnt_500 == 1

        # 3. 21x24 rectangle -> area 504 px^2 in Zone 1
        frame_504 = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(frame_504, (400, 50), (421, 74), (0, 255, 255), -1)
        occ_504, cnt_504 = detector.detect(frame_504)
        assert occ_504 == [0, 1, 0, 0, 0, 0]
        assert cnt_504 == 1

    def test_cv_stress_12_exact_circle_disk_area_threshold(self):
        """
        Verify circular disk area threshold:
        - Circle of radius r=12 (raster area ~440 px^2) is REJECTED (< 500).
        - Circle of radius r=14 (raster area ~590 px^2) is ACCEPTED (>= 500).
        """
        detector = ColorDetector(min_contour_area=500.0)

        # Small circle (r=12) in Zone 2
        frame_c_small = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.circle(frame_c_small, (150, 200), 12, (0, 255, 255), -1)
        occ_s, cnt_s = detector.detect(frame_c_small)
        assert occ_s == [0, 0, 0, 0, 0, 0]
        assert cnt_s == 0

        # Large circle (r=14) in Zone 2
        frame_c_large = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.circle(frame_c_large, (150, 200), 14, (0, 255, 255), -1)
        occ_l, cnt_l = detector.detect(frame_c_large)
        assert occ_l == [0, 0, 1, 0, 0, 0]
        assert cnt_l == 1

    def test_cv_stress_13_irregular_polygon_area_threshold(self):
        """Verify non-rectangular polygon area filtering."""
        detector = ColorDetector(min_contour_area=500.0)

        # Triangle with base=30, height=30 -> area ~ 450 px^2 (REJECTED)
        frame_tri_small = np.zeros((400, 600, 3), dtype=np.uint8)
        tri_pts_small = np.array([[100, 100], [130, 100], [115, 70]], dtype=np.int32)
        cv2.fillPoly(frame_tri_small, [tri_pts_small], (0, 255, 255))
        occ_t1, cnt_t1 = detector.detect(frame_tri_small)
        assert cnt_t1 == 0
        assert occ_t1 == [0, 0, 0, 0, 0, 0]

        # Triangle with base=40, height=30 -> area ~ 600 px^2 (ACCEPTED)
        frame_tri_large = np.zeros((400, 600, 3), dtype=np.uint8)
        tri_pts_large = np.array([[100, 100], [140, 100], [120, 70]], dtype=np.int32)
        cv2.fillPoly(frame_tri_large, [tri_pts_large], (0, 255, 255))
        occ_t2, cnt_t2 = detector.detect(frame_tri_large)
        assert cnt_t2 == 1
        assert occ_t2[0] == 1

    def test_cv_stress_14_custom_min_contour_area_configuration(self):
        """Verify ColorDetector respects custom min_contour_area overrides."""
        # Custom 200.0
        det_200 = ColorDetector(min_contour_area=200.0)
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 50), (120, 65), (0, 255, 255), -1)  # 20x15 = 300 px^2
        occ, cnt = det_200.detect(frame)
        assert cnt == 1

        # Custom 1000.0
        det_1000 = ColorDetector(min_contour_area=1000.0)
        occ_1000, cnt_1000 = det_1000.detect(frame)  # 300 px^2 is < 1000
        assert cnt_1000 == 0
        assert occ_1000 == [0, 0, 0, 0, 0, 0]


# =============================================================================
# SUITE 3: COORDINATE BOUNDARY CLAMPING & ZONE PARTITIONS
# =============================================================================
class TestAdversarialCoordinateBoundaryClamping:
    """Stress tests exact zone partition transitions and boundary clamping."""

    def test_cv_stress_15_all_24_canonical_border_intersections(self):
        """
        Verify zone lookup across exact specified boundary coordinates:
        x in [0, 299, 300, 600]
        y in [0, 132, 133, 265, 266, 400]
        """
        expected_matrix = {
            # y = 0 (Row 0)
            (0, 0): 0, (299, 0): 0, (300, 0): 1, (600, 0): 1,
            # y = 132 (Row 0)
            (0, 132): 0, (299, 132): 0, (300, 132): 1, (600, 132): 1,
            # y = 133 (Row 1)
            (0, 133): 2, (299, 133): 2, (300, 133): 3, (600, 133): 3,
            # y = 265 (Row 1)
            (0, 265): 2, (299, 265): 2, (300, 265): 3, (600, 265): 3,
            # y = 266 (Row 2)
            (0, 266): 4, (299, 266): 4, (300, 266): 5, (600, 266): 5,
            # y = 400 (Row 2)
            (0, 400): 4, (299, 400): 4, (300, 400): 5, (600, 400): 5,
        }

        for (x, y), expected_zone in expected_matrix.items():
            actual = get_zone_id(x, y)
            assert actual == expected_zone, f"Failed at ({x}, {y}): expected {expected_zone}, got {actual}"

    def test_cv_stress_16_micro_epsilon_zone_transitions(self):
        """Verify epsilon perturbations across row and column boundaries."""
        eps = 1e-6
        # Col boundary at x=300
        assert get_zone_id(300.0 - eps, 50) == 0
        assert get_zone_id(300.0, 50) == 1
        assert get_zone_id(300.0 + eps, 50) == 1

        # Row boundary at y=133
        assert get_zone_id(150, 133.0 - eps) == 0
        assert get_zone_id(150, 133.0) == 2
        assert get_zone_id(150, 133.0 + eps) == 2

        # Row boundary at y=266
        assert get_zone_id(150, 266.0 - eps) == 2
        assert get_zone_id(150, 266.0) == 4
        assert get_zone_id(150, 266.0 + eps) == 4

    def test_cv_stress_17_extreme_out_of_bounds_clamping(self):
        """Verify robust clamping for extreme out-of-bounds coordinates."""
        # Far negative -> Zone 0 (Row 0, Col 0)
        assert get_zone_id(-1.0, -1.0) == 0
        assert get_zone_id(-999999.0, -999999.0) == 0

        # Far top-right -> Zone 1 (Row 0, Col 1)
        assert get_zone_id(10000.0, -500.0) == 1

        # Far bottom-left -> Zone 4 (Row 2, Col 0)
        assert get_zone_id(-500.0, 10000.0) == 4

        # Far bottom-right -> Zone 5 (Row 2, Col 1)
        assert get_zone_id(10000.0, 10000.0) == 5

    def test_cv_stress_18_special_float_and_nan_invariants(self):
        """Verify NaN, +/- Inf, string inputs, None return safe zone IDs in range [0..5]."""
        assert get_zone_id(float('nan'), float('nan')) == 0
        assert get_zone_id(float('nan'), 200.0) == 0
        assert get_zone_id(200.0, float('nan')) == 0
        assert get_zone_id(float('-inf'), float('-inf')) == 0
        assert get_zone_id(float('inf'), float('inf')) == 5
        assert get_zone_id(float('inf'), 50.0) == 1
        assert get_zone_id(float('-inf'), 350.0) == 4
        assert get_zone_id(None, None) == 0
        assert get_zone_id("150", "200") == 2
        assert get_zone_id("bad_coord", 200) == 0


# =============================================================================
# SUITE 4: MULTI-CUP STRESS TESTING
# =============================================================================
class TestAdversarialMultiCupStress:
    """Stress tests detection with multiple yellow cups placed simultaneously."""

    def test_cv_stress_19_ten_cups_spread_all_zones(self):
        """10 distinct yellow cups distributed across all 6 zones must be fully detected."""
        detector = ColorDetector(min_contour_area=500.0)
        frame = np.zeros((400, 600, 3), dtype=np.uint8)

        cup_positions = [
            (50, 40), (180, 40),     # Zone 0 (2 cups)
            (350, 40), (500, 40),    # Zone 1 (2 cups)
            (100, 180),              # Zone 2 (1 cup)
            (350, 180), (500, 180),  # Zone 3 (2 cups)
            (100, 320),              # Zone 4 (1 cup)
            (350, 320), (500, 320)   # Zone 5 (2 cups)
        ]

        for cx, cy in cup_positions:
            cv2.rectangle(frame, (cx - 15, cy - 15), (cx + 15, cy + 15), (0, 255, 255), -1)

        occupied, total = detector.detect(frame)
        assert occupied == [1, 1, 1, 1, 1, 1], f"Expected all zones occupied, got {occupied}"
        assert total == 10, f"Expected 10 cups, got {total}"

    def test_cv_stress_20_ten_cups_concentrated_in_single_zone(self):
        """10 distinct yellow cups all located inside Zone 0."""
        detector = ColorDetector(min_contour_area=500.0)
        frame = np.zeros((400, 600, 3), dtype=np.uint8)

        # 5x2 grid inside Zone 0 [0..300, 0..133]
        for i in range(10):
            gx = 20 + (i % 5) * 50
            gy = 20 + (i // 5) * 50
            cv2.rectangle(frame, (gx, gy), (gx + 25, gy + 25), (0, 255, 255), -1)

        occupied, total = detector.detect(frame)
        assert occupied == [1, 0, 0, 0, 0, 0]
        assert total == 10

    def test_cv_stress_21_twenty_four_cups_dense_grid(self):
        """24 cups (4 cups in each of the 6 zones) densely populated across the 600x400 model."""
        detector = ColorDetector(min_contour_area=500.0)
        frame = np.zeros((400, 600, 3), dtype=np.uint8)

        # Zone origins:
        zone_origins = [
            (0, 0), (300, 0),        # Row 0
            (0, 133), (300, 133),    # Row 1
            (0, 266), (300, 266),    # Row 2
        ]

        for zx, zy in zone_origins:
            # 4 cups in this zone (2x2 mini grid)
            for cx_off, cy_off in [(40, 25), (180, 25), (40, 75), (180, 75)]:
                x = zx + cx_off
                y = zy + cy_off
                cv2.rectangle(frame, (x, y), (x + 25, y + 25), (0, 255, 255), -1)

        occupied, total = detector.detect(frame)
        assert occupied == [1, 1, 1, 1, 1, 1]
        assert total == 24


# =============================================================================
# SUITE 5: MORPHOLOGICAL FILTERING & NOISE RESILIENCE
# =============================================================================
class TestAdversarialMorphologicalResilience:
    """Stress tests noise filtering and morphology operations (OPEN/CLOSE)."""

    def test_cv_stress_22_salt_and_pepper_dense_noise_resilience(self):
        """5,000 salt-and-pepper yellow noise pixels must be eliminated by morphology and area threshold."""
        detector = ColorDetector(min_contour_area=500.0)
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        rng = np.random.default_rng(seed=999)

        # Scatter 5,000 random yellow pixels
        xs = rng.integers(0, 600, size=5000)
        ys = rng.integers(0, 400, size=5000)
        for x, y in zip(xs, ys):
            frame[y, x] = [0, 255, 255]

        # Alone: 0 detections
        occ, cnt = detector.detect(frame)
        assert occ == [0, 0, 0, 0, 0, 0]
        assert cnt == 0

        # With 1 genuine yellow cup in Zone 3
        cv2.rectangle(frame, (435, 185), (465, 215), (0, 255, 255), -1)
        occ_cup, cnt_cup = detector.detect(frame)
        assert occ_cup == [0, 0, 0, 1, 0, 0]
        assert cnt_cup == 1

    def test_cv_stress_23_thin_yellow_lines_eroded_by_morphology(self):
        """Thin 1-2px yellow lines across the canvas must be eroded by 5x5 MORPH_OPEN."""
        detector = ColorDetector(min_contour_area=500.0)
        frame = np.zeros((400, 600, 3), dtype=np.uint8)

        # 1px horizontal and vertical lines across canvas
        cv2.line(frame, (0, 200), (599, 200), (0, 255, 255), 1)
        cv2.line(frame, (300, 0), (300, 399), (0, 255, 255), 1)

        occ, cnt = detector.detect(frame)
        assert occ == [0, 0, 0, 0, 0, 0]
        assert cnt == 0

    def test_cv_stress_24_hollow_annulus_cup_marker_morphology_behavior(self):
        """
        Verify structural filtering on annular shapes:
        - Thin ring (< 5px wall): eroded by 5x5 MORPH_OPEN -> 0 detections (rejects thin circular artifacts).
        - Thick ring (>= 6px wall): survives 5x5 MORPH_OPEN and is bridged by MORPH_CLOSE -> 1 detection.
        """
        detector = ColorDetector(min_contour_area=500.0)

        # 1. Thin ring (outer r=20, inner r=16 -> 4px wall) -> Eroded by 5x5 kernel
        frame_thin = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.circle(frame_thin, (150, 330), 20, (0, 255, 255), -1)
        cv2.circle(frame_thin, (150, 330), 16, (0, 0, 0), -1)
        occ_thin, cnt_thin = detector.detect(frame_thin)
        assert occ_thin == [0, 0, 0, 0, 0, 0]
        assert cnt_thin == 0

        # 2. Thick ring (outer r=22, inner r=16 -> 6px wall) in Zone 4 (Row 2, Col 0: cx=150, cy=330)
        frame_thick = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.circle(frame_thick, (150, 330), 22, (0, 255, 255), -1)
        cv2.circle(frame_thick, (150, 330), 16, (0, 0, 0), -1)
        occ_thick, cnt_thick = detector.detect(frame_thick)
        assert occ_thick == [0, 0, 0, 0, 1, 0]
        assert cnt_thick == 1


# =============================================================================
# SUITE 6: DETECTOR & PERSPECTIVE PIPELINE INTEGRATION
# =============================================================================
class TestAdversarialPipelineIntegration:
    """Stress tests YOLOv8OpenVINODetector, PerspectiveTransformer, and cross-module integration."""

    def test_cv_stress_25_color_detector_with_perspective_transformer(self):
        """Verify ColorDetector seamlessly transforms raw 640x480 coordinates through homography."""
        src_pts = np.array([[140, 90], [500, 90], [580, 410], [60, 410]], dtype=np.float32)
        dst_pts = np.array([[0, 0], [600, 0], [600, 400], [0, 400]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        transformer = PerspectiveTransformer()
        transformer.matrix = M

        detector = ColorDetector(min_contour_area=500.0)

        # Place yellow cup in native frame at (320, 250) -> maps near center of 600x400 (Zone 2 or 3)
        raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(raw_frame, (300, 230), (340, 270), (0, 255, 255), -1)

        occupied, count = detector.detect(raw_frame, transformer=transformer)
        assert count == 1
        assert sum(occupied) == 1
        assert occupied[2] == 1 or occupied[3] == 1

    def test_cv_stress_26_transformer_singular_matrix_resilience(self):
        """Singular / degenerate perspective matrix must not crash transformer or detector."""
        transformer = PerspectiveTransformer()
        # Singular matrix (determinant == 0)
        transformer.matrix = np.zeros((3, 3), dtype=np.float32)

        pt = transformer.transform_point(100.0, 100.0)
        assert isinstance(pt, tuple) and len(pt) == 2

        # Detector should catch transform exception and fall back cleanly
        detector = ColorDetector()
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 50), (150, 90), (0, 255, 255), -1)
        occ, cnt = detector.detect(frame, transformer=transformer)
        assert cnt == 1
        assert sum(occ) == 1

    def test_cv_stress_27_openvino_yolo_detector_simulation_and_real_mode(self):
        """Verify YOLOv8OpenVINODetector functions safely in both simulation and real mode."""
        # Simulation mode
        mock_det = YOLOv8OpenVINODetector(is_simulation=True)
        assert not mock_det.use_real_model
        occ_m, cnt_m = mock_det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert occ_m == [0, 0, 0, 0, 0, 0]
        assert cnt_m == 0

        # Real model mode
        real_det = YOLOv8OpenVINODetector(is_simulation=False)
        if real_det.use_real_model:
            occ_r, cnt_r = real_det.detect(np.zeros((480, 640, 3), dtype=np.uint8))
            assert isinstance(occ_r, list) and len(occ_r) == 6
            assert isinstance(cnt_r, int)

    def test_cv_stress_28_concurrent_multithreaded_vision_inference(self):
        """20 concurrent threads hammering ColorDetector.detect and get_mask simultaneously."""
        detector = ColorDetector()
        stop_event = threading.Event()
        errors = queue.Queue()

        def vision_worker(thread_id):
            try:
                for i in range(50):
                    if stop_event.is_set():
                        break
                    # Alternating test frames
                    if (thread_id + i) % 2 == 0:
                        frame = np.zeros((400, 600, 3), dtype=np.uint8)
                        cv2.rectangle(frame, (100, 50), (140, 85), (0, 255, 255), -1)
                    else:
                        frame = np.full((400, 600, 3), 255, dtype=np.uint8)

                    occ, cnt = detector.detect(frame)
                    mask = detector.get_mask(frame)
                    assert len(occ) == 6
                    assert mask.shape == (400, 600)
            except Exception as e:
                errors.put((thread_id, e))

        threads = [threading.Thread(target=vision_worker, args=(tid,)) for tid in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive(), "Vision thread deadlock detected"

        assert errors.empty(), f"Thread errors: {[errors.get() for _ in range(errors.qsize())]}"
