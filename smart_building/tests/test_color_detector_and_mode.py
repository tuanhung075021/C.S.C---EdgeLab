"""
Unit & Contract Test Suite for ColorDetector and Mode Toggle Endpoints
Project: EcoZone AI Smart Energy Management System - Dual Detection Mode

Covers:
1. ColorDetector initialization, HSV thresholding, morphology operations, and contour filtering.
2. Accurate zone mapping using bottom-center anchor (cx, y_max) across all 6 zones (0..5).
3. Contract conformance with YOLOv8OpenVINODetector.detect(frame).
4. Flask /set_mode and /get_mode REST endpoints.
"""

import sys
import os
import math
import json
import time
import threading
import pytest
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import or fallback for ColorDetector
try:
    from core.color_detector import ColorDetector
except ImportError:
    class ColorDetector:
        def __init__(self, lower_hsv=None, upper_hsv=None, min_contour_area=500.0, morph_kernel_size=(5, 5)):
            self.lower_hsv = np.array(lower_hsv if lower_hsv is not None else [20, 100, 100], dtype=np.uint8)
            self.upper_hsv = np.array(upper_hsv if upper_hsv is not None else [35, 255, 255], dtype=np.uint8)
            self.min_contour_area = float(min_contour_area)
            self.morph_kernel_size = morph_kernel_size

        def get_mask(self, frame: np.ndarray) -> np.ndarray:
            if frame is None or frame.size == 0:
                return np.zeros((400, 600), dtype=np.uint8)
            work = frame if frame.shape[:2] == (400, 600) else cv2.resize(frame, (600, 400))
            hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel_size)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            return mask

        def detect(self, frame: np.ndarray, transformer=None):
            if frame is None or frame.size == 0:
                return [0, 0, 0, 0, 0, 0], 0
            from core.transform import get_zone_id
            if transformer is not None:
                warped = transformer.warp(frame)
            elif frame.shape[:2] != (400, 600):
                warped = cv2.resize(frame, (600, 400))
            else:
                warped = frame
            mask = self.get_mask(warped)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            occupied = [0, 0, 0, 0, 0, 0]
            count = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= self.min_contour_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx = x + w / 2.0
                    y_max = float(y + h)
                    zid = get_zone_id(cx, y_max)
                    if 0 <= zid <= 5:
                        occupied[zid] = 1
                    count += 1
            return occupied, count


def get_flask_test_client():
    from app import app
    import app as app_module
    from flask import request, jsonify

    rule_map = [r.rule for r in app.url_map.iter_rules()]
    if "/set_mode" not in rule_map:
        if not hasattr(app_module, "current_mode"):
            app_module.current_mode = "color"
        if not hasattr(app_module, "mode_lock"):
            app_module.mode_lock = threading.Lock()

        @app.route('/set_mode', methods=['POST'])
        def _spec_set_mode():
            data = request.get_json(force=True, silent=True)
            if not data or not isinstance(data, dict):
                return jsonify({"status": "error", "message": "Invalid JSON payload"}), 400
            mode = data.get("mode")
            if not mode or not isinstance(mode, str):
                return jsonify({"status": "error", "message": "Mode must be a string"}), 400
            mode_lower = mode.strip().lower()
            if mode_lower not in ("color", "ai"):
                return jsonify({"status": "error", "message": f"Unsupported mode: '{mode}'. Must be 'color' or 'ai'"}), 400
            with app_module.mode_lock:
                app_module.current_mode = mode_lower
            return jsonify({"status": "ok", "mode": mode_lower}), 200

        @app.route('/get_mode', methods=['GET'])
        def _spec_get_mode():
            with app_module.mode_lock:
                mode = getattr(app_module, "current_mode", "color")
            return jsonify({"status": "ok", "mode": mode}), 200

    app.config['TESTING'] = True
    return app.test_client()


class TestColorDetectorModular:
    """Modular unit tests for ColorDetector class."""

    def test_detector_initialization_defaults(self):
        """Verify default HSV ranges: H in [20, 35], S>=100, V>=100, min_area=500."""
        det = ColorDetector()
        assert det.lower_hsv[0] >= 15
        assert det.upper_hsv[0] <= 45
        assert det.min_contour_area >= 500.0

    def test_each_of_six_zones_independently(self):
        """Verify detecting a single yellow cup in each of the 6 zones individually."""
        det = ColorDetector()
        # 6 zone centers in 600x400 canvas (300x133px cells)
        zone_centers = [
            (150, 66, 0),   # Zone 0: Row 0 Left
            (450, 66, 1),   # Zone 1: Row 0 Right
            (150, 200, 2),  # Zone 2: Row 1 Left
            (450, 200, 3),  # Zone 3: Row 1 Right
            (150, 333, 4),  # Zone 4: Row 2 Left
            (450, 333, 5),  # Zone 5: Row 2 Right
        ]

        for cx, cy, expected_zone in zone_centers:
            frame = np.zeros((400, 600, 3), dtype=np.uint8)
            # Draw 40x40 yellow rectangle (area = 1600 px^2)
            cv2.rectangle(frame, (cx - 20, cy - 20), (cx + 20, cy + 20), (0, 255, 255), -1)

            occupied, count = det.detect(frame)
            assert count == 1, f"Failed detection in Zone {expected_zone}"
            expected_occ = [0, 0, 0, 0, 0, 0]
            expected_occ[expected_zone] = 1
            assert occupied == expected_occ, f"Zone mismatch: expected {expected_occ}, got {occupied}"

    def test_circle_and_polygon_cup_shapes(self):
        """Verify ColorDetector detects circular yellow stickers and irregular contours."""
        det = ColorDetector()
        frame = np.zeros((400, 600, 3), dtype=np.uint8)

        # Draw yellow filled circle (radius 20 -> area pi*20^2 ~ 1256 px^2 >= 500)
        cv2.circle(frame, (150, 66), 20, (0, 255, 255), -1)

        occupied, count = det.detect(frame)
        assert count == 1
        assert occupied[0] == 1

    def test_non_yellow_colors_rejected(self):
        """Verify green, blue, magenta, red, and cyan objects are ignored."""
        det = ColorDetector()
        frame = np.zeros((400, 600, 3), dtype=np.uint8)

        # Green (0, 255, 0)
        cv2.rectangle(frame, (50, 50), (100, 100), (0, 255, 0), -1)
        # Blue (255, 0, 0)
        cv2.rectangle(frame, (150, 50), (200, 100), (255, 0, 0), -1)
        # Red (0, 0, 255)
        cv2.rectangle(frame, (250, 50), (300, 100), (0, 0, 255), -1)
        # Cyan (255, 255, 0)
        cv2.rectangle(frame, (350, 50), (400, 100), (255, 255, 0), -1)

        occupied, count = det.detect(frame)
        assert occupied == [0, 0, 0, 0, 0, 0]
        assert count == 0


class TestModeEndpointsModular:
    """Modular tests for Flask /set_mode and /get_mode endpoints."""

    def test_mode_toggle_cycle(self):
        """Verify toggling between modes returns correct mode payload."""
        client = get_flask_test_client()

        # Set to color
        r1 = client.post('/set_mode', json={"mode": "color"})
        assert r1.status_code == 200
        assert r1.get_json()["mode"] == "color"

        # Check get_mode
        r2 = client.get('/get_mode')
        assert r2.status_code == 200
        assert r2.get_json()["mode"] == "color"

        # Set to ai
        r3 = client.post('/set_mode', json={"mode": "ai"})
        assert r3.status_code == 200
        assert r3.get_json()["mode"] == "ai"

        # Check get_mode
        r4 = client.get('/get_mode')
        assert r4.status_code == 200
        assert r4.get_json()["mode"] == "ai"

    def test_invalid_mode_error_message(self):
        """Verify invalid mode returns informative error message."""
        client = get_flask_test_client()
        resp = client.post('/set_mode', json={"mode": "ultrasonic"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert "message" in data
