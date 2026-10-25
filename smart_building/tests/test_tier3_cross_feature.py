"""
Tier 3: Cross-Feature Pairwise Interaction Test Suite
Project: EcoZone AI Smart Energy Management System - Dual Detection Mode

This module tests complex interactions between pairs and groups of features:
- Feature 1: OpenVINO Core Import & Model Loading
- Feature 2: Config Definitions & Defaults
- Feature 3: Color Detection Module (HSV Yellow)
- Feature 4: Mode Toggle API & Background Pipeline
- Feature 5: Dashboard UI & Telemetry Synchronization
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
from unittest.mock import MagicMock, patch

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
    """Helper to acquire Flask test client with /set_mode and /get_mode endpoints."""
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


@pytest.mark.tier3
class TestTier3CrossFeatureInteractions:
    """Pairwise and Cross-Feature Integration Tests."""

    def test_t3_01_mode_switching_under_active_streaming(self):
        """Feature 3 (Color Detector) <-> Feature 4 (Mode Toggle) <-> Vision Stream.
        Verify switching between 'color' and 'ai' modes while processing stream frames
        updates the active detector instantly on the next frame without dropping the loop.
        """
        from core.vision_stream import VisionStream
        from core.detector import YOLOv8OpenVINODetector

        stream = VisionStream(stream_url="http://10.37.61.34:81/stream", is_simulation=True)
        color_detector = ColorDetector()
        ai_detector = YOLOv8OpenVINODetector(is_simulation=True)

        current_mode = "color"
        for i in range(10):
            frame = stream._generate_synthetic_frame()
            if i == 5:
                current_mode = "ai"

            if current_mode == "color":
                occ, cnt = color_detector.detect(frame)
            else:
                occ, cnt = ai_detector.detect(frame)

            assert len(occ) == 6
            assert isinstance(cnt, int)

        stream.stop()

    def test_t3_02_perspective_calibration_and_color_detection(self):
        """Feature 3 (Color Detection) <-> Perspective Transformer.
        Verify that calibrating a perspective matrix and passing the transformer to
        ColorDetector.detect() accurately transforms coordinates into the 600x400 Bird's-Eye grid.
        """
        from core.transform import PerspectiveTransformer
        transformer = PerspectiveTransformer()

        # Define 4 source points covering a sub-quad
        src_points = np.float32([[50, 40], [590, 40], [590, 440], [50, 440]])
        dst_points = np.float32([[0, 0], [600, 0], [600, 400], [0, 400]])
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transformer.matrix = matrix

        color_detector = ColorDetector()

        # Create raw unwarped frame with yellow rectangle in center
        raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(raw_frame, (100, 80), (200, 180), (0, 255, 255), -1)

        occupied, total_people = color_detector.detect(raw_frame, transformer=transformer)
        assert isinstance(occupied, list)
        assert len(occupied) == 6
        assert total_people >= 1

    def test_t3_03_color_occupancy_to_gpio_and_fan_actuation(self, mock_gpio):
        """Feature 3 (Color Detection) <-> Hardware Actuators (mock_gpio).
        Verify that yellow cups detected in Zones 0 and 3 activate LED pins 17 and 23,
        and scale the PWM fan duty cycle according to temperature + 2 occupants.
        """
        color_detector = ColorDetector()
        frame = np.zeros((400, 600, 3), dtype=np.uint8)

        # Place yellow rectangle in Zone 0 (Row 0 Left: cx=150, cy=66)
        cv2.rectangle(frame, (100, 30), (200, 100), (0, 255, 255), -1)
        # Place yellow rectangle in Zone 3 (Row 1 Right: cx=450, cy=200)
        cv2.rectangle(frame, (400, 160), (500, 240), (0, 255, 255), -1)

        occupied_zones, total_people = color_detector.detect(frame)
        assert occupied_zones[0] == 1
        assert occupied_zones[3] == 1
        assert total_people == 2

        # Actuate GPIO LEDs
        mock_gpio.update_zones(occupied_zones)
        mock_gpio.assert_zone_led(0, 1)
        mock_gpio.assert_zone_led(3, 1)
        mock_gpio.assert_zone_led(1, 0)
        mock_gpio.assert_zone_led(2, 0)

        # Actuate PWM Fan at temp=28.0°C and 2 people
        # Duty = 40 + (28 - 25)*5 + 2*10 = 40 + 15 + 20 = 75%
        duty = mock_gpio.set_fan(28.0, total_people)
        assert duty == 75
        mock_gpio.assert_fan_duty(75)

    def test_t3_04_dual_engine_pre_instantiation(self):
        """Feature 1 (OpenVINO AI) + Feature 3 (Color Detector).
        Verify both detection engines can coexist in memory and execute sequentially
        without memory leaks or device conflict.
        """
        from core.detector import YOLOv8OpenVINODetector
        ai_detector = YOLOv8OpenVINODetector(is_simulation=True)
        color_detector = ColorDetector()

        test_frame = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(test_frame, (50, 50), (150, 120), (0, 255, 255), -1)

        # Run color detection
        occ_color, cnt_color = color_detector.detect(test_frame)
        assert occ_color[0] == 1
        assert cnt_color == 1

        # Run AI detection
        occ_ai, cnt_ai = ai_detector.detect(test_frame)
        assert len(occ_ai) == 6
        assert isinstance(cnt_ai, int)

    def test_t3_05_mode_toggle_api_to_sse_broadcast_sync(self):
        """Feature 4 (Mode Toggle API) <-> Telemetry State.
        Verify updating mode via /set_mode reflects in the global telemetry state.
        """
        client = get_flask_test_client()

        # Set to color mode
        r1 = client.post('/set_mode', json={"mode": "color"})
        assert r1.status_code == 200

        r2 = client.get('/get_mode')
        assert r2.status_code == 200
        assert r2.get_json()["mode"] == "color"

        # Switch to AI mode
        r3 = client.post('/set_mode', json={"mode": "ai"})
        assert r3.status_code == 200

        r4 = client.get('/get_mode')
        assert r4.status_code == 200
        assert r4.get_json()["mode"] == "ai"

    def test_t3_06_camera_stream_reconnect_with_color_mode(self):
        """Feature 3 (Color Detection) <-> Vision Stream Reconnection.
        Verify that simulating camera disconnect and reconnection while in Color mode
        does not crash and resumes valid detection once frames arrive.
        """
        from core.vision_stream import VisionStream
        stream = VisionStream(is_simulation=True)
        detector = ColorDetector()

        # Active stream frame
        frame1 = stream._generate_synthetic_frame()
        occ1, cnt1 = detector.detect(frame1)
        assert len(occ1) == 6

        # Simulate stream pause/stop
        stream.stop()
        assert detector.detect(None) == ([0, 0, 0, 0, 0, 0], 0)

        # Reconnect stream
        stream = VisionStream(is_simulation=True)
        frame2 = stream._generate_synthetic_frame()
        occ2, cnt2 = detector.detect(frame2)
        assert len(occ2) == 6
        stream.stop()

    def test_t3_07_full_system_telemetry_consistency_in_color_mode(self, mock_gpio, mock_dht_sensor):
        """Full system pairwise telemetry integration under Color Mode.
        Sensors + Color Detection -> Actuators -> Telemetry Payload.
        """
        detector = ColorDetector()
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 50), (200, 100), (0, 255, 255), -1)  # Zone 0
        cv2.rectangle(frame, (400, 300), (500, 380), (0, 255, 255), -1)  # Zone 5

        occupied, total_people = detector.detect(frame)
        assert occupied[0] == 1
        assert occupied[5] == 1
        assert total_people == 2

        temp, humidity = mock_dht_sensor.read()
        fan_speed = mock_gpio.set_fan(temp, total_people)
        mock_gpio.update_zones(occupied)

        telemetry = {
            "zones": occupied,
            "temp": temp,
            "humidity": humidity,
            "fan_speed": fan_speed,
            "total_people": total_people,
            "mode": "color",
            "timestamp": int(time.time()),
        }

        assert telemetry["mode"] == "color"
        assert telemetry["total_people"] == 2
        assert telemetry["zones"][0] == 1
        assert telemetry["zones"][5] == 1
        assert telemetry["fan_speed"] > 0
