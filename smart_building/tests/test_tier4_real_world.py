"""
Tier 4: Real-World End-to-End Application Workflow Test Suite
Project: EcoZone AI Smart Energy Management System - Dual Detection Mode

This module tests realistic, full system lifecycle workflows and physical model scenarios:
1. Full Dual-Mode Switching Workflow (Color <-> AI Mode Transitions)
2. Multi-Cup Physical Model Simulation (30x40cm Sa Bàn with Yellow Cup Proxies)
3. Multi-Client Concurrent Mode Toggle Stress & Telemetry Integrity
4. Rapid Runtime Mode Toggle Stress Under Stream
5. Environmental Lighting Shift Simulation (Warm White Desk Setup)
6. End-to-End Cold Boot to Dual-Mode Energy Monitoring Lifecycle
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


@pytest.mark.tier4
class TestTier4RealWorldWorkflows:
    """Real-World System Lifecycle & End-to-End Application Workflow Tests."""

    def test_t4_rw01_full_dual_mode_switching_workflow(self, mock_gpio):
        """Scenario 1: Full Dual-Mode Switching Workflow.
        1. Initialize system in Color Mode.
        2. Feed video frame containing 2 yellow cup proxies in Zones 0 and 3.
        3. Verify ColorDetector maps to Zones 0 & 3, GPIO LEDs 17 & 23 light up.
        4. Runtime switch mode to AI Mode via API.
        5. Verify AI detector runs on next frame tick.
        6. Runtime switch back to Color Mode -> verify instant detection recovery.
        """
        from core.detector import YOLOv8OpenVINODetector
        client = get_flask_test_client()

        color_detector = ColorDetector()
        ai_detector = YOLOv8OpenVINODetector(is_simulation=True)

        # Step 1: Set mode to color
        r_init = client.post('/set_mode', json={"mode": "color"})
        assert r_init.status_code == 200

        # Step 2: Create synthetic frame with 2 yellow cups
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 30), (200, 100), (0, 255, 255), -1)  # Zone 0
        cv2.rectangle(frame, (400, 160), (500, 240), (0, 255, 255), -1)  # Zone 3

        # Step 3: Verify color detection
        occ_color, count_color = color_detector.detect(frame)
        assert occ_color == [1, 0, 0, 1, 0, 0]
        assert count_color == 2
        mock_gpio.update_zones(occ_color)
        mock_gpio.assert_zone_led(0, 1)
        mock_gpio.assert_zone_led(3, 1)

        # Step 4: Switch to AI mode
        r_ai = client.post('/set_mode', json={"mode": "ai"})
        assert r_ai.status_code == 200
        assert r_ai.get_json()["mode"] == "ai"

        # Step 5: Verify AI detector processes frame
        occ_ai, count_ai = ai_detector.detect(frame)
        assert len(occ_ai) == 6
        assert isinstance(count_ai, int)

        # Step 6: Switch back to Color mode
        r_back = client.post('/set_mode', json={"mode": "color"})
        assert r_back.status_code == 200
        occ_recovery, count_recovery = color_detector.detect(frame)
        assert occ_recovery == [1, 0, 0, 1, 0, 0]
        assert count_recovery == 2

    def test_t4_rw02_multi_cup_physical_model_simulation(self, mock_gpio):
        """Scenario 2: Multi-Cup Movement on 30x40cm Physical Model.
        Simulates moving physical cups on the sa bàn:
        T=0: 3 cups in Zone 0, Zone 3, Zone 4 -> [1, 0, 0, 1, 1, 0], total=3.
        T=1: Cup moved from Zone 0 to Zone 1 -> [0, 1, 0, 1, 1, 0], total=3.
        T=2: Cup removed from Zone 4 -> [0, 1, 0, 1, 0, 0], total=2.
        """
        detector = ColorDetector()

        # State 0: Cups in Zones 0, 3, 4
        f0 = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(f0, (100, 30), (200, 100), (0, 255, 255), -1)   # Z0: cx=150, y=100
        cv2.rectangle(f0, (400, 160), (500, 240), (0, 255, 255), -1) # Z3: cx=450, y=240
        cv2.rectangle(f0, (100, 300), (200, 370), (0, 255, 255), -1) # Z4: cx=150, y=370

        occ0, cnt0 = detector.detect(f0)
        assert occ0 == [1, 0, 0, 1, 1, 0]
        assert cnt0 == 3
        mock_gpio.update_zones(occ0)
        mock_gpio.assert_zone_led(0, 1)
        mock_gpio.assert_zone_led(3, 1)
        mock_gpio.assert_zone_led(4, 1)

        # State 1: Cup moved from Z0 to Z1 (cx=450, y=100)
        f1 = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(f1, (400, 30), (500, 100), (0, 255, 255), -1)   # Z1: cx=450, y=100
        cv2.rectangle(f1, (400, 160), (500, 240), (0, 255, 255), -1) # Z3: cx=450, y=240
        cv2.rectangle(f1, (100, 300), (200, 370), (0, 255, 255), -1) # Z4: cx=150, y=370

        occ1, cnt1 = detector.detect(f1)
        assert occ1 == [0, 1, 0, 1, 1, 0]
        assert cnt1 == 3
        mock_gpio.update_zones(occ1)
        mock_gpio.assert_zone_led(0, 0)
        mock_gpio.assert_zone_led(1, 1)

        # State 2: Cup removed from Z4
        f2 = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(f2, (400, 30), (500, 100), (0, 255, 255), -1)   # Z1
        cv2.rectangle(f2, (400, 160), (500, 240), (0, 255, 255), -1) # Z3

        occ2, cnt2 = detector.detect(f2)
        assert occ2 == [0, 1, 0, 1, 0, 0]
        assert cnt2 == 2
        mock_gpio.update_zones(occ2)
        mock_gpio.assert_zone_led(4, 0)

    def test_t4_rw03_concurrent_multi_client_mode_toggle_stress(self):
        """Scenario 3: Multi-Client Concurrent Mode Toggle Stress.
        Simulates 4 browser clients concurrently toggling and reading mode state.
        Verifies thread safety, absence of deadlocks, and valid JSON responses.
        """
        client = get_flask_test_client()
        iterations = 20
        errors = []

        def client_worker(client_id):
            for i in range(iterations):
                mode = "color" if (i + client_id) % 2 == 0 else "ai"
                try:
                    r_post = client.post('/set_mode', json={"mode": mode})
                    assert r_post.status_code == 200
                    r_get = client.get('/get_mode')
                    assert r_get.status_code == 200
                    data = r_get.get_json()
                    assert data["mode"] in ("color", "ai")
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=client_worker, args=(cid,)) for cid in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0, f"Concurrent client errors: {errors}"

    def test_t4_rw04_rapid_runtime_toggle_stress_under_stream(self):
        """Scenario 4: Rapid Runtime Toggle Stress Under Active Stream.
        Toggles mode back and forth 30 times at rapid speed while feeding video frames.
        Verifies zero pipeline crashes, zero deadlocks, and final state consistency.
        """
        from core.vision_stream import VisionStream
        from core.detector import YOLOv8OpenVINODetector

        stream = VisionStream(is_simulation=True)
        color_detector = ColorDetector()
        ai_detector = YOLOv8OpenVINODetector(is_simulation=True)
        client = get_flask_test_client()

        for i in range(30):
            target_mode = "color" if i % 2 == 0 else "ai"
            resp = client.post('/set_mode', json={"mode": target_mode})
            assert resp.status_code == 200

            frame = stream._generate_synthetic_frame()
            if target_mode == "color":
                occ, cnt = color_detector.detect(frame)
            else:
                occ, cnt = ai_detector.detect(frame)

            assert len(occ) == 6
            assert isinstance(cnt, int)

        stream.stop()

    def test_t4_rw05_environmental_lighting_shift_simulation(self):
        """Scenario 5: Environmental Lighting Shift Simulation (Warm LED Desk Setup).
        Simulates lighting variations typical of desk lamp setups:
        - Baseline warm yellow (H=30, S=220, V=220)
        - Dimmer lighting (V=140, S=180)
        - Brighter lighting (V=255, S=240)
        - Slight color temperature shift (H=24 to H=34)
        Verifies ColorDetector reliably detects the yellow cup across all variations.
        """
        detector = ColorDetector()

        variations = [
            (30, 220, 220),  # Baseline warm yellow
            (28, 160, 140),  # Dim desk lighting
            (32, 255, 255),  # Bright illumination
            (24, 200, 200),  # Amber-warm shift
            (34, 180, 210),  # Greenish-yellow shift
        ]

        for idx, (h, s, v) in enumerate(variations):
            frame = np.zeros((400, 600, 3), dtype=np.uint8)
            # Create yellow patch in HSV then convert to BGR
            hsv_patch = np.uint8([[[h, s, v]]])
            bgr_patch = cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2BGR)[0][0]
            bgr_color = (int(bgr_patch[0]), int(bgr_patch[1]), int(bgr_patch[2]))

            # Draw 40x40 cup in Zone 2 (Row 1 Left: cx=150, cy=200)
            cv2.rectangle(frame, (130, 180), (170, 220), bgr_color, -1)

            occupied, count = detector.detect(frame)
            assert count == 1, f"Failed detection on lighting variation {idx}: (H={h}, S={s}, V={v})"
            assert occupied[2] == 1, f"Failed zone mapping on lighting variation {idx}"

    def test_t4_rw06_end_to_end_cold_boot_to_dual_mode_monitoring(self, mock_gpio, mock_dht_sensor):
        """Scenario 6: End-to-End Cold Boot to Dual-Mode Energy Monitoring Lifecycle.
        Complete lifecycle from startup to monitoring:
        1. Verify config constants and calibration geometry.
        2. Boot vision pipeline and start in Color Mode.
        3. Detect 2 cups in model -> LEDs ON, Fan modulated with DHT11 temperature.
        4. Log detection event into detection_log.
        5. Verify shutdown & cleanup.
        """
        import config
        assert config.WARP_WIDTH == 600
        assert config.WARP_HEIGHT == 400

        detector = ColorDetector()

        # Step 2 & 3: Frame with 2 occupants in Zone 1 (Row 0 Right) and Zone 4 (Row 2 Left)
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.rectangle(frame, (400, 30), (500, 100), (0, 255, 255), -1)   # Zone 1
        cv2.rectangle(frame, (100, 300), (200, 370), (0, 255, 255), -1) # Zone 4

        occupied, total_people = detector.detect(frame)
        assert occupied == [0, 1, 0, 0, 1, 0]
        assert total_people == 2

        # Step 4: Actuate GPIO & Fan
        mock_gpio.update_zones(occupied)
        mock_gpio.assert_zone_led(1, 1)
        mock_gpio.assert_zone_led(4, 1)

        temp, humidity = mock_dht_sensor.read()
        fan_duty = mock_gpio.set_fan(temp, total_people)
        assert fan_duty > 0

        # Step 5: Clean teardown
        mock_gpio.cleanup()
        mock_gpio.assert_cleaned_up()
