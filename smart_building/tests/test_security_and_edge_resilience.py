"""
Regression & Resilience Tests for EcoZone AI Security, Hardware Safety,
Self-Healing VisionStream, ROI Filtering, Glare Recovery, and Edge Endpoints.
"""

import json
import os
import socket
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import app as app_module
from app import app as flask_app, apply_zone_hysteresis
from core.color_detector import ColorDetector
from core.detector import YOLOv8OpenVINODetector
from core.transform import PerspectiveTransformer, get_zone_id
from core.vision_stream import VisionStream


class TestOccupancyHysteresisFSM:
    """Verify hold-time hysteresis prevents relay chatter without delaying turn-on."""

    def test_immediate_turn_on_and_grace_period_hold(self):
        with app_module.zone_hysteresis_lock:
            for i in range(6):
                app_module.zone_last_seen_ts[i] = None

        # Frame 1 at t=10.0: Zone 2 occupied -> immediate ON
        res1 = apply_zone_hysteresis([0, 0, 1, 0, 0, 0], now=10.0, hold_seconds=3.0)
        assert res1 == [0, 0, 1, 0, 0, 0]

        # Frame 2 at t=11.5: detection drops briefly -> stays ON within 3.0s grace period
        res2 = apply_zone_hysteresis([0, 0, 0, 0, 0, 0], now=11.5, hold_seconds=3.0)
        assert res2 == [0, 0, 1, 0, 0, 0]

        # Frame 3 at t=13.1: grace period expired -> turns OFF
        res3 = apply_zone_hysteresis([0, 0, 0, 0, 0, 0], now=13.1, hold_seconds=3.0)
        assert res3 == [0, 0, 0, 0, 0, 0]

    def test_zero_hold_turns_off_immediately(self):
        with app_module.zone_hysteresis_lock:
            for i in range(6):
                app_module.zone_last_seen_ts[i] = None

        assert apply_zone_hysteresis([1, 0, 0, 0, 0, 0], now=20.0, hold_seconds=0.0) == [1, 0, 0, 0, 0, 0]
        assert apply_zone_hysteresis([0, 0, 0, 0, 0, 0], now=20.1, hold_seconds=0.0) == [0, 0, 0, 0, 0, 0]


class TestGlareRecoveryAndROIFiltering:
    """Verify glare-resilient yellow detection and out-of-bounds ROI exclusion."""

    def test_color_detector_recovers_led_glare_center(self):
        detector = ColorDetector()
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        # Yellow cup rim in Zone 0 (BGR: 0, 255, 255)
        cv2.rectangle(frame, (100, 40), (160, 100), (0, 255, 255), -1)
        # Simulated LED specular glare core (H=30, S=55, V=245 -> BGR approx (200, 245, 245))
        cv2.rectangle(frame, (115, 55), (145, 85), (200, 245, 245), -1)

        occupied, count = detector.detect(frame)
        assert occupied == [1, 0, 0, 0, 0, 0]
        assert count == 1

    def test_color_detector_filters_out_of_bounds_after_transform(self):
        detector = ColorDetector()
        transformer = PerspectiveTransformer()
        # Shift all points right by +700 px (outside 600x400 canvas)
        transformer.matrix = np.array([
            [1.0, 0.0, 700.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 50), (160, 110), (0, 255, 255), -1)

        occupied, count = detector.detect(frame, transformer=transformer)
        assert occupied == [0, 0, 0, 0, 0, 0]
        assert count == 0

    def test_yolo_mock_filters_out_of_bounds_after_transform(self):
        detector = YOLOv8OpenVINODetector(is_simulation=True)
        transformer = PerspectiveTransformer()
        # Shift points outside [0..600, 0..400]
        transformer.matrix = np.array([
            [1.0, 0.0, -500.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 50), (150, 120), (0, 255, 0), -1)

        occupied, count = detector.detect(frame, transformer=transformer)
        assert occupied == [0, 0, 0, 0, 0, 0]
        assert count == 0

    def test_yolo_openvino_roi_filter_preserves_inside_excludes_outside(self):
        detector = YOLOv8OpenVINODetector(is_simulation=True)
        detector.use_real_model = True
        detector.output_layer = "out"

        # Build synthetic OpenVINO output tensor shape (1, 84, 2):
        # Detection 0: inside table at (150, 66) -> Zone 0
        # Detection 1: outside hallway at (-120, 200) -> must be excluded
        preds = np.zeros((1, 84, 2), dtype=np.float32)
        # In 640x640 network space on a 640x400 frame: scale_x=1, scale_y=400/640
        preds[0, 0, 0] = 150.0  # cx
        preds[0, 1, 0] = 50.0   # cy
        preds[0, 2, 0] = 40.0   # w
        preds[0, 3, 0] = 80.0   # h -> y_max = (50+40)*400/640 = 56.25
        preds[0, 4, 0] = 0.90   # class 0 confidence

        preds[0, 0, 1] = 10.0   # cx (will be shifted to -190 by transformer)
        preds[0, 1, 1] = 200.0
        preds[0, 2, 1] = 40.0
        preds[0, 3, 1] = 80.0
        preds[0, 4, 1] = 0.92

        detector.compiled_model = MagicMock(return_value={"out": preds})

        transformer = PerspectiveTransformer()
        # Shift x by -200 so detection 0 -> -50 (outside!) wait: shift only -100 so 150->50 (inside Zone 0) and 10->-90 (outside!)
        transformer.matrix = np.array([
            [1.0, 0.0, -100.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        frame = np.zeros((400, 640, 3), dtype=np.uint8)
        occupied, count = detector.detect(frame, transformer=transformer)
        assert occupied == [1, 0, 0, 0, 0, 0]
        assert count == 1

    def test_get_zone_id_invariant1_clamping_preserved(self):
        assert get_zone_id(-100, -100) == 0
        assert get_zone_id(9999, 9999) == 5


class TestVisionStreamUDPSecurity:
    """Verify UDP camera auto-discovery rejects spoofed or malformed packets."""

    def test_rejects_spoofed_ip_mismatch(self):
        vs = VisionStream(is_simulation=False)
        fake_sock = MagicMock()
        fake_sock.recvfrom.return_value = (b"ECOZONE_CAM:10.0.0.99", ("192.168.1.50", 12345))

        with patch("core.vision_stream.socket.socket", return_value=fake_sock):
            assert vs._discover_camera_ip(timeout=0.1) is None

    def test_rejects_malformed_ip(self):
        vs = VisionStream(is_simulation=False)
        fake_sock = MagicMock()
        fake_sock.recvfrom.return_value = (b"ECOZONE_CAM:evil.com/path?", ("192.168.1.50", 12345))

        with patch("core.vision_stream.socket.socket", return_value=fake_sock):
            assert vs._discover_camera_ip(timeout=0.1) is None

    def test_accepts_matching_sender_ip(self):
        vs = VisionStream(is_simulation=False)
        fake_sock = MagicMock()
        fake_sock.recvfrom.return_value = (b"ECOZONE_CAM:172.20.10.2", ("172.20.10.2", 12345))

        with patch("core.vision_stream.socket.socket", return_value=fake_sock):
            assert vs._discover_camera_ip(timeout=0.1) == "http://172.20.10.2:81/stream"


class TestCalibrationAndFeedEndpoints:
    """Verify new endpoints (/raw_feed, /debug_feed, /calibrate/reset, /detection_log/clear) and convexity check."""

    def test_calibrate_save_rejects_non_convex_bowtie(self):
        flask_app.config["TESTING"] = True
        client = flask_app.test_client()
        bowtie_pts = [[100, 100], [500, 400], [500, 100], [100, 400]]
        res = client.post("/calibrate/save", json={
            "points": bowtie_pts,
            "img_width": 640,
            "img_height": 480,
        })
        assert res.status_code == 400
        assert res.get_json()["status"] == "error"

    def test_calibrate_reset_restores_identity(self):
        flask_app.config["TESTING"] = True
        client = flask_app.test_client()
        res = client.post("/calibrate/reset")
        assert res.status_code == 200
        assert res.get_json()["status"] == "ok"

    def test_raw_and_debug_feeds_return_mjpeg(self):
        flask_app.config["TESTING"] = True
        client = flask_app.test_client()
        for route in ("/raw_feed", "/debug_feed"):
            res = client.get(route)
            assert res.status_code == 200
            assert "multipart/x-mixed-replace" in res.mimetype
            first_chunk = next(res.iter_encoded())
            assert b"Content-Type: image/jpeg" in first_chunk

    def test_detection_log_clear_empties_buffer(self):
        flask_app.config["TESTING"] = True
        client = flask_app.test_client()
        with app_module.detection_log_lock:
            app_module.detection_log.append({"ts": "12:00:00", "people": 1, "zones": ["Z0"], "fan": 50, "temp": 28.0})
        res = client.post("/detection_log/clear")
        assert res.status_code == 200
        res2 = client.get("/detection_log")
        assert res2.get_json()["log"] == []


class TestRound2AIModelAndAlgorithmUpgrades:
    """Verify Round 2 CV/AI algorithm upgrades, input sanitization, and calibration safety."""

    def test_geodesic_glare_recovery_half_glared_cup(self):
        import cv2
        import numpy as np
        from core.color_detector import ColorDetector

        detector = ColorDetector(min_contour_area=500.0, enable_glare_recovery=True)
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        # Zone 1: 26x26 cup (area 676 > 500), left half pure yellow, right half washed-out glare (S~26, V=250)
        frame[50:76, 420:433] = (0, 255, 255)
        frame[50:76, 433:446] = (225, 250, 250)

        occupied, count = detector.detect(frame)
        assert occupied == [0, 1, 0, 0, 0, 0]
        assert count == 1

    def test_geodesic_glare_recovery_4px_rim_s25_center_recovered(self):
        import numpy as np
        from core.color_detector import ColorDetector

        detector = ColorDetector(min_contour_area=500.0, enable_glare_recovery=True)
        frame = np.zeros((400, 600, 3), dtype=np.uint8)
        # Zone 2: 30x30 cup centered at (150, 200) -> x=[135..165), y=[185..215)
        # Inner glare core S~25-26, V=250 surrounded by thin 4px yellow rim
        frame[185:215, 135:165] = (225, 250, 250)
        frame[185:189, 135:165] = (0, 255, 255)
        frame[211:215, 135:165] = (0, 255, 255)
        frame[185:215, 135:139] = (0, 255, 255)
        frame[185:215, 161:165] = (0, 255, 255)

        mask = detector.get_mask(frame)
        assert mask[200, 150] == 255
        last_mask = detector.get_last_mask()
        assert last_mask is not None
        assert last_mask[200, 150] == 255

        occupied, count = detector.detect(frame)
        assert occupied == [0, 0, 1, 0, 0, 0]
        assert count == 1

    def test_detectors_reject_non_array_1d_2d_grayscale_and_bgra_safely(self):
        import numpy as np
        from core.color_detector import ColorDetector
        from core.detector import YOLOv8OpenVINODetector

        cd = ColorDetector()
        yd = YOLOv8OpenVINODetector(is_simulation=True)

        nan_frame = np.zeros((400, 600, 3), dtype=np.float32)
        nan_frame[10, 10] = np.nan
        nan_frame[20, 20] = np.inf

        invalid_inputs = [
            None,
            "not_an_array",
            np.zeros((100,), dtype=np.uint8),
            np.zeros((400, 600), dtype=np.uint8),
            np.zeros((400, 600, 4), dtype=np.uint8),
            nan_frame,
        ]
        for bad in invalid_inputs:
            assert cd.detect(bad) == ([0, 0, 0, 0, 0, 0], 0)
            assert yd.detect(bad) == ([0, 0, 0, 0, 0, 0], 0)

    def test_yolo_force_mock_and_zero_copy_openvino_path(self):
        import cv2
        import numpy as np
        from core.detector import YOLOv8OpenVINODetector
        from core.transform import PerspectiveTransformer

        yd = YOLOv8OpenVINODetector(is_simulation=True)
        out = np.zeros((1, 84, 8400), dtype=np.float32)
        out[0, 0, 0] = 480.0
        out[0, 1, 0] = 100.0
        out[0, 2, 0] = 60.0
        out[0, 3, 0] = 80.0
        out[0, 4, 0] = 0.90  # Class 0 (person)
        mock_compiled = MagicMock(return_value={"out": out})
        yd.use_real_model = True
        yd.compiled_model = mock_compiled
        yd.output_layer = "out"

        # Green mock target in Zone 0
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 50), (140, 120), (0, 255, 0), -1)

        occ_mock, cnt_mock = yd.detect(frame, force_mock=True)
        mock_compiled.assert_not_called()
        assert occ_mock == [1, 0, 0, 0, 0, 0]
        assert cnt_mock == 1

        # OpenVINO zero-copy path with Identity transformer -> (480, 140) -> Zone 1
        identity_tr = PerspectiveTransformer()
        identity_tr.matrix = np.eye(3, dtype=np.float32)
        occ_ov, cnt_ov = yd.detect(frame, transformer=identity_tr, force_mock=False)
        mock_compiled.assert_called_once()
        assert occ_ov == [0, 1, 0, 0, 0, 0]
        assert cnt_ov == 1

    def test_transform_matrix_setter_rejects_bad_shape(self):
        import numpy as np
        import pytest
        from core.transform import PerspectiveTransformer

        tr = PerspectiveTransformer()
        with pytest.raises(ValueError):
            tr.matrix = np.eye(2, dtype=np.float32)

    def test_calibrate_save_rejects_ccw_and_nonfinite_coords(self):
        flask_app.config["TESTING"] = True
        client = flask_app.test_client()

        # Counter-clockwise TL -> BL -> BR -> TR
        ccw_pts = [[100, 100], [100, 400], [500, 400], [500, 100]]
        res_ccw = client.post("/calibrate/save", json={
            "points": ccw_pts,
            "img_width": 640,
            "img_height": 480,
        })
        assert res_ccw.status_code == 400
        assert res_ccw.get_json()["status"] == "error"

        # Non-finite coordinates ("NaN" string -> float("nan"), and 1e40 -> float32 inf)
        for bad_val in ("NaN", 1e40):
            res_inf = client.post("/calibrate/save", json={
                "points": [[bad_val, 100], [500, 100], [500, 400], [100, 400]],
                "img_width": 640,
                "img_height": 480,
            })
            assert res_inf.status_code == 400
            assert res_inf.get_json()["status"] == "error"

    def test_vision_stream_synthetic_covers_all_six_zones(self):
        from core.color_detector import ColorDetector

        vs = VisionStream(is_simulation=True)
        cd = ColorDetector(min_contour_area=500.0)
        seen_zones = set()
        for _ in range(220):
            frame = vs._generate_synthetic_frame()
            warped = frame[:400, :600]
            occupied, _ = cd.detect(warped)
            seen_zones.update(i for i, val in enumerate(occupied) if val)
        assert seen_zones == {0, 1, 2, 3, 4, 5}

    def test_calibration_module_synthetic_grid_and_cw_rotation_sign(self):
        import calibration
        import inspect
        import numpy as np

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        with patch("core.vision_stream.VisionStream", side_effect=RuntimeError("offline")), \
             patch("calibration.cv2.VideoCapture", return_value=mock_cap):
            syn = calibration.acquire_frame(source=None)

        assert syn.shape == (480, 640, 3)
        # 3x2 interior grid lines at x=300 and y=150, 250 inside [50..550, 50..350]
        assert np.array_equal(syn[100, 300], np.array([100, 100, 100], dtype=np.uint8))
        assert np.array_equal(syn[150, 100], np.array([100, 100, 100], dtype=np.uint8))
        assert np.array_equal(syn[250, 100], np.array([100, 100, 100], dtype=np.uint8))
        assert "-rotation_index" in inspect.getsource(calibration.main)

    def test_color_detector_update_config_and_api_endpoints(self):
        from core.color_detector import ColorDetector

        detector = ColorDetector(lower_hsv=[20, 100, 100], upper_hsv=[35, 255, 255], min_contour_area=500.0)
        cfg = detector.update_config(lower_hsv=[24, 140, 120], upper_hsv=[35, 255, 255], min_contour_area=600.0, enable_glare_recovery=False)
        assert cfg["lower_hsv"] == [24, 140, 120]
        assert cfg["min_contour_area"] == 600.0
        assert cfg["enable_glare_recovery"] is False

        flask_app.config["TESTING"] = True
        client = flask_app.test_client()

        with patch.object(app_module, "color_detector_instance", detector):
            res_get = client.get("/api/color_config")
            assert res_get.status_code == 200
            assert res_get.get_json()["status"] == "ok"

            res_post = client.post("/api/color_config", json={
                "lower_hsv": [25, 140, 120],
                "upper_hsv": [35, 255, 255],
                "min_contour_area": 550.0,
                "enable_glare_recovery": True,
            })
            assert res_post.status_code == 200
            assert res_post.get_json()["status"] == "ok"
            assert res_post.get_json()["config"]["lower_hsv"] == [25, 140, 120]
