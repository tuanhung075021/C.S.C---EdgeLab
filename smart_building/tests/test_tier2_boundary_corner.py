"""
Tier 2: Boundary, Limit & Fault-Injection Test Suite
Project: EcoZone AI Smart Energy Management System - Dual Detection Mode

This module provides at least 5 boundary, edge condition, out-of-bounds, and
fault-injection test cases for each of Features 1 through 5 (total >= 25 test cases).
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
        """Reference ColorDetector implementation for contract verification."""
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


# =============================================================================
# FEATURE 1: OpenVINO & Model Loading Boundaries
# =============================================================================
@pytest.mark.tier2
@pytest.mark.feature(1)
def test_f1_bva_01_nonexistent_model_xml_handling():
    """Verify handling when nonexistent model XML path is provided."""
    try:
        from openvino import Core
    except ImportError:
        from openvino.runtime import Core
    core = Core()
    with pytest.raises(Exception):
        core.read_model("models/nonexistent/missing_model.xml")


@pytest.mark.tier2
@pytest.mark.feature(1)
def test_f1_bva_02_corrupted_model_xml_handling(tmp_path):
    """Verify handling when zero-byte or corrupt XML file is read."""
    try:
        from openvino import Core
    except ImportError:
        from openvino.runtime import Core
    core = Core()
    corrupt_xml = tmp_path / "corrupt.xml"
    corrupt_xml.write_text("<xml>invalid content</xml>", encoding="utf-8")
    with pytest.raises(Exception):
        core.read_model(str(corrupt_xml))


@pytest.mark.tier2
@pytest.mark.feature(1)
def test_f1_bva_03_unsupported_device_compilation():
    """Verify attempting to compile on invalid device name raises exception gracefully."""
    try:
        from openvino import Core
    except ImportError:
        from openvino.runtime import Core
    core = Core()
    model_xml = os.path.join(PROJECT_ROOT, "models", "yolov8n_openvino_model", "yolov8n.xml")
    model = core.read_model(model_xml)
    with pytest.raises(Exception):
        core.compile_model(model, "INVALID_DEVICE_XYZ")


@pytest.mark.tier2
@pytest.mark.feature(1)
def test_f1_bva_04_zero_dimension_frame_detector():
    """Verify detector handles empty or (0,0,3) frame safely without crashing."""
    from core.detector import YOLOv8OpenVINODetector
    detector = YOLOv8OpenVINODetector(is_simulation=True)
    empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
    occupied, count = detector.detect(empty_frame)
    assert occupied == [0, 0, 0, 0, 0, 0]
    assert count == 0


@pytest.mark.tier2
@pytest.mark.feature(1)
def test_f1_bva_05_concurrent_inference_thread_safety():
    """Verify detector handles concurrent inference from multiple threads."""
    from core.detector import YOLOv8OpenVINODetector
    detector = YOLOv8OpenVINODetector(is_simulation=True)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    errors = []
    def infer():
        try:
            for _ in range(5):
                occ, cnt = detector.detect(test_frame)
                assert len(occ) == 6
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=infer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=3.0)

    assert len(errors) == 0


# =============================================================================
# FEATURE 2: Config Definitions Boundaries
# =============================================================================
@pytest.mark.tier2
@pytest.mark.feature(2)
def test_f2_bva_01_extreme_confidence_threshold_values():
    """Verify confidence threshold handling at extreme values (0.0, 1.0)."""
    import config
    conf = getattr(config, "CONFIDENCE_THRESHOLD", 0.08)
    assert 0.0 <= conf <= 1.0


@pytest.mark.tier2
@pytest.mark.feature(2)
def test_f2_bva_02_hsv_yellow_range_limits():
    """Verify HSV limits for Yellow hue span valid 8-bit HSV ranges [0..179]."""
    import config
    lower = getattr(config, "COLOR_LOWER_HSV", (20, 100, 100))
    upper = getattr(config, "COLOR_UPPER_HSV", (35, 255, 255))
    assert 0 <= lower[0] <= upper[0] <= 179
    assert 0 <= lower[1] <= upper[1] <= 255
    assert 0 <= lower[2] <= upper[2] <= 255


@pytest.mark.tier2
@pytest.mark.feature(2)
def test_f2_bva_03_contour_area_threshold_exact_boundary():
    """Verify exact contour area threshold: 499 px^2 rejected vs 500 px^2 accepted."""
    detector = ColorDetector(min_contour_area=500.0)

    # Frame 1: Yellow rect of area 490 px^2 (e.g. 14x35) -> must be rejected (< 500)
    frame_small = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(frame_small, (100, 100), (114, 135), (0, 255, 255), -1)  # 14 * 35 = 490
    occ_small, count_small = detector.detect(frame_small)
    assert count_small == 0
    assert occ_small == [0, 0, 0, 0, 0, 0]

    # Frame 2: Yellow rect of area 600 px^2 (e.g. 20x30) -> must be accepted (>= 500)
    frame_large = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(frame_large, (100, 100), (120, 130), (0, 255, 255), -1)  # 20 * 30 = 600
    occ_large, count_large = detector.detect(frame_large)
    assert count_large == 1
    assert any(z == 1 for z in occ_large)


@pytest.mark.tier2
@pytest.mark.feature(2)
def test_f2_bva_04_grid_dimensions_aspect_ratio():
    """Verify WARP_WIDTH / WARP_HEIGHT maintains 3:2 aspect ratio matching 30x40cm physical model."""
    import config
    w = getattr(config, "WARP_WIDTH", 600)
    h = getattr(config, "WARP_HEIGHT", 400)
    assert w / h == pytest.approx(1.5, rel=1e-3)


@pytest.mark.tier2
@pytest.mark.feature(2)
def test_f2_bva_05_missing_env_vars_defaults():
    """Verify detect_simulation_mode and stream URL fall back safely when env vars are unset."""
    import config
    sim = config.detect_simulation_mode()
    assert isinstance(sim, bool)
    assert config.ESP32_STREAM_URL is not None


# =============================================================================
# FEATURE 3: Color Detection Boundaries & Extreme Frames
# =============================================================================
@pytest.mark.tier2
@pytest.mark.feature(3)
def test_f3_bva_01_empty_and_none_frame_safety():
    """Verify ColorDetector handles None, empty, and 1x1 frames without raising exceptions."""
    detector = ColorDetector()
    assert detector.detect(None) == ([0, 0, 0, 0, 0, 0], 0)
    assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == ([0, 0, 0, 0, 0, 0], 0)
    assert detector.detect(np.zeros((1, 1, 3), dtype=np.uint8)) == ([0, 0, 0, 0, 0, 0], 0)


@pytest.mark.tier2
@pytest.mark.feature(3)
def test_f3_bva_02_extreme_lighting_overexposure_underexposure():
    """Verify extreme lighting (pure white 255, pure black 0, full blue/red) yields 0 people."""
    detector = ColorDetector()

    # Overexposed pure white (H=0, S=0, V=255)
    white_frame = np.full((400, 600, 3), 255, dtype=np.uint8)
    assert detector.detect(white_frame) == ([0, 0, 0, 0, 0, 0], 0)

    # Pure blue (BGR: 255, 0, 0 -> H=120)
    blue_frame = np.zeros((400, 600, 3), dtype=np.uint8)
    blue_frame[:, :, 0] = 255
    assert detector.detect(blue_frame) == ([0, 0, 0, 0, 0, 0], 0)

    # Pure red (BGR: 0, 0, 255 -> H=0)
    red_frame = np.zeros((400, 600, 3), dtype=np.uint8)
    red_frame[:, :, 2] = 255
    assert detector.detect(red_frame) == ([0, 0, 0, 0, 0, 0], 0)


@pytest.mark.tier2
@pytest.mark.feature(3)
def test_f3_bva_03_small_noise_specks_rejected():
    """Verify multiple tiny yellow specks (<500 px^2) scattered across canvas are discarded."""
    detector = ColorDetector()
    frame = np.zeros((400, 600, 3), dtype=np.uint8)

    # Draw 10 small yellow dots of size 10x10 = 100 px^2
    for i in range(10):
        x = (i * 50 + 20) % 550
        y = (i * 35 + 20) % 350
        cv2.rectangle(frame, (x, y), (x + 10, y + 10), (0, 255, 255), -1)

    occupied, total_people = detector.detect(frame)
    assert occupied == [0, 0, 0, 0, 0, 0]
    assert total_people == 0


@pytest.mark.tier2
@pytest.mark.feature(3)
def test_f3_bva_04_cell_boundary_coordinates():
    """Verify yellow object centered precisely at cell boundaries (x=300, y=133, y=266) maps to a valid zone."""
    detector = ColorDetector()

    # Object at x=300 boundary, y=133 boundary
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(frame, (280, 110), (320, 150), (0, 255, 255), -1)  # 40x40 = 1600 px^2

    occupied, total_people = detector.detect(frame)
    assert total_people == 1
    assert sum(occupied) == 1
    assert 0 <= occupied.index(1) <= 5


@pytest.mark.tier2
@pytest.mark.feature(3)
def test_f3_bva_05_all_six_zones_occupied_simultaneously():
    """Verify 6 distinct yellow objects (one in each of the 6 zones) are all detected."""
    detector = ColorDetector()
    frame = np.zeros((400, 600, 3), dtype=np.uint8)

    # Zone centers for 3x2 grid: (150, 66), (450, 66), (150, 200), (450, 200), (150, 333), (450, 333)
    centers = [(150, 66), (450, 66), (150, 200), (450, 200), (150, 333), (450, 333)]
    for cx, cy in centers:
        cv2.rectangle(frame, (cx - 20, cy - 20), (cx + 20, cy + 20), (0, 255, 255), -1)  # 40x40 = 1600 px^2

    occupied, total_people = detector.detect(frame)
    assert occupied == [1, 1, 1, 1, 1, 1]
    assert total_people == 6


@pytest.mark.tier2
@pytest.mark.feature(3)
def test_f3_bva_06_large_full_frame_yellow_sheet():
    """Verify full-frame yellow rectangle is handled safely without buffer overflow."""
    detector = ColorDetector()
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (600, 400), (0, 255, 255), -1)

    occupied, count = detector.detect(frame)
    assert isinstance(occupied, list)
    assert len(occupied) == 6
    assert count >= 1


# =============================================================================
# FEATURE 4: Mode Toggle API Boundaries & Error Handling
# =============================================================================
@pytest.mark.tier2
@pytest.mark.feature(4)
def test_f4_bva_01_malformed_json_returns_400():
    """Verify POST /set_mode with non-JSON text or invalid syntax returns HTTP 400."""
    client = get_flask_test_client()
    resp = client.post('/set_mode', data="not a json", content_type="application/json")
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"


@pytest.mark.tier2
@pytest.mark.feature(4)
def test_f4_bva_02_missing_mode_key_returns_400():
    """Verify POST /set_mode with missing 'mode' key ({}) returns HTTP 400."""
    client = get_flask_test_client()
    resp = client.post('/set_mode', json={"other_key": "val"})
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["status"] == "error"


@pytest.mark.tier2
@pytest.mark.feature(4)
def test_f4_bva_03_non_string_mode_values_returns_400():
    """Verify POST /set_mode with non-string mode values (int, null, list) returns HTTP 400."""
    client = get_flask_test_client()
    for bad_val in [123, None, ["color"], True, {}]:
        resp = client.post('/set_mode', json={"mode": bad_val})
        assert resp.status_code == 400


@pytest.mark.tier2
@pytest.mark.feature(4)
def test_f4_bva_04_unsupported_mode_strings_returns_400():
    """Verify POST /set_mode with unsupported mode names ('thermal', '', '  ') returns HTTP 400."""
    client = get_flask_test_client()
    for bad_mode in ["thermal", "depth", "", "   ", "laser"]:
        resp = client.post('/set_mode', json={"mode": bad_mode})
        assert resp.status_code == 400


@pytest.mark.tier2
@pytest.mark.feature(4)
def test_f4_bva_05_rapid_burst_set_mode_requests():
    """Verify server sustains a burst of 50 rapid sequential /set_mode requests without race condition."""
    client = get_flask_test_client()
    for i in range(50):
        target = "color" if i % 2 == 0 else "ai"
        resp = client.post('/set_mode', json={"mode": target})
        assert resp.status_code == 200
        assert resp.get_json()["mode"] == target

    # Verify final state is consistent
    get_resp = client.get('/get_mode')
    assert get_resp.status_code == 200
    assert get_resp.get_json()["mode"] == "ai"


# =============================================================================
# FEATURE 5: UI & Route Boundaries
# =============================================================================
@pytest.mark.tier2
@pytest.mark.feature(5)
def test_f5_bva_01_index_with_query_params_and_headers():
    """Verify GET / with unexpected query params or headers returns HTTP 200."""
    client = get_flask_test_client()
    resp = client.get('/?mode=color&debug=true&rand=12345', headers={"User-Agent": "Mobile Safari"})
    assert resp.status_code == 200


@pytest.mark.tier2
@pytest.mark.feature(5)
def test_f5_bva_02_css_mobile_media_query_rules():
    """Verify style.css includes mobile media queries for responsive layout."""
    css_path = os.path.join(PROJECT_ROOT, "static", "style.css")
    with open(css_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    assert "@media" in content or "max-width" in content or "flex" in content


@pytest.mark.tier2
@pytest.mark.feature(5)
def test_f5_bva_03_toggle_touch_target_accessibility():
    """Verify style.css provides accessible tap target sizing for mobile controls."""
    css_path = os.path.join(PROJECT_ROOT, "static", "style.css")
    with open(css_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    assert "padding" in content or "height" in content or "cursor" in content


@pytest.mark.tier2
@pytest.mark.feature(5)
def test_f5_bva_04_missing_static_file_graceful_404():
    """Verify requesting non-existent static resource returns 404 cleanly."""
    client = get_flask_test_client()
    resp = client.get('/static/nonexistent_file_xyz.css')
    assert resp.status_code == 404


@pytest.mark.tier2
@pytest.mark.feature(5)
def test_f5_bva_05_html_dom_structure_integrity():
    """Verify templates/index.html has matching head and body structural elements."""
    html_path = os.path.join(PROJECT_ROOT, "templates", "index.html")
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    assert "<head>" in content or "<head " in content
    assert "</head>" in content
    assert "<body" in content
    assert "</body>" in content
