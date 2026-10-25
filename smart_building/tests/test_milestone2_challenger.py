"""
Milestone 2 Challenger Empirical Verification Test Harness
EcoZone AI Smart Energy Management System

Tests:
1. Static validation of HTML/CSS/JS syntax & DOM structure in templates and static files.
2. End-to-end perspective calibration pipeline (POST /calibrate/save, PerspectiveTransformer, warp, 3x2 zone mapping 0..5).
3. Concurrency & multi-threading stress test on raw_frame_lock, annotated_frame_lock, and telemetry_lock.
4. Proper cleanup of perspective_matrix.npy.
"""

import os
import sys
import time
import json
import subprocess
import threading
import queue
import pytest
import numpy as np
import cv2

# Add root directory to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from core.transform import get_zone_id, PerspectiveTransformer
import app as flask_app_module
from app import app as flask_app


@pytest.fixture(autouse=True, scope="module")
def module_matrix_cleanup():
    """Ensure perspective_matrix.npy does not exist before and after module test run."""
    matrix_path = os.path.join(PROJECT_ROOT, "perspective_matrix.npy")
    if os.path.exists(matrix_path):
        try:
            os.remove(matrix_path)
        except Exception:
            pass
    yield
    if os.path.exists(matrix_path):
        try:
            os.remove(matrix_path)
        except Exception:
            pass


class TestMilestone2WebAssets:
    """Task 1: Verify HTML/CSS/JS syntax and structure."""

    def test_calibrate_html_structure_and_elements(self):
        """Verify templates/calibrate.html has valid structure and all required UI elements."""
        path = os.path.join(PROJECT_ROOT, "templates", "calibrate.html")
        assert os.path.exists(path), f"calibrate.html missing at {path}"

        with open(path, "r", encoding="utf-8") as f:
            html = f.read()

        # Canvas element
        assert '<canvas id="calibrationCanvas"' in html or '<canvas id="calibrationCanvas"' in html
        assert 'id="canvas-wrapper"' in html

        # Required buttons
        assert 'id="btn-refresh"' in html
        assert 'id="btn-undo"' in html
        assert 'id="btn-reset"' in html
        assert 'id="btn-save"' in html

        # Step indicator and checklist
        assert 'id="step-badge"' in html
        assert 'id="step-text"' in html
        for i in range(4):
            assert f'id="item-{i}"' in html
            assert f'id="coord-{i}"' in html

        # Notification banner & back link
        assert 'id="notification-banner"' in html
        assert 'href="/"' in html or 'href="{{ url_for(' in html

        # Touch support & pointer events
        assert 'pointerdown' in html or 'touchstart' in html
        assert 'touch-action: none' in html

        # Check endpoints called in JS
        assert '/calibrate/snapshot' in html
        assert '/calibrate/save' in html

    def test_index_html_3x2_grid_and_calibrate_link(self):
        """Verify templates/index.html contains 3x2 zone grid (Zones 0..5) and Calibrate button."""
        path = os.path.join(PROJECT_ROOT, "templates", "index.html")
        assert os.path.exists(path), f"index.html missing at {path}"

        with open(path, "r", encoding="utf-8") as f:
            html = f.read()

        # Calibrate link / button
        assert 'href="/calibrate"' in html
        assert 'Calibrate Camera' in html

        # 6 Zone cards
        for i in range(6):
            assert f'id="zone-{i}"' in html
            assert f'id="zone-badge-{i}"' in html
            assert f'id="zone-act-{i}"' in html

        # 3x2 Zone row descriptions
        assert "Row 0 (Left)" in html
        assert "Row 0 (Right)" in html
        assert "Row 1 (Left)" in html
        assert "Row 1 (Right)" in html
        assert "Row 2 (Left)" in html
        assert "Row 2 (Right)" in html

        # SSE stream client & Video feed
        assert 'EventSource(' in html
        assert '/stream' in html
        assert 'src="/video_feed"' in html

    def test_style_css_3x2_grid_definition(self):
        """Verify static/style.css defines .zone-grid with 2 columns x 3 rows."""
        path = os.path.join(PROJECT_ROOT, "static", "style.css")
        assert os.path.exists(path), f"style.css missing at {path}"

        with open(path, "r", encoding="utf-8") as f:
            css = f.read()

        assert "grid-template-columns: repeat(2, 1fr)" in css
        assert "grid-template-rows: repeat(3, 1fr)" in css
        assert ".btn-calibrate" in css

    def test_javascript_syntax_with_node(self):
        """Extract inline JS from calibrate.html and index.html and validate with Node.js."""
        for template_name in ["calibrate.html", "index.html"]:
            path = os.path.join(PROJECT_ROOT, "templates", template_name)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract script blocks
            import re
            scripts = re.findall(r"<script>(.*?)</script>", content, re.DOTALL)
            assert len(scripts) > 0, f"No <script> tag found in {template_name}"

            for idx, js_code in enumerate(scripts):
                # Clean Jinja tags if any in JS
                js_clean = re.sub(r"\{\{.*?\}\}", "'placeholder'", js_code)
                js_clean = re.sub(r"\{%.*?%\}", "", js_clean)

                # Run node syntax check via node -e
                # Wrap in Function to test parsing
                eval_script = f"new Function({json.dumps(js_clean)});"
                res = subprocess.run(
                    ["node", "-e", eval_script],
                    capture_output=True,
                    text=True
                )
                assert res.returncode == 0, f"JS syntax error in {template_name} (script block {idx}):\n{res.stderr}"


class TestMilestone2PerspectivePipeline:
    """Task 2: Test end-to-end perspective calibration pipeline."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown_matrix(self):
        """Ensure clean matrix state before and after each test."""
        matrix_path = os.path.join(PROJECT_ROOT, "perspective_matrix.npy")
        if os.path.exists(matrix_path):
            try:
                os.remove(matrix_path)
            except Exception:
                pass
        yield
        if os.path.exists(matrix_path):
            try:
                os.remove(matrix_path)
            except Exception:
                pass

    def test_calibrate_routes_get(self):
        """Verify GET /calibrate and GET /calibrate/snapshot return HTTP 200."""
        client = flask_app.test_client()

        # 1. GET /calibrate
        res_cal = client.get("/calibrate")
        assert res_cal.status_code == 200
        assert b"EcoZone AI" in res_cal.data
        assert b"calibrationCanvas" in res_cal.data

        # 2. GET /calibrate/snapshot
        res_snap = client.get("/calibrate/snapshot")
        assert res_snap.status_code == 200
        assert res_snap.content_type == "image/jpeg"
        # Decode JPEG bytes to numpy image
        img_array = np.frombuffer(res_snap.data, dtype=np.uint8)
        decoded = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        assert decoded is not None
        assert decoded.shape[0] == 480
        assert decoded.shape[1] == 640
        assert decoded.shape[2] == 3

    def test_calibrate_save_validation_errors(self):
        """Verify POST /calibrate/save rejects invalid input formats."""
        client = flask_app.test_client()

        # Invalid JSON
        r1 = client.post("/calibrate/save", data="bad-json", content_type="application/json")
        assert r1.status_code == 400

        # Missing points
        r2 = client.post("/calibrate/save", json={"img_width": 640, "img_height": 480})
        assert r2.status_code == 400

        # Only 3 points
        r3 = client.post("/calibrate/save", json={"points": [[0,0], [600,0], [600,400]]})
        assert r3.status_code == 400

        # 5 points
        r4 = client.post("/calibrate/save", json={"points": [[0,0], [600,0], [600,400], [0,400], [100,100]]})
        assert r4.status_code == 400

        # Malformed point coordinate
        r5 = client.post("/calibrate/save", json={"points": [[0,0], "bad", [600,400], [0,400]]})
        assert r5.status_code == 400

    def test_e2e_perspective_calibration_cycle_and_zone_mapping(self):
        """
        Simulate full calibration cycle:
        1. Define realistic trapezoid in 640x480 native frame.
        2. POST /calibrate/save with coordinates.
        3. Verify saved perspective_matrix.npy.
        4. Load matrix into PerspectiveTransformer.
        5. Warp test grid image.
        6. Verify transformed coordinates and 3x2 zone mapping for physical 30x40cm model.
        """
        client = flask_app.test_client()
        matrix_path = os.path.join(PROJECT_ROOT, "perspective_matrix.npy")

        # Trapezoid coordinates representing camera looking down at angle
        # Order: TL, TR, BR, BL
        src_points = [
            [140.0, 90.0],   # Top-Left
            [500.0, 90.0],   # Top-Right
            [580.0, 410.0],  # Bottom-Right
            [60.0, 410.0]    # Bottom-Left
        ]

        payload = {
            "points": src_points,
            "img_width": 640,
            "img_height": 480
        }

        # Step 1: POST to /calibrate/save
        res = client.post("/calibrate/save", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "ok"

        # Step 2: Verify file exists on disk with correct shape and type
        assert os.path.exists(matrix_path), "perspective_matrix.npy was not created"
        saved_M = np.load(matrix_path)
        assert isinstance(saved_M, np.ndarray)
        assert saved_M.shape == (3, 3)
        assert saved_M.dtype in (np.float32, np.float64)

        # Step 3: Load into PerspectiveTransformer
        transformer = PerspectiveTransformer(matrix_path=matrix_path)
        assert transformer.matrix.shape == (3, 3)

        # Step 4: Verify 4 corner transformations map to 600x400 canvas
        # Target: TL -> (0,0), TR -> (600,0), BR -> (600,400), BL -> (0,400)
        tl_w = transformer.transform_point(140.0, 90.0)
        tr_w = transformer.transform_point(500.0, 90.0)
        br_w = transformer.transform_point(580.0, 410.0)
        bl_w = transformer.transform_point(60.0, 410.0)

        assert abs(tl_w[0] - 0.0) < 0.5 and abs(tl_w[1] - 0.0) < 0.5, f"TL warped wrong: {tl_w}"
        assert abs(tr_w[0] - 600.0) < 0.5 and abs(tr_w[1] - 0.0) < 0.5, f"TR warped wrong: {tr_w}"
        assert abs(br_w[0] - 600.0) < 0.5 and abs(br_w[1] - 400.0) < 0.5, f"BR warped wrong: {br_w}"
        assert abs(bl_w[0] - 0.0) < 0.5 and abs(bl_w[1] - 400.0) < 0.5, f"BL warped wrong: {bl_w}"

        # Step 5: Test warped image generation
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a distinctive marker inside the room area
        cv2.circle(test_frame, (320, 250), 10, (0, 255, 0), -1)
        warped = transformer.warp(test_frame)
        assert warped.shape == (400, 600, 3)

        # Step 6: Verify points in the 30x40cm model map to expected zones (0..5)
        # Grid is 3 rows x 2 cols:
        # Zone 0: Row 0, Col 0 -> X: [0..300), Y: [0..133)
        # Zone 1: Row 0, Col 1 -> X: [300..600), Y: [0..133)
        # Zone 2: Row 1, Col 0 -> X: [0..300), Y: [133..266)
        # Zone 3: Row 1, Col 1 -> X: [300..600), Y: [133..266)
        # Zone 4: Row 2, Col 0 -> X: [0..300), Y: [266..400)
        # Zone 5: Row 2, Col 1 -> X: [300..600), Y: [266..400)

        zone_test_cases = [
            # Zone 0
            (150, 66, 0),
            (20, 20, 0),
            (299, 132, 0),
            # Zone 1
            (450, 66, 1),
            (300, 20, 1),
            (599, 132, 1),
            # Zone 2
            (150, 200, 2),
            (20, 133, 2),
            (299, 265, 2),
            # Zone 3
            (450, 200, 3),
            (300, 133, 3),
            (599, 265, 3),
            # Zone 4
            (150, 333, 4),
            (20, 266, 4),
            (299, 399, 4),
            # Zone 5
            (450, 333, 5),
            (300, 266, 5),
            (599, 399, 5),
        ]

        for x, y, expected_zone in zone_test_cases:
            actual_zone = get_zone_id(x, y)
            assert actual_zone == expected_zone, f"Point ({x}, {y}) mapped to Zone {actual_zone}, expected {expected_zone}"

        # Step 7: Test Inverse Mapping (Camera coordinates -> Warped coordinates -> Zone ID)
        inv_M = np.linalg.inv(saved_M)

        # For each zone center in warped space, project to camera space and back
        zone_centers_warped = [
            (150.0, 66.0, 0),
            (450.0, 66.0, 1),
            (150.0, 200.0, 2),
            (450.0, 200.0, 3),
            (150.0, 333.0, 4),
            (450.0, 333.0, 5)
        ]

        for wx, wy, expected_z in zone_centers_warped:
            # Map warped center back to camera frame coordinates
            cam_pt = cv2.perspectiveTransform(np.array([[[wx, wy]]], dtype=np.float32), inv_M)[0][0]
            cx, cy = float(cam_pt[0]), float(cam_pt[1])

            # Forward transform back using transformer
            reprojected_wx, reprojected_wy = transformer.transform_point(cx, cy)
            assert abs(reprojected_wx - wx) < 1.0
            assert abs(reprojected_wy - wy) < 1.0

            # Calculate zone
            zid = get_zone_id(reprojected_wx, reprojected_wy)
            assert zid == expected_z, f"Reprojected camera point ({cx:.1f}, {cy:.1f}) -> warped ({reprojected_wx:.1f}, {reprojected_wy:.1f}) gave Zone {zid}, expected {expected_z}"

    def test_scaled_display_coordinates_calibration(self):
        """Verify calibration properly handles scaled display sizes (e.g. mobile view)."""
        client = flask_app.test_client()

        # Display size is half (320x240) of native (640x480)
        scaled_points = [
            [70.0, 45.0],    # Scaled TL (140 / 2, 90 / 2)
            [250.0, 45.0],   # Scaled TR (500 / 2, 90 / 2)
            [290.0, 205.0],  # Scaled BR (580 / 2, 410 / 2)
            [30.0, 205.0]    # Scaled BL (60 / 2, 410 / 2)
        ]

        payload = {
            "points": scaled_points,
            "img_width": 320,
            "img_height": 240
        }

        res = client.post("/calibrate/save", json=payload)
        assert res.status_code == 200

        transformer = PerspectiveTransformer()
        # Verify that scaled clicks correctly map native (140, 90) -> (0, 0)
        tl_w = transformer.transform_point(140.0, 90.0)
        assert abs(tl_w[0] - 0.0) < 0.5
        assert abs(tl_w[1] - 0.0) < 0.5


class TestMilestone2Concurrency:
    """Task 3: Test concurrency to ensure no thread deadlocks on raw_frame_lock."""

    @pytest.fixture(autouse=True)
    def clean_matrix(self):
        matrix_path = os.path.join(PROJECT_ROOT, "perspective_matrix.npy")
        yield
        if os.path.exists(matrix_path):
            try:
                os.remove(matrix_path)
            except Exception:
                pass

    def test_concurrent_snapshot_stream_and_save_stress(self):
        """
        Simulate simultaneous snapshot requests, MJPEG video streaming,
        calibration saves, and telemetry streaming under heavy background lock contention.
        """
        client = flask_app.test_client()
        stop_event = threading.Event()
        errors = queue.Queue()

        # Producer thread: rapidly updates latest_raw_frame and latest_annotated_frame
        def mock_producer():
            sample_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
            _, sample_jpeg = cv2.imencode('.jpg', sample_frame)
            sample_bytes = sample_jpeg.tobytes()

            while not stop_event.is_set():
                with flask_app_module.raw_frame_lock:
                    flask_app_module.latest_raw_frame = sample_frame.copy()

                with flask_app_module.annotated_frame_lock:
                    flask_app_module.latest_annotated_frame = sample_bytes

                with flask_app_module.telemetry_lock:
                    flask_app_module.latest_telemetry["timestamp"] = int(time.time())

                time.sleep(0.005)  # 200 FPS stress

        # Worker thread 1: Snapshot requests
        def snapshot_requester(count=30):
            try:
                for _ in range(count):
                    if stop_event.is_set():
                        break
                    res = client.get("/calibrate/snapshot")
                    assert res.status_code == 200
                    assert res.content_type == "image/jpeg"
                    assert len(res.data) > 0
                    time.sleep(0.01)
            except Exception as e:
                errors.put(("snapshot_requester", e))

        # Worker thread 2: Calibration save requests
        def calibration_saver(count=15):
            try:
                pts = [[100, 50], [500, 50], [550, 400], [80, 400]]
                for _ in range(count):
                    if stop_event.is_set():
                        break
                    res = client.post("/calibrate/save", json={
                        "points": pts,
                        "img_width": 640,
                        "img_height": 480
                    })
                    assert res.status_code == 200
                    time.sleep(0.02)
            except Exception as e:
                errors.put(("calibration_saver", e))

        # Worker thread 3: MJPEG stream consumer
        def mjpeg_consumer():
            try:
                res = client.get("/video_feed")
                assert res.status_code == 200
                chunks_read = 0
                for chunk in res.response:
                    if stop_event.is_set() or chunks_read >= 25:
                        break
                    assert len(chunk) > 0
                    chunks_read += 1
            except Exception as e:
                errors.put(("mjpeg_consumer", e))

        # Launch producer
        prod_thread = threading.Thread(target=mock_producer, daemon=True)
        prod_thread.start()

        # Launch 10 snapshot requesters, 5 calibration savers, 5 video stream consumers
        threads = []
        for _ in range(10):
            threads.append(threading.Thread(target=snapshot_requester, args=(25,)))
        for _ in range(5):
            threads.append(threading.Thread(target=calibration_saver, args=(10,)))
        for _ in range(5):
            threads.append(threading.Thread(target=mjpeg_consumer))

        start_time = time.time()
        for t in threads:
            t.start()

        # Join all threads with a strict 15-second deadlock timeout
        for t in threads:
            t.join(timeout=15.0)
            assert not t.is_alive(), "Deadlock detected! Thread did not finish within timeout."

        stop_event.set()
        prod_thread.join(timeout=2.0)

        elapsed = time.time() - start_time
        print(f"\n[STRESS TEST] Concurrency test finished in {elapsed:.2f}s with {len(threads)} concurrent threads.")

        # Check for any exceptions caught during concurrent execution
        collected_errors = []
        while not errors.empty():
            collected_errors.append(errors.get())

        assert len(collected_errors) == 0, f"Concurrency errors encountered: {collected_errors}"


class TestMilestone2Cleanup:
    """Task 4: Ensure perspective_matrix.npy is removed and fallback works."""

    def test_stale_matrix_removal_and_identity_fallback(self):
        """Verify perspective_matrix.npy removal leaves system functioning with identity fallback."""
        matrix_path = os.path.join(PROJECT_ROOT, "perspective_matrix.npy")
        if os.path.exists(matrix_path):
            os.remove(matrix_path)

        assert not os.path.exists(matrix_path), "perspective_matrix.npy still exists"

        # Transformer must fall back to 3x3 identity matrix
        transformer = PerspectiveTransformer(matrix_path=matrix_path)
        assert np.array_equal(transformer.matrix, np.eye(3, dtype=np.float32))

        # Warping should still succeed without crashing
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        warped = transformer.warp(dummy_frame)
        assert warped.shape == (400, 600, 3)
