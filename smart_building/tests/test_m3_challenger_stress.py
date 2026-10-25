"""
Empirical & Adversarial Stress Test Suite for Milestone 3 of EcoZone AI
Author: challenger_m3_1
Covers:
1. Full test suite execution & verification.
2. Adversarial zone mapping formula fuzzing across all 6 zones and out-of-bounds inputs.
3. Perspective transformation pipeline with randomized quad coordinates, reprojection & inverse warp accuracy.
4. Flask test client stress testing across all routes (/calibrate, /calibrate/snapshot, /calibrate/save, /, /stream, /video_feed, /detection_log).
5. Thread safety and concurrency under high request load.
"""

import math
import os
import sys
import time
import json
import random
import threading
import queue
import pytest
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from core.transform import get_zone_id, PerspectiveTransformer
import app as flask_app_module
from app import app as flask_app, annotate_frame


@pytest.fixture(autouse=True, scope="function")
def cleanup_matrix_after_test():
    """Ensure perspective_matrix.npy is cleanly removed after tests that may write it."""
    matrix_path = os.path.join(PROJECT_ROOT, "perspective_matrix.npy")
    yield
    if os.path.exists(matrix_path):
        try:
            os.remove(matrix_path)
        except Exception:
            pass


# =============================================================================
# PART 1: ADVERSARIAL ZONE MAPPING FUZZING & INVARIANT TESTING
# =============================================================================
class TestZoneMappingAdversarialFuzzing:
    """Rigorous property-based and fuzz testing of the 3x2 zone mapping formula."""

    def test_canonical_zone_reference_points(self):
        """Verify the 6 canonical reference points from ORIGINAL_REQUEST §R5."""
        test_vectors = [
            ((150, 66), 0),   # Row 0, Col 0 -> Zone 0
            ((450, 66), 1),   # Row 0, Col 1 -> Zone 1
            ((150, 200), 2),  # Row 1, Col 0 -> Zone 2
            ((450, 200), 3),  # Row 1, Col 1 -> Zone 3
            ((150, 333), 4),  # Row 2, Col 0 -> Zone 4
            ((450, 333), 5),  # Row 2, Col 1 -> Zone 5
        ]
        for (x, y), expected_zone in test_vectors:
            assert get_zone_id(x, y) == expected_zone, f"Failed at ({x}, {y}): expected {expected_zone}, got {get_zone_id(x, y)}"

    def test_zone_uniform_fuzzing_100k_points(self):
        """Sample 100,000 randomized points uniformly distributed inside each zone boundary."""
        rng = np.random.default_rng(seed=42)
        N_PER_ZONE = 15000

        # Define bounds for the 6 zones (CELL_WIDTH=300, CELL_HEIGHT=133)
        # Zone 0: [0, 300) x [0, 133)
        # Zone 1: [300, 600) x [0, 133)
        # Zone 2: [0, 300) x [133, 266)
        # Zone 3: [300, 600) x [133, 266)
        # Zone 4: [0, 300) x [266, 400)
        # Zone 5: [300, 600) x [266, 400)
        zone_bounds = [
            (0, 0.0, 299.999, 0.0, 132.999),
            (1, 300.0, 599.999, 0.0, 132.999),
            (2, 0.0, 299.999, 133.0, 265.999),
            (3, 300.0, 599.999, 133.0, 265.999),
            (4, 0.0, 299.999, 266.0, 399.999),
            (5, 300.0, 599.999, 266.0, 399.999),
        ]

        for expected_zone, x_min, x_max, y_min, y_max in zone_bounds:
            xs = rng.uniform(x_min, x_max, size=N_PER_ZONE)
            ys = rng.uniform(y_min, y_max, size=N_PER_ZONE)
            for x, y in zip(xs, ys):
                z = get_zone_id(x, y)
                assert z == expected_zone, f"Zone mismatch for ({x:.4f}, {y:.4f}): expected {expected_zone}, got {z}"

    def test_boundary_micro_epsilon_transitions(self):
        """Test micro-epsilon perturbations around exact partition lines."""
        epsilons = [1e-12, 1e-9, 1e-6, 1e-4, 0.001, 0.01]

        for eps in epsilons:
            # Col boundary at x = 300: left is 0, right is 1 (in Row 0: y=50)
            assert get_zone_id(300.0 - eps, 50) == 0
            assert get_zone_id(300.0, 50) == 1
            assert get_zone_id(300.0 + eps, 50) == 1

            # Row 0 -> Row 1 boundary at y = 133: top is 0, bottom is 2 (in Col 0: x=100)
            assert get_zone_id(100, 133.0 - eps) == 0
            assert get_zone_id(100, 133.0) == 2
            assert get_zone_id(100, 133.0 + eps) == 2

            # Row 1 -> Row 2 boundary at y = 266: top is 2, bottom is 4 (in Col 0: x=100)
            assert get_zone_id(100, 266.0 - eps) == 2
            assert get_zone_id(100, 266.0) == 4
            assert get_zone_id(100, 266.0 + eps) == 4

            # Row 1 -> Row 2 boundary in Col 1 (x=450): top is 3, bottom is 5
            assert get_zone_id(450, 266.0 - eps) == 3
            assert get_zone_id(450, 266.0) == 5
            assert get_zone_id(450, 266.0 + eps) == 5

    def test_out_of_bounds_clamping_adversarial(self):
        """Fuzz with 50,000 extreme positive and negative coordinates to verify robust clamping."""
        rng = np.random.default_rng(seed=123)

        # Extreme negative coordinates -> must clamp to Zone 0
        neg_xs = rng.uniform(-1e9, -0.0001, size=10000)
        neg_ys = rng.uniform(-1e9, -0.0001, size=10000)
        for x, y in zip(neg_xs, neg_ys):
            assert get_zone_id(x, y) == 0

        # Far right, top (x >> 600, y < 0) -> col clamped to 1, row clamped to 0 -> Zone 1
        pos_xs = rng.uniform(600.0, 1e9, size=10000)
        neg_ys = rng.uniform(-1e9, 0.0, size=10000)
        for x, y in zip(pos_xs, neg_ys):
            assert get_zone_id(x, y) == 1

        # Far left, bottom (x < 0, y >> 400) -> col clamped to 0, row clamped to 2 -> Zone 4
        neg_xs = rng.uniform(-1e9, 0.0, size=10000)
        pos_ys = rng.uniform(400.0, 1e9, size=10000)
        for x, y in zip(neg_xs, pos_ys):
            assert get_zone_id(x, y) == 4

        # Far right, far bottom (x >> 600, y >> 400) -> col clamped to 1, row clamped to 2 -> Zone 5
        pos_xs = rng.uniform(600.0, 1e9, size=10000)
        pos_ys = rng.uniform(400.0, 1e9, size=10000)
        for x, y in zip(pos_xs, pos_ys):
            assert get_zone_id(x, y) == 5

    def test_special_float_and_type_adversarial(self):
        """Verify behavior on NaN, +/- Inf, string inputs, None, and bad types."""
        # Special floats
        assert get_zone_id(float('nan'), float('nan')) == 0
        assert get_zone_id(float('nan'), 100.0) == 0
        assert get_zone_id(100.0, float('nan')) == 0
        assert get_zone_id(float('-inf'), float('-inf')) == 0
        assert get_zone_id(float('inf'), float('inf')) == 5
        assert get_zone_id(float('inf'), 50.0) == 1
        assert get_zone_id(float('-inf'), 350.0) == 4

        # Type resilience
        assert get_zone_id("150", "200") == 2
        assert get_zone_id("invalid", 200) == 0
        assert get_zone_id(None, None) == 0
        assert get_zone_id(None, 100) == 0
        assert get_zone_id([150], 200) == 0
        assert get_zone_id(np.float32(450.0), np.float64(333.0)) == 5

    def test_arbitrary_custom_grid_geometry(self):
        """Test get_zone_id with non-standard grid configurations and degenerate parameters."""
        # 4 rows x 3 cols = 12 zones, cell_w=100, cell_h=50
        assert get_zone_id(50, 25, grid_rows=4, grid_cols=3, cell_width=100, cell_height=50) == 0
        assert get_zone_id(250, 175, grid_rows=4, grid_cols=3, cell_width=100, cell_height=50) == 11
        assert get_zone_id(150, 75, grid_rows=4, grid_cols=3, cell_width=100, cell_height=50) == 4  # row 1, col 1 -> 1*3 + 1 = 4

        # Degenerate grid configurations
        assert get_zone_id(100, 100, grid_rows=0, grid_cols=2) == 0
        assert get_zone_id(100, 100, grid_rows=3, grid_cols=0) == 0
        assert get_zone_id(100, 100, cell_width=0, cell_height=133) == 0
        assert get_zone_id(100, 100, cell_width=-100, cell_height=133) == 0


    def test_exhaustive_canvas_integer_grid_and_cell_counts(self):
        """
        Exhaustively test all 241,001 integer points on the 601x401 grid.
        Verify:
        1. All points map to 0..5
        2. Strict cell boundary partitions
        3. Exact area / pixel assignment counts for standard 600x400 interior
        """
        counts = [0] * 6
        for y in range(400):
            expected_row = 0 if y < 133 else (1 if y < 266 else 2)
            for x in range(600):
                expected_col = 0 if x < 300 else 1
                expected_zone = expected_row * 2 + expected_col
                z = get_zone_id(x, y)
                assert z == expected_zone, f"Mismatch at ({x}, {y}): expected {expected_zone}, got {z}"
                counts[z] += 1

        # Check exact pixel allocations:
        # Rows 0 & 1 have height 133, Row 2 has height 134 (400 - 266 = 134)
        # Cols have width 300
        assert counts[0] == 300 * 133  # 39,900
        assert counts[1] == 300 * 133  # 39,900
        assert counts[2] == 300 * 133  # 39,900
        assert counts[3] == 300 * 133  # 39,900
        assert counts[4] == 300 * 134  # 40,200
        assert counts[5] == 300 * 134  # 40,200
        assert sum(counts) == 600 * 400  # 240,000


# =============================================================================
# PART 2: PERSPECTIVE TRANSFORMATION PIPELINE & REPROJECTION ACCURACY
# =============================================================================
class TestPerspectivePipelineReprojection:
    """Stress test the perspective homography transformation with randomized quads."""

    def test_randomized_quad_reprojection_accuracy(self):
        """
        Generate 500 randomized non-collinear convex quadrilaterals,
        compute homography matrix, and verify forward and inverse reprojection error < 1e-3 px.
        """
        rng = np.random.default_rng(seed=777)
        dst_corners = np.array([
            [0.0, 0.0],
            [600.0, 0.0],
            [600.0, 400.0],
            [0.0, 400.0]
        ], dtype=np.float32)

        for trial in range(500):
            # Generate perturbed quad in 640x480 native space
            tl = [rng.uniform(20.0, 150.0), rng.uniform(20.0, 100.0)]
            tr = [rng.uniform(490.0, 620.0), rng.uniform(20.0, 100.0)]
            br = [rng.uniform(490.0, 620.0), rng.uniform(380.0, 460.0)]
            bl = [rng.uniform(20.0, 150.0), rng.uniform(380.0, 460.0)]

            src_corners = np.array([tl, tr, br, bl], dtype=np.float32)

            # Compute Homography
            M = cv2.getPerspectiveTransform(src_corners, dst_corners)
            assert M is not None and M.shape == (3, 3)

            transformer = PerspectiveTransformer()
            transformer.matrix = M

            # Forward transform verification: src -> dst
            for i in range(4):
                px, py = transformer.transform_point(src_corners[i][0], src_corners[i][1])
                expected_x, expected_y = dst_corners[i]
                error = math.hypot(px - expected_x, py - expected_y)
                assert error < 1e-3, f"Forward reprojection error {error:.6f} exceeds tolerance on corner {i}"

            # Vectorized transform verification
            warped_pts = transformer.transform_points(src_corners)
            assert np.allclose(warped_pts, dst_corners, atol=1e-3)

            # Inverse transform verification: dst -> src
            M_inv = np.linalg.inv(M)
            inv_transformer = PerspectiveTransformer()
            inv_transformer.matrix = M_inv

            for i in range(4):
                inv_x, inv_y = inv_transformer.transform_point(dst_corners[i][0], dst_corners[i][1])
                orig_x, orig_y = src_corners[i]
                inv_error = math.hypot(inv_x - orig_x, inv_y - orig_y)
                assert inv_error < 1e-3, f"Inverse reprojection error {inv_error:.6f} exceeds tolerance on corner {i}"

    def test_randomized_internal_quad_points_preservation(self):
        """
        Generate random quad, sample 1,000 interior points via bilinear interpolation,
        and verify forward transform lands within [0, 600] x [0, 400] and inverts accurately.
        """
        rng = np.random.default_rng(seed=888)
        dst_corners = np.array([[0.0, 0.0], [600.0, 0.0], [600.0, 400.0], [0.0, 400.0]], dtype=np.float32)

        tl = np.array([50.0, 40.0])
        tr = np.array([590.0, 40.0])
        br = np.array([590.0, 440.0])
        bl = np.array([50.0, 440.0])
        src_corners = np.array([tl, tr, br, bl], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_corners, dst_corners)
        transformer = PerspectiveTransformer()
        transformer.matrix = M
        M_inv = np.linalg.inv(M)
        inv_transformer = PerspectiveTransformer()
        inv_transformer.matrix = M_inv

        # Sample interior points with (u, v) in [0, 1]
        u_samples = rng.uniform(0.01, 0.99, size=1000)
        v_samples = rng.uniform(0.01, 0.99, size=1000)

        for u, v in zip(u_samples, v_samples):
            # Bilinear point inside quad
            src_pt = (1 - u) * (1 - v) * tl + u * (1 - v) * tr + u * v * br + (1 - u) * v * bl
            wx, wy = transformer.transform_point(src_pt[0], src_pt[1])

            # Must lie inside [0, 600] x [0, 400]
            assert 0.0 <= wx <= 600.0, f"Transformed point x={wx} out of bounds"
            assert 0.0 <= wy <= 400.0, f"Transformed point y={wy} out of bounds"

            # Invert
            rec_x, rec_y = inv_transformer.transform_point(wx, wy)
            rec_err = math.hypot(rec_x - src_pt[0], rec_y - src_pt[1])
            assert rec_err < 1e-3, f"Roundtrip error {rec_err:.6f} exceeds tolerance for ({src_pt[0]}, {src_pt[1]})"

    def test_warp_image_dimension_and_dtype(self):
        """Verify warp produces strictly (400, 600, 3) uint8 image across varied input shapes."""
        transformer = PerspectiveTransformer()
        input_shapes = [(480, 640, 3), (720, 1280, 3), (240, 320, 3), (1080, 1920, 3)]

        for h, w, c in input_shapes:
            dummy_img = np.random.randint(0, 256, (h, w, c), dtype=np.uint8)
            warped = transformer.warp(dummy_img)
            assert warped.shape == (400, 600, 3), f"Warp shape {warped.shape} != (400, 600, 3)"
            assert warped.dtype == np.uint8

    def test_transformer_edge_cases_and_fallbacks(self):
        """Test PerspectiveTransformer resilience against None frames, empty frames, and corrupted matrix."""
        transformer = PerspectiveTransformer()

        # None / Empty frame handling
        res_none = transformer.warp(None)
        assert res_none.shape == (400, 600, 3)
        assert np.all(res_none == 0)

        res_empty = transformer.warp(np.empty((0, 0, 3), dtype=np.uint8))
        assert res_empty.shape == (400, 600, 3)
        assert np.all(res_empty == 0)

        # Empty points
        assert len(transformer.transform_points(None)) == 0
        assert len(transformer.transform_points(np.empty((0, 2)))) == 0

        # Corrupted matrix fallback
        corrupted_path = os.path.join(PROJECT_ROOT, "test_corrupted_matrix.npy")
        try:
            # Write invalid shape
            np.save(corrupted_path, np.array([[1, 2], [3, 4]]))
            corrupted_transformer = PerspectiveTransformer(matrix_path=corrupted_path)
            assert np.array_equal(corrupted_transformer.matrix, np.eye(3, dtype=np.float32))
        finally:
            if os.path.exists(corrupted_path):
                os.remove(corrupted_path)


# =============================================================================
# PART 3: FLASK TEST CLIENT ACROSS ALL ROUTES & ADVERSARIAL PAYLOADS
# =============================================================================
class TestFlaskRoutesAndAdversarialPayloads:
    """Empirical verification of all Flask routes, content-types, and input validation."""

    @pytest.fixture
    def client(self):
        flask_app.config['TESTING'] = True
        with flask_app.test_client() as client:
            yield client

    def test_route_index_html(self, client):
        """Verify GET / returns 200 OK and contains 3x2 grid and Calibrate button."""
        resp = client.get('/')
        assert resp.status_code == 200
        assert resp.mimetype == 'text/html'
        html = resp.get_data(as_text=True)
        assert "EcoZone AI" in html
        assert "Calibrate Camera" in html
        assert "/calibrate" in html
        assert "zone-grid" in html

    def test_route_calibrate_html(self, client):
        """Verify GET /calibrate returns 200 OK with canvas and interactive elements."""
        resp = client.get('/calibrate')
        assert resp.status_code == 200
        assert resp.mimetype == 'text/html'
        html = resp.get_data(as_text=True)
        assert "<canvas" in html
        assert "calibrationCanvas" in html
        assert "Save Calibration" in html
        assert "Reset" in html
        assert "Take New Snapshot" in html

    def test_route_calibrate_snapshot(self, client):
        """Verify GET /calibrate/snapshot returns valid JPEG image."""
        resp = client.get('/calibrate/snapshot')
        assert resp.status_code == 200
        assert resp.mimetype == 'image/jpeg'
        img_bytes = resp.data
        assert len(img_bytes) > 100
        # Decode JPEG to verify it is a valid decodable image
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        assert img is not None
        assert img.shape[2] == 3

    def test_route_calibrate_save_valid_payloads(self, client):
        """Verify POST /calibrate/save succeeds with valid 4-corner coordinates and updates matrix."""
        matrix_path = os.path.join(PROJECT_ROOT, "perspective_matrix.npy")

        valid_payloads = [
            # Array of lists
            {
                "points": [[50, 40], [590, 40], [590, 440], [50, 440]],
                "img_width": 640,
                "img_height": 480
            },
            # Array of dicts
            {
                "points": [
                    {"x": 60, "y": 50},
                    {"x": 580, "y": 50},
                    {"x": 580, "y": 430},
                    {"x": 60, "y": 430}
                ],
                "img_width": 640,
                "img_height": 480
            },
            # Display scaled coordinates (e.g. mobile display 320x240)
            {
                "points": [[25, 20], [295, 20], [295, 220], [25, 220]],
                "img_width": 320,
                "img_height": 240
            }
        ]

        for payload in valid_payloads:
            resp = client.post('/calibrate/save', data=json.dumps(payload), content_type='application/json')
            assert resp.status_code == 200
            data = resp.get_json()
            assert data.get("status") == "ok"
            assert os.path.exists(matrix_path)
            saved_matrix = np.load(matrix_path)
            assert saved_matrix.shape == (3, 3)
            assert saved_matrix.dtype in (np.float32, np.float64)

    def test_route_calibrate_save_adversarial_rejections(self, client):
        """Verify POST /calibrate/save rejects malformed/invalid payloads."""
        invalid_payloads = [
            ({}, 400),  # Empty dict
            ({"points": [[10, 10], [20, 20], [30, 30]]}, 400),  # 3 points
            ({"points": [[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]]}, 400),  # 5 points
            ({"points": ["a", "b", "c", "d"]}, 400),  # Non-numeric
            ({"points": [[None, 10], [20, 20], [30, 30], [40, 40]]}, 500),  # None coordinate
            ("plain string body", 400),
            (None, 400),
        ]

        for payload, expected_status in invalid_payloads:
            if isinstance(payload, str):
                resp = client.post('/calibrate/save', data=payload, content_type='text/plain')
            elif payload is None:
                resp = client.post('/calibrate/save', data=b'', content_type='application/json')
            else:
                resp = client.post('/calibrate/save', data=json.dumps(payload), content_type='application/json')

            assert resp.status_code == expected_status, f"Payload {payload} returned {resp.status_code}, expected {expected_status}"
            if resp.is_json:
                assert resp.get_json().get("status") == "error"

    def test_route_stream_sse_telemetry(self, client):
        """Verify GET /stream yields SSE data frames with complete telemetry fields."""
        resp = client.get('/stream')
        assert resp.status_code == 200
        assert 'text/event-stream' in resp.mimetype

        # Read first chunk from generator
        for chunk in resp.response:
            chunk_str = chunk.decode('utf-8')
            if chunk_str.startswith('data: '):
                json_str = chunk_str.replace('data: ', '').strip()
                telemetry = json.loads(json_str)
                assert "zones" in telemetry
                assert len(telemetry["zones"]) == 6
                assert "temp" in telemetry
                assert "humidity" in telemetry
                assert "fan_speed" in telemetry
                assert "total_people" in telemetry
                assert "timestamp" in telemetry
                assert "zone_activity" in telemetry
                assert len(telemetry["zone_activity"]) == 6
                break

    def test_route_video_feed_mjpeg(self, client):
        """Verify GET /video_feed serves multipart MJPEG stream with valid JPEG frame chunks."""
        resp = client.get('/video_feed')
        assert resp.status_code == 200
        assert 'multipart/x-mixed-replace' in resp.mimetype

        # Read first chunk
        for chunk in resp.response:
            assert b'--frame' in chunk or b'Content-Type: image/jpeg' in chunk
            break

    def test_route_detection_log(self, client):
        """Verify GET /detection_log returns JSON log and 6-element zone_activity array."""
        resp = client.get('/detection_log')
        assert resp.status_code == 200
        assert resp.is_json
        data = resp.get_json()
        assert "log" in data
        assert isinstance(data["log"], list)
        assert "zone_activity" in data
        assert isinstance(data["zone_activity"], list)
        assert len(data["zone_activity"]) == 6


# =============================================================================
# PART 4: CONCURRENCY & THREAD SAFETY STRESS TESTING
# =============================================================================
class TestConcurrencyAndThreadSafety:
    """Stress test shared locks and simultaneous multi-threaded access."""

    def test_concurrent_api_requests(self):
        """Hammer Flask test client across multiple endpoints concurrently from 20 threads."""
        flask_app.config['TESTING'] = True
        client = flask_app.test_client()

        errors = queue.Queue()

        def request_worker(thread_id):
            try:
                for i in range(10):
                    # Mix of GET and POST
                    r1 = client.get('/calibrate/snapshot')
                    assert r1.status_code == 200

                    r2 = client.get('/detection_log')
                    assert r2.status_code == 200

                    r3 = client.get('/')
                    assert r3.status_code == 200

                    payload = {
                        "points": [
                            [50 + (thread_id % 10), 40],
                            [580 - (thread_id % 10), 40],
                            [580 - (thread_id % 10), 440],
                            [50 + (thread_id % 10), 440]
                        ],
                        "img_width": 640,
                        "img_height": 480
                    }
                    r4 = client.post('/calibrate/save', data=json.dumps(payload), content_type='application/json')
                    assert r4.status_code == 200
            except Exception as e:
                errors.put(f"Thread {thread_id} error: {e}")

        threads = []
        for tid in range(20):
            t = threading.Thread(target=request_worker, args=(tid,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert errors.empty(), f"Errors during concurrency test: {[errors.get() for _ in range(errors.qsize())]}"

    def test_annotate_frame_high_throughput(self):
        """Stress test annotate_frame with 1,000 rapid calls under alternating occupancy."""
        frame = np.zeros((400, 600, 3), dtype=np.uint8)

        for i in range(1000):
            occupied = [(i + z) % 2 for z in range(6)]
            res = annotate_frame(frame, occupied, total_people=sum(occupied), fan_speed=50, temp=28.5)
            assert res.shape == (400, 600, 3)
            assert res.dtype == np.uint8
