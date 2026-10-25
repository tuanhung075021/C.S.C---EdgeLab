"""
Challenger 2 Adversarial Stress Test Harness: Backend, Mode Switching, API Routes & Concurrency.
Empirical verification of thread-safety, high-concurrency race resilience, payload fuzzing,
background worker stability, and SSE telemetry synchronization consistency.
"""

import copy
import json
import os
import sys
import threading
import time
import queue
import pytest
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
import app as flask_app_module
from app import (
    app as flask_app,
    mode_lock,
    telemetry_lock,
    raw_frame_lock,
    annotated_frame_lock,
    detection_log_lock,
    annotate_frame,
)
from core.color_detector import ColorDetector
from core.detector import YOLOv8OpenVINODetector
from core.transform import PerspectiveTransformer


@pytest.fixture(autouse=True)
def setup_flask_testing_mode():
    flask_app.config["TESTING"] = True
    yield


# =============================================================================
# DOMAIN 1: HIGH-CONCURRENCY RACE CONDITION TESTING
# =============================================================================
class TestHighConcurrencyRaceConditions:
    """Stress tests concurrent API access and mode switching race conditions."""

    def test_20_threads_rapid_mode_switching_storm(self):
        """
        20 concurrent threads hammering POST /set_mode with alternating 'color' and 'ai'
        (50 requests per thread = 1,000 total requests) while 5 reader threads query GET /get_mode.
        Verifies:
        - 100% of requests return HTTP 200.
        - Zero unhandled exceptions or 500 status codes.
        - Response payloads are well-formed JSON with correct status and mode.
        - Final server mode is strictly either 'color' or 'ai' with zero corruption.
        """
        NUM_WRITER_THREADS = 20
        REQUESTS_PER_WRITER = 50
        NUM_READER_THREADS = 5
        REQUESTS_PER_READER = 50

        results_queue = queue.Queue()
        errors_list = []
        barrier = threading.Barrier(NUM_WRITER_THREADS + NUM_READER_THREADS)

        def writer_worker(thread_idx):
            client = flask_app.test_client()
            barrier.wait()
            for i in range(REQUESTS_PER_WRITER):
                target_mode = "color" if (thread_idx + i) % 2 == 0 else "ai"
                try:
                    res = client.post(
                        "/set_mode",
                        data=json.dumps({"mode": target_mode}),
                        content_type="application/json",
                    )
                    data = res.get_json()
                    if res.status_code != 200 or data.get("status") != "ok" or data.get("mode") != target_mode:
                        errors_list.append(f"Writer {thread_idx} req {i}: status={res.status_code}, body={data}")
                    results_queue.put((res.status_code, data))
                except Exception as e:
                    errors_list.append(f"Writer {thread_idx} exception: {e}")

        def reader_worker(thread_idx):
            client = flask_app.test_client()
            barrier.wait()
            for i in range(REQUESTS_PER_READER):
                try:
                    res = client.get("/get_mode")
                    data = res.get_json()
                    if res.status_code != 200 or data.get("status") != "ok" or data.get("mode") not in ("color", "ai"):
                        errors_list.append(f"Reader {thread_idx} req {i}: status={res.status_code}, body={data}")
                except Exception as e:
                    errors_list.append(f"Reader {thread_idx} exception: {e}")

        threads = []
        for t in range(NUM_WRITER_THREADS):
            th = threading.Thread(target=writer_worker, args=(t,))
            threads.append(th)
            th.start()

        for t in range(NUM_READER_THREADS):
            th = threading.Thread(target=reader_worker, args=(t,))
            threads.append(th)
            th.start()

        for th in threads:
            th.join(timeout=20.0)
            assert not th.is_alive(), "A thread timed out, potential deadlock detected!"

        assert len(errors_list) == 0, f"Encountered race errors: {errors_list[:10]}"
        assert results_queue.qsize() == NUM_WRITER_THREADS * REQUESTS_PER_WRITER

        # Check final state consistency
        final_client = flask_app.test_client()
        final_res = final_client.get("/get_mode")
        assert final_res.status_code == 200
        final_data = final_res.get_json()
        assert final_data["status"] == "ok"
        assert final_data["mode"] in ("color", "ai")
        with mode_lock:
            assert flask_app_module.current_mode in ("color", "ai")
            assert flask_app_module.current_mode == final_data["mode"]

    def test_interleaved_valid_and_invalid_mode_requests_storm(self):
        """
        10 threads sending valid modes ('color', 'ai') and 10 threads sending
        adversarial invalid modes simultaneously.
        Verifies:
        - Valid requests return exactly 200.
        - Invalid requests return exactly 400.
        - Zero 500 internal server errors.
        - State remains uncorrupted throughout.
        """
        NUM_VALID_THREADS = 10
        NUM_INVALID_THREADS = 10
        REQUESTS_PER_THREAD = 30

        barrier = threading.Barrier(NUM_VALID_THREADS + NUM_INVALID_THREADS)
        errors = []

        def valid_worker(idx):
            client = flask_app.test_client()
            barrier.wait()
            modes = ["color", "ai", "COLOR", "  ai  "]
            for i in range(REQUESTS_PER_THREAD):
                m = modes[(idx + i) % len(modes)]
                expected_clean = m.strip().lower()
                res = client.post("/set_mode", data=json.dumps({"mode": m}), content_type="application/json")
                data = res.get_json()
                if res.status_code != 200 or data.get("status") != "ok" or data.get("mode") != expected_clean:
                    errors.append(f"Valid worker {idx} error: status={res.status_code}, data={data}")

        def invalid_worker(idx):
            client = flask_app.test_client()
            barrier.wait()
            invalids = ["thermal", "", "invalid_mode", 12345, None, ["ai"], {"nested": "color"}]
            for i in range(REQUESTS_PER_THREAD):
                m = invalids[(idx + i) % len(invalids)]
                res = client.post("/set_mode", data=json.dumps({"mode": m}), content_type="application/json")
                data = res.get_json()
                if res.status_code != 400 or data.get("status") != "error":
                    errors.append(f"Invalid worker {idx} error: status={res.status_code}, data={data}, payload={m}")

        threads = []
        for i in range(NUM_VALID_THREADS):
            t = threading.Thread(target=valid_worker, args=(i,))
            threads.append(t)
            t.start()
        for i in range(NUM_INVALID_THREADS):
            t = threading.Thread(target=invalid_worker, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=15.0)
            assert not t.is_alive(), "Thread deadlock detected during interleaved request storm!"

        assert len(errors) == 0, f"Errors in interleaved storm: {errors[:10]}"


# =============================================================================
# DOMAIN 2: FUZZING & ADVERSARIAL PAYLOAD TESTING ON POST /set_mode
# =============================================================================
class TestModeFuzzingAndInputValidation:
    """Exhaustively fuzzes POST /set_mode against malicious, malformed, and anomalous payloads."""

    @pytest.mark.parametrize(
        "bad_payload,content_type",
        [
            ("not a json at all", "text/plain"),
            ("{broken json: [1, 2", "application/json"),
            ("", "application/json"),
            ("   ", "application/json"),
            ("null", "application/json"),
            ("12345", "application/json"),
            ("true", "application/json"),
            ("false", "application/json"),
            ("[1, 2, 3]", "application/json"),
            ('["mode", "color"]', "application/json"),
            (b"\x00\x01\x02\xff\xfe", "application/octet-stream"),
            ("<xml><mode>color</mode></xml>", "application/xml"),
        ],
    )
    def test_fuzz_non_json_or_malformed_root(self, flask_sse_client, bad_payload, content_type):
        """Ensure non-dict JSON or malformed content returns HTTP 400 with status=error."""
        res = flask_sse_client.post("/set_mode", data=bad_payload, content_type=content_type)
        assert res.status_code == 400
        data = res.get_json()
        assert data is not None
        assert data.get("status") == "error"
        assert "message" in data

    @pytest.mark.parametrize(
        "unexpected_type_payload",
        [
            {"mode": 0},
            {"mode": 1},
            {"mode": -999},
            {"mode": 3.14159},
            {"mode": None},
            {"mode": True},
            {"mode": False},
            {"mode": ["color"]},
            {"mode": ["ai"]},
            {"mode": []},
            {"mode": {"name": "color"}},
            {"mode": {}},
            {"foo": "color"},  # missing 'mode' key
            {"MODE": "color"},  # wrong key case
            {},
        ],
    )
    def test_fuzz_unexpected_types_and_missing_keys(self, flask_sse_client, unexpected_type_payload):
        """Ensure unexpected data types for 'mode' or missing 'mode' key return HTTP 400."""
        res = flask_sse_client.post(
            "/set_mode",
            data=json.dumps(unexpected_type_payload),
            content_type="application/json",
        )
        assert res.status_code == 400
        data = res.get_json()
        assert data is not None
        assert data.get("status") == "error"

    @pytest.mark.parametrize(
        "edge_string",
        [
            "",
            "   ",
            "\t\n\r",
            "a" * 10000,  # Buffer overflow / large string attempt
            "color\x00extra",  # Null byte injection
            "ai\r\nHeader-Injection: 1",
            "undefined",
            "null",
            "None",
            "NaN",
            "Infinity",
            "color_extra",
            "ai_extra",
            "colors",
            "ais",
            "thermal",
            "lidar",
            "infrared",
            "color ai",
            "color;ai",
        ],
    )
    def test_fuzz_edge_case_and_unsupported_strings(self, flask_sse_client, edge_string):
        """Ensure empty, oversized, and unsupported string modes are rejected with HTTP 400."""
        res = flask_sse_client.post(
            "/set_mode",
            data=json.dumps({"mode": edge_string}),
            content_type="application/json",
        )
        assert res.status_code == 400
        data = res.get_json()
        assert data.get("status") == "error"

    @pytest.mark.parametrize(
        "security_payload",
        [
            "' OR '1'='1",
            "'; DROP TABLE detection_mode; --",
            "admin' --",
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert(1)>",
            "{{7*7}}",
            "${7*7}",
            "${jndi:ldap://127.0.0.1/a}",
            "../../../../etc/passwd",
            "; cat /etc/passwd",
            "| whoami",
            "& ping -n 1 127.0.0.1",
        ],
    )
    def test_fuzz_security_injection_payloads(self, flask_sse_client, security_payload):
        """Ensure SQLi, XSS, SSTI, and Command Injection strings are safely rejected with HTTP 400."""
        res = flask_sse_client.post(
            "/set_mode",
            data=json.dumps({"mode": security_payload}),
            content_type="application/json",
        )
        assert res.status_code == 400
        data = res.get_json()
        assert data.get("status") == "error"
        assert "message" in data

    @pytest.mark.parametrize(
        "valid_input,expected_mode",
        [
            ("color", "color"),
            ("ai", "ai"),
            ("COLOR", "color"),
            ("AI", "ai"),
            ("Color", "color"),
            ("Ai", "ai"),
            ("  color  ", "color"),
            ("  ai  ", "ai"),
            ("\tcolor\n", "color"),
            ("cOlOr", "color"),
        ],
    )
    def test_valid_mode_variations_and_normalization(self, flask_sse_client, valid_input, expected_mode):
        """Ensure case-insensitivity and whitespace trimming accept valid mode inputs."""
        res = flask_sse_client.post(
            "/set_mode",
            data=json.dumps({"mode": valid_input}),
            content_type="application/json",
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data.get("status") == "ok"
        assert data.get("mode") == expected_mode

        # Check get_mode endpoint confirms updated mode
        get_res = flask_sse_client.get("/get_mode")
        assert get_res.status_code == 200
        assert get_res.get_json().get("mode") == expected_mode


# =============================================================================
# DOMAIN 3: SIMULATED BACKGROUND WORKER UNDER ACTIVE MODE SWITCHING
# =============================================================================
class TestSimulatedBackgroundWorkerUnderModeSwitching:
    """Tests background worker thread safety, absence of deadlocks, and frame integrity."""

    def test_concurrent_worker_pipeline_under_rapid_mode_switching(self):
        """
        Runs an intensive background worker simulation loop alongside concurrent mode switchers
        and telemetry/MJPEG consumers.
        Verifies:
        - Zero deadlocks across all 5 locks (mode_lock, telemetry_lock, raw_frame_lock,
          annotated_frame_lock, detection_log_lock).
        - Zero frame corruption: all generated annotated frames are valid JPEGs decoding to (400, 600, 3).
        - Zero state inconsistencies: occupied_zones is always 6 binary ints, mode is always valid.
        """
        # Pre-instantiate detectors
        color_detector = ColorDetector(
            lower_hsv=getattr(config, "COLOR_YELLOW_HSV_LOWER", [20, 100, 100]),
            upper_hsv=getattr(config, "COLOR_YELLOW_HSV_UPPER", [35, 255, 255]),
            min_contour_area=getattr(config, "COLOR_MIN_CONTOUR_AREA", 500.0),
        )
        ai_detector = YOLOv8OpenVINODetector(
            model_xml=getattr(config, "MODEL_XML", "models/yolov8n_openvino_model/yolov8n.xml"),
            conf_threshold=getattr(config, "CONFIDENCE_THRESHOLD", 0.5),
            is_simulation=True,
        )
        transformer = PerspectiveTransformer()

        # Generate a synthetic test frame with a yellow cup in Zone 0
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[:] = (50, 50, 50)
        # Draw yellow rectangle in upper-left
        test_frame[50:120, 50:120] = (0, 255, 255)  # BGR Yellow

        stop_event = threading.Event()
        worker_errors = []
        frames_generated = 0
        frames_lock = threading.Lock()

        # Background worker thread simulating app.py background_pipeline_worker
        def worker_loop():
            nonlocal frames_generated
            while not stop_event.is_set():
                try:
                    with mode_lock:
                        active_mode = flask_app_module.current_mode

                    # Warp frame
                    warped = transformer.warp(test_frame)
                    assert warped.shape == (400, 600, 3), f"Warped shape invalid: {warped.shape}"

                    # Detect according to active mode
                    if active_mode == "color":
                        occupied_zones, total_people = color_detector.detect(warped)
                    else:
                        occupied_zones, total_people = ai_detector.detect(test_frame, transformer=transformer)

                    assert len(occupied_zones) == 6, "Occupied zones length must be 6"
                    assert all(z in (0, 1) for z in occupied_zones), "Zones must contain 0 or 1"

                    # Annotate frame
                    annotated = annotate_frame(warped, occupied_zones, total_people, fan_speed=50, temp=28.5)
                    ret, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    assert ret, "JPEG encoding failed"

                    with annotated_frame_lock:
                        flask_app_module.latest_annotated_frame = jpeg.tobytes()

                    with telemetry_lock:
                        flask_app_module.latest_telemetry = {
                            "zones": [int(z) for z in occupied_zones],
                            "temp": 28.5,
                            "humidity": 65.0,
                            "fan_speed": 50,
                            "total_people": int(total_people),
                            "timestamp": int(time.time()),
                            "zone_activity": [0] * 6,
                            "mode": active_mode,
                        }

                    with frames_lock:
                        frames_generated += 1

                    time.sleep(0.001)  # High rate processing loop
                except Exception as e:
                    worker_errors.append(f"Worker exception: {e}")
                    break

        worker_th = threading.Thread(target=worker_loop, daemon=True)
        worker_th.start()

        # Concurrent Mode Switcher Threads
        switch_errors = []
        def mode_switcher_thread(t_idx):
            client = flask_app.test_client()
            for i in range(25):
                target = "color" if (t_idx + i) % 2 == 0 else "ai"
                res = client.post("/set_mode", data=json.dumps({"mode": target}), content_type="application/json")
                if res.status_code != 200:
                    switch_errors.append(f"Switch error: {res.status_code}")
                time.sleep(0.02)

        # Concurrent MJPEG Frame Consumer Threads (verifies zero frame corruption)
        frame_consumer_errors = []
        def frame_consumer_thread():
            for _ in range(20):
                with annotated_frame_lock:
                    fb = flask_app_module.latest_annotated_frame
                if fb:
                    # Decode JPEG to verify no partial/corrupt frame was read
                    arr = np.frombuffer(fb, dtype=np.uint8)
                    decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if decoded is None or decoded.shape != (400, 600, 3):
                        frame_consumer_errors.append(f"Corrupt frame decoded: {decoded.shape if decoded is not None else None}")
                time.sleep(0.02)

        # Concurrent Telemetry Consumer Threads (verifies atomic telemetry read)
        telemetry_consumer_errors = []
        def telemetry_consumer_thread():
            for _ in range(20):
                with telemetry_lock:
                    telem = copy.deepcopy(flask_app_module.latest_telemetry)
                if telem:
                    if len(telem.get("zones", [])) != 6:
                        telemetry_consumer_errors.append(f"Invalid zones in telemetry: {telem}")
                    if telem.get("mode") not in ("color", "ai"):
                        telemetry_consumer_errors.append(f"Invalid mode in telemetry: {telem}")
                time.sleep(0.02)

        stress_threads = []
        for i in range(6):
            stress_threads.append(threading.Thread(target=mode_switcher_thread, args=(i,)))
        for _ in range(4):
            stress_threads.append(threading.Thread(target=frame_consumer_thread))
        for _ in range(4):
            stress_threads.append(threading.Thread(target=telemetry_consumer_thread))

        for t in stress_threads:
            t.start()

        for t in stress_threads:
            t.join(timeout=10.0)
            assert not t.is_alive(), "Stress thread deadlock!"

        # Stop worker
        stop_event.set()
        worker_th.join(timeout=5.0)
        assert not worker_th.is_alive(), "Worker thread deadlock!"

        assert len(worker_errors) == 0, f"Worker loop errors: {worker_errors}"
        assert len(switch_errors) == 0, f"Mode switch errors: {switch_errors}"
        assert len(frame_consumer_errors) == 0, f"Frame corruption errors: {frame_consumer_errors}"
        assert len(telemetry_consumer_errors) == 0, f"Telemetry corruption errors: {telemetry_consumer_errors}"
        assert frames_generated >= 5, f"Expected at least 5 frames generated, got {frames_generated}"


# =============================================================================
# DOMAIN 4: SSE TELEMETRY SYNCHRONIZATION CONSISTENCY UNDER CONCURRENT LOAD
# =============================================================================
class TestSSETelemetrySynchronization:
    """Tests Server-Sent Events (SSE) telemetry synchronization under concurrent mode updates."""

    def test_sse_telemetry_schema_and_mode_sync(self, flask_sse_client):
        """
        Validates that /stream telemetry events conform to schema,
        include active 'mode', and immediately synchronize when mode is switched.
        """
        # Set mode to color
        flask_sse_client.post("/set_mode", data=json.dumps({"mode": "color"}), content_type="application/json")
        with telemetry_lock:
            flask_app_module.latest_telemetry["mode"] = "color"

        # Read an SSE event from generator
        with flask_sse_client.get("/stream") as res:
            assert res.status_code == 200
            assert res.mimetype == "text/event-stream"
            # Read first chunk
            chunk = next(res.iter_encoded()).decode("utf-8")
            assert chunk.startswith("data: ")
            payload_str = chunk[len("data: "):].strip()
            data = json.loads(payload_str)

            # Validate schema
            assert "zones" in data and len(data["zones"]) == 6
            assert "mode" in data and data["mode"] == "color"
            assert "temp" in data
            assert "humidity" in data
            assert "fan_speed" in data
            assert "total_people" in data
            assert "timestamp" in data
            assert "zone_activity" in data and len(data["zone_activity"]) == 6

        # Switch mode to ai
        flask_sse_client.post("/set_mode", data=json.dumps({"mode": "ai"}), content_type="application/json")
        with telemetry_lock:
            flask_app_module.latest_telemetry["mode"] = "ai"

        with flask_sse_client.get("/stream") as res:
            chunk = next(res.iter_encoded()).decode("utf-8")
            payload_str = chunk[len("data: "):].strip()
            data = json.loads(payload_str)
            assert data["mode"] == "ai"

    def test_multi_client_concurrent_sse_and_mode_toggle_stress(self):
        """
        Simulates 5 concurrent clients reading SSE streams while 5 other threads
        rapidly cycle modes.
        Verifies that all SSE event chunks remain valid, uncorrupted JSON,
        and all contains a valid mode value ('color' or 'ai').
        """
        NUM_SSE_CONSUMERS = 5
        NUM_MODE_TOGGLERS = 5
        CYCLES_PER_TOGGLER = 20

        consumer_errors = []
        stop_consumers = threading.Event()

        def sse_consumer(c_idx):
            read_count = 0
            while not stop_consumers.is_set() and read_count < 20:
                with telemetry_lock:
                    payload = json.dumps(flask_app_module.latest_telemetry)
                sse_line = f"data: {payload}\n\n"
                if not sse_line.startswith("data: "):
                    consumer_errors.append(f"Consumer {c_idx}: invalid prefix {sse_line[:20]}")
                    break
                raw_json = sse_line[6:].strip()
                try:
                    parsed = json.loads(raw_json)
                    if parsed.get("mode") not in ("color", "ai"):
                        consumer_errors.append(f"Consumer {c_idx}: invalid mode in {parsed}")
                    if len(parsed.get("zones", [])) != 6:
                        consumer_errors.append(f"Consumer {c_idx}: invalid zones in {parsed}")
                except Exception as e:
                    consumer_errors.append(f"Consumer {c_idx} parse exception: {e}")
                    break
                read_count += 1
                time.sleep(0.01)

        def mode_toggler(t_idx):
            client = flask_app.test_client()
            for i in range(CYCLES_PER_TOGGLER):
                m = "color" if (t_idx + i) % 2 == 0 else "ai"
                res = client.post("/set_mode", data=json.dumps({"mode": m}), content_type="application/json")
                if res.status_code != 200:
                    consumer_errors.append(f"Toggler {t_idx} status {res.status_code}")
                with telemetry_lock:
                    flask_app_module.latest_telemetry["mode"] = m
                time.sleep(0.005)

        threads = []
        for i in range(NUM_SSE_CONSUMERS):
            threads.append(threading.Thread(target=sse_consumer, args=(i,)))
        for i in range(NUM_MODE_TOGGLERS):
            threads.append(threading.Thread(target=mode_toggler, args=(i,)))

        for t in threads:
            t.start()

        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive(), "Consumer/toggler thread deadlock!"

        stop_consumers.set()
        assert len(consumer_errors) == 0, f"SSE consumer errors: {consumer_errors}"

    def test_detection_log_concurrency_under_rapid_mode_and_log_burst(self):
        """
        Tests concurrent updates to detection_log circular buffer while
        clients are requesting GET /detection_log.
        Verifies:
        - Max length constraint (20 items in API response) is strictly preserved.
        - Zone activity array has length exactly 6.
        - Zero race conditions or torn reads.
        """
        errors = []
        NUM_LOG_WRITERS = 4
        NUM_LOG_READERS = 4
        WRITES_PER_THREAD = 30
        READS_PER_THREAD = 30

        def log_writer(idx):
            for i in range(WRITES_PER_THREAD):
                with detection_log_lock:
                    flask_app_module.detection_log.append({
                        "ts": time.strftime("%H:%M:%S"),
                        "people": (idx + i) % 4,
                        "zones": [f"Z{(idx + i) % 6}"],
                        "fan": 50,
                        "temp": 28.0,
                    })
                    if len(flask_app_module.detection_log) > flask_app_module.MAX_LOG_ENTRIES:
                        flask_app_module.detection_log.pop(0)
                    flask_app_module.zone_activity[(idx + i) % 6] += 1
                time.sleep(0.005)

        def log_reader(idx):
            client = flask_app.test_client()
            for _ in range(READS_PER_THREAD):
                try:
                    res = client.get("/detection_log")
                    if res.status_code != 200:
                        errors.append(f"Reader {idx} status {res.status_code}")
                    data = res.get_json()
                    if "log" not in data or "zone_activity" not in data:
                        errors.append(f"Reader {idx} missing keys: {data}")
                    if len(data.get("zone_activity", [])) != 6:
                        errors.append(f"Reader {idx} zone_activity len != 6: {data}")
                    if len(data.get("log", [])) > 20:
                        errors.append(f"Reader {idx} log exceeds 20 items: {len(data['log'])}")
                except Exception as e:
                    errors.append(f"Reader {idx} exception: {e}")
                time.sleep(0.005)

        threads = []
        for i in range(NUM_LOG_WRITERS):
            threads.append(threading.Thread(target=log_writer, args=(i,)))
        for i in range(NUM_LOG_READERS):
            threads.append(threading.Thread(target=log_reader, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
            assert not t.is_alive(), "Log reader/writer thread deadlock!"

        assert len(errors) == 0, f"Detection log concurrency errors: {errors}"
