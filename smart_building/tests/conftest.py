"""
pytest configuration and shared offline mock fixtures for EcoZone AI test suite.
Provides simulation testing fixtures for 3x2 grid (GRID_ROWS=3, GRID_COLS=2, CELL_WIDTH=300, CELL_HEIGHT=133),
GPIO, DHT sensor, MJPEG stream, YOLO detector, and Flask SSE/Calibration client.
"""

import sys
import os
import math
import json
import time
import threading
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Standard BCM pin mappings for 3x2 grid (6 zones)
ZONE_LEDS_MAP = {
    0: 17,  # Zone 0: Row0-Left
    1: 27,  # Zone 1: Row0-Right
    2: 22,  # Zone 2: Row1-Left
    3: 23,  # Zone 3: Row1-Right
    4: 24,  # Zone 4: Row2-Left
    5: 25,  # Zone 5: Row2-Right
}
FAN_ENA_PIN = 18
FAN_IN1_PIN = 5
FAN_IN2_PIN = 6
DHT_PIN = 4

# 3x2 Canvas Constants
WARP_WIDTH = 600
WARP_HEIGHT = 400
GRID_ROWS = 3
GRID_COLS = 2
CELL_WIDTH = 300
CELL_HEIGHT = 133


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "tier1: Tier 1 Functional & Feature Coverage Tests")
    config.addinivalue_line("markers", "tier2: Tier 2 Boundary, Limit & Fault-Injection Tests")
    config.addinivalue_line("markers", "tier3: Tier 3 Cross-Feature Pairwise Interaction Tests")
    config.addinivalue_line("markers", "tier4: Tier 4 Real-World End-to-End Tests")
    config.addinivalue_line("markers", "feature(id): Filter tests by Feature ID (1 to 11)")


# -----------------------------------------------------------------------------
# Module Stubs Helper
# -----------------------------------------------------------------------------
def setup_module_stubs():
    """Ensure modules exist in sys.modules with spec-compliant fallbacks if not yet created on disk."""
    if "config" not in sys.modules:
        try:
            import config
        except ImportError:
            config_stub = MagicMock()
            config_stub.IS_SIMULATION = True
            config_stub.ESP32_STREAM_URL = "http://10.37.61.34:81/stream"
            config_stub.ZONE_LEDS = ZONE_LEDS_MAP
            config_stub.FAN_ENA_PIN = 18
            config_stub.FAN_IN1_PIN = 5
            config_stub.FAN_IN2_PIN = 6
            config_stub.DHT_PIN = 4
            config_stub.WARP_WIDTH = 600
            config_stub.WARP_HEIGHT = 400
            config_stub.GRID_ROWS = 3
            config_stub.GRID_COLS = 2
            config_stub.CELL_WIDTH = 300
            config_stub.CELL_HEIGHT = 133
            config_stub.CONFIDENCE_THRESHOLD = 0.25
            config_stub.PERSON_CLASS_ID = 0
            config_stub.MODEL_DIR = "models/yolov8n_openvino_model"
            config_stub.MODEL_XML = "models/yolov8n_openvino_model/yolov8n.xml"
            config_stub.PERSPECTIVE_MATRIX_PATH = "perspective_matrix.npy"
            sys.modules["config"] = config_stub


setup_module_stubs()


# -----------------------------------------------------------------------------
# Autouse Fixture: Matrix Cleanup
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def clean_matrix_environment():
    """Back up any pre-existing perspective_matrix.npy, provide clean state per test, and restore afterward."""
    matrix_path = os.path.join(PROJECT_ROOT, "perspective_matrix.npy")
    backup_bytes = None
    if os.path.exists(matrix_path):
        try:
            with open(matrix_path, "rb") as f:
                backup_bytes = f.read()
            os.remove(matrix_path)
        except Exception:
            pass
    yield matrix_path
    if backup_bytes is not None:
        try:
            with open(matrix_path, "wb") as f:
                f.write(backup_bytes)
        except Exception:
            pass
    elif os.path.exists(matrix_path):
        try:
            os.remove(matrix_path)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Fixture: 3x2 Grid Spec Definitions
# -----------------------------------------------------------------------------
@pytest.fixture
def grid_config_3x2():
    """Returns the canonical 3x2 physical room model configuration dictionary."""
    return {
        "GRID_ROWS": 3,
        "GRID_COLS": 2,
        "WARP_WIDTH": 600,
        "WARP_HEIGHT": 400,
        "CELL_WIDTH": 300,
        "CELL_HEIGHT": 133,
        "NUM_ZONES": 6,
        "ZONE_CENTERS": [
            (150, 66, 0),    # Zone 0: Row0-Left
            (450, 66, 1),    # Zone 1: Row0-Right
            (150, 200, 2),   # Zone 2: Row1-Left
            (450, 200, 3),   # Zone 3: Row1-Right
            (150, 333, 4),   # Zone 4: Row2-Left
            (450, 333, 5),   # Zone 5: Row2-Right
        ]
    }


# -----------------------------------------------------------------------------
# Fixture 1: mock_gpio
# -----------------------------------------------------------------------------
class MockGPIOController:
    """Offline Mock GPIO Controller tracking pin states and PWM duty cycle for 3x2 grid."""
    def __init__(self, is_simulation=True):
        self.is_simulation = is_simulation
        self.pin_modes = {}
        self.pin_states = {pin: 0 for pin in list(ZONE_LEDS_MAP.values()) + [FAN_ENA_PIN, FAN_IN1_PIN, FAN_IN2_PIN, DHT_PIN]}
        self.current_zones = [0, 0, 0, 0, 0, 0]
        self.pwm_frequency = 1000
        self.pwm_duty_cycle = 0
        self.current_fan_duty = 0
        self.cleaned_up = False
        self.log_history = []

        # Initialize pin setup
        for pin in list(ZONE_LEDS_MAP.values()) + [FAN_ENA_PIN, FAN_IN1_PIN, FAN_IN2_PIN]:
            self.pin_modes[pin] = "OUT"

    def update_zones(self, zones: list):
        if not isinstance(zones, (list, tuple)) or len(zones) != 6:
            raise ValueError("Zones must be a list of 6 integers (0 or 1)")
        for idx, val in enumerate(zones):
            if val not in (0, 1):
                raise ValueError(f"Zone state must be 0 or 1, got {val}")
            pin = ZONE_LEDS_MAP[idx]
            self.pin_states[pin] = int(val)
            log_msg = f"[SIMULATION] Zone LED {idx} (GPIO {pin}) -> {'HIGH' if val else 'LOW'}"
            self.log_history.append(log_msg)
        self.current_zones = list(zones)

    def calculate_fan_duty(self, temp: float, total_people: int) -> int:
        if total_people <= 0:
            return 0
        raw_duty = 40.0 + (temp - 25.0) * 5.0 + total_people * 10.0
        return int(min(100, max(0, round(raw_duty))))

    def set_fan(self, temp: float, total_people: int) -> int:
        duty = self.calculate_fan_duty(temp, total_people)
        self.pwm_duty_cycle = duty
        self.current_fan_duty = duty
        if duty > 0:
            self.pin_states[FAN_IN1_PIN] = 1
            self.pin_states[FAN_IN2_PIN] = 0
        else:
            self.pin_states[FAN_IN1_PIN] = 0
            self.pin_states[FAN_IN2_PIN] = 0
        log_msg = f"[SIMULATION] Fan PWM duty cycle set to {duty}%"
        self.log_history.append(log_msg)
        return duty

    def cleanup(self):
        self.cleaned_up = True
        for pin in self.pin_states:
            self.pin_states[pin] = 0
        self.current_zones = [0, 0, 0, 0, 0, 0]
        self.pwm_duty_cycle = 0
        self.current_fan_duty = 0
        self.log_history.append("[SIMULATION] GPIO cleanup completed")

    def assert_zone_led(self, zone_id: int, expected_state: int):
        pin = ZONE_LEDS_MAP[zone_id]
        actual = self.pin_states.get(pin, 0)
        assert actual == expected_state, f"Zone {zone_id} (pin {pin}) state expected {expected_state}, got {actual}"

    def assert_fan_duty(self, expected_duty: int):
        assert self.pwm_duty_cycle == expected_duty, f"Fan duty cycle expected {expected_duty}, got {self.pwm_duty_cycle}"

    def assert_cleaned_up(self):
        assert self.cleaned_up is True, "GPIO cleanup was not executed"


@pytest.fixture
def mock_gpio():
    """Fixture providing an offline MockGPIOController instance."""
    controller = MockGPIOController(is_simulation=True)
    yield controller
    if not controller.cleaned_up:
        controller.cleanup()


# -----------------------------------------------------------------------------
# Fixture 2: mock_dht_sensor
# -----------------------------------------------------------------------------
class MockDHTSensor:
    """Offline Mock DHT11 Environment Sensor with failure injection and retry tracking."""
    def __init__(self, pin: int = 4, is_simulation: bool = True):
        self.pin = pin
        self.is_simulation = is_simulation
        self.temperature = 28.0
        self.humidity = 65.0
        self.fail_count = 0
        self.call_counter = 0
        self.retry_count = 0
        self.last_valid = (28.0, 65.0)

    def set_reading(self, temp: float, humidity: float):
        self.temperature = float(temp)
        self.humidity = float(humidity)
        if (-10.0 <= self.temperature <= 85.0) and (0.0 <= self.humidity <= 100.0):
            self.last_valid = (float(temp), float(humidity))

    def inject_failures(self, count: int):
        self.fail_count = count

    def read(self) -> tuple:
        self.call_counter += 1
        attempts = 0
        while attempts < 3:
            attempts += 1
            if self.fail_count > 0:
                self.fail_count -= 1
                self.retry_count += 1
                continue
            # Check for out of bounds logic
            if not (-10.0 <= self.temperature <= 85.0) or not (0.0 <= self.humidity <= 100.0):
                return self.last_valid
            self.last_valid = (self.temperature, self.humidity)
            return (self.temperature, self.humidity)
        return self.last_valid

    def cleanup(self):
        pass


@pytest.fixture
def mock_dht_sensor():
    """Fixture providing a MockDHTSensor instance."""
    sensor = MockDHTSensor(pin=DHT_PIN, is_simulation=True)
    return sensor


# -----------------------------------------------------------------------------
# Fixture 3: mock_mjpeg_stream
# -----------------------------------------------------------------------------
class MockMJPEGStream:
    """Offline Mock MJPEG Stream Server and Frame Generator for 3x2 layout."""
    def __init__(self, width=640, height=480, fps=15):
        self.width = width
        self.height = height
        self.fps = fps
        self.is_connected = True
        self.frame_count = 0

    def generate_synthetic_frame(self, person_zones=None) -> np.ndarray:
        """Generates a 640x480 synthetic BGR image frame with 3 rows x 2 columns grid."""
        self.frame_count += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :] = (30, 30, 30)

        # Draw grid lines for 3 rows x 2 columns zones
        cell_w = self.width // 2   # 320
        cell_h = self.height // 3  # 160
        for r in range(1, 3):
            frame[r * cell_h:r * cell_h + 2, :] = (200, 200, 200)
        for c in range(1, 2):
            frame[:, c * cell_w:c * cell_w + 2] = (200, 200, 200)

        # Draw synthetic person boxes if requested
        if person_zones:
            for zone_id in person_zones:
                r = zone_id // 2
                c = zone_id % 2
                cx = c * cell_w + cell_w // 2
                cy = r * cell_h + cell_h // 2
                frame[cy - 40:cy + 40, cx - 30:cx + 30] = (0, 255, 0)

        return frame

    def read_frame(self) -> tuple:
        if not self.is_connected:
            return False, None
        return True, self.generate_synthetic_frame()

    def get_http_response_headers(self) -> dict:
        return {
            "Content-Type": "multipart/x-mixed-replace; boundary=frame",
            "Server": "ESP32-S3 MJPEG Server",
            "Connection": "close"
        }

    def simulate_disconnect(self):
        self.is_connected = False

    def reconnect(self):
        self.is_connected = True

    def stop(self):
        self.is_connected = False


@pytest.fixture
def mock_mjpeg_stream():
    """Fixture providing a MockMJPEGStream instance."""
    return MockMJPEGStream()


# -----------------------------------------------------------------------------
# Fixture 4: mock_detector
# -----------------------------------------------------------------------------
class MockYOLOv8OpenVINODetector:
    """Offline Mock OpenVINO YOLOv8 Person Detector for 3x2 grid."""
    def __init__(self, model_xml="models/yolov8n_openvino_model/yolov8n.xml", conf_threshold=0.25):
        self.model_xml = model_xml
        self.conf_threshold = conf_threshold
        self.person_class_id = 0
        self.preset_detections = []  # list of (x1, y1, x2, y2, conf, class_id)

    def set_preset_boxes(self, boxes: list):
        """Boxes format: [(x1, y1, x2, y2, conf, class_id), ...]"""
        self.preset_detections = boxes

    def detect(self, frame: np.ndarray) -> tuple:
        occupied = [0, 0, 0, 0, 0, 0]
        people_count = 0

        for box in self.preset_detections:
            x1, y1, x2, y2, conf, class_id = box
            if class_id == self.person_class_id and conf >= self.conf_threshold:
                x_center = (x1 + x2) / 2.0
                y_max = float(y2)
                # Compute zone id formula for 3 rows x 2 cols (CELL_WIDTH=300, CELL_HEIGHT=133)
                col = min(1, max(0, int(math.floor(x_center / 300.0))))
                row = min(2, max(0, int(math.floor(y_max / 133.0))))
                zone_id = min(5, max(0, row * 2 + col))
                if 0 <= zone_id <= 5:
                    occupied[zone_id] = 1
                people_count += 1

        return occupied, people_count


@pytest.fixture
def mock_detector():
    """Fixture providing a MockYOLOv8OpenVINODetector instance."""
    return MockYOLOv8OpenVINODetector()


# -----------------------------------------------------------------------------
# Fixture 5: flask_sse_client
# -----------------------------------------------------------------------------
@pytest.fixture
def flask_sse_client():
    """Fixture providing a clean test client for app.py with SSE & calibration endpoints."""
    from app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client
