"""
EcoZone AI Smart Energy Management System
Central Configuration Module (config.py)
"""

import os
import platform
import sys


def detect_simulation_mode() -> bool:
    """
    Auto-detects whether the system is running in simulation mode.
    Returns True for non-ARM platforms (Windows, macOS, x86_64 Linux)
    or if hardware libraries are missing.
    Can be overridden by setting environment variable IS_SIMULATION=1/0 or ECOZONE_SIMULATION=1/0.
    """
    env_sim = os.getenv("IS_SIMULATION") or os.getenv("ECOZONE_SIMULATION")
    if env_sim is not None:
        return env_sim.lower() in ("true", "1", "yes")

    is_linux = sys.platform.startswith("linux")
    machine = platform.machine().lower()
    is_arm = ("arm" in machine) or ("aarch64" in machine)

    if not is_linux or not is_arm:
        return True

    # On Linux ARM, test RPi.GPIO import safety
    try:
        import RPi.GPIO  # type: ignore # noqa: F401
        return False
    except ImportError:
        return True


# Dynamic Simulation Flag
IS_SIMULATION: bool = detect_simulation_mode()

# ESP32 MJPEG Camera Stream
ESP32_STREAM_URL: str = os.getenv(
    "ESP32_STREAM_URL", "http://172.20.10.2:81/stream"
)

# Raspberry Pi BCM GPIO Pin Assignments
# 6 Lighting Zone LEDs (30x40cm physical layout: 3 rows x 2 columns)
ZONE_LEDS: dict[int, int] = {
    0: 17,  # Zone 0: Row0-Left
    1: 27,  # Zone 1: Row0-Right
    2: 22,  # Zone 2: Row1-Left
    3: 23,  # Zone 3: Row1-Right
    4: 24,  # Zone 4: Row2-Left
    5: 25,  # Zone 5: Row2-Right
}

# L298N Dual H-Bridge Fan Driver Pins
FAN_ENA_PIN: int = 18  # Hardware PWM Pin
FAN_IN1_PIN: int = 5   # Direction Pin 1
FAN_IN2_PIN: int = 6   # Direction Pin 2
FAN_PWM_FREQ: int = 1000  # 1kHz PWM Frequency

# DHT11 Environmental Sensor Pin
DHT_PIN: int = 4  # GPIO4 Data Pin

# Relay Hardware Configuration
RELAY_ACTIVE_LOW: bool = os.getenv("RELAY_ACTIVE_LOW", "0").lower() in ("1", "true", "yes")
ZONE_HOLD_SECONDS: float = float(os.getenv("ZONE_HOLD_SECONDS", "3.0"))
SYNTHETIC_HARDWARE_LOCKOUT: bool = os.getenv("SYNTHETIC_HARDWARE_LOCKOUT", "1").lower() in ("1", "true", "yes")
ESP32_REPROBE_INTERVAL: float = float(os.getenv("ESP32_REPROBE_INTERVAL", "5.0"))

# Energy Telemetry Configuration (Watts per Zone & Max Fan Power)
ZONE_LIGHT_WATTS: float = float(os.getenv("ZONE_LIGHT_WATTS", "10.0"))
FAN_MAX_WATTS: float = float(os.getenv("FAN_MAX_WATTS", "5.0"))

# Vision & Perspective Transform Parameters
WARP_WIDTH: int = 600
WARP_HEIGHT: int = 400
GRID_ROWS: int = 3
GRID_COLS: int = 2
CELL_WIDTH: int = WARP_WIDTH // GRID_COLS  # 300px
CELL_HEIGHT: int = WARP_HEIGHT // GRID_ROWS  # 133px

# AI Detection Parameters
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))  # Clean person detection threshold
PERSON_CLASS_ID: int = 0  # COCO class ID for person
VALID_CLASS_IDS: list[int] = [0]  # Person only

# Color Detection Parameters (Yellow Cup Stickers Proxy)
COLOR_YELLOW_HSV_LOWER: list[int] = [20, 100, 100]
COLOR_YELLOW_HSV_UPPER: list[int] = [35, 255, 255]
COLOR_MIN_CONTOUR_AREA: int = 500

# Default Detection Mode ('color' or 'ai')
DEFAULT_DETECTION_MODE: str = os.getenv("DEFAULT_DETECTION_MODE", "color")

# Base Directory Anchor
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# Path Definitions
MODEL_DIR: str = os.path.join(BASE_DIR, "models", "yolov8n_openvino_model")
MODEL_XML: str = os.path.join(MODEL_DIR, "yolov8n.xml")
PERSPECTIVE_MATRIX_PATH: str = os.path.join(BASE_DIR, "perspective_matrix.npy")


