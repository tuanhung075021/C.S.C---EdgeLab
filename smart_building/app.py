"""
EcoZone AI - Flask SSE Server & Real-time Web Dashboard
Milestone 2 Implementation (app.py)
"""

import json
import logging
import os
import signal
import sys
import threading
import time
from typing import Optional
from flask import Flask, Response, render_template, jsonify, request
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
logger = logging.getLogger(__name__)

# Try importing configuration and core/hardware modules with robust fallbacks
try:
    import config
    IS_SIMULATION = getattr(config, 'IS_SIMULATION', True)
    DHT_PIN = getattr(config, 'DHT_PIN', 4)
    ESP32_STREAM_URL = getattr(config, 'ESP32_STREAM_URL', "http://192.168.137.31:81/stream")
    MODEL_XML = getattr(config, 'MODEL_XML', "models/yolov8n_openvino_model/yolov8n.xml")
    CONFIDENCE_THRESHOLD = getattr(config, 'CONFIDENCE_THRESHOLD', 0.5)
    WARP_WIDTH = getattr(config, 'WARP_WIDTH', 600)
    WARP_HEIGHT = getattr(config, 'WARP_HEIGHT', 400)
    GRID_ROWS = getattr(config, 'GRID_ROWS', 3)
    GRID_COLS = getattr(config, 'GRID_COLS', 2)
    CELL_WIDTH = getattr(config, 'CELL_WIDTH', 300)
    CELL_HEIGHT = getattr(config, 'CELL_HEIGHT', 133)
    PERSPECTIVE_MATRIX_PATH = getattr(config, 'PERSPECTIVE_MATRIX_PATH', "perspective_matrix.npy")
except ImportError:
    config = None
    IS_SIMULATION = True
    DHT_PIN = 4
    ESP32_STREAM_URL = "http://192.168.137.31:81/stream"
    MODEL_XML = "models/yolov8n_openvino_model/yolov8n.xml"
    CONFIDENCE_THRESHOLD = 0.5
    WARP_WIDTH = 600
    WARP_HEIGHT = 400
    GRID_ROWS = 3
    GRID_COLS = 2
    CELL_WIDTH = 300
    CELL_HEIGHT = 133
    PERSPECTIVE_MATRIX_PATH = "perspective_matrix.npy"

# Module imports
try:
    from core.vision_stream import VisionStream
except ImportError:
    VisionStream = None

try:
    from core.transform import PerspectiveTransformer
except ImportError:
    PerspectiveTransformer = None

try:
    from core.detector import YOLOv8OpenVINODetector
except ImportError:
    YOLOv8OpenVINODetector = None

try:
    from core.color_detector import ColorDetector
except ImportError:
    ColorDetector = None

try:
    from hardware.gpio_controller import GPIOController
except ImportError:
    GPIOController = None

try:
    from hardware.dht_sensor import DHTSensor
except ImportError:
    DHTSensor = None

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Shared state and thread locks
# ---------------------------------------------------------------------------
mode_lock = threading.Lock()
current_mode = getattr(config, "DEFAULT_DETECTION_MODE", "color")

telemetry_lock = threading.Lock()
latest_telemetry = {
    "zones": [0, 0, 0, 0, 0, 0],
    "temp": 28.0,
    "humidity": 65.0,
    "fan_speed": 0,
    "total_people": 0,
    "timestamp": int(time.time()),
    "zone_activity": [0, 0, 0, 0, 0, 0],
    "mode": current_mode,
}

# Unwarped raw frame buffer for perspective calibration
raw_frame_lock = threading.Lock()
latest_raw_frame: Optional[np.ndarray] = None

# Annotated MJPEG frame buffer
annotated_frame_lock = threading.Lock()
latest_annotated_frame: bytes = b''

# Detection event log (circular buffer)
detection_log_lock = threading.Lock()
detection_log = []
zone_activity = [0] * 6
MAX_LOG_ENTRIES = 50

is_running = True

# Global component instances
gpio_ctrl_instance = None
dht_sensor_instance = None
vision_stream_instance = None
transformer_instance = None
detector_instance = None
ai_detector_instance = None
color_detector_instance = None


# ---------------------------------------------------------------------------
# Fallback mock classes
# ---------------------------------------------------------------------------
class FallbackGPIOController:
    """Mock GPIO Controller if hardware.gpio_controller is missing."""
    def __init__(self, is_simulation=True, relay_active_low=False):
        self.is_simulation = is_simulation
        self.relay_active_low = relay_active_low
        logger.info("Initialized FallbackGPIOController (simulation mode)")

    def update_zones(self, zones):
        pass

    def set_fan(self, temp, total_people):
        if total_people == 0:
            return 0
        duty = min(100, max(0, int(40 + (temp - 25) * 5 + total_people * 10)))
        return duty

    def cleanup(self):
        logger.info("FallbackGPIOController cleanup complete")


class FallbackDHTSensor:
    """Mock DHT Sensor if hardware.dht_sensor is missing."""
    def __init__(self, pin=4, is_simulation=True, min_interval=2.5):
        self.pin = pin
        self.is_simulation = is_simulation
        self.min_interval = min_interval

    def read(self):
        return (28.0, 65.0)

    def cleanup(self):
        pass


# ---------------------------------------------------------------------------
# Pipeline initialization
# ---------------------------------------------------------------------------
def initialize_pipeline_components():
    global gpio_ctrl_instance, dht_sensor_instance, vision_stream_instance, transformer_instance
    global detector_instance, ai_detector_instance, color_detector_instance

    # GPIO Controller
    if GPIOController is not None:
        try:
            gpio_ctrl_instance = GPIOController(
                is_simulation=IS_SIMULATION,
                relay_active_low=getattr(config, "RELAY_ACTIVE_LOW", False)
            )
        except Exception as e:
            logger.warning(f"Failed to initialize GPIOController: {e}, falling back to mock")
            gpio_ctrl_instance = FallbackGPIOController(is_simulation=True)
    else:
        gpio_ctrl_instance = FallbackGPIOController(is_simulation=True)

    # DHT Sensor
    if DHTSensor is not None:
        try:
            dht_sensor_instance = DHTSensor(pin=DHT_PIN, is_simulation=IS_SIMULATION)
        except Exception as e:
            logger.warning(f"Failed to initialize DHTSensor: {e}, falling back to mock")
            dht_sensor_instance = FallbackDHTSensor(pin=DHT_PIN, is_simulation=True)
    else:
        dht_sensor_instance = FallbackDHTSensor(pin=DHT_PIN, is_simulation=True)

    # Vision Stream
    if VisionStream is not None:
        try:
            # Force is_simulation=False to use ESP32 camera instead of laptop webcam
            vision_stream_instance = VisionStream(stream_url=ESP32_STREAM_URL, is_simulation=False)
            vision_stream_instance.start()
        except Exception as e:
            logger.warning(f"Failed to initialize VisionStream: {e}")
            vision_stream_instance = None

    # Perspective Transformer
    if PerspectiveTransformer is not None:
        try:
            transformer_instance = PerspectiveTransformer(matrix_path=PERSPECTIVE_MATRIX_PATH)
        except Exception as e:
            logger.warning(f"Failed to initialize PerspectiveTransformer: {e}")
            transformer_instance = None

    # YOLO AI Detector
    if YOLOv8OpenVINODetector is not None:
        try:
            # Force is_simulation=False for the detector so it loads YOLOv8 on Windows
            ai_detector_instance = YOLOv8OpenVINODetector(
                model_xml=MODEL_XML,
                conf_threshold=CONFIDENCE_THRESHOLD,
                is_simulation=False
            )
        except Exception as e:
            logger.warning(f"Failed to initialize YOLOv8OpenVINODetector: {e}")
            ai_detector_instance = None
    else:
        ai_detector_instance = None
    detector_instance = ai_detector_instance

    # Color Detector
    if ColorDetector is not None:
        try:
            color_detector_instance = ColorDetector(
                lower_hsv=getattr(config, "COLOR_YELLOW_HSV_LOWER", [20, 100, 100]),
                upper_hsv=getattr(config, "COLOR_YELLOW_HSV_UPPER", [35, 255, 255]),
                min_contour_area=getattr(config, "COLOR_MIN_CONTOUR_AREA", 500.0)
            )
        except Exception as e:
            logger.warning(f"Failed to initialize ColorDetector: {e}")
            color_detector_instance = None
    else:
        color_detector_instance = None

    return gpio_ctrl_instance, dht_sensor_instance, vision_stream_instance, transformer_instance, ai_detector_instance


# ---------------------------------------------------------------------------
# Frame annotation helper (3 rows x 2 columns grid)
# ---------------------------------------------------------------------------
def annotate_frame(frame, occupied_zones, total_people, fan_speed, temp):
    """
    Draw 3x2 zone grid overlay, occupancy markers, and telemetry HUD on a BGR frame.
    Returns annotated BGR frame at 600x400.
    """
    if frame is None or frame.size == 0:
        out = np.zeros((WARP_HEIGHT, WARP_WIDTH, 3), dtype=np.uint8)
    elif frame.shape[:2] != (WARP_HEIGHT, WARP_WIDTH):
        out = cv2.resize(frame.copy(), (WARP_WIDTH, WARP_HEIGHT))
    else:
        out = frame.copy()

    h, w = out.shape[:2]
    cell_w = w // GRID_COLS  # 300 px
    cell_h = h // GRID_ROWS  # 133 px

    # Draw 6 zone cells (3 rows x 2 columns) with semi-transparent overlays
    for zone_id in range(6):
        row, col = divmod(zone_id, GRID_COLS)  # row: 0..2, col: 0..1
        x1, y1 = col * cell_w, row * cell_h
        x2 = x1 + cell_w
        y2 = h if row == GRID_ROWS - 1 else y1 + cell_h
        is_occ = bool(occupied_zones[zone_id]) if zone_id < len(occupied_zones) else False

        color = (20, 200, 120) if is_occ else (80, 80, 100)
        alpha = 0.28 if is_occ else 0.08

        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)

        # Zone ID label (Z0 to Z5)
        cv2.putText(out, f"Z{zone_id}", (x1 + 10, y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 255), 1, cv2.LINE_AA)

        # Occupied / empty text
        status_color = (20, 220, 130) if is_occ else (100, 100, 120)
        cv2.putText(out, "OCCUPIED" if is_occ else "empty", (x1 + 10, y1 + 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1, cv2.LINE_AA)

        # Green dot indicator
        if is_occ:
            cv2.circle(out, (x2 - 16, y1 + 16), 7, (20, 200, 120), -1)
            cv2.circle(out, (x2 - 16, y1 + 16), 7, (255, 255, 255), 1)

    # HUD bar at bottom
    hud_y = h - 32
    cv2.rectangle(out, (0, hud_y), (w, h), (10, 10, 30), -1)
    hud_text = f"People: {total_people}  |  Temp: {temp:.1f}\u00b0C  |  Fan: {fan_speed}%"
    cv2.putText(out, hud_text, (10, hud_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 220, 255), 1, cv2.LINE_AA)

    ts = time.strftime("%H:%M:%S")
    ts_w = cv2.getTextSize(ts, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0][0]
    cv2.putText(out, ts, (w - ts_w - 8, hud_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 140, 180), 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------
def background_pipeline_worker():
    """Continuous background processing loop: Vision -> Raw Buffer -> Transform -> Detector -> GPIO -> Telemetry."""
    global latest_telemetry, is_running, latest_annotated_frame, latest_raw_frame, detection_log, zone_activity
    global current_mode, ai_detector_instance, color_detector_instance

    gpio_ctrl, dht_sensor, vision_stream, transformer, detector = initialize_pipeline_components()
    logger.info("Background pipeline worker thread started")

    while is_running:
        try:
            occupied_zones = [0, 0, 0, 0, 0, 0]
            total_people = 0
            warped_frame = None

            # Read active mode under lock
            with mode_lock:
                active_mode = current_mode

            # Vision pipeline
            if vision_stream is not None:
                ret, frame = vision_stream.read_frame()
                if ret and frame is not None and frame.size > 0:
                    # Update thread-safe unwarped frame buffer before transform
                    with raw_frame_lock:
                        latest_raw_frame = frame.copy()

                    if transformer is not None:
                        warped_frame = transformer.warp(frame)
                    else:
                        warped_frame = cv2.resize(frame, (WARP_WIDTH, WARP_HEIGHT))

                    # Branch based on active detection mode
                    if active_mode == "color" and color_detector_instance is not None:
                        occupied_zones, total_people = color_detector_instance.detect(warped_frame)
                    elif active_mode == "ai" and ai_detector_instance is not None:
                        occupied_zones, total_people = ai_detector_instance.detect(frame, transformer=transformer)

            # Read sensors
            temp, humidity = dht_sensor.read() if dht_sensor is not None else (28.0, 65.0)

            # Control hardware
            fan_speed = gpio_ctrl.set_fan(temp, total_people) if gpio_ctrl is not None else 0
            if gpio_ctrl is not None:
                gpio_ctrl.update_zones(occupied_zones)

            # Update zone activity counters + detection log
            with detection_log_lock:
                for i, occ in enumerate(occupied_zones):
                    if occ:
                        zone_activity[i] += 1
                if total_people > 0:
                    occ_names = [f"Z{i}" for i, v in enumerate(occupied_zones) if v]
                    detection_log.append({
                        "ts": time.strftime("%H:%M:%S"),
                        "people": total_people,
                        "zones": occ_names,
                        "fan": fan_speed,
                        "temp": round(temp, 1),
                    })
                    if len(detection_log) > MAX_LOG_ENTRIES:
                        detection_log.pop(0)

            # Build annotated MJPEG frame
            if warped_frame is not None:
                annotated = annotate_frame(warped_frame, occupied_zones, total_people, fan_speed, temp)
                _, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
                with annotated_frame_lock:
                    latest_annotated_frame = jpeg.tobytes()

            # Update shared telemetry with active mode
            with telemetry_lock:
                latest_telemetry = {
                    "zones": [int(z) for z in occupied_zones],
                    "temp": float(temp),
                    "humidity": float(humidity),
                    "fan_speed": int(fan_speed),
                    "total_people": int(total_people),
                    "timestamp": int(time.time()),
                    "zone_activity": list(zone_activity),
                    "mode": active_mode,
                }

        except Exception as e:
            logger.error(f"Error in background pipeline worker loop: {e}")

        time.sleep(0.1)

    logger.info("Background pipeline worker thread stopped")


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    """Serves the main EcoZone AI Web Dashboard UI."""
    return render_template('index.html')


@app.route('/calibrate')
def calibrate():
    """Serves the interactive web camera calibration page."""
    return render_template('calibrate.html')


@app.route('/calibrate/snapshot')
def calibrate_snapshot():
    """
    Captures one unwarped/raw frame from VisionStream (or latest raw frame buffer),
    encodes as JPEG and returns with Content-Type: image/jpeg.
    Handles fallbacks if stream is not ready or warming up.
    """
    frame = None
    with raw_frame_lock:
        if latest_raw_frame is not None and latest_raw_frame.size > 0:
            frame = latest_raw_frame.copy()

    # Fallback to direct read from vision_stream_instance if available
    if frame is None and vision_stream_instance is not None:
        try:
            ret, vs_frame = vision_stream_instance.read_frame()
            if ret and vs_frame is not None and vs_frame.size > 0:
                frame = vs_frame.copy()
        except Exception as e:
            logger.warning(f"Error reading frame directly from VisionStream: {e}")

    # Fallback to synthetic 640x480 test pattern if stream is offline/warming up
    if frame is None or frame.size == 0:
        logger.info("VisionStream not ready; returning synthetic fallback snapshot")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (40, 40, 50)
        cv2.rectangle(frame, (50, 40), (590, 440), (180, 180, 180), 2)
        cv2.line(frame, (320, 40), (320, 440), (80, 80, 90), 1)
        cv2.line(frame, (50, 240), (590, 240), (80, 80, 90), 1)
        cv2.putText(frame, "STANDBY / SYNTHETIC CALIBRATION SNAPSHOT", (70, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, "Click 4 corners: TL -> TR -> BR -> BL", (120, 465),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ret or jpeg is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, jpeg = cv2.imencode('.jpg', frame)

    return Response(
        jpeg.tobytes(),
        mimetype='image/jpeg',
        headers={
            'Cache-Control': 'no-cache, no-store, must-revalidate',
            'Pragma': 'no-cache',
            'Expires': '0'
        }
    )


@app.route('/calibrate/save', methods=['POST'])
def calibrate_save():
    """
    Accepts JSON with 4 corner points and display dimensions,
    calculates homography/perspective transform matrix mapping to 600x400 Birds-Eye canvas
    using cv2.getPerspectiveTransform(src_points, dst_points)
    where dst_points are [[0,0],[600,0],[600,400],[0,400]],
    saves matrix to perspective_matrix.npy as (3,3) float, updates live transformer,
    returns JSON {"status": "ok"}.
    """
    try:
        data = request.get_json(force=True, silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Invalid JSON payload"}), 400

        points = data.get("points")
        if not points or not isinstance(points, (list, tuple)) or len(points) != 4:
            return jsonify({"status": "error", "message": "Exactly 4 corner points required"}), 400

        # Extract [x, y] coordinates
        parsed_points = []
        for idx, pt in enumerate(points):
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                x, y = float(pt[0]), float(pt[1])
            elif isinstance(pt, dict) and 'x' in pt and 'y' in pt:
                x, y = float(pt['x']), float(pt['y'])
            else:
                return jsonify({"status": "error", "message": f"Malformed coordinate at index {idx}"}), 400
            parsed_points.append([x, y])

        # Coordinate scaling from display canvas to native image resolution
        disp_w = float(data.get("img_width") or 0)
        disp_h = float(data.get("img_height") or 0)

        # Get native snapshot resolution
        native_w, native_h = 640.0, 480.0
        with raw_frame_lock:
            if latest_raw_frame is not None and latest_raw_frame.size > 0:
                native_h, native_w = float(latest_raw_frame.shape[0]), float(latest_raw_frame.shape[1])

        if disp_w > 0 and disp_h > 0:
            scale_x = native_w / disp_w
            scale_y = native_h / disp_h
        else:
            scale_x, scale_y = 1.0, 1.0

        src_points = np.float32([[pt[0] * scale_x, pt[1] * scale_y] for pt in parsed_points])

        # Destination points for 600x400 Bird's-Eye perspective view: TL, TR, BR, BL
        dst_points = np.float32([
            [0.0, 0.0],
            [float(WARP_WIDTH), 0.0],
            [float(WARP_WIDTH), float(WARP_HEIGHT)],
            [0.0, float(WARP_HEIGHT)]
        ])

        # Calculate 3x3 Homography Perspective Matrix
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        if M is None or M.shape != (3, 3):
            return jsonify({"status": "error", "message": "Perspective transform matrix computation failed"}), 400

        # Ensure float32 representation
        M = M.astype(np.float32)

        # Save to disk
        np.save(PERSPECTIVE_MATRIX_PATH, M)
        logger.info("Saved new perspective matrix to %s (shape: %s)", PERSPECTIVE_MATRIX_PATH, M.shape)

        # Dynamically update running transformer in-memory
        if transformer_instance is not None:
            transformer_instance.matrix = M
            logger.info("Updated live transformer instance with new perspective matrix")

        return jsonify({
            "status": "ok",
            "message": "Calibration matrix saved successfully"
        }), 200

    except cv2.error as cv_err:
        logger.error(f"OpenCV error during perspective transform: {cv_err}")
        return jsonify({"status": "error", "message": f"Geometry error (collinear points?): {cv_err}"}), 400
    except Exception as e:
        logger.error(f"Unexpected error during calibration save: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/stream')
def stream():
    """Server-Sent Events (SSE) streaming endpoint for telemetry."""
    def event_generator():
        while is_running:
            with telemetry_lock:
                payload = json.dumps(latest_telemetry)
            yield f"data: {payload}\n\n"
            time.sleep(0.1)

    return Response(
        event_generator(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


@app.route('/video_feed')
def video_feed():
    """MJPEG stream of the annotated (zone-overlaid) camera view."""
    def generate():
        # Build a placeholder frame once
        ph = np.zeros((WARP_HEIGHT, WARP_WIDTH, 3), dtype=np.uint8)
        cv2.putText(ph, 'Warming up AI pipeline...', (80, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 120, 160), 2)
        _, enc = cv2.imencode('.jpg', ph)
        placeholder = enc.tobytes()

        while is_running:
            with annotated_frame_lock:
                frame_bytes = latest_annotated_frame

            chosen = frame_bytes if frame_bytes else placeholder
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + chosen + b'\r\n')
            time.sleep(0.1)

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={'Cache-Control': 'no-cache'})


@app.route('/detection_log')
def get_detection_log():
    """Returns the last 20 detection events + per-zone activity counts as JSON."""
    with detection_log_lock:
        return jsonify({'log': list(detection_log[-20:]), 'zone_activity': list(zone_activity)})


@app.route('/set_mode', methods=['POST'])
def set_mode():
    """
    Sets the active detection mode ('color' or 'ai').
    Accepts JSON: {"mode": "color"} or {"mode": "ai"}.
    Thread-safely updates current_mode and returns status.
    """
    global current_mode
    try:
        data = request.get_json(force=True, silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"status": "error", "message": "Invalid mode. Must be 'color' or 'ai'"}), 400

        mode = data.get("mode")
        if not mode or not isinstance(mode, str):
            return jsonify({"status": "error", "message": "Invalid mode. Must be 'color' or 'ai'"}), 400

        mode_clean = mode.strip().lower()
        if mode_clean not in ("color", "ai"):
            return jsonify({"status": "error", "message": "Invalid mode. Must be 'color' or 'ai'"}), 400

        with mode_lock:
            current_mode = mode_clean

        logger.info(f"Detection mode updated to '{current_mode}'")
        return jsonify({"status": "ok", "mode": current_mode}), 200

    except Exception as e:
        logger.error(f"Error updating detection mode: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/get_mode', methods=['GET'])
def get_mode():
    """
    Returns the current active detection mode ('color' or 'ai').
    """
    with mode_lock:
        mode = current_mode
    return jsonify({"status": "ok", "mode": mode}), 200


# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
def handle_shutdown_signal(sig, frame):
    global is_running, gpio_ctrl_instance, dht_sensor_instance, vision_stream_instance
    logger.info(f"Received signal {sig}, initiating graceful shutdown...")
    is_running = False
    if gpio_ctrl_instance is not None:
        try:
            gpio_ctrl_instance.cleanup()
        except Exception as e:
            logger.error(f"Error during GPIO cleanup: {e}")
    if dht_sensor_instance is not None and hasattr(dht_sensor_instance, "cleanup"):
        try:
            dht_sensor_instance.cleanup()
        except Exception as e:
            logger.error(f"Error during DHT cleanup: {e}")
    if vision_stream_instance is not None and hasattr(vision_stream_instance, "stop"):
        try:
            vision_stream_instance.stop()
        except Exception as e:
            logger.error(f"Error during VisionStream cleanup: {e}")
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    worker_thread = threading.Thread(target=background_pipeline_worker, daemon=True)
    worker_thread.start()

    logger.info("Starting EcoZone AI Flask SSE Web Server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
