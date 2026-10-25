# ARCHITECTURAL INVARIANTS & GUIDELINES FOR AI CODING AGENTS

**Project:** EcoZone AI - Edge Smart Energy Management  
**Target Platform:** Raspberry Pi 4 Model B (Debian Trixie 64-bit) & ESP32-S3 Camera  
**Applicability:** Mandatory for all AI assistants, autonomous coding agents, and human contributors.

---

## MISSION & OPERATING PRINCIPLE

This document defines the strict architectural invariants, mathematical models, hardware safety rules, and coding conventions governing the **EcoZone AI** repository. All modifications must adhere to these rules without exception. Regressions or integrity violations (e.g. dummy test stubs, bypassed checks) are unacceptable.

---

## 1. INVARIANT 1: PHYSICAL MODEL GEOMETRY & 3×2 GRID MAPPING

1. **Physical Model Dimensions:**
   - The physical scale model table is strictly **$30\text{ cm}$ (Width) $\times$ $40\text{ cm}$ (Depth)**.
   - It is divided into **3 Rows $\times$ 2 Columns** (total 6 zones).
   - **NEVER change or assume a 2 Rows $\times$ 3 Columns layout.**

2. **Digital Canvas & Cell Coordinates:**
   - Warped perspective canvas size: **$600\text{ px}$ (Width) $\times$ $400\text{ px}$ (Height)** (`WARP_WIDTH = 600`, `WARP_HEIGHT = 400`).
   - Cell dimensions: **$300\text{ px}$ (Width) $\times$ $133.33\text{ px}$ (Height)** (`CELL_WIDTH = 300`, `CELL_HEIGHT = 133`).

3. **Zone Indexing & Mathematical Formula:**
   ```text
   +------------------------------+------------------------------+
   |            ZONE 0            |            ZONE 1            |
   |   Row 0, Col 0 [0..300, 0..133] | Row 0, Col 1 [300..600, 0..133]|
   +------------------------------+------------------------------+
   |            ZONE 2            |            ZONE 3            |
   |  Row 1, Col 0 [0..300, 133..266]| Row 1, Col 1 [300..600, 133..266]|
   +------------------------------+------------------------------+
   |            ZONE 4            |            ZONE 5            |
   |  Row 2, Col 0 [0..300, 266..400]| Row 2, Col 1 [300..600, 266..400]|
   +------------------------------+------------------------------+
   ```
   For any coordinate $(x, y)$ in the $600 \times 400$ canvas, zone assignment in `core/transform.py` must strictly follow:
   $$\text{Col} = \max\left(0, \min\left(1, \left\lfloor \frac{x}{300} \right\rfloor\right)\right)$$
   $$\text{Row} = \max\left(0, \min\left(2, \left\lfloor \frac{y}{133.33} \right\rfloor\right)\right)$$
   $$\text{Zone ID} = \text{Row} \times 2 + \text{Col} \quad (\text{clamped to } [0, 5])$$

---

## 2. INVARIANT 2: DUAL DETECTION PIPELINE COORDINATE SYSTEMS

The system supports two distinct detection modes (`DEFAULT_DETECTION_MODE = "color"`, toggled via `POST /set_mode`):

1. **Color Mode (`ColorDetector`):**
   - **Target:** Scale-model proxies (yellow sticker cups, markers).
   - **Input Frame:** **Warped Bird's-Eye Frame ($600 \times 400$)**.
   - **Algorithm:** HSV filtering (`[20, 100, 100]` to `[35, 255, 255]`), morphological filtering (`cv2.morphologyEx`), contour area thresholding (`min_contour_area = 500.0`), and direct contour center/anchor mapping on the warped canvas.

2. **AI Person Mode (`YOLOv8OpenVINODetector`):**
   - **Target:** Real humans (`COCO Class ID 0: person`).
   - **Configuration:** `CONFIDENCE_THRESHOLD = 0.35`, `VALID_CLASS_IDS = [0]`.
   - **Input Frame:** **Raw Camera Frame ($640 \times 480$)**.
   - **Coordinate Projection:** The neural network infers on the raw frame to preserve perspective cues. For each detection, the bottom-center anchor point $(x_{\text{center}}, y_{\text{max}})$ represents the contact point with the floor. This single anchor point is projected through the $3 \times 3$ homography matrix $H$ via `transformer.transform_point(...)` into the $600 \times 400$ coordinate space, where `get_zone_id()` assigns the zone.
   - **RULE:** Never pass the warped frame into YOLOv8 without adjusting models, and never pass raw unwarped coordinates directly into `get_zone_id()`.

---

## 3. INVARIANT 3: NON-BLOCKING DHT11 ENVIRONMENTAL SENSING

1. **Sensor Driver Contract (`hardware/dht_sensor.py`):**
   - Polling the DHT11 sensor must **NEVER stall the video processing pipeline** or worker threads.
   - **Caching Window:** Cache readings for `min_interval: float = 2.5` seconds. Calls within this window must return `self.last_valid_reading` immediately ($< 0.1\text{ ms}$).
   - **Zero Blocking Sleep on Failure:** On hardware communication failure (CRC / timeout), do **NOT** invoke `time.sleep()`. Catch exceptions, log at `DEBUG` level, and immediately return `self.last_valid_reading`. The entire `read()` method must complete in $< 0.05\text{ s}$ even during failure.
2. **Resource Teardown:**
   - In `app.py`'s `handle_shutdown_signal()`, always call `dht_sensor_instance.cleanup()` to properly release `libgpiod` line allocations and prevent kernel line lockups (`RuntimeError: Line 4 is busy`).

---

## 4. INVARIANT 4: CONFIGURABLE RELAY POLARITY (`RELAY_ACTIVE_LOW`)

1. **Hardware Polarity Flag:**
   - Define and respect `RELAY_ACTIVE_LOW: bool` in `config.py` and `hardware/gpio_controller.py`:
     ```python
     RELAY_ACTIVE_LOW = os.getenv("RELAY_ACTIVE_LOW", "0").lower() in ("1", "true", "yes")
     ```
2. **Behavioral Contract:**
   - **Active-LOW (`relay_active_low=True` / `1` - Commercial Optocoupled Relay Boards):**
     - Initial setup state: `GPIO.HIGH` ($3.3\text{V}$, relay de-energized / OFF).
     - Occupied zone: `GPIO.LOW` ($0\text{V}$, relay energized / ON).
     - Empty zone: `GPIO.HIGH` ($3.3\text{V}$, relay de-energized / OFF).
   - **Active-HIGH (`relay_active_low=False` / `0` - Direct Breadboard LEDs):**
     - Initial setup state: `GPIO.LOW` ($0\text{V}$, LED OFF).
     - Occupied zone: `GPIO.HIGH` ($3.3\text{V}$, LED ON).
     - Empty zone: `GPIO.LOW` ($0\text{V}$, LED OFF).
3. **Safety Notice:** Initializing an active-LOW relay board with `initial=GPIO.LOW` will energize all 6 relays at boot, causing a massive power spike. Always honor the polarity setting.

---

## 5. INVARIANT 5: RASPBERRY PI OS DEBIAN TRIXIE (64-BIT) COMPATIBILITY

1. **Linux Kernel 6.x & Modern GPIO:**
   - The target OS is **Raspberry Pi OS 64-bit (Debian Trixie, Linux Kernel 6.18+)**.
   - In modern kernels, `/dev/gpiomem` and the legacy `RPi.GPIO` library are obsolete and non-functional.
   - **Always use `rpi-lgpio>=0.6`** as a drop-in replacement on the Pi. It routes calls through the kernel character device via `libgpiod2`.
   - **Always use `adafruit-circuitpython-dht>=4.0.0`** with system package `libgpiod2`. Never use legacy `Adafruit-DHT`.
2. **Headless OpenCV:**
   - On Raspberry Pi OS Console/Headless installations, use **`opencv-python-headless>=4.8.0`** (specified in `requirements-pi.txt`) to avoid pulling heavy X11/Qt libraries.

---

## 6. INVARIANT 6: SECRET HYGIENE & GIT CLEANLINESS

1. **No Plain-Text Credentials:**
   - Never commit WiFi SSIDs, passwords, API tokens, or secrets to the Git repository.
   - ESP32 credentials belong in `smart_building/esp32_firmware/esp32_cam_server/wifi_secrets.h`, which is explicitly tracked in `.gitignore`.
   - Always commit `wifi_secrets.example.h` with placeholder credentials.
2. **Transient Artifact Exclusion:**
   - The `.gitignore` must ignore `.venv/`, `__pycache__/`, `.pytest_cache/`, `*.npy`, `debug_*.jpg`, `test_report.json`, and `.agents/`.
   - Do not commit calibration output matrices (`perspective_matrix.npy`) to version control; calibration must be performed per physical deployment.

---

## 7. INVARIANT 7: TEST SUITE VERIFICATION & ZERO REGRESSIONS

1. **Deterministic Path Anchoring:**
   - `MODEL_DIR`, `MODEL_XML`, and `PERSPECTIVE_MATRIX_PATH` in `config.py` must be anchored using `os.path.dirname(os.path.abspath(__file__))` so tests and services resolve cleanly regardless of Current Working Directory (CWD).
2. **Execution & Validation Standard:**
   - Before completing any task, run the test suite:
     ```bash
     python smart_building/run_tests.py
     ```
   - **100% pass rate is required** (all 227+ existing tests + new unit tests).
   - Source code must compile without errors:
     ```bash
     python -m py_compile smart_building/config.py smart_building/app.py smart_building/core/*.py smart_building/hardware/*.py
     ```
3. **No Cheating:** Never fake test results, hardcode test outputs, or delete failing tests. Fix the underlying root cause.
