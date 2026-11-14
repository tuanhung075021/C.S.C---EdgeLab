"""
Vision Stream Ingestion Engine for EcoZone AI.
Handles MJPEG decoding, auto-reconnection, and simulation fallbacks (webcam, video file, synthetic frame generator).
"""

import os
import math
import time
import logging
import threading
from typing import Tuple, Optional
import numpy as np
import cv2
import socket

try:
    import config
except ImportError:
    try:
        from .. import config  # type: ignore
    except ImportError:
        config = None

logger = logging.getLogger(__name__)


class VisionStream:
    """
    Decodes MJPEG video stream from ESP32 camera, local webcam, video file, or synthetic generator.
    Provides auto-reconnection and a thread-safe frame producer interface.
    """

    def __init__(
        self,
        stream_url: Optional[str] = None,
        is_simulation: Optional[bool] = None,
        reconnect_interval: float = 2.0,
        webcam_index: int = 0,
        video_path: Optional[str] = None,
    ) -> None:
        if config is not None:
            default_url = getattr(config, "ESP32_STREAM_URL", "http://192.168.137.31:81/stream")
            default_sim = getattr(config, "IS_SIMULATION", False)
        else:
            default_url = "http://192.168.137.31:81/stream"
            default_sim = False

        self.stream_url = stream_url if stream_url is not None else default_url
        self.is_simulation = is_simulation if is_simulation is not None else default_sim
        self.reconnect_interval = reconnect_interval
        self.webcam_index = webcam_index
        self.video_path = video_path

        self.cap: Optional[cv2.VideoCapture] = None
        self.active_source_type: str = "none"

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ret: bool = False
        self._sim_step: int = 0

    def start(self) -> "VisionStream":
        """Start the background frame capture producer thread."""
        with self._lock:
            if self._running:
                return self
            self._running = True
            self._thread = threading.Thread(
                target=self._capture_loop, daemon=True, name="VisionStreamThread"
            )
            self._thread.start()
            logger.info("VisionStream background capture thread started.")
            return self

    def stop(self) -> None:
        """Stop background capture thread and release resources."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self._lock:
            self._release_cap()
            logger.info("VisionStream stopped and resources released.")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Thread-safe method to retrieve the latest frame.
        Returns:
            (ret, frame): tuple where ret is bool and frame is BGR numpy array (640x480).
        """
        with self._lock:
            if self._latest_frame is None:
                return False, None
            return self._latest_ret, self._latest_frame.copy()

    def is_connected(self) -> bool:
        """Check if frame acquisition is currently successful."""
        with self._lock:
            return self._latest_ret and self._latest_frame is not None

    def get_source_type(self) -> str:
        """Get the description of active video source type ('esp32', 'webcam', 'video', 'synthetic')."""
        with self._lock:
            return self.active_source_type

    def _release_cap(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logger.warning("Error releasing VideoCapture: %s", e)
            self.cap = None

    def _discover_camera_ip(self, timeout: float = 3.0) -> Optional[str]:
        """Listens for UDP broadcasts from the ESP32 to automatically discover its IP."""
        logger.info("Starting UDP Auto-Discovery for ESP32 camera...")
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Bind to all interfaces on port 12345
            sock.bind(("0.0.0.0", 12345))
            sock.settimeout(timeout)
            
            # Wait for broadcast packet
            data, addr = sock.recvfrom(1024)
            msg = data.decode('utf-8').strip()
            
            if msg.startswith("ECOZONE_CAM:"):
                ip = msg.split(":")[1]
                url = f"http://{ip}:81/stream"
                logger.info("Auto-Discovery SUCCESS! Camera found at %s", url)
                return url
        except socket.timeout:
            logger.warning("UDP Auto-Discovery timed out (no camera broadcast detected).")
        except Exception as e:
            logger.warning("UDP Auto-Discovery error: %s", e)
        finally:
            sock.close()
            
        return None

    def _init_source(self) -> bool:
        self._release_cap()

        if not self.is_simulation:
            # Attempt UDP Auto-Discovery before using config URL
            discovered_url = self._discover_camera_ip()
            if discovered_url:
                self.stream_url = discovered_url
                
            logger.info("Attempting ESP32 MJPEG stream connection: %s", self.stream_url)
            self.cap = cv2.VideoCapture(self.stream_url)
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    self.active_source_type = "esp32"
                    logger.info("Successfully connected to ESP32 stream.")
                    return True
            logger.warning("Failed to open ESP32 stream at %s", self.stream_url)
            self._release_cap()

            force_esp32 = os.getenv("FORCE_ESP32_ONLY", "0").lower() in ("1", "true", "yes")
            if force_esp32:
                logger.info("FORCE_ESP32_ONLY is active. Skipping fallbacks.")
                return False

            logger.info("Auto-fallback engaged: attempting webcam, video, and synthetic sources.")

        # Fallback Sequence (Simulation or Unreachable ESP32)
        # 1. Try local webcam
        logger.info("Trying local webcam index %d", self.webcam_index)
        self.cap = cv2.VideoCapture(self.webcam_index)
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                self.active_source_type = "webcam"
                logger.info("Webcam open successful.")
                return True
            self._release_cap()

        # 2. Try pre-recorded video file if specified
        if self.video_path:
            logger.info("Trying video file: %s", self.video_path)
            self.cap = cv2.VideoCapture(self.video_path)
            if self.cap.isOpened():
                self.active_source_type = "video"
                logger.info("Video file open successful.")
                return True
            self._release_cap()

        # 3. Synthetic frame generator fallback
        self.active_source_type = "synthetic"
        logger.info("Webcam/video unavailable. Using synthetic frame generator fallback.")
        return True

    def _capture_loop(self) -> None:
        consecutive_failures = 0

        while self._running:
            if self.active_source_type == "none" or (self.cap is None and self.active_source_type != "synthetic"):
                if not self._init_source():
                    time.sleep(self.reconnect_interval)
                    continue

            if self.active_source_type == "synthetic":
                frame = self._generate_synthetic_frame()
                ret = True
                time.sleep(0.033)  # ~30 FPS timing
            else:
                ret, frame = self.cap.read()
                if not ret or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    logger.warning("Frame read failure count: %d", consecutive_failures)

                    if self.active_source_type == "video" and self.cap is not None:
                        # Loop video
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue

                    if consecutive_failures >= 5:
                        logger.error("Too many frame failures (%d). Re-initializing source.", consecutive_failures)
                        self.active_source_type = "none"
                        self._release_cap()
                        consecutive_failures = 0
                        time.sleep(self.reconnect_interval)
                    continue
                else:
                    consecutive_failures = 0

            with self._lock:
                self._latest_frame = frame
                self._latest_ret = ret

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generates a 640x480 synthetic frame with grid lines and a moving person target."""
        width, height = 640, 480
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (46, 26, 26)  # Dark purple-blue background

        # Draw simulated zone grid lines (3 rows x 2 cols)
        cv2.line(frame, (int(width / 2.0), 0), (int(width / 2.0), height), (80, 80, 80), 1)
        for r in range(1, 3):
            y = int(r * (height / 3.0))
            cv2.line(frame, (0, y), (width, y), (80, 80, 80), 1)

        # Calculate moving synthetic person position
        self._sim_step += 1
        cx = int(320 + 200 * math.sin(self._sim_step * 0.05))
        cy = int(240 + 120 * math.cos(self._sim_step * 0.05))

        # Draw synthetic person (head + body) in green (BGR: 0, 255, 0)
        cv2.circle(frame, (cx, cy - 30), 15, (0, 255, 0), -1)  # Green head
        cv2.rectangle(frame, (cx - 20, cy - 15), (cx + 20, cy + 40), (0, 255, 0), -1)  # Green body

        # Draw yellow proxy marker (BGR: 0, 255, 255) for Color Mode testing
        # Radius 16 gives area pi * 16^2 = 804 px^2 (> COLOR_MIN_CONTOUR_AREA=500)
        cv2.circle(frame, (cx, cy + 12), 16, (0, 255, 255), -1)

        # Text overlay
        cv2.putText(frame, "SIMULATION MODE - SYNTHETIC STREAM", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Step: {self._sim_step}", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def __enter__(self) -> "VisionStream":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
