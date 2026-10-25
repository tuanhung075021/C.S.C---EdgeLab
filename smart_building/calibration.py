"""
EcoZone AI Smart Energy Management System
Camera Calibration Tool (Requirement R6)

Captures camera frame, prompts user to click 4 corners of physical table (TL, TR, BR, BL),
computes 3x3 Homography Matrix mapping to 600x400, and saves to perspective_matrix.npy.
Supports interactive GUI and non-interactive headless CLI modes.
"""

import argparse
import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Attempt to load config module with fallbacks
try:
    import config
    WARP_WIDTH = getattr(config, "WARP_WIDTH", 600)
    WARP_HEIGHT = getattr(config, "WARP_HEIGHT", 400)
    ESP32_STREAM_URL = getattr(config, "ESP32_STREAM_URL", "http://192.168.137.31:81/stream")
except ImportError:
    WARP_WIDTH = 600
    WARP_HEIGHT = 400
    ESP32_STREAM_URL = "http://192.168.137.31:81/stream"

# Default table corner points (TL, TR, BR, BL)
DEFAULT_POINTS = np.float32([[50, 50], [550, 50], [550, 350], [50, 350]])

# Base Destination points for 600x400 perspective transformation
# These can be rotated dynamically to handle different camera view angles.
BASE_DST_POINTS = np.float32([
    [0, 0],
    [WARP_WIDTH, 0],
    [WARP_WIDTH, WARP_HEIGHT],
    [0, WARP_HEIGHT]
])


def acquire_frame(source: str = None, headless: bool = False) -> np.ndarray:
    """
    Frame acquisition with 3-tier fallback hierarchy:
    Tier 1: core.vision_stream.VisionStream
    Tier 2: OpenCV cv2.VideoCapture (CLI source, config URL, webcam 0)
    Tier 3: Synthetic 640x480 calibration image
    """
    if not headless:
        # Tier 1: Try core.vision_stream.VisionStream
        try:
            from core.vision_stream import VisionStream
            stream_url = source or ESP32_STREAM_URL
            vs = VisionStream(stream_url=stream_url, is_simulation=False)
            vs.start()
            for _ in range(25):
                ret, frame = vs.read_frame()
                if ret and frame is not None and frame.size > 0:
                    vs.stop()
                    print(f"[calibration] Acquired frame via VisionStream (source type: {vs.get_source_type()})")
                    return frame
                time.sleep(0.1)
            vs.stop()
        except Exception as err:
            print(f"[calibration] VisionStream acquisition attempt notice: {err}")

        # Tier 2: OpenCV cv2.VideoCapture fallbacks
        sources_to_try = []
        if source:
            sources_to_try.append(source)
        if ESP32_STREAM_URL and ESP32_STREAM_URL not in sources_to_try:
            sources_to_try.append(ESP32_STREAM_URL)
        sources_to_try.append(0)

        for src in sources_to_try:
            try:
                cap = cv2.VideoCapture(src if isinstance(src, int) or (isinstance(src, str) and not src.isdigit()) else int(src))
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None and frame.size > 0:
                        print(f"[calibration] Acquired frame via cv2.VideoCapture({src})")
                        return frame
            except Exception as err:
                print(f"[calibration] cv2.VideoCapture({src}) failed: {err}")

    # Tier 3: Synthetic 640x480 test image fallback
    print("[calibration] Using synthetic 640x480 test image for calibration.")
    synthetic = np.zeros((480, 640, 3), dtype=np.uint8)
    synthetic[:] = (40, 40, 50)
    # Visual grid lines and border
    cv2.rectangle(synthetic, (50, 50), (550, 350), (200, 200, 200), 2)
    cv2.line(synthetic, (216, 50), (216, 350), (100, 100, 100), 1)
    cv2.line(synthetic, (383, 50), (383, 350), (100, 100, 100), 1)
    cv2.line(synthetic, (50, 200), (550, 200), (100, 100, 100), 1)
    cv2.putText(synthetic, "CALIBRATION TEST FRAME (640x480)", (120, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return synthetic


def run_interactive_calibration(frame: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Opens an interactive OpenCV GUI window titled 'Camera Calibration - EcoZone AI'.
    Registers mouse callback for picking 4 corners.
    Handles 'r' to reset, 't' to toggle rotation, 'q'/ESC to cancel, ENTER to confirm.
    Returns (src_points, rotation_index).
    """
    clicked_points = []
    point_labels = ["1: Top-Left", "2: Top-Right", "3: Bottom-Right", "4: Bottom-Left"]
    rotation_index = 0  # 0, 1, 2, 3 (each is 90 deg rotation)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
            clicked_points.append((x, y))
            lbl = point_labels[len(clicked_points) - 1]
            print(f"[calibration] Point {len(clicked_points)} selected ({lbl}): ({x}, {y})")

    win_name = "Camera Calibration - EcoZone AI"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, mouse_callback)

    display_frame = frame.copy()

    print("[calibration] Interactive window open. Click 4 points in order:")
    print("  1: Top-Left -> 2: Top-Right -> 3: Bottom-Right -> 4: Bottom-Left")
    print("  Press 'r' to reset, 't' to rotate mapping 90deg, 'q'/ESC to cancel, ENTER to confirm.")

    while True:
        draw_frame = display_frame.copy()

        for idx, (px, py) in enumerate(clicked_points):
            cv2.circle(draw_frame, (px, py), 6, (0, 255, 0), -1)
            cv2.circle(draw_frame, (px, py), 8, (255, 255, 255), 1)
            cv2.putText(draw_frame, f"{idx+1}", (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if len(clicked_points) > 1:
            pts_array = np.array(clicked_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(draw_frame, [pts_array], isClosed=(len(clicked_points) == 4), color=(0, 255, 255), thickness=2)

        if len(clicked_points) < 4:
            next_step = point_labels[len(clicked_points)]
            cv2.putText(draw_frame, f"Click point: {next_step}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            rot_deg = rotation_index * 90
            cv2.putText(draw_frame, f"4 points ok! Rotation: {rot_deg}deg. Press 't' to rotate, ENTER confirm", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw preview of destination corners
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)] # BGR
            for i in range(4):
                px, py = clicked_points[i]
                map_idx = (i + rotation_index) % 4
                dst_pt = BASE_DST_POINTS[map_idx]
                cv2.putText(draw_frame, f"->({int(dst_pt[0])},{int(dst_pt[1])})", (px + 10, py + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[map_idx], 2)

        cv2.imshow(win_name, draw_frame)
        key = cv2.waitKey(30) & 0xFF

        if key in (13, 10, 32):  # ENTER
            if len(clicked_points) == 4:
                print(f"[calibration] Confirmed. Applying rotation: {rotation_index * 90} degrees.")
                break
        elif key in (ord('r'), ord('R')):
            clicked_points.clear()
        elif key in (ord('t'), ord('T')):
            rotation_index = (rotation_index + 1) % 4
            print(f"[calibration] Toggled mapping rotation to {rotation_index * 90} degrees.")
        elif key in (ord('q'), ord('Q'), 27):
            cv2.destroyAllWindows()
            sys.exit(1)

    cv2.destroyAllWindows()
    return np.float32(clicked_points), rotation_index


def parse_args():
    parser = argparse.ArgumentParser(description="EcoZone AI Camera Calibration Tool")
    parser.add_argument("--headless", action="store_true", help="Run in headless non-interactive mode")
    parser.add_argument("--points", type=str, help="Comma-separated 8 floats: x1,y1,x2,y2,x3,y3,x4,y4")
    parser.add_argument("--source", type=str, help="Video stream URL, video file path, or camera index")
    parser.add_argument("--output", type=str, default="perspective_matrix.npy", help="Output numpy file path")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine 4 corner source points
    src_pts = None
    rotation_index = 0

    if args.points:
        try:
            coords = [float(p.strip()) for p in args.points.split(",")]
            if len(coords) != 8:
                raise ValueError("Expected exactly 8 comma-separated float numbers")
            src_pts = np.float32(coords).reshape((4, 2))
            print(f"[calibration] Using CLI provided points:\n{src_pts}")
        except Exception as err:
            print(f"[calibration] ERROR parsing --points: {err}")
            sys.exit(1)

    if src_pts is None and not args.headless:
        frame = acquire_frame(source=args.source, headless=False)
        try:
            src_pts, rotation_index = run_interactive_calibration(frame)
        except (cv2.error, Exception) as err:
            print(f"[calibration] GUI window creation failed ({err}). Falling back to defaults.")
            src_pts = DEFAULT_POINTS.copy()

    if src_pts is None:
        print("[calibration] Headless mode active. Using default table corners:")
        src_pts = DEFAULT_POINTS.copy()

    # Apply rotation mapping to destination points
    dst_pts = np.roll(BASE_DST_POINTS, rotation_index, axis=0)

    # Compute 3x3 Homography Matrix
    print(f"[calibration] Computing Homography matrix with rotation {rotation_index*90}deg...")
    print(f"[calibration] Mapping source points to destination:\n{dst_pts}")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Save matrix to file
    output_path = args.output
    np.save(output_path, M)

    # Verification: reload saved file and check shape
    try:
        reloaded_M = np.load(output_path)
        assert reloaded_M.shape == (3, 3), f"Matrix shape mismatch: {reloaded_M.shape}"
        print(f"[calibration] SUCCESS: Perspective matrix calculated and saved to '{output_path}'")
        print(f"[calibration] Matrix shape: {reloaded_M.shape}, dtype: {reloaded_M.dtype}")
        print("[calibration] Homography Matrix M:")
        print(reloaded_M)
        return 0
    except Exception as err:
        print(f"[calibration] ERROR verifying saved matrix: {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
