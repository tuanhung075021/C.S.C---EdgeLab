"""
EcoZone AI Smart Energy Management System
One-time Model Setup Script (Requirement R5)

Downloads yolov8n.pt from Ultralytics if missing and exports it to OpenVINO IR FP16 format.
Idempotent: skips download and export if models/yolov8n_openvino_model/ contains valid yolov8n.xml and yolov8n.bin files.
"""

import logging
import os
import shutil
import sys
from pathlib import Path

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("setup_model")

PROJECT_ROOT = Path(__file__).parent.resolve()
MODELS_DIR = PROJECT_ROOT / "models"
OPENVINO_DIR = MODELS_DIR / "yolov8n_openvino_model"
XML_FILE = OPENVINO_DIR / "yolov8n.xml"
BIN_FILE = OPENVINO_DIR / "yolov8n.bin"
PT_FILE = MODELS_DIR / "yolov8n.pt"


def is_model_ready() -> bool:
    """Check if OpenVINO IR model directory exists and contains non-empty .xml and .bin files."""
    if not OPENVINO_DIR.is_dir():
        return False

    # Check exact expected files
    exact_ready = (
        XML_FILE.is_file()
        and XML_FILE.stat().st_size > 0
        and BIN_FILE.is_file()
        and BIN_FILE.stat().st_size > 0
    )
    if exact_ready:
        return True

    # Fallback glob check in case files exist under model directory
    xml_files = [f for f in OPENVINO_DIR.glob("*.xml") if f.stat().st_size > 0]
    bin_files = [f for f in OPENVINO_DIR.glob("*.bin") if f.stat().st_size > 0]
    return len(xml_files) > 0 and len(bin_files) > 0


def setup_model() -> bool:
    """Main execution function for model setup."""
    # 1. Idempotency check
    if is_model_ready():
        logger.info(
            f"OpenVINO model already exists at '{OPENVINO_DIR}'. Skipping setup."
        )
        print(f"[setup_model] OpenVINO model already exists in '{OPENVINO_DIR}'. Skipping export.")
        return True

    logger.info("Starting OpenVINO YOLOv8 model setup...")
    print("[setup_model] OpenVINO model not found. Initializing setup...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Check dependencies
    try:
        from ultralytics import YOLO
    except ImportError as err:
        logger.error(
            f"Missing required package 'ultralytics' or 'openvino': {err}. "
            "Please install requirements via: pip install ultralytics openvino"
        )
        print(f"[setup_model] ERROR: 'ultralytics' package is not installed: {err}")
        return False

    try:
        # 3. Load / download PyTorch weights
        if PT_FILE.exists():
            logger.info(f"Loading PyTorch weights from '{PT_FILE}'...")
            print(f"[setup_model] Found PyTorch model at '{PT_FILE}'")
            model = YOLO(str(PT_FILE))
        else:
            logger.info(f"Downloading PyTorch weights to '{PT_FILE}'...")
            print("[setup_model] Downloading 'yolov8n.pt'...")
            model = YOLO("yolov8n.pt")
            # If downloaded to current working directory, relocate to MODELS_DIR
            root_pt = PROJECT_ROOT / "yolov8n.pt"
            if root_pt.exists() and not PT_FILE.exists():
                shutil.move(str(root_pt), str(PT_FILE))
                logger.info(f"Moved downloaded 'yolov8n.pt' to '{PT_FILE}'")

        # 4. Export to OpenVINO FP16
        logger.info("Exporting YOLOv8n to OpenVINO FP16 format (imgsz=640)...")
        print("[setup_model] Exporting to OpenVINO IR format (FP16, imgsz=640)...")
        export_res = model.export(format="openvino", half=True, imgsz=640)
        logger.info(f"Export command finished. Export result path: {export_res}")
        print(f"[setup_model] Export completed: {export_res}")

        # 5. Post-export alignment: check if export output landed in root directory
        root_export_dir = PROJECT_ROOT / "yolov8n_openvino_model"
        if root_export_dir.exists() and root_export_dir.resolve() != OPENVINO_DIR.resolve():
            if not OPENVINO_DIR.exists():
                logger.info(f"Moving exported model from '{root_export_dir}' to '{OPENVINO_DIR}'...")
                shutil.move(str(root_export_dir), str(OPENVINO_DIR))
            else:
                for item in root_export_dir.iterdir():
                    shutil.move(str(item), str(OPENVINO_DIR / item.name))
                shutil.rmtree(str(root_export_dir))

        # Check if exported inside models/models/yolov8n_openvino_model by mistake
        nested_export_dir = MODELS_DIR / "models" / "yolov8n_openvino_model"
        if nested_export_dir.exists() and not is_model_ready():
            if not OPENVINO_DIR.exists():
                shutil.move(str(nested_export_dir), str(OPENVINO_DIR))
            else:
                for item in nested_export_dir.iterdir():
                    shutil.move(str(item), str(OPENVINO_DIR / item.name))

        # 6. Verification
        if is_model_ready():
            logger.info(f"SUCCESS: OpenVINO model successfully verified at '{OPENVINO_DIR}'.")
            print(f"[setup_model] SUCCESS: Model exported and verified successfully at '{OPENVINO_DIR}'")
            return True
        else:
            logger.error(f"Export completed, but expected files missing at '{OPENVINO_DIR}'.")
            print(f"[setup_model] ERROR: Export completed, but expected files missing at '{OPENVINO_DIR}'")
            return False

    except Exception as err:
        logger.error(f"Error during model setup: {err}", exc_info=True)
        print(f"[setup_model] ERROR during model setup: {err}")
        return False


if __name__ == "__main__":
    success = setup_model()
    sys.exit(0 if success else 1)
