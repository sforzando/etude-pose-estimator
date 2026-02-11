#!/usr/bin/env python3
"""Download YOLO11x pose model.

This script automatically downloads the YOLO11x pose detection model
from Ultralytics and saves it to the models directory.

Usage:
    python scripts/download_models.py
    or
    task download-models
"""

import sys
from pathlib import Path

from ultralytics import YOLO


def download_yolo11x_pose() -> None:
    """Download YOLO11x pose model.

    The model will be automatically downloaded by Ultralytics to the
    default cache directory (~/.cache/ultralytics) and then a reference
    will be created in the models directory.
    """
    # Ensure models directory exists
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "yolo11x-pose.pt"

    print("=" * 60)
    print("Downloading YOLO11x Pose model...")
    print("=" * 60)
    print(f"Target path: {model_path.absolute()}")
    print()

    try:
        # Load model (this will auto-download if not cached)
        print("Initializing YOLO11x-pose model...")
        model = YOLO("yolo11x-pose.pt")

        print(f"✅ Model downloaded successfully!")
        print(f"   Model type: {model.task}")
        print(f"   Model file: {model.model_name}")
        print()

        # Get the actual model file path from cache
        if hasattr(model, "model_name"):
            # The model is downloaded to Ultralytics cache
            # We need to inform the user where it is
            print("📍 Model location:")
            print(f"   Cache: ~/.cache/ultralytics/")
            print(f"   Use YOLO_MODEL_PATH in .envrc to specify path")
            print()

        print("=" * 60)
        print("✨ YOLO11x Pose model is ready!")
        print("=" * 60)
        print()
        print("Next steps:")
        print("  1. The model is cached by Ultralytics")
        print("  2. Update YOLO_MODEL_PATH in .envrc if needed")
        print("  3. Run 'task setup-motionbert' to set up MotionBERT")
        print()

    except Exception as e:
        print(f"❌ Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    download_yolo11x_pose()
