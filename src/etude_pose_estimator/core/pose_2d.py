"""2D pose detection using YOLO11x.

This module provides pose detection functionality using YOLO11x Pose model,
which has been proven to work with costumes (82% confidence with Ultraman).
MediaPipe was rejected due to inability to detect poses in costumes.
"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO


class YOLO11xPoseDetector:
    """2D pose detector using YOLO11x Pose model.

    This detector uses YOLO11x Pose to detect 17 COCO keypoints from images.
    It has been proven to work with costumes at 82% confidence, unlike MediaPipe
    which failed to detect poses in costume scenarios.

    COCO 17 Keypoints:
        0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
        5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
        9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
        13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

    Attributes:
        model: YOLO11x Pose model instance
        model_path: Path to the model file
    """

    def __init__(self, model_path: Path) -> None:
        """Initialize YOLO11x pose detector.

        Args:
            model_path: Path to YOLO11x pose model file (.pt)

        Raises:
            FileNotFoundError: If model file does not exist
        """
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO11x model not found at {model_path}")

        self.model_path = model_path
        self.model = YOLO(str(model_path))

    def detect(
        self,
        image_path: Path,
        conf_threshold: float = 0.5,
    ) -> np.ndarray | None:
        """Detect 2D pose from an image.

        Args:
            image_path: Path to input image file
            conf_threshold: Confidence threshold for detection (0.0-1.0)

        Returns:
            NumPy array of shape (17, 3) containing [x, y, confidence] for each
            keypoint in normalized coordinates (0-1 range), or None if no pose
            is detected above the confidence threshold.

        Raises:
            FileNotFoundError: If image file does not exist
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Run inference
        results = self.model(str(image_path), verbose=False)

        # Check if any poses were detected
        if len(results) == 0 or results[0].keypoints is None:
            return None

        # Get keypoints for the first person only
        keypoints = results[0].keypoints

        if keypoints.data is None or len(keypoints.data) == 0:
            return None

        # Extract keypoints: shape (num_people, 17, 3) -> (17, 3) for first person
        kpts = keypoints.data[0].cpu().numpy()

        # Check if confidence is above threshold
        if kpts[:, 2].mean() < conf_threshold:
            return None

        # Normalize coordinates to 0-1 range
        # YOLO returns pixel coordinates, need to normalize by image size
        img_height, img_width = results[0].orig_shape
        kpts[:, 0] /= img_width  # Normalize x
        kpts[:, 1] /= img_height  # Normalize y

        return kpts

    def detect_detailed(
        self,
        image_path: Path,
        conf_threshold: float = 0.5,
    ) -> tuple[np.ndarray, tuple[float, float, float, float], float] | None:
        """Detect 2D pose with bounding box and confidence.

        Args:
            image_path: Path to input image file
            conf_threshold: Confidence threshold for detection (0.0-1.0)

        Returns:
            Tuple of (keypoints, bbox, confidence) or None if no pose detected:
            - keypoints: NumPy array (17, 3) [x, y, confidence] in normalized coords
            - bbox: Tuple (x_min, y_min, x_max, y_max) in normalized coords
            - confidence: Detection confidence (0-1)

        Raises:
            FileNotFoundError: If image file does not exist
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        # Run inference
        results = self.model(str(image_path), verbose=False)

        # Check if any poses were detected
        if len(results) == 0 or results[0].keypoints is None:
            return None

        # Get keypoints for the first person only
        keypoints = results[0].keypoints

        if keypoints.data is None or len(keypoints.data) == 0:
            return None

        # Extract keypoints: shape (num_people, 17, 3) -> (17, 3) for first person
        kpts = keypoints.data[0].cpu().numpy()

        # Check if confidence is above threshold
        avg_conf = kpts[:, 2].mean()
        if avg_conf < conf_threshold:
            return None

        # Get bounding box (from boxes if available)
        bbox = None
        detection_conf = avg_conf

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            # Get first box
            box = results[0].boxes[0]
            box_coords = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
            detection_conf = float(box.conf[0].cpu().numpy())

            # Normalize bbox coordinates
            img_height, img_width = results[0].orig_shape
            bbox = (
                float(box_coords[0] / img_width),  # x_min
                float(box_coords[1] / img_height),  # y_min
                float(box_coords[2] / img_width),  # x_max
                float(box_coords[3] / img_height),  # y_max
            )
        else:
            # Calculate bbox from keypoints
            valid_kpts = kpts[kpts[:, 2] > 0.3]  # Filter by confidence
            if len(valid_kpts) > 0:
                x_coords = valid_kpts[:, 0]
                y_coords = valid_kpts[:, 1]
                bbox = (
                    float(x_coords.min()),
                    float(y_coords.min()),
                    float(x_coords.max()),
                    float(y_coords.max()),
                )

        # Normalize coordinates to 0-1 range
        img_height, img_width = results[0].orig_shape
        kpts[:, 0] /= img_width  # Normalize x
        kpts[:, 1] /= img_height  # Normalize y

        return kpts, bbox, detection_conf

    def detect_batch(
        self,
        image_paths: list[Path],
        conf_threshold: float = 0.5,
    ) -> list[np.ndarray | None]:
        """Detect 2D poses from multiple images.

        Args:
            image_paths: List of paths to input image files
            conf_threshold: Confidence threshold for detection (0.0-1.0)

        Returns:
            List of NumPy arrays, one for each image. Each array has shape
            (17, 3) or None if no pose was detected.

        Raises:
            FileNotFoundError: If any image file does not exist
        """
        results = []
        for image_path in image_paths:
            result = self.detect(image_path, conf_threshold)
            results.append(result)
        return results
