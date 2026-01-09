"""Pose estimation using MediaPipe Pose Landmarker."""

from typing import List, Optional

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseEstimator:
    """Pose estimator using MediaPipe Pose Landmarker."""

    def __init__(self):
        """Initialize the pose landmarker."""
        # Create options for pose landmarker
        base_options = python.BaseOptions(
            model_asset_path=self._download_model()
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def _download_model(self) -> str:
        """Download MediaPipe pose landmarker model if not exists."""
        import urllib.request
        from pathlib import Path

        model_path = Path("pose_landmarker.task")
        
        if not model_path.exists():
            print("Downloading MediaPipe Pose Landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, model_path)
            print("Model downloaded successfully.")
        
        return str(model_path)

    def detect_pose(self, image_path: str) -> Optional[List[List[float]]]:
        """
        Detect pose landmarks from an image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            List of landmarks (x, y, z coordinates) or None if no pose detected
        """
        # Load the image
        image = mp.Image.create_from_file(image_path)
        
        # Detect pose landmarks
        detection_result = self.landmarker.detect(image)
        
        # Check if any pose was detected
        if not detection_result.pose_landmarks:
            return None
        
        # Get the first detected pose (33 landmarks)
        landmarks = detection_result.pose_landmarks[0]
        
        # Convert landmarks to list format
        landmarks_list = [
            [lm.x, lm.y, lm.z] for lm in landmarks
        ]
        
        return landmarks_list
