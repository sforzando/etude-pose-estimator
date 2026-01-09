"""Pose estimation using MediaPipe Pose Landmarker."""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseEstimator:
    """Pose estimator using MediaPipe Pose Landmarker."""

    def __init__(self):
        """Initialize the pose landmarker."""
        # Create options for pose landmarker
        base_options = python.BaseOptions(model_asset_path=self._download_model())
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
            
            # Create a request with proper headers
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                },
            )
            
            try:
                with urllib.request.urlopen(req) as response:
                    with open(model_path, "wb") as out_file:
                        out_file.write(response.read())
                print("Model downloaded successfully.")
            except Exception as e:
                print(f"Error downloading model: {e}")
                print("Please download the model manually from:")
                print(url)
                print(f"and save it as {model_path}")
                raise

        return str(model_path)

    def detect_pose(self, image_path: str) -> list[list[float]] | None:
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
        landmarks_list = [[lm.x, lm.y, lm.z] for lm in landmarks]

        return landmarks_list
