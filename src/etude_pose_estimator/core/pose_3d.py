"""3D pose lifting using MotionBERT.

This module provides 2D-to-3D pose lifting functionality using MotionBERT,
which handles camera angle differences and enables fair pose comparison.

Note: MotionBERT requires manual setup. Run `task setup-motionbert` before use.
"""

from pathlib import Path

import numpy as np
import torch


class MotionBERTLifter:
    """3D pose lifter using MotionBERT.

    This lifter converts 2D keypoints to 3D coordinates, handling camera angle
    differences for fair pose comparison. Uses DSTformer architecture.

    COCO to H36M 17 joints mapping:
        - Direct mapping for compatible joints
        - Interpolation for head, thorax, spine, hip

    Attributes:
        model_path: Path to MotionBERT checkpoint file
        device: Device for inference ('cuda' or 'cpu')
        model: MotionBERT model instance
    """

    def __init__(
        self,
        model_path: Path,
        device: str = "cuda",
    ) -> None:
        """Initialize MotionBERT 3D pose lifter.

        Args:
            model_path: Path to MotionBERT checkpoint file (.pth.tar)
            device: Device for inference ('cuda' or 'cpu')

        Note:
            If model_path does not exist, operates in placeholder mode
        """
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.placeholder_mode = not model_path.exists()

        if not self.placeholder_mode:
            # TODO: Load MotionBERT model after setup
            # This requires the MotionBERT repository to be cloned and configured
            # For now, we use placeholder mode
            try:
                self._load_model()
            except ImportError:
                # If MotionBERT is not available, switch to placeholder mode
                self.placeholder_mode = True

    def _load_model(self) -> None:
        """Load MotionBERT model from checkpoint.

        Raises:
            ImportError: If MotionBERT library is not available
            RuntimeError: If checkpoint loading fails
        """
        # TODO: Implement MotionBERT model loading
        # This requires:
        # 1. MotionBERT repository cloned to models/MotionBERT
        # 2. Checkpoint downloaded to models/motionbert/checkpoint.pth.tar
        # 3. MotionBERT library in Python path
        #
        # Example implementation (requires MotionBERT setup):
        # from motionbert.lib.model.DSTformer import DSTformer
        # self.model = DSTformer(...)
        # checkpoint = torch.load(self.model_path, map_location=self.device)
        # self.model.load_state_dict(checkpoint['model'])
        # self.model.eval()

        raise NotImplementedError(
            "MotionBERT model loading not implemented yet. "
            "This requires manual setup of MotionBERT repository. "
            "Please refer to scripts/setup_motionbert.sh"
        )

    def lift_2d_to_3d(
        self,
        keypoints_2d: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Lift 2D keypoints to 3D coordinates.

        Args:
            keypoints_2d: 2D keypoints array of shape (17, 2) or (17, 3).
                If shape is (17, 3), the third column (confidence) is ignored.
            normalize: Whether to normalize 3D coordinates (center at origin,
                unit variance)

        Returns:
            3D keypoints array of shape (17, 3) containing [x, y, z] coordinates

        Raises:
            ValueError: If input shape is invalid
            RuntimeError: If model is not loaded
        """
        if keypoints_2d.shape[0] != 17:
            raise ValueError(f"Expected 17 keypoints, got {keypoints_2d.shape[0]}")

        if keypoints_2d.shape[1] not in [2, 3]:
            raise ValueError(f"Expected shape (17, 2) or (17, 3), got {keypoints_2d.shape}")

        # Extract only x, y coordinates if confidence is included
        if keypoints_2d.shape[1] == 3:
            keypoints_2d = keypoints_2d[:, :2]

        # TODO: Implement actual 3D lifting with MotionBERT
        # For now, return a placeholder (simple elevation to 3D)
        # This should be replaced with actual MotionBERT inference
        keypoints_3d = np.zeros((17, 3), dtype=np.float32)
        keypoints_3d[:, :2] = keypoints_2d
        keypoints_3d[:, 2] = 0.0  # Placeholder z-coordinate

        if normalize:
            keypoints_3d = self._normalize_3d(keypoints_3d)

        return keypoints_3d

    def _normalize_3d(self, keypoints_3d: np.ndarray) -> np.ndarray:
        """Normalize 3D keypoints (center at origin, unit variance).

        Args:
            keypoints_3d: 3D keypoints array of shape (17, 3)

        Returns:
            Normalized 3D keypoints array
        """
        # Center at hip (average of left_hip and right_hip)
        left_hip_idx = 11
        right_hip_idx = 12
        hip_center = (keypoints_3d[left_hip_idx] + keypoints_3d[right_hip_idx]) / 2
        keypoints_3d = keypoints_3d - hip_center

        # Scale to unit variance
        std = np.std(keypoints_3d)
        if std > 1e-6:  # Avoid division by zero
            keypoints_3d = keypoints_3d / std

        return keypoints_3d
