"""3D pose lifting using MotionBERT.

This module provides 2D-to-3D pose lifting functionality using MotionBERT,
which handles camera angle differences and enables fair pose comparison.

Note: MotionBERT requires manual setup. Run `task setup-motionbert` before use.
"""

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


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
        try:
            from etude_pose_estimator.motionbert import DSTformer
        except ImportError as e:
            raise ImportError(
                "MotionBERT model code not found. "
                "Please ensure DSTformer.py and drop.py are in "
                "src/etude_pose_estimator/motionbert/"
            ) from e

        logger.info(f"Loading MotionBERT model from {self.model_path}")

        # Initialize DSTformer model
        # Parameters from MotionBERT's MB_ft_h36m.yaml configuration
        self.model = DSTformer(
            dim_in=3,  # Input dimension (x, y, confidence)
            dim_out=3,  # Output dimension (x, y, z)
            dim_feat=512,  # Feature dimension
            dim_rep=512,  # Representation dimension
            depth=5,  # Transformer block depth
            num_heads=8,  # Multi-head attention heads
            mlp_ratio=2,  # MLP hidden layer expansion ratio
            num_joints=17,  # H36M format (17 joints)
            maxlen=243,  # Max sequence length
            att_fuse=True,  # Enable attention fusion
        )

        # Load checkpoint
        try:
            checkpoint = torch.load(
                self.model_path,
                map_location=self.device,
                weights_only=False,  # Required for OrderedDict
            )

            # Extract model weights (key: 'model_pos')
            if "model_pos" in checkpoint:
                state_dict = checkpoint["model_pos"]
                logger.info("✓ Found 'model_pos' key in checkpoint")
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
                logger.info("✓ Found 'model' key in checkpoint")
            else:
                raise RuntimeError(
                    f"Checkpoint does not contain 'model_pos' or 'model' key. "
                    f"Available keys: {list(checkpoint.keys())}"
                )

            # Remove 'module.' prefix if present (from DataParallel training)
            if any(k.startswith("module.") for k in state_dict.keys()):
                logger.info("Removing 'module.' prefix from state dict keys")
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }

            # Load weights into model
            self.model.load_state_dict(state_dict, strict=True)
            logger.info("✓ Model weights loaded successfully")

            # Set model to evaluation mode
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"✓ MotionBERT model loaded on {self.device}")

        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    def _convert_coco_to_h36m(self, coco_kp: np.ndarray) -> np.ndarray:
        """Convert COCO 17 keypoints to H36M 17 keypoints format.

        COCO format (17 joints):
            0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
            5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
            9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
            13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle

        H36M format (17 joints):
            0:pelvis, 1:right_hip, 2:right_knee, 3:right_ankle,
            4:left_hip, 5:left_knee, 6:left_ankle, 7:torso, 8:neck,
            9:nose, 10:head, 11:left_shoulder, 12:left_elbow, 13:left_wrist,
            14:right_shoulder, 15:right_elbow, 16:right_wrist

        Args:
            coco_kp: COCO keypoints array of shape (17, 2) or (17, 3)

        Returns:
            H36M keypoints array of shape (17, 2)
        """
        h36m_kp = np.zeros((17, 2), dtype=np.float32)

        # Direct mappings
        h36m_kp[1] = coco_kp[12, :2]  # right_hip
        h36m_kp[2] = coco_kp[14, :2]  # right_knee
        h36m_kp[3] = coco_kp[16, :2]  # right_ankle
        h36m_kp[4] = coco_kp[11, :2]  # left_hip
        h36m_kp[5] = coco_kp[13, :2]  # left_knee
        h36m_kp[6] = coco_kp[15, :2]  # left_ankle
        h36m_kp[9] = coco_kp[0, :2]  # nose
        h36m_kp[11] = coco_kp[5, :2]  # left_shoulder
        h36m_kp[12] = coco_kp[7, :2]  # left_elbow
        h36m_kp[13] = coco_kp[9, :2]  # left_wrist
        h36m_kp[14] = coco_kp[6, :2]  # right_shoulder
        h36m_kp[15] = coco_kp[8, :2]  # right_elbow
        h36m_kp[16] = coco_kp[10, :2]  # right_wrist

        # Interpolated joints
        h36m_kp[0] = (coco_kp[11, :2] + coco_kp[12, :2]) / 2  # pelvis (hip center)
        h36m_kp[8] = (coco_kp[5, :2] + coco_kp[6, :2]) / 2  # neck (shoulder center)
        h36m_kp[7] = (h36m_kp[0] + h36m_kp[8]) / 2  # torso (midpoint between pelvis and neck)
        h36m_kp[10] = (coco_kp[1, :2] + coco_kp[2, :2]) / 2  # head (eye center)

        return h36m_kp

    def lift_2d_to_3d(
        self,
        keypoints_2d: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """Lift 2D keypoints to 3D coordinates.

        Args:
            keypoints_2d: 2D keypoints array of shape (17, 2) or (17, 3) in COCO format.
                If shape is (17, 3), the third column (confidence) is ignored.
            normalize: Whether to normalize 3D coordinates (center at origin,
                unit variance)

        Returns:
            3D keypoints array of shape (17, 3) containing [x, y, z] coordinates
                in H36M format

        Raises:
            ValueError: If input shape is invalid
            RuntimeError: If model is not loaded
        """
        if keypoints_2d.shape[0] != 17:
            raise ValueError(f"Expected 17 keypoints, got {keypoints_2d.shape[0]}")

        if keypoints_2d.shape[1] not in [2, 3]:
            raise ValueError(f"Expected shape (17, 2) or (17, 3), got {keypoints_2d.shape}")

        # Convert COCO format to H36M format
        h36m_2d = self._convert_coco_to_h36m(keypoints_2d)

        # Use placeholder mode if model is not loaded
        if self.placeholder_mode:
            logger.warning("MotionBERT not available - using placeholder (z=0)")
            keypoints_3d = np.zeros((17, 3), dtype=np.float32)
            keypoints_3d[:, :2] = h36m_2d
            keypoints_3d[:, 2] = 0.0
            if normalize:
                keypoints_3d = self._normalize_3d(keypoints_3d)
            return keypoints_3d

        # Prepare input for MotionBERT
        input_2d = h36m_2d.copy()

        # MotionBERT expects shape: [batch, frames, joints, channels]
        # We have single image, so: [1, 1, 17, 3]
        # Third channel is confidence (set to 1.0 for all joints)
        batch_input = np.zeros((1, 1, 17, 3), dtype=np.float32)
        batch_input[0, 0, :, :2] = input_2d
        batch_input[0, 0, :, 2] = 1.0  # Confidence = 1.0

        # Convert to torch tensor
        input_tensor = torch.from_numpy(batch_input).to(self.device)

        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)  # Shape: [1, 1, 17, 3]

        # Extract 3D coordinates
        keypoints_3d = output[0, 0].cpu().numpy()  # Shape: (17, 3)

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
