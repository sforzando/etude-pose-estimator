"""Pose analysis: joint angles and similarity comparison."""

import numpy as np
from scipy.spatial import procrustes


class PoseAnalyzer:
    """Analyzer for pose joint angles and similarity."""

    # MediaPipe Pose landmark indices
    # Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    def calculate_angle(
        self, point1: list[float], point2: list[float], point3: list[float]
    ) -> float:
        """
        Calculate angle between three points in degrees.

        Args:
            point1: First point (x, y, z)
            point2: Middle point (vertex of angle)
            point3: Third point (x, y, z)

        Returns:
            Angle in degrees
        """
        # Convert to numpy arrays
        p1 = np.array(point1[:2])  # Use only x, y coordinates
        p2 = np.array(point2[:2])
        p3 = np.array(point3[:2])

        # Calculate vectors
        v1 = p1 - p2
        v2 = p3 - p2

        # Check for zero-length vectors to avoid division by zero
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            # Return 180 degrees for straight/undefined angles
            return 180.0

        # Calculate angle using dot product
        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)

        # Clamp to [-1, 1] to avoid numerical errors
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Calculate angle in degrees
        angle = np.arccos(cos_angle) * 180 / np.pi

        return float(angle)

    def calculate_joint_angles(self, landmarks: list[list[float]]) -> dict[str, float]:
        """
        Calculate key joint angles from pose landmarks.

        Args:
            landmarks: List of 33 landmarks (x, y, z)

        Returns:
            Dictionary of joint angles
        """
        angles = {}

        # Left elbow angle
        angles["left_elbow"] = self.calculate_angle(
            landmarks[self.LEFT_SHOULDER],
            landmarks[self.LEFT_ELBOW],
            landmarks[self.LEFT_WRIST],
        )

        # Right elbow angle
        angles["right_elbow"] = self.calculate_angle(
            landmarks[self.RIGHT_SHOULDER],
            landmarks[self.RIGHT_ELBOW],
            landmarks[self.RIGHT_WRIST],
        )

        # Left shoulder angle
        angles["left_shoulder"] = self.calculate_angle(
            landmarks[self.LEFT_ELBOW],
            landmarks[self.LEFT_SHOULDER],
            landmarks[self.LEFT_HIP],
        )

        # Right shoulder angle
        angles["right_shoulder"] = self.calculate_angle(
            landmarks[self.RIGHT_ELBOW],
            landmarks[self.RIGHT_SHOULDER],
            landmarks[self.RIGHT_HIP],
        )

        # Left knee angle
        angles["left_knee"] = self.calculate_angle(
            landmarks[self.LEFT_HIP],
            landmarks[self.LEFT_KNEE],
            landmarks[self.LEFT_ANKLE],
        )

        # Right knee angle
        angles["right_knee"] = self.calculate_angle(
            landmarks[self.RIGHT_HIP],
            landmarks[self.RIGHT_KNEE],
            landmarks[self.RIGHT_ANKLE],
        )

        return angles

    def calculate_similarity(
        self, landmarks1: list[list[float]], landmarks2: list[list[float]]
    ) -> float:
        """
        Calculate similarity between two poses using Procrustes analysis.

        Args:
            landmarks1: First pose landmarks
            landmarks2: Second pose landmarks (reference)

        Returns:
            Similarity score (0-100, higher is more similar)
        """
        # Convert landmarks to numpy arrays (use only x, y coordinates)
        pose1 = np.array([[lm[0], lm[1]] for lm in landmarks1])
        pose2 = np.array([[lm[0], lm[1]] for lm in landmarks2])

        # Perform Procrustes analysis
        mtx1, mtx2, disparity = procrustes(pose1, pose2)

        # Convert disparity to similarity score (0-100)
        # Disparity is typically in range [0, 2], where 0 is identical
        similarity = max(0, 100 * (1 - disparity / 2))

        return round(similarity, 2)
