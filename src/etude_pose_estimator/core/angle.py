"""Joint angle calculation from 3D poses.

This module provides joint angle calculation functionality, which is more
intuitive for feedback than raw distance measurements.
"""

import numpy as np


class JointAngleCalculator:
    """Joint angle calculator for 3D poses.

    Calculates joint angles from 3D coordinates using the dot product formula.
    Angles are returned in degrees (0-180 range).

    H36M/COCO 17 joint indices:
        5: left_shoulder, 6: right_shoulder
        7: left_elbow, 8: right_elbow
        9: left_wrist, 10: right_wrist
        11: left_hip, 12: right_hip
        13: left_knee, 14: right_knee
        15: left_ankle, 16: right_ankle
    """

    # Joint angle definitions: (joint_name, point1_idx, joint_idx, point2_idx)
    JOINT_ANGLES = [
        # Elbows
        ("left_elbow", 5, 7, 9),  # shoulder -> elbow -> wrist
        ("right_elbow", 6, 8, 10),
        # Shoulders
        ("left_shoulder", 7, 5, 11),  # elbow -> shoulder -> hip
        ("right_shoulder", 8, 6, 12),
        # Knees
        ("left_knee", 11, 13, 15),  # hip -> knee -> ankle
        ("right_knee", 12, 14, 16),
        # Hips
        ("left_hip", 5, 11, 13),  # shoulder -> hip -> knee
        ("right_hip", 6, 12, 14),
    ]

    def calculate_angles(self, pose_3d: np.ndarray) -> dict[str, float]:
        """Calculate all joint angles from 3D pose.

        Args:
            pose_3d: 3D pose array of shape (17, 3)

        Returns:
            Dictionary mapping joint names to angles in degrees

        Raises:
            ValueError: If pose shape is invalid
        """
        if pose_3d.shape != (17, 3):
            raise ValueError(f"Expected pose shape (17, 3), got {pose_3d.shape}")

        angles = {}
        for joint_name, p1_idx, joint_idx, p2_idx in self.JOINT_ANGLES:
            angle = self._calculate_angle(
                pose_3d[p1_idx],
                pose_3d[joint_idx],
                pose_3d[p2_idx],
            )
            angles[joint_name] = angle

        return angles

    def _calculate_angle(
        self,
        point1: np.ndarray,
        point2: np.ndarray,
        point3: np.ndarray,
    ) -> float:
        """Calculate angle at point2 formed by three 3D points.

        Uses the dot product formula:
        cos(θ) = (v1 · v2) / (|v1| |v2|)

        Args:
            point1: First point (3D coordinates)
            point2: Joint point (vertex of the angle)
            point3: Third point (3D coordinates)

        Returns:
            Angle in degrees (0-180 range)
        """
        # Create vectors from joint to other points
        vector1 = point1 - point2
        vector2 = point3 - point2

        # Calculate magnitudes
        mag1 = np.linalg.norm(vector1)
        mag2 = np.linalg.norm(vector2)

        # Avoid division by zero
        epsilon = 1e-6
        if mag1 < epsilon or mag2 < epsilon:
            return 0.0

        # Calculate dot product
        dot_product = np.dot(vector1, vector2)

        # Calculate cosine of angle
        cos_angle = dot_product / (mag1 * mag2)

        # Clamp to [-1, 1] for numerical stability
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

        # Calculate angle in radians and convert to degrees
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return float(angle_deg)

    def compare_angles(
        self,
        angles1: dict[str, float],
        angles2: dict[str, float],
    ) -> dict[str, float]:
        """Compare two sets of joint angles.

        Args:
            angles1: First set of joint angles
            angles2: Second set of joint angles

        Returns:
            Dictionary mapping joint names to absolute angle differences

        Raises:
            ValueError: If angle dictionaries have different keys
        """
        if set(angles1.keys()) != set(angles2.keys()):
            raise ValueError("Angle dictionaries must have the same joint names")

        differences = {}
        for joint_name in angles1:
            diff = abs(angles1[joint_name] - angles2[joint_name])
            differences[joint_name] = diff

        return differences
