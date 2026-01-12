"""Pose comparison using Procrustes analysis.

This module provides pose comparison functionality using Procrustes analysis,
which automatically removes translation, rotation, and scaling differences.
"""

from typing import Any

import numpy as np
from scipy.spatial import procrustes

from etude_pose_estimator.core.joint_names import translate_joint_name


class ProcrustesComparator:
    """Pose comparator using Procrustes analysis.

    Procrustes analysis removes translation, rotation, and scaling differences,
    allowing fair comparison between poses from different body sizes and
    camera angles (after 3D lifting).

    The similarity score is computed as 1 - disparity, where:
    - 1.0 indicates identical poses
    - 0.0 indicates completely different poses
    """

    def compare(
        self,
        pose1: np.ndarray,
        pose2: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compare two 3D poses using Procrustes analysis.

        Args:
            pose1: First pose array of shape (N, 3)
            pose2: Second pose array of shape (N, 3)

        Returns:
            Tuple containing:
            - similarity_score: Float between 0.0 and 1.0 (1.0 = identical)
            - standardized_pose1: Standardized first pose
            - standardized_pose2: Standardized second pose

        Raises:
            ValueError: If poses have different shapes or invalid dimensions
        """
        if pose1.shape != pose2.shape:
            raise ValueError(f"Pose shapes must match: {pose1.shape} != {pose2.shape}")

        if pose1.ndim != 2 or pose1.shape[1] != 3:
            raise ValueError(f"Poses must be 2D arrays with shape (N, 3), got {pose1.shape}")

        # Perform Procrustes analysis
        # Returns: (mtx1, mtx2, disparity)
        # mtx1, mtx2: Standardized matrices
        # disparity: Sum of squared differences after alignment
        standardized_pose1, standardized_pose2, disparity = procrustes(
            pose1,
            pose2,
        )

        # Convert disparity to similarity score (1.0 = identical, 0.0 = very different)
        # Using a linear scale with steeper penalty for differences
        # disparity = 0 → score = 1.0 (100%)
        # disparity = 0.1 → score = 0.7 (70%)
        # disparity = 0.2 → score = 0.4 (40%)
        # disparity >= 0.33 → score = 0 (0%)
        similarity_score = max(0.0, 1.0 - disparity * 3.0)

        return similarity_score, standardized_pose1, standardized_pose2

    def compare_with_metrics(
        self,
        pose1: np.ndarray,
        pose2: np.ndarray,
        joint_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare two 3D poses with detailed metrics.

        Args:
            pose1: First pose array of shape (N, 3)
            pose2: Second pose array of shape (N, 3)
            joint_names: Optional list of joint names for labeling distances.
                Must have length N if provided.

        Returns:
            Dictionary containing:
            - similarity_score: Overall similarity (0.0-1.0)
            - disparity: Procrustes disparity value
            - joint_distances: Per-joint Euclidean distances
            - mean_distance: Mean of joint distances
            - max_distance: Maximum joint distance
            - max_distance_joint: Name/index of joint with maximum distance

        Raises:
            ValueError: If poses have different shapes or joint_names length mismatch
        """
        similarity_score, std_pose1, std_pose2 = self.compare(pose1, pose2)

        # Calculate per-joint Euclidean distances
        joint_distances = np.linalg.norm(std_pose1 - std_pose2, axis=1)

        # Find joint with maximum distance
        max_distance_idx = np.argmax(joint_distances)
        if joint_names is not None:
            # Use provided joint names and translate to Japanese
            max_distance_joint = translate_joint_name(joint_names[max_distance_idx])
        else:
            # Use index and translate to Japanese
            max_distance_joint = translate_joint_name(max_distance_idx)

        # Compute disparity (for reference)
        disparity = 1.0 / similarity_score - 1.0 if similarity_score > 0 else float("inf")

        return {
            "similarity_score": float(similarity_score),
            "disparity": float(disparity),
            "joint_distances": joint_distances.tolist(),
            "mean_distance": float(np.mean(joint_distances)),
            "max_distance": float(joint_distances[max_distance_idx]),
            "max_distance_joint": max_distance_joint,
        }
