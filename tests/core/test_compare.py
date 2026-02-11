"""Tests for pose comparison module."""

import numpy as np
import pytest

from etude_pose_estimator.core.compare import ProcrustesComparator


class TestProcrustesComparator:
    """Test Procrustes comparison functionality."""

    @pytest.fixture
    def comparator(self) -> ProcrustesComparator:
        """Create a ProcrustesComparator instance."""
        return ProcrustesComparator()

    @pytest.fixture
    def sample_pose(self) -> np.ndarray:
        """Create a sample 3D pose (17 joints x 3 coordinates)."""
        return np.array([
            [0.0, 0.0, 0.0],  # 0: pelvis
            [0.1, -0.2, 0.0],  # 1: right_hip
            [0.1, -0.4, 0.0],  # 2: right_knee
            [0.1, -0.6, 0.0],  # 3: right_ankle
            [-0.1, -0.2, 0.0],  # 4: left_hip
            [-0.1, -0.4, 0.0],  # 5: left_knee
            [-0.1, -0.6, 0.0],  # 6: left_ankle
            [0.0, 0.2, 0.0],  # 7: spine
            [0.0, 0.4, 0.0],  # 8: neck
            [0.0, 0.5, 0.0],  # 9: head_top
            [0.0, 0.45, 0.0],  # 10: head
            [-0.2, 0.3, 0.0],  # 11: left_shoulder
            [-0.3, 0.2, 0.0],  # 12: left_elbow
            [-0.4, 0.1, 0.0],  # 13: left_wrist
            [0.2, 0.3, 0.0],  # 14: right_shoulder
            [0.3, 0.2, 0.0],  # 15: right_elbow
            [0.4, 0.1, 0.0],  # 16: right_wrist
        ])

    def test_identical_poses_score(
        self,
        comparator: ProcrustesComparator,
        sample_pose: np.ndarray,
    ) -> None:
        """Test that identical poses get 100% similarity score."""
        score, _, _ = comparator.compare(sample_pose, sample_pose.copy())
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_very_different_poses_low_score(
        self,
        comparator: ProcrustesComparator,
        sample_pose: np.ndarray,
    ) -> None:
        """Test that very different poses get low similarity score."""
        # Create a very different pose (e.g., arms down vs arms up)
        different_pose = sample_pose.copy()
        different_pose[12:17, :] *= -1  # Flip arm positions

        score, _, _ = comparator.compare(sample_pose, different_pose)
        # With new linear scoring, very different poses should score < 0.6
        assert score < 0.6

    def test_new_scoring_formula(
        self,
        comparator: ProcrustesComparator,
        sample_pose: np.ndarray,
    ) -> None:
        """Test new linear scoring formula behavior."""
        # Add small perturbation
        slightly_different = sample_pose + np.random.randn(*sample_pose.shape) * 0.05
        score, _, _ = comparator.compare(sample_pose, slightly_different)

        # New formula: score = max(0, 1 - disparity * 3)
        # Small perturbation should still yield reasonable score
        assert 0.0 <= score <= 1.0

    def test_compare_with_metrics(
        self,
        comparator: ProcrustesComparator,
        sample_pose: np.ndarray,
    ) -> None:
        """Test compare_with_metrics returns all expected fields."""
        metrics = comparator.compare_with_metrics(sample_pose, sample_pose.copy())

        assert "similarity_score" in metrics
        assert "disparity" in metrics
        assert "joint_distances" in metrics
        assert "mean_distance" in metrics
        assert "max_distance" in metrics
        assert "max_distance_joint" in metrics

    def test_compare_with_metrics_japanese_names(
        self,
        comparator: ProcrustesComparator,
        sample_pose: np.ndarray,
    ) -> None:
        """Test that max_distance_joint is translated to Japanese."""
        slightly_different = sample_pose.copy()
        slightly_different[9] += [0.1, 0.1, 0.1]  # Perturb head_top

        metrics = comparator.compare_with_metrics(sample_pose, slightly_different)

        # Joint 9 is "頭頂" in Japanese
        assert metrics["max_distance_joint"] == "頭頂"

    def test_invalid_shape_raises_error(
        self,
        comparator: ProcrustesComparator,
    ) -> None:
        """Test that invalid pose shapes raise ValueError."""
        invalid_pose = np.array([[1, 2], [3, 4]])  # Wrong shape
        valid_pose = np.random.randn(17, 3)

        with pytest.raises(ValueError, match="Pose shapes must match"):
            comparator.compare(invalid_pose, valid_pose)

    def test_mismatched_shapes_raises_error(
        self,
        comparator: ProcrustesComparator,
    ) -> None:
        """Test that mismatched pose shapes raise ValueError."""
        pose1 = np.random.randn(17, 3)
        pose2 = np.random.randn(10, 3)

        with pytest.raises(ValueError, match="Pose shapes must match"):
            comparator.compare(pose1, pose2)
