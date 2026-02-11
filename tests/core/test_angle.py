"""Tests for joint angle calculation module."""

import numpy as np
import pytest

from etude_pose_estimator.core.angle import JointAngleCalculator


class TestJointAngleCalculator:
    """Test joint angle calculation functionality."""

    @pytest.fixture
    def calculator(self) -> JointAngleCalculator:
        """Create a JointAngleCalculator instance."""
        return JointAngleCalculator()

    @pytest.fixture
    def sample_pose_3d(self) -> np.ndarray:
        """Create a sample 3D pose with known angles."""
        # Create a simple T-pose for predictable angles
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
            [0.0, 0.5, 0.0],  # 9: head_top (left_wrist for angles)
            [0.0, 0.45, 0.0],  # 10: head (right_wrist for angles)
            [-0.2, 0.3, 0.0],  # 11: left_hip for shoulder angle
            [-0.3, 0.3, 0.0],  # 12: left_knee for shoulder angle
            [-0.4, 0.3, 0.0],  # 13: left_ankle for shoulder angle
            [0.2, 0.3, 0.0],  # 14: right_hip for shoulder angle
            [0.3, 0.3, 0.0],  # 15: right_knee for shoulder angle
            [0.4, 0.3, 0.0],  # 16: right_ankle for shoulder angle
        ])

    def test_calculate_angles_returns_dict(
        self,
        calculator: JointAngleCalculator,
        sample_pose_3d: np.ndarray,
    ) -> None:
        """Test that calculate_angles returns a dictionary."""
        angles = calculator.calculate_angles(sample_pose_3d)

        assert isinstance(angles, dict)
        assert len(angles) == 8  # 8 joint angles defined

    def test_calculate_angles_has_expected_joints(
        self,
        calculator: JointAngleCalculator,
        sample_pose_3d: np.ndarray,
    ) -> None:
        """Test that all expected joint angles are calculated."""
        angles = calculator.calculate_angles(sample_pose_3d)

        expected_joints = [
            "left_elbow",
            "right_elbow",
            "left_shoulder",
            "right_shoulder",
            "left_knee",
            "right_knee",
            "left_hip",
            "right_hip",
        ]

        for joint in expected_joints:
            assert joint in angles

    def test_angle_values_in_valid_range(
        self,
        calculator: JointAngleCalculator,
        sample_pose_3d: np.ndarray,
    ) -> None:
        """Test that angle values are in the valid range [0, 180]."""
        angles = calculator.calculate_angles(sample_pose_3d)

        for joint_name, angle in angles.items():
            assert 0.0 <= angle <= 180.0, f"{joint_name} angle {angle} out of range"

    def test_straight_line_angle(self, calculator: JointAngleCalculator) -> None:
        """Test angle calculation for straight line (180 degrees)."""
        # Three points in a straight line
        point1 = np.array([0.0, 0.0, 0.0])
        point2 = np.array([1.0, 0.0, 0.0])
        point3 = np.array([2.0, 0.0, 0.0])

        angle = calculator._calculate_angle(point1, point2, point3)
        assert angle == pytest.approx(180.0, abs=1e-6)

    def test_right_angle(self, calculator: JointAngleCalculator) -> None:
        """Test angle calculation for right angle (90 degrees)."""
        # Three points forming a right angle
        point1 = np.array([0.0, 0.0, 0.0])
        point2 = np.array([1.0, 0.0, 0.0])
        point3 = np.array([1.0, 1.0, 0.0])

        angle = calculator._calculate_angle(point1, point2, point3)
        assert angle == pytest.approx(90.0, abs=1e-6)

    def test_compare_angles_returns_japanese_names(
        self,
        calculator: JointAngleCalculator,
    ) -> None:
        """Test that compare_angles returns Japanese joint names."""
        angles1 = {
            "left_elbow": 90.0,
            "right_elbow": 85.0,
            "left_knee": 120.0,
        }
        angles2 = {
            "left_elbow": 95.0,
            "right_elbow": 80.0,
            "left_knee": 125.0,
        }

        differences = calculator.compare_angles(angles1, angles2)

        # Check that Japanese names are used
        assert "左肘" in differences
        assert "右肘" in differences
        assert "左膝" in differences

        # Check that English names are NOT used
        assert "left_elbow" not in differences
        assert "right_elbow" not in differences

    def test_compare_angles_calculates_differences(
        self,
        calculator: JointAngleCalculator,
    ) -> None:
        """Test that angle differences are calculated correctly."""
        angles1 = {
            "left_elbow": 90.0,
            "right_elbow": 85.0,
        }
        angles2 = {
            "left_elbow": 95.0,
            "right_elbow": 80.0,
        }

        differences = calculator.compare_angles(angles1, angles2)

        assert differences["左肘"] == pytest.approx(5.0)
        assert differences["右肘"] == pytest.approx(5.0)

    def test_compare_angles_mismatched_keys_raises_error(
        self,
        calculator: JointAngleCalculator,
    ) -> None:
        """Test that mismatched angle dictionaries raise ValueError."""
        angles1 = {"left_elbow": 90.0}
        angles2 = {"right_elbow": 85.0}

        with pytest.raises(ValueError, match="same joint names"):
            calculator.compare_angles(angles1, angles2)

    def test_invalid_pose_shape_raises_error(
        self,
        calculator: JointAngleCalculator,
    ) -> None:
        """Test that invalid pose shape raises ValueError."""
        invalid_pose = np.random.randn(10, 3)  # Should be (17, 3)

        with pytest.raises(ValueError, match="Expected pose shape"):
            calculator.calculate_angles(invalid_pose)
