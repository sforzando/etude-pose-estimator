"""Tests for joint name translation module."""

from etude_pose_estimator.core.joint_names import (
    JOINT_INDEX_NAMES,
    JOINT_NAMES_JA,
    translate_joint_name,
)


class TestJointNamesTranslation:
    """Test joint name translation functionality."""

    def test_translate_english_joint_name(self) -> None:
        """Test translating English joint names to Japanese."""
        assert translate_joint_name("left_elbow") == "左肘"
        assert translate_joint_name("right_shoulder") == "右肩"
        assert translate_joint_name("left_knee") == "左膝"
        assert translate_joint_name("right_hip") == "右股関節"

    def test_translate_joint_index(self) -> None:
        """Test translating joint indices to Japanese names."""
        assert translate_joint_name(0) == "骨盤中心"
        assert translate_joint_name(9) == "頭頂"
        assert translate_joint_name(11) == "左肩"
        assert translate_joint_name(16) == "右手首"

    def test_translate_string_index(self) -> None:
        """Test translating string indices to Japanese names."""
        assert translate_joint_name("0") == "骨盤中心"
        assert translate_joint_name("9") == "頭頂"
        assert translate_joint_name("11") == "左肩"

    def test_translate_unknown_name(self) -> None:
        """Test translating unknown joint names returns original."""
        assert translate_joint_name("unknown_joint") == "unknown_joint"
        assert translate_joint_name("xyz") == "xyz"

    def test_translate_unknown_index(self) -> None:
        """Test translating unknown index returns formatted string."""
        assert translate_joint_name(99) == "関節99"
        assert translate_joint_name("99") == "関節99"

    def test_joint_names_ja_coverage(self) -> None:
        """Test that JOINT_NAMES_JA contains expected entries."""
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
            assert joint in JOINT_NAMES_JA

    def test_joint_index_names_coverage(self) -> None:
        """Test that JOINT_INDEX_NAMES contains all 17 H36M joints."""
        assert len(JOINT_INDEX_NAMES) == 17
        for i in range(17):
            assert i in JOINT_INDEX_NAMES
            assert isinstance(JOINT_INDEX_NAMES[i], str)
            assert len(JOINT_INDEX_NAMES[i]) > 0
