#!/usr/bin/env python3
"""Test script to validate the application functionality."""

import json
from pathlib import Path

# Test data - sample landmarks for a standing pose
SAMPLE_LANDMARKS = [
    [0.5, 0.2, 0.0],  # nose
    [0.48, 0.18, 0.0],  # left_eye_inner
    [0.47, 0.18, 0.0],  # left_eye
    [0.46, 0.18, 0.0],  # left_eye_outer
    [0.52, 0.18, 0.0],  # right_eye_inner
    [0.53, 0.18, 0.0],  # right_eye
    [0.54, 0.18, 0.0],  # right_eye_outer
    [0.43, 0.20, 0.0],  # left_ear
    [0.57, 0.20, 0.0],  # right_ear
    [0.48, 0.23, 0.0],  # mouth_left
    [0.52, 0.23, 0.0],  # mouth_right
    [0.40, 0.35, 0.0],  # left_shoulder
    [0.60, 0.35, 0.0],  # right_shoulder
    [0.35, 0.50, 0.0],  # left_elbow
    [0.65, 0.50, 0.0],  # right_elbow
    [0.30, 0.65, 0.0],  # left_wrist
    [0.70, 0.65, 0.0],  # right_wrist
    [0.28, 0.68, 0.0],  # left_pinky
    [0.72, 0.68, 0.0],  # right_pinky
    [0.29, 0.67, 0.0],  # left_index
    [0.71, 0.67, 0.0],  # right_index
    [0.30, 0.66, 0.0],  # left_thumb
    [0.70, 0.66, 0.0],  # right_thumb
    [0.42, 0.55, 0.0],  # left_hip
    [0.58, 0.55, 0.0],  # right_hip
    [0.40, 0.75, 0.0],  # left_knee
    [0.60, 0.75, 0.0],  # right_knee
    [0.38, 0.95, 0.0],  # left_ankle
    [0.62, 0.95, 0.0],  # right_ankle
    [0.37, 0.98, 0.0],  # left_heel
    [0.63, 0.98, 0.0],  # right_heel
    [0.39, 0.99, 0.0],  # left_foot_index
    [0.61, 0.99, 0.0],  # right_foot_index
]


def test_pose_analyzer():
    """Test the PoseAnalyzer functionality."""
    from app.pose_analyzer import PoseAnalyzer

    analyzer = PoseAnalyzer()

    # Test joint angle calculation
    print("Testing joint angle calculation...")
    angles = analyzer.calculate_joint_angles(SAMPLE_LANDMARKS)
    print(f"Calculated angles: {angles}")

    assert "left_elbow" in angles
    assert "right_elbow" in angles
    assert "left_shoulder" in angles
    assert "right_shoulder" in angles
    assert "left_knee" in angles
    assert "right_knee" in angles

    for joint, angle in angles.items():
        assert 0 <= angle <= 180, f"Invalid angle for {joint}: {angle}"

    print("✓ Joint angle calculation works correctly\n")

    # Test Procrustes similarity
    print("Testing Procrustes similarity...")
    similarity = analyzer.calculate_similarity(SAMPLE_LANDMARKS, SAMPLE_LANDMARKS)
    print(f"Similarity (identical poses): {similarity}%")
    assert similarity > 99.0, f"Expected ~100% similarity, got {similarity}%"

    # Test with slightly different pose
    modified_landmarks = [lm.copy() for lm in SAMPLE_LANDMARKS]
    modified_landmarks[13][0] += 0.1  # Move left elbow slightly
    similarity2 = analyzer.calculate_similarity(SAMPLE_LANDMARKS, modified_landmarks)
    print(f"Similarity (slightly different): {similarity2}%")
    assert 80 < similarity2 < 100, f"Expected 80-100% similarity, got {similarity2}%"

    print("✓ Procrustes similarity calculation works correctly\n")


def test_reference_pose_storage():
    """Test reference pose storage."""
    print("Testing reference pose storage...")

    ref_dir = Path("reference_poses")
    ref_dir.mkdir(exist_ok=True)

    # Save a reference pose
    ref_name = "test_standing"
    ref_path = ref_dir / f"{ref_name}.json"

    with open(ref_path, "w") as f:
        json.dump({"name": ref_name, "landmarks": SAMPLE_LANDMARKS}, f, indent=2)

    # Read it back
    with open(ref_path, "r") as f:
        data = json.load(f)

    assert data["name"] == ref_name
    assert len(data["landmarks"]) == 33
    assert data["landmarks"] == SAMPLE_LANDMARKS

    print(f"✓ Reference pose '{ref_name}' saved and loaded successfully\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Pose Estimator Components")
    print("=" * 60 + "\n")

    try:
        test_pose_analyzer()
        test_reference_pose_storage()

        print("=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
