"""Pose visualization utilities.

This module provides functionality for drawing pose keypoints and skeletons
on images for visualization purposes.
"""

from pathlib import Path

import cv2
import numpy as np


class PoseVisualizer:
    """Visualizer for pose keypoints and skeletons.

    Draws COCO 17-keypoint pose skeletons on images for debugging and
    visual feedback.

    COCO 17 keypoints:
        0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
        5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
        9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
        13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    """

    # COCO skeleton definition: pairs of keypoint indices to connect
    SKELETON = [
        # Head
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        # Torso
        (5, 6),
        (5, 11),
        (6, 12),
        (11, 12),
        # Left arm
        (5, 7),
        (7, 9),
        # Right arm
        (6, 8),
        (8, 10),
        # Left leg
        (11, 13),
        (13, 15),
        # Right leg
        (12, 14),
        (14, 16),
    ]

    # Color scheme (BGR format for OpenCV)
    KEYPOINT_COLOR = (0, 255, 0)  # Green
    SKELETON_COLOR = (255, 0, 0)  # Blue
    BBOX_COLOR = (255, 0, 0)  # Blue
    PERSON_COLOR = (0, 255, 0)  # Green
    LOW_CONFIDENCE_COLOR = (128, 128, 128)  # Gray

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        keypoint_radius: int = 5,
        line_thickness: int = 2,
    ) -> None:
        """Initialize pose visualizer.

        Args:
            confidence_threshold: Minimum confidence to draw keypoints/lines
            keypoint_radius: Radius of keypoint circles
            line_thickness: Thickness of skeleton lines
        """
        self.confidence_threshold = confidence_threshold
        self.keypoint_radius = keypoint_radius
        self.line_thickness = line_thickness

    def draw_pose(
        self,
        image_path: Path,
        keypoints_2d: np.ndarray,
        output_path: Path | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        confidence: float | None = None,
        show_bbox: bool = True,
        show_keypoint_numbers: bool = False,
    ) -> np.ndarray:
        """Draw pose keypoints and skeleton on image.

        Args:
            image_path: Path to input image
            keypoints_2d: 2D keypoints array of shape (17, 3) [x, y, confidence]
            output_path: Optional path to save output image
            bbox: Optional bounding box (x_min, y_min, x_max, y_max) in normalized coords
            confidence: Optional detection confidence (0-1)
            show_bbox: Whether to draw bounding box and confidence
            show_keypoint_numbers: Whether to show keypoint numbers

        Returns:
            Image array with pose drawn (BGR format)

        Raises:
            ValueError: If keypoints have invalid shape
        """
        if keypoints_2d.shape != (17, 3):
            raise ValueError(f"Expected keypoints shape (17, 3), got {keypoints_2d.shape}")

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        height, width = image.shape[:2]

        # Draw bounding box if provided
        if show_bbox and bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            # Convert normalized coordinates to pixel coordinates
            x1 = int(x_min * width)
            y1 = int(y_min * height)
            x2 = int(x_max * width)
            y2 = int(y_max * height)

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), self.BBOX_COLOR, 2)

            # Draw label with confidence
            if confidence is not None:
                label = f"person {confidence:.2f}"
                label_size, _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                label_y = max(y1, label_size[1] + 10)

                # Draw label background
                cv2.rectangle(
                    image,
                    (x1, label_y - label_size[1] - 10),
                    (x1 + label_size[0] + 10, label_y),
                    self.BBOX_COLOR,
                    -1,
                )

                # Draw label text
                cv2.putText(
                    image,
                    label,
                    (x1 + 5, label_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        # Draw skeleton lines
        for pt1_idx, pt2_idx in self.SKELETON:
            pt1 = keypoints_2d[pt1_idx]
            pt2 = keypoints_2d[pt2_idx]

            # Check confidence for both points
            if pt1[2] >= self.confidence_threshold and pt2[2] >= self.confidence_threshold:
                # Convert normalized coordinates to pixel coordinates
                x1, y1 = int(pt1[0] * width), int(pt1[1] * height)
                x2, y2 = int(pt2[0] * width), int(pt2[1] * height)

                # Draw line
                cv2.line(
                    image,
                    (x1, y1),
                    (x2, y2),
                    self.SKELETON_COLOR,
                    self.line_thickness,
                )

        # Draw keypoints (on top of skeleton)
        for i, (x, y, conf) in enumerate(keypoints_2d):
            if conf >= self.confidence_threshold:
                # Convert normalized coordinates to pixel coordinates
                px, py = int(x * width), int(y * height)

                # Draw circle
                cv2.circle(
                    image,
                    (px, py),
                    self.keypoint_radius,
                    self.KEYPOINT_COLOR,
                    -1,  # Filled circle
                )

                # Draw keypoint number (for debugging)
                if show_keypoint_numbers:
                    cv2.putText(
                        image,
                        str(i),
                        (px + 5, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 255, 255),
                        1,
                    )

        # Save if output path specified
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), image)

        return image

    def draw_comparison(
        self,
        image1_path: Path,
        keypoints1: np.ndarray,
        image2_path: Path,
        keypoints2: np.ndarray,
        output_path: Path | None = None,
    ) -> np.ndarray:
        """Draw side-by-side comparison of two poses.

        Args:
            image1_path: Path to first image
            keypoints1: Keypoints for first image (17, 3)
            image2_path: Path to second image
            keypoints2: Keypoints for second image (17, 3)
            output_path: Optional path to save output image

        Returns:
            Combined image array with both poses drawn (BGR format)
        """
        # Draw poses on individual images
        img1 = self.draw_pose(image1_path, keypoints1)
        img2 = self.draw_pose(image2_path, keypoints2)

        # Resize images to same height
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        target_height = max(h1, h2)

        if h1 != target_height:
            scale = target_height / h1
            img1 = cv2.resize(img1, (int(w1 * scale), target_height))

        if h2 != target_height:
            scale = target_height / h2
            img2 = cv2.resize(img2, (int(w2 * scale), target_height))

        # Concatenate horizontally
        combined = np.hstack([img1, img2])

        # Save if output path specified
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), combined)

        return combined
