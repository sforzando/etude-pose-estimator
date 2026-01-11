"""Reference pose management.

This module provides functionality for storing and retrieving reference poses
as JSON files. Simple, database-free, human-readable, and easy to backup.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


class ReferencePoseManager:
    """Manager for reference pose storage and retrieval.

    Stores reference poses as JSON files in a designated directory.
    Each pose includes 2D keypoints, 3D coordinates, joint angles, and metadata.

    JSON file structure:
    {
        "name": "Specium Beam",
        "created_at": "2026-01-11T12:00:00Z",
        "pose_2d": [[x, y, conf], ...],
        "pose_3d": [[x, y, z], ...],
        "angles": {"left_elbow": 120.5, ...},
        "metadata": {"character": "Ultraman", ...}
    }

    Attributes:
        reference_dir: Directory for storing reference pose JSON files
    """

    def __init__(self, reference_dir: Path) -> None:
        """Initialize reference pose manager.

        Args:
            reference_dir: Directory for storing reference pose JSON files

        Note:
            Creates the directory if it doesn't exist
        """
        self.reference_dir = reference_dir
        self.reference_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        pose_2d: np.ndarray,
        pose_3d: np.ndarray,
        angles: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a reference pose.

        Args:
            name: Unique name for the reference pose
            pose_2d: 2D keypoints array (17, 3) [x, y, confidence]
            pose_3d: 3D keypoints array (17, 3) [x, y, z]
            angles: Joint angles dictionary
            metadata: Optional additional metadata

        Raises:
            ValueError: If a pose with the same name already exists
            ValueError: If pose arrays have invalid shapes
        """
        # Validate inputs
        if pose_2d.shape != (17, 3):
            raise ValueError(f"pose_2d must have shape (17, 3), got {pose_2d.shape}")

        if pose_3d.shape != (17, 3):
            raise ValueError(f"pose_3d must have shape (17, 3), got {pose_3d.shape}")

        # Check for existing pose with same name
        file_path = self._get_file_path(name)
        if file_path.exists():
            raise ValueError(f"Reference pose '{name}' already exists")

        # Prepare data
        data = {
            "name": name,
            "created_at": datetime.now(UTC).isoformat(),
            "pose_2d": pose_2d.tolist(),
            "pose_3d": pose_3d.tolist(),
            "angles": angles,
            "metadata": metadata or {},
        }

        # Save as JSON
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, name: str) -> dict[str, Any]:
        """Load a reference pose.

        Args:
            name: Name of the reference pose to load

        Returns:
            Dictionary containing:
            - name: Pose name
            - created_at: ISO 8601 timestamp
            - pose_2d: NumPy array (17, 3)
            - pose_3d: NumPy array (17, 3)
            - angles: Joint angles dictionary
            - metadata: Additional metadata

        Raises:
            FileNotFoundError: If reference pose does not exist
        """
        file_path = self._get_file_path(name)
        if not file_path.exists():
            raise FileNotFoundError(f"Reference pose '{name}' not found")

        # Load from JSON
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        # Convert lists back to NumPy arrays
        data["pose_2d"] = np.array(data["pose_2d"], dtype=np.float32)
        data["pose_3d"] = np.array(data["pose_3d"], dtype=np.float32)

        return data

    def list_references(self) -> list[dict[str, Any]]:
        """List all reference poses.

        Returns:
            List of dictionaries, each containing:
            - name: Pose name
            - created_at: ISO 8601 timestamp
            - metadata: Additional metadata

        Note:
            Does not load full pose data for performance
        """
        references = []

        for file_path in self.reference_dir.glob("*.json"):
            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)

                references.append(
                    {
                        "name": data["name"],
                        "created_at": data["created_at"],
                        "metadata": data.get("metadata", {}),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                # Skip invalid files
                continue

        # Sort by creation time (newest first)
        references.sort(
            key=lambda x: x["created_at"],
            reverse=True,
        )

        return references

    def delete(self, name: str) -> None:
        """Delete a reference pose.

        Args:
            name: Name of the reference pose to delete

        Raises:
            FileNotFoundError: If reference pose does not exist
        """
        file_path = self._get_file_path(name)
        if not file_path.exists():
            raise FileNotFoundError(f"Reference pose '{name}' not found")

        file_path.unlink()

    def exists(self, name: str) -> bool:
        """Check if a reference pose exists.

        Args:
            name: Name of the reference pose

        Returns:
            True if the reference pose exists, False otherwise
        """
        return self._get_file_path(name).exists()

    def _get_file_path(self, name: str) -> Path:
        """Get file path for a reference pose.

        Args:
            name: Name of the reference pose

        Returns:
            Path to the JSON file
        """
        # Sanitize name for filesystem safety
        safe_name = "".join(c if c.isalnum() or c in [" ", "-", "_"] else "_" for c in name)
        return self.reference_dir / f"{safe_name}.json"
