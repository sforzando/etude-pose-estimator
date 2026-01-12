"""Joint name translations and mappings.

This module provides centralized joint name translations for UI display.
"""

from typing import Final

import numpy as np

# Joint name translations (English -> Japanese)
JOINT_NAMES_JA: Final[dict[str, str]] = {
    # Angle-based joint names (from angle.py)
    "left_elbow": "左肘",
    "right_elbow": "右肘",
    "left_shoulder": "左肩",
    "right_shoulder": "右肩",
    "left_knee": "左膝",
    "right_knee": "右膝",
    "left_hip": "左股関節",
    "right_hip": "右股関節",
    # Position-based joint names (from H36M/COCO indices)
    "0": "骨盤中心",
    "1": "右股関節",
    "2": "右膝",
    "3": "右足首",
    "4": "左股関節",
    "5": "左膝",
    "6": "左足首",
    "7": "胴体",
    "8": "首",
    "9": "頭頂",
    "10": "頭部",
    "11": "左肩",
    "12": "左肘",
    "13": "左手首",
    "14": "右肩",
    "15": "右肘",
    "16": "右手首",
}

# Joint index to name mapping (H36M format)
JOINT_INDEX_NAMES: Final[dict[int, str]] = {
    0: "骨盤中心",
    1: "右股関節",
    2: "右膝",
    3: "右足首",
    4: "左股関節",
    5: "左膝",
    6: "左足首",
    7: "胴体",
    8: "首",
    9: "頭頂",
    10: "頭部",
    11: "左肩",
    12: "左肘",
    13: "左手首",
    14: "右肩",
    15: "右肘",
    16: "右手首",
}


def translate_joint_name(name: str | int) -> str:
    """Translate joint name to Japanese.

    Args:
        name: English joint name (str) or joint index (int)

    Returns:
        Japanese joint name, or original name if translation not found
    """
    # Handle numpy integer types and Python integers
    if isinstance(name, (int, np.integer)):
        return JOINT_INDEX_NAMES.get(int(name), f"関節{name}")

    # Must be string from here
    # Try direct lookup first
    if name in JOINT_NAMES_JA:
        return JOINT_NAMES_JA[name]

    # Try as string index
    if name.isdigit():
        idx = int(name)
        return JOINT_INDEX_NAMES.get(idx, f"関節{name}")

    # Return original if not found
    return name
