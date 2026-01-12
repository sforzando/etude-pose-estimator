"""Pose improvement advice generation using Gemini 3 Flash.

This module provides advice generation functionality using Google's Gemini 3 Flash,
offering context-aware, bilingual (Japanese/English) improvement suggestions.
"""

from typing import Any

from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel, Field


class AdviceResponse(BaseModel):
    """Structured response model for pose improvement advice."""

    overall: str = Field(
        description="Overall assessment of the pose (1-2 sentences)",
    )
    improvements: list[str] = Field(
        description="List of 3 specific improvement points with measured angle data",
        min_length=1,
        max_length=3,
    )
    priority_joints: str = Field(
        description="2-3 joint names that need most attention, comma-separated",
    )


class GeminiAdviceGenerator:
    """Pose improvement advice generator using Gemini 3 Flash.

    Generates natural language advice based on pose similarity scores,
    angle differences, and joint position differences. Supports both
    Japanese (primary) and English (secondary) output.

    Attributes:
        client: Gemini API client
        model_name: Gemini model identifier
    """

    def __init__(self, api_key: str) -> None:
        """Initialize Gemini advice generator.

        Args:
            api_key: Gemini API key

        Raises:
            ValueError: If API key is empty
        """
        if not api_key:
            raise ValueError("Gemini API key cannot be empty")

        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-3-flash-preview"

    def generate_advice(
        self,
        similarity_score: float,
        angle_differences: dict[str, float],
        joint_distances: dict[str, float],
        language: str = "ja",
    ) -> dict[str, Any]:
        """Generate improvement advice for a pose.

        Args:
            similarity_score: Overall similarity score (0.0-1.0)
            angle_differences: Dictionary of joint name to angle difference (degrees)
            joint_distances: Dictionary of joint name to position difference
            language: Target language ('ja' for Japanese, 'en' for English)

        Returns:
            Dictionary containing:
            - overall: Overall assessment (1-2 sentences)
            - improvements: List of top 3 specific improvements (prioritized)
            - priority_joints: Priority joints that need most attention

        Raises:
            ValueError: If language is not supported
            RuntimeError: If API call fails
        """
        if language not in ["ja", "en"]:
            raise ValueError(f"Unsupported language: {language}")

        prompt = self._build_prompt(
            similarity_score,
            angle_differences,
            joint_distances,
            language,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                    response_schema=AdviceResponse,
                ),
            )

            # Parse JSON response into Pydantic model
            advice_model = AdviceResponse.model_validate_json(response.text)

            return {
                "overall": advice_model.overall,
                "improvements": advice_model.improvements,
                "priority_joints": advice_model.priority_joints,
            }

        except Exception as e:
            raise RuntimeError(f"Failed to generate advice: {e}") from e

    def _build_prompt(
        self,
        similarity_score: float,
        angle_differences: dict[str, float],
        joint_distances: dict[str, float],
        language: str,
    ) -> str:
        """Build structured prompt for Gemini.

        Args:
            similarity_score: Overall similarity score (0.0-1.0)
            angle_differences: Joint angle differences
            joint_distances: Joint position differences
            language: Target language

        Returns:
            Formatted prompt string
        """
        # Sort by magnitude for priority
        sorted_angles = sorted(
            angle_differences.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        lang_instruction = "Respond in Japanese." if language == "ja" else "Respond in English."

        # Map joint names from angle differences to more descriptive names
        joint_names_ja = {
            "Left Elbow": "左肘",
            "Right Elbow": "右肘",
            "Left Shoulder": "左肩",
            "Right Shoulder": "右肩",
            "Left Knee": "左膝",
            "Right Knee": "右膝",
            "Left Hip": "左股関節",
            "Right Hip": "右股関節",
        }

        # Format top angles with Japanese names for better context
        formatted_angles = []
        for joint, angle in sorted_angles[:5]:
            ja_name = joint_names_ja.get(joint, joint)
            formatted_angles.append(f"- {ja_name} ({joint}): {angle:.1f}度のずれ")
        angles_text = "\n".join(formatted_angles)

        # Build data-driven prompt for structured JSON output
        prompt = f"""{lang_instruction}

あなたはヒーローショーのアクター向けポーズコーチです。
3D姿勢推定で測定された関節角度データに基づき、基準ポーズと比較した具体的な改善アドバイスを提供してください。

【測定データ】
類似度スコア: {similarity_score * 100:.1f}%

関節角度のずれ（測定値）:
{angles_text}

【必須ルール】
1. 改善ポイント(improvements)では、上記の測定データから実際の角度差分を**必ず引用**すること
2. 「〜度ほど」ではなく、「測定値では○○度のずれ」のように具体的に記載
3. ずれが大きい順（15度以上→5〜15度→5度未満）に優先順位をつける
4. 各改善ポイントは簡潔な1文で完結させること

【レスポンススキーマ】
- overall: 類似度{similarity_score * 100:.1f}%を踏まえた1-2文の総合評価
- improvements: 測定データに基づく具体的な改善点3つ（配列）
- priority_joints: ずれが大きい関節名を2-3個、カンマ区切り（文字列）

【良い改善ポイントの例】
"左肩は測定値で18.5度のずれがあり、後方に引いて胸を張った姿勢にすると力強さが増します"

【悪い改善ポイントの例】
"左肩をもっと開いて胸を張りましょう"（測定値なし、曖昧）
"""
        return prompt

    def _format_dict(self, data: dict[str, float]) -> str:
        """Format dictionary for prompt.

        Args:
            data: Dictionary to format

        Returns:
            Formatted string
        """
        return "\n".join([f"- {k}: {v:.2f}" for k, v in data.items()])

    def generate_bilingual_advice(
        self,
        similarity_score: float,
        angle_differences: dict[str, float],
        joint_distances: dict[str, float],
    ) -> dict[str, dict[str, Any]]:
        """Generate both Japanese and English advice.

        Args:
            similarity_score: Overall similarity score (0.0-1.0)
            angle_differences: Joint angle differences
            joint_distances: Joint position differences

        Returns:
            Dictionary with 'ja' and 'en' keys, each containing advice dict
        """
        ja_advice = self.generate_advice(
            similarity_score,
            angle_differences,
            joint_distances,
            language="ja",
        )
        en_advice = self.generate_advice(
            similarity_score,
            angle_differences,
            joint_distances,
            language="en",
        )

        return {
            "ja": ja_advice,
            "en": en_advice,
        }
