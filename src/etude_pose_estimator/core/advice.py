"""Pose improvement advice generation using Gemini 3 Flash.

This module provides advice generation functionality using Google's Gemini 3 Flash,
offering context-aware, bilingual (Japanese/English) improvement suggestions.
"""

from typing import Any

from google import genai
from google.genai.types import GenerateContentConfig


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
        self.model_name = "gemini-2.0-flash-exp"

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
                ),
            )

            advice = self._parse_response(response.text, language)
            return advice

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
        sorted_distances = sorted(
            joint_distances.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        lang_instruction = "Respond in Japanese." if language == "ja" else "Respond in English."

        prompt = f"""{lang_instruction}

You are a professional pose coach for hero shows (like Ultraman).

Similarity Score: {similarity_score * 100:.1f}%

Angle Differences (degrees):
{self._format_dict(dict(sorted_angles[:5]))}

Joint Position Differences:
{self._format_dict(dict(sorted_distances[:5]))}

Provide improvement advice in the following format:

1. Overall assessment (1-2 sentences about the pose quality)
2. Top 3 specific improvements (ordered by priority, be specific about body parts and movements)
3. Priority joints (which joints need the most attention)

Keep the advice concise, actionable, and encouraging.
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

    def _parse_response(self, response_text: str, language: str) -> dict[str, Any]:
        """Parse Gemini response into structured advice.

        Args:
            response_text: Raw response from Gemini
            language: Language of the response

        Returns:
            Parsed advice dictionary
        """
        # Simple parsing - split by numbered sections
        lines = response_text.strip().split("\n")

        overall = ""
        improvements = []
        priority_joints = ""

        current_section = None
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "1." in line or "overall" in line.lower() or "総合" in line:
                current_section = "overall"
                # Extract content after the marker
                content = line.split(".", 1)[-1].strip()
                if content:
                    overall += content + " "
            elif "2." in line or "improvement" in line.lower() or "改善" in line:
                current_section = "improvements"
            elif "3." in line or "priority" in line.lower() or "優先" in line:
                current_section = "priority"
            elif current_section == "overall":
                overall += line + " "
            elif current_section == "improvements":
                if line.startswith("-") or line.startswith("*") or line[0].isdigit():
                    improvements.append(line.lstrip("-*0123456789. "))
            elif current_section == "priority":
                priority_joints += line + " "

        return {
            "overall": overall.strip() or "Pose analysis complete.",
            "improvements": improvements[:3]
            if improvements
            else ["Continue practicing for better form."],
            "priority_joints": priority_joints.strip() or "Focus on overall posture.",
        }

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
