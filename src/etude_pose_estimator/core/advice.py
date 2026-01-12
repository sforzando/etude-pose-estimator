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

        # Build clearer prompt with explicit formatting requirements
        prompt = f"""{lang_instruction}

あなたはヒーローショー（ウルトラマンなど）の着ぐるみアクター向けのポーズコーチです。
3D姿勢推定データに基づいて、決めポーズの品質を高めるための具体的で実用的なアドバイスを提供してください。

【分析データ】
- 類似度スコア: {similarity_score * 100:.1f}%
- 関節角度のずれ（上位5つ）:
{angles_text}

【評価基準】
- 角度のずれが5度未満: ほぼ完璧（優秀）
- 角度のずれが5〜15度: 要改善
- 角度のずれが15度以上: 要注意（大幅に修正が必要）

【重要な指針】
1. 観客から見た「見栄え」と「力強さ」を重視
2. 着ぐるみの視界制限や可動域制限を考慮
3. 左右対称性が求められるかどうかを見極める
4. 具体的な改善方法を数値（角度）で示す

【出力フォーマット】
以下の形式で、Markdownを使わずに平文で出力してください：

1. 総合評価
[類似度スコアと全体的なポーズの完成度について1-2文で評価。励ましの言葉を含む]

2. 改善ポイント
- [最も重要な改善点。「〜を〜度ほど〜」のように具体的に]
- [2番目に重要な改善点]
- [3番目に重要な改善点]

3. 重点部位
[最も注意すべき関節名を2-3個、カンマ区切りで]

【例】
1. 総合評価
類似度85%と良好なポーズです。あと一歩で完璧な決めポーズになります！

2. 改善ポイント
- 左肘をあと12度ほど上げると、より力強く見栄えのある印象になります
- 右膝の角度をあと8度ほど深くして、安定感のある着地姿勢を目指しましょう
- 左肩をあと5度ほど後ろに引いて、胸を張った堂々とした姿勢に

3. 重点部位
左肘、右膝、左肩
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

        # Provide Japanese fallback messages
        default_overall = (
            "ポーズの分析が完了しました。"
            if language == "ja"
            else "Pose analysis complete."
        )
        default_improvements = (
            ["練習を続けて、フォームをさらに改善していきましょう。"]
            if language == "ja"
            else ["Continue practicing for better form."]
        )
        default_priority = (
            "全体の姿勢に注目してください。" if language == "ja" else "Focus on overall posture."
        )

        return {
            "overall": overall.strip() or default_overall,
            "improvements": improvements[:3] if improvements else default_improvements,
            "priority_joints": priority_joints.strip() or default_priority,
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
