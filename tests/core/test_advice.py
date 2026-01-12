"""Tests for advice generation module."""

from unittest.mock import Mock, patch

import pytest

from etude_pose_estimator.core.advice import GeminiAdviceGenerator


class TestGeminiAdviceGenerator:
    """Test Gemini advice generation functionality."""

    @pytest.fixture
    def generator(self) -> GeminiAdviceGenerator:
        """Create a GeminiAdviceGenerator instance with mock API key."""
        return GeminiAdviceGenerator(api_key="test_api_key")

    def test_initialization_with_api_key(self) -> None:
        """Test that generator initializes with valid API key."""
        generator = GeminiAdviceGenerator(api_key="test_api_key")
        assert generator.model_name == "gemini-3-flash-preview"

    def test_initialization_without_api_key_raises_error(self) -> None:
        """Test that empty API key raises ValueError."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            GeminiAdviceGenerator(api_key="")

    def test_build_prompt_includes_similarity_score(
        self,
        generator: GeminiAdviceGenerator,
    ) -> None:
        """Test that prompt includes similarity score."""
        angle_diffs = {"左肘": 10.0, "右肘": 5.0}
        joint_distances = {"joint_0": 0.1, "joint_9": 0.05}

        prompt = generator._build_prompt(
            similarity_score=0.85,
            angle_differences=angle_diffs,
            joint_distances=joint_distances,
            language="ja",
        )

        assert "85.0%" in prompt
        assert "左肘" in prompt
        assert "10.0度のずれ" in prompt

    def test_build_prompt_japanese_language(
        self,
        generator: GeminiAdviceGenerator,
    ) -> None:
        """Test that Japanese prompt is generated correctly."""
        angle_diffs = {"左肘": 10.0}
        joint_distances = {}

        prompt = generator._build_prompt(
            similarity_score=0.85,
            angle_differences=angle_diffs,
            joint_distances=joint_distances,
            language="ja",
        )

        assert "Respond in Japanese" in prompt
        assert "ヒーローショー" in prompt
        assert "ポーズコーチ" in prompt

    def test_build_prompt_english_language(
        self,
        generator: GeminiAdviceGenerator,
    ) -> None:
        """Test that English prompt is generated correctly."""
        angle_diffs = {"left_elbow": 10.0}
        joint_distances = {}

        prompt = generator._build_prompt(
            similarity_score=0.85,
            angle_differences=angle_diffs,
            joint_distances=joint_distances,
            language="en",
        )

        assert "Respond in English" in prompt

    def test_parse_response_extracts_sections(
        self,
        generator: GeminiAdviceGenerator,
    ) -> None:
        """Test that response parsing extracts all sections."""
        response_text = """
1. 総合評価
類似度85%と良好なポーズです。あと一歩で完璧です。

2. 改善ポイント
- 左肘をあと12度ほど上げましょう
- 右膝の角度を深くしてください
- 左肩を後ろに引いてください

3. 重点部位
左肘、右膝、左肩
"""
        advice = generator._parse_response(response_text, language="ja")

        assert "overall" in advice
        assert "improvements" in advice
        assert "priority_joints" in advice
        assert "類似度85%" in advice["overall"]
        assert len(advice["improvements"]) == 3
        assert "左肘" in advice["priority_joints"]

    def test_parse_response_fallback_messages_japanese(
        self,
        generator: GeminiAdviceGenerator,
    ) -> None:
        """Test that Japanese fallback messages are used when parsing fails."""
        response_text = ""  # Empty response

        advice = generator._parse_response(response_text, language="ja")

        assert "ポーズの分析が完了しました" in advice["overall"]
        assert "練習を続けて" in advice["improvements"][0]
        assert "全体の姿勢" in advice["priority_joints"]

    def test_parse_response_fallback_messages_english(
        self,
        generator: GeminiAdviceGenerator,
    ) -> None:
        """Test that English fallback messages are used when parsing fails."""
        response_text = ""  # Empty response

        advice = generator._parse_response(response_text, language="en")

        assert "Pose analysis complete" in advice["overall"]
        assert "Continue practicing" in advice["improvements"][0]
        assert "Focus on overall posture" in advice["priority_joints"]

    @patch("etude_pose_estimator.core.advice.genai.Client")
    def test_generate_advice_calls_api(
        self,
        mock_client_class: Mock,
        generator: GeminiAdviceGenerator,
    ) -> None:
        """Test that generate_advice calls Gemini API."""
        # Setup mock
        mock_response = Mock()
        mock_response.text = """
1. 総合評価
良好です。

2. 改善ポイント
- テスト改善1
- テスト改善2

3. 重点部位
テスト部位
"""
        mock_client = Mock()
        mock_client.models.generate_content.return_value = mock_response
        generator.client = mock_client

        # Call generate_advice
        advice = generator.generate_advice(
            similarity_score=0.85,
            angle_differences={"左肘": 10.0},
            joint_distances={"joint_0": 0.1},
            language="ja",
        )

        # Verify API was called
        mock_client.models.generate_content.assert_called_once()
        assert "overall" in advice
        assert "improvements" in advice

    def test_generate_advice_unsupported_language_raises_error(
        self,
        generator: GeminiAdviceGenerator,
    ) -> None:
        """Test that unsupported language raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported language"):
            generator.generate_advice(
                similarity_score=0.85,
                angle_differences={},
                joint_distances={},
                language="fr",  # Unsupported
            )
