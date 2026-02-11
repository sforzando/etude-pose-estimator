"""Tests for advice generation module."""

import json
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

    @patch("etude_pose_estimator.core.advice.genai.Client")
    def test_generate_advice_calls_api(
        self,
        mock_client_class: Mock,
        generator: GeminiAdviceGenerator,
    ) -> None:
        """Test that generate_advice calls Gemini API with structured output."""
        # Setup mock with JSON response
        mock_response = Mock()
        mock_response.text = json.dumps({
            "overall": "類似度85.0%と良好なポーズです。あと一歩で完璧です。",
            "improvements": [
                "左肘は測定値で10.0度のずれがあり、もう少し高く上げると力強さが増します",
                "右膝の角度を少し深くして、安定感のある姿勢を目指しましょう",
                "左肩を後ろに引いて、胸を張った堂々とした姿勢にしましょう",
            ],
            "priority_joints": "左肘、右膝、左肩",
        })
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
        assert "priority_joints" in advice
        assert "類似度85.0%" in advice["overall"]
        assert len(advice["improvements"]) == 3
        assert "左肘" in advice["priority_joints"]

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
