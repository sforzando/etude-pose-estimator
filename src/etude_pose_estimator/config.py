"""Application configuration management using pydantic-settings.

This module provides centralized configuration management for the pose estimator
application, loading settings from environment variables with type validation.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        gemini_api_key: API key for Gemini 3 Flash
        port: Server port number
        host: Server host address
        debug: Debug mode flag
        yolo_model_path: Path to YOLO11x pose model file
        motionbert_model_path: Path to MotionBERT checkpoint file
        max_upload_size: Maximum file upload size in bytes
        allowed_extensions: Comma-separated allowed file extensions
        reference_dir: Directory for storing reference pose JSON files
        upload_dir: Directory for temporary uploaded files
    """

    model_config = SettingsConfigDict(
        env_file=".envrc",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Gemini API Configuration
    gemini_api_key: str = Field(
        ...,
        description="API key for Gemini 3 Flash",
    )

    # Server Configuration
    port: int = Field(
        default=8888,
        description="Server port number",
        ge=1,
        le=65535,
    )
    host: str = Field(
        default="0.0.0.0",
        description="Server host address",
    )
    debug: bool = Field(
        default=False,
        description="Debug mode flag",
    )

    # Model Paths
    yolo_model_path: Path = Field(
        default=Path("models/yolo11x-pose.pt"),
        description="Path to YOLO11x pose model file",
    )
    motionbert_model_path: Path = Field(
        default=Path("models/motionbert/checkpoint.pth.tar"),
        description="Path to MotionBERT checkpoint file",
    )

    # File Upload Configuration
    max_upload_size: int = Field(
        default=104857600,  # 100MB
        description="Maximum file upload size in bytes",
        ge=0,
    )
    allowed_extensions: str = Field(
        default=".jpg,.jpeg,.png,.mp4,.mov",
        description="Comma-separated allowed file extensions",
    )

    # Data Directory
    reference_dir: Path = Field(
        default=Path("data/references"),
        description="Directory for storing reference pose JSON files",
    )
    upload_dir: Path = Field(
        default=Path("data/uploads"),
        description="Directory for temporary uploaded files",
    )

    def get_allowed_extensions_list(self) -> list[str]:
        """Get allowed file extensions as a list.

        Returns:
            List of allowed file extensions (e.g., ['.jpg', '.jpeg', '.png'])
        """
        return [ext.strip() for ext in self.allowed_extensions.split(",")]


# Global settings instance
settings = Settings()
