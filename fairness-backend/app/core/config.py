"""
Configuration settings for the Fairness Evaluation Platform API
"""

from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import List
import os
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",  # React development server
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "https://your-frontend-domain.com"  # Production frontend
    ]
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS: List[str] = [
        ".csv", ".json", ".pkl", ".joblib", ".xlsx", ".xls", ".parquet"
    ]
    UPLOAD_DIR: str = "uploads"
    
    # Model Settings
    SUPPORTED_MODEL_FORMATS: List[str] = [".pkl", ".joblib"]
    MODEL_TIMEOUT: int = 300  # 5 minutes for model operations
    
    # Analysis Settings
    DEFAULT_SIGNIFICANCE_LEVEL: float = 0.05
    MAX_CONCURRENT_ANALYSES: int = 5
    ANALYSIS_TIMEOUT: int = 1800  # 30 minutes
    MAX_ANALYSIS_TIME: int = 300  # 5 minutes - added from .env
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///./fairness_platform.db"
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # AI/LLM Settings (optional)
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    ENABLE_AI_EXPLANATIONS: bool = False  # Added from .env
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    model_config = ConfigDict(env_file=".env", case_sensitive=True)


# Create global settings instance
settings = Settings()

# Ensure upload directories exist
for directory in [
    settings.UPLOAD_DIR,
    f"{settings.UPLOAD_DIR}/datasets",
    f"{settings.UPLOAD_DIR}/models",
    f"{settings.UPLOAD_DIR}/temp"
]:
    Path(directory).mkdir(parents=True, exist_ok=True)
