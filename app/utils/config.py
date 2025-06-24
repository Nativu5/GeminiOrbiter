import os
import sys
from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

CONFIG_PATH = "config/config.yaml"


class ServerConfig(BaseModel):
    """Server configuration"""

    host: str = Field(default="0.0.0.0", description="Server host address")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port number")
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication, if set, will enable API key validation",
    )


class GeminiClientSettings(BaseModel):
    """Credential set for one Gemini client."""

    id: str = Field(..., description="Unique identifier for the client")
    secure_1psid: str = Field(..., description="Gemini Secure 1PSID")
    secure_1psidts: str = Field(..., description="Gemini Secure 1PSIDTS")


class GeminiConfig(BaseModel):
    """Gemini API configuration"""

    clients: list[GeminiClientSettings] = Field(
        ..., description="List of Gemini client credential pairs"
    )
    timeout: int = Field(default=60, ge=1, description="Init timeout")
    auto_refresh: bool = Field(True, description="Enable auto-refresh for Gemini cookies")
    refresh_interval: int = Field(
        default=540, ge=1, description="Interval in seconds to refresh Gemini cookies"
    )
    verbose: bool = Field(False, description="Enable verbose logging for Gemini API requests")


class CORSConfig(BaseModel):
    """CORS configuration"""

    enabled: bool = Field(default=True, description="Enable CORS support")
    allow_origins: list[str] = Field(
        default=["*"], description="List of allowed origins for CORS requests"
    )
    allow_credentials: bool = Field(default=True, description="Allow credentials in CORS requests")
    allow_methods: list[str] = Field(
        default=["*"], description="List of allowed HTTP methods for CORS requests"
    )
    allow_headers: list[str] = Field(
        default=["*"], description="List of allowed headers for CORS requests"
    )


class StorageConfig(BaseModel):
    path: str = Field(
        default="data/lmdb",
        description="Path to the storage directory where data will be saved",
    )
    max_size: int = Field(
        default=1024**2 * 128,  # 128 MB
        ge=1,
        description="Maximum size of the storage in bytes",
    )


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="DEBUG",
        description="Logging level",
    )


class Config(BaseSettings):
    """Application configuration"""

    # Server configuration
    server: ServerConfig = Field(
        default=ServerConfig(),
        description="Server configuration, including host, port, and API key",
    )

    # CORS configuration
    cors: CORSConfig = Field(
        default=CORSConfig(),
        description="CORS configuration, allows cross-origin requests",
    )

    # Gemini API configuration
    gemini: GeminiConfig = Field(..., description="Gemini API configuration, must be set")

    storage: StorageConfig = Field(
        default=StorageConfig(),
        description="Storage configuration, defines where and how data will be stored",
    )

    # Logging configuration
    logging: LoggingConfig = Field(
        default=LoggingConfig(),
        description="Logging configuration",
    )

    model_config = SettingsConfigDict(
        env_prefix="CONFIG_",
        env_nested_delimiter="__",
        yaml_file=os.getenv("CONFIG_PATH", CONFIG_PATH),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        """Read settings: env -> yaml -> default"""
        return (env_settings, YamlConfigSettingsSource(settings_cls))


def initialize_config() -> Config:
    """
    Initialize the configuration.

    Returns:
        Config: Configuration object
    """
    try:
        # Using environment variables and YAML file for configuration
        return Config()  # type: ignore
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e!s}")
        sys.exit(1)
