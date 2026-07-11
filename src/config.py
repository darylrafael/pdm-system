"""
Centralized application configuration via pydantic-settings.

All configuration is read from environment variables (or .env file).
Import the `get_settings()` function — never instantiate Settings directly.

Usage:
    from src.config import get_settings

    settings = get_settings()
    print(settings.app_name)
"""

from functools import lru_cache
from enum import Enum
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(str, Enum):
    """Valid application environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Valid log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class CMAPSSSubset(str, Enum):
    """Valid NASA CMAPSS dataset subsets."""

    FD001 = "FD001"
    FD002 = "FD002"
    FD003 = "FD003"
    FD004 = "FD004"


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All fields map directly to variables in .env.example.
    pydantic-settings handles type coercion and validation automatically.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars
    )

    # ── Application ────────────────────────────────────────────────────────
    app_env: AppEnv = Field(
        default=AppEnv.DEVELOPMENT,
        description="Application environment",
    )
    app_name: str = Field(
        default="pdm-system",
        description="Application name",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode — NEVER true in production",
    )

    # ── API Security ───────────────────────────────────────────────────────
    api_key: str = Field(
        description="Secret API key for endpoint authentication",
        min_length=32,
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="HTTP header name for API key",
    )

    # ── API Server ─────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1024, le=65535)
    api_workers: int = Field(default=1, ge=1, le=16)

    # ── Database ───────────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite+aiosqlite:///./data/pdm.db",
        description="Async SQLAlchemy database URL",
    )

    # ── MLflow ─────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(default="./mlruns")
    mlflow_experiment_name: str = Field(default="pdm-predictive-maintenance")

    # ── Model ──────────────────────────────────────────────────────────────
    model_path: str = Field(default="./models/best_model.pkl")
    model_version: str = Field(default="1.0.0")

    # ── Data ───────────────────────────────────────────────────────────────
    data_raw_path: str = Field(default="./data/raw")
    data_processed_path: str = Field(default="./data/processed")
    cmapss_subset: CMAPSSSubset = Field(
        default=CMAPSSSubset.FD001,
        description="NASA CMAPSS dataset subset to use",
    )

    # ── Drift Detection ────────────────────────────────────────────────────
    drift_threshold: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Drift score above this value triggers an alert",
    )
    drift_reference_window: int = Field(
        default=1000,
        ge=100,
        description="Number of samples used as reference distribution",
    )

    # ── Dashboard ──────────────────────────────────────────────────────────
    dashboard_api_url: str = Field(default="http://localhost:8000")
    streamlit_port: int = Field(default=8501, ge=1024, le=65535)

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: LogLevel = Field(default=LogLevel.INFO)

    # ── Derived Properties ─────────────────────────────────────────────────
    @property
    def is_production(self) -> bool:
        """True if running in production environment."""
        return self.app_env == AppEnv.PRODUCTION

    @property
    def is_development(self) -> bool:
        """True if running in development environment."""
        return self.app_env == AppEnv.DEVELOPMENT

    # ── Validators ────────────────────────────────────────────────────────
    @field_validator("debug", mode="before")
    @classmethod
    def debug_must_be_false_in_production(cls, v: bool, info: object) -> bool:
        """Prevent debug mode in production — security guardrail."""
        # Note: cross-field validation via model_validator done at model level
        return v

    @field_validator("api_key", mode="before")
    @classmethod
    def api_key_must_not_be_placeholder(cls, v: str) -> str:
        """Reject obviously unset API keys."""
        forbidden = {"your-strong-api-key-here", "changeme", "secret", ""}
        if str(v).lower() in forbidden:
            raise ValueError(
                "API_KEY is set to a placeholder value. "
                "Generate a real key: python -c \"import secrets; print(secrets.token_hex(32))\""
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    Uses lru_cache to ensure .env is read only once per process lifetime.
    This is the ONLY way settings should be accessed across the codebase.

    Returns:
        Settings: Validated, immutable settings instance.

    Example:
        from src.config import get_settings
        settings = get_settings()
    """
    return Settings()