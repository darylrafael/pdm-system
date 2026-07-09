"""
Unit tests for src/config.py

Tests cover:
- Valid settings load from environment
- Field validation (API key, drift threshold, ports)
- Enum validation (AppEnv, LogLevel, CMAPSSSubset)
- Derived properties (is_production, is_development)
- Singleton behavior of get_settings()
- Security: placeholder API key rejection
"""

import pytest
from unittest.mock import patch
from pydantic import ValidationError

from src.config import (
    AppEnv,
    CMAPSSSubset,
    LogLevel,
    Settings,
    get_settings,
)

# ── Fixtures ───────────────────────────────────────────────────────────────

VALID_ENV = {
    "APP_ENV": "development",
    "APP_NAME": "pdm-system",
    "APP_VERSION": "0.1.0",
    "DEBUG": "true",
    "API_KEY": "a" * 32,  # 32-char valid key
    "API_KEY_HEADER": "X-API-Key",
    "API_HOST": "0.0.0.0",
    "API_PORT": "8000",
    "API_WORKERS": "1",
    "DATABASE_URL": "sqlite+aiosqlite:///./data/test.db",
    "MLFLOW_TRACKING_URI": "./mlruns",
    "MLFLOW_EXPERIMENT_NAME": "test-experiment",
    "MODEL_PATH": "./models/test_model.pkl",
    "MODEL_VERSION": "0.1.0",
    "DATA_RAW_PATH": "./data/raw",
    "DATA_PROCESSED_PATH": "./data/processed",
    "CMAPSS_SUBSET": "FD001",
    "DRIFT_THRESHOLD": "0.15",
    "DRIFT_REFERENCE_WINDOW": "1000",
    "DASHBOARD_API_URL": "http://localhost:8000",
    "STREAMLIT_PORT": "8501",
    "LOG_LEVEL": "INFO",
}


@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    """Clear lru_cache before each test to ensure isolation."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ── Valid Configuration ────────────────────────────────────────────────────

class TestValidSettings:
    def test_loads_all_fields_correctly(self) -> None:
        with patch.dict("os.environ", VALID_ENV, clear=True):
            s = Settings()
            assert s.app_name == "pdm-system"
            assert s.api_port == 8000
            assert s.drift_threshold == 0.15
            assert s.cmapss_subset == CMAPSSSubset.FD001
            assert s.log_level == LogLevel.INFO
            assert s.app_env == AppEnv.DEVELOPMENT

    def test_default_values_applied(self) -> None:
        minimal_env = {"API_KEY": "b" * 32}
        with patch.dict("os.environ", minimal_env, clear=True):
            s = Settings()
            assert s.app_name == "pdm-system"
            assert s.api_port == 8000
            assert s.streamlit_port == 8501
            assert s.drift_threshold == 0.15
            assert s.cmapss_subset == CMAPSSSubset.FD001


# ── Derived Properties ─────────────────────────────────────────────────────

class TestDerivedProperties:
    def test_is_development_true_in_dev(self) -> None:
        with patch.dict("os.environ", {**VALID_ENV, "APP_ENV": "development"}, clear=True):
            s = Settings()
            assert s.is_development is True
            assert s.is_production is False

    def test_is_production_true_in_prod(self) -> None:
        with patch.dict("os.environ", {**VALID_ENV, "APP_ENV": "production"}, clear=True):
            s = Settings()
            assert s.is_production is True
            assert s.is_development is False


# ── Enum Validation ────────────────────────────────────────────────────────

class TestEnumValidation:
    def test_invalid_app_env_raises(self) -> None:
        with patch.dict("os.environ", {**VALID_ENV, "APP_ENV": "invalid"}, clear=True):
            with pytest.raises(ValidationError):
                Settings()

    def test_invalid_log_level_raises(self) -> None:
        with patch.dict("os.environ", {**VALID_ENV, "LOG_LEVEL": "VERBOSE"}, clear=True):
            with pytest.raises(ValidationError):
                Settings()

    def test_invalid_cmapss_subset_raises(self) -> None:
        with patch.dict("os.environ", {**VALID_ENV, "CMAPSS_SUBSET": "FD999"}, clear=True):
            with pytest.raises(ValidationError):
                Settings()

    def test_all_valid_cmapss_subsets_accepted(self) -> None:
        for subset in ["FD001", "FD002", "FD003", "FD004"]:
            with patch.dict("os.environ", {**VALID_ENV, "CMAPSS_SUBSET": subset}, clear=True):
                s = Settings()
                assert s.cmapss_subset.value == subset


# ── Security Validators ────────────────────────────────────────────────────

class TestSecurityValidation:
    def test_placeholder_api_key_rejected(self) -> None:
        bad_keys = [
            "your-strong-api-key-here",
            "changeme",
            "secret",
        ]
        for bad_key in bad_keys:
            with patch.dict("os.environ", {**VALID_ENV, "API_KEY": bad_key}, clear=True):
                with pytest.raises(ValidationError, match="placeholder"):
                    Settings()

    def test_short_api_key_rejected(self) -> None:
        with patch.dict("os.environ", {**VALID_ENV, "API_KEY": "tooshort"}, clear=True):
            with pytest.raises(ValidationError):
                Settings()

    def test_valid_32_char_api_key_accepted(self) -> None:
        valid_key = "x" * 32
        with patch.dict("os.environ", {**VALID_ENV, "API_KEY": valid_key}, clear=True):
            s = Settings()
            assert s.api_key == valid_key


# ── Field Range Validation ─────────────────────────────────────────────────

class TestFieldValidation:
    def test_drift_threshold_above_1_rejected(self) -> None:
        with patch.dict("os.environ", {**VALID_ENV, "DRIFT_THRESHOLD": "1.5"}, clear=True):
            with pytest.raises(ValidationError):
                Settings()

    def test_drift_threshold_below_0_rejected(self) -> None:
        with patch.dict("os.environ", {**VALID_ENV, "DRIFT_THRESHOLD": "-0.1"}, clear=True):
            with pytest.raises(ValidationError):
                Settings()

    def test_invalid_port_rejected(self) -> None:
        with patch.dict("os.environ", {**VALID_ENV, "API_PORT": "80"}, clear=True):
            with pytest.raises(ValidationError):
                Settings()

    def test_drift_reference_window_minimum(self) -> None:
        with patch.dict("os.environ", {**VALID_ENV, "DRIFT_REFERENCE_WINDOW": "50"}, clear=True):
            with pytest.raises(ValidationError):
                Settings()


# ── Singleton Behavior ─────────────────────────────────────────────────────

class TestSingleton:
    def test_get_settings_returns_same_instance(self) -> None:
        with patch.dict("os.environ", VALID_ENV, clear=True):
            s1 = get_settings()
            s2 = get_settings()
            assert s1 is s2

    def test_cache_clear_allows_reload(self) -> None:
        with patch.dict("os.environ", VALID_ENV, clear=True):
            s1 = get_settings()
            get_settings.cache_clear()
            s2 = get_settings()
            # Different instances but same values
            assert s1.app_name == s2.app_name
            assert s1 is not s2