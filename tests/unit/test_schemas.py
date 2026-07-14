"""
Unit tests for src/models/schemas.py

Tests cover:
- Field validation (ranges, patterns, lengths)
- Cross-field validators (risk_level consistency, CI ordering)
- RiskLevel.from_rul() classification boundaries
- JSON serialization round-trip
- Schema rejection of invalid inputs
"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    EvaluationResult,
    ModelMetadata,
    PredictionRequest,
    PredictionResponse,
    RiskLevel,
)

# ── Shared Fixtures ────────────────────────────────────────────────────────

NOW = datetime.now(timezone.utc)

VALID_SENSOR_READINGS: list[float] = [
    518.67, 641.82, 1589.70, 1400.60, 14.62,
    21.61, 554.36, 2388.02, 9046.19, 1.30,
    47.47, 521.66, 2388.02, 8138.62, 8.4195,
    0.03, 392.0, 2388.0, 100.0, 39.06, 23.419
]

VALID_OP_SETTINGS: list[float] = [0.0, 0.0, 100.0]


# ── RiskLevel ──────────────────────────────────────────────────────────────

class TestRiskLevel:
    def test_rul_below_30_is_high(self) -> None:
        assert RiskLevel.from_rul(0) == RiskLevel.HIGH
        assert RiskLevel.from_rul(29.9) == RiskLevel.HIGH

    def test_rul_exactly_30_is_medium(self) -> None:
        assert RiskLevel.from_rul(30) == RiskLevel.MEDIUM

    def test_rul_between_30_and_60_is_medium(self) -> None:
        assert RiskLevel.from_rul(45) == RiskLevel.MEDIUM
        assert RiskLevel.from_rul(59.9) == RiskLevel.MEDIUM

    def test_rul_exactly_60_is_normal(self) -> None:
        assert RiskLevel.from_rul(60) == RiskLevel.NORMAL

    def test_rul_above_60_is_normal(self) -> None:
        assert RiskLevel.from_rul(100) == RiskLevel.NORMAL
        assert RiskLevel.from_rul(999) == RiskLevel.NORMAL

    def test_risk_level_is_string_enum(self) -> None:
        assert RiskLevel.HIGH == "HIGH"
        assert RiskLevel.MEDIUM == "MEDIUM"
        assert RiskLevel.NORMAL == "NORMAL"


# ── PredictionRequest ──────────────────────────────────────────────────────

class TestPredictionRequest:
    def _valid(self, **overrides: object) -> dict:
        base = {
            "unit_id": 1,
            "cycle": 50,
            "operational_settings": VALID_OP_SETTINGS,
            "sensor_readings": VALID_SENSOR_READINGS,
        }
        base.update(overrides)
        return base

    def test_valid_request_accepted(self) -> None:
        req = PredictionRequest(**self._valid())
        assert req.unit_id == 1
        assert req.cycle == 50
        assert len(req.sensor_readings) == 21

    def test_unit_id_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unit_id"):
            PredictionRequest(**self._valid(unit_id=0))

    def test_unit_id_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PredictionRequest(**self._valid(unit_id=-1))

    def test_cycle_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PredictionRequest(**self._valid(cycle=0))

    def test_wrong_sensor_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PredictionRequest(**self._valid(sensor_readings=[1.0] * 20))

    def test_too_many_sensors_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PredictionRequest(**self._valid(sensor_readings=[1.0] * 22))

    def test_wrong_op_setting_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PredictionRequest(**self._valid(operational_settings=[0.0, 0.0]))

    def test_nan_sensor_rejected(self) -> None:
        bad = VALID_SENSOR_READINGS.copy()
        bad[3] = float("nan")
        with pytest.raises(ValidationError, match="non-finite"):
            PredictionRequest(**self._valid(sensor_readings=bad))

    def test_inf_sensor_rejected(self) -> None:
        bad = VALID_SENSOR_READINGS.copy()
        bad[0] = float("inf")
        with pytest.raises(ValidationError, match="non-finite"):
            PredictionRequest(**self._valid(sensor_readings=bad))

    def test_nan_op_setting_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-finite"):
            PredictionRequest(**self._valid(
                operational_settings=[float("nan"), 0.0, 100.0]
            ))

    def test_json_serializable(self) -> None:
        req = PredictionRequest(**self._valid())
        json_str = req.model_dump_json()
        assert "unit_id" in json_str
        assert "sensor_readings" in json_str

    def test_round_trip_serialization(self) -> None:
        req = PredictionRequest(**self._valid())
        restored = PredictionRequest.model_validate_json(req.model_dump_json())
        assert restored.unit_id == req.unit_id
        assert restored.sensor_readings == req.sensor_readings


# ── PredictionResponse ─────────────────────────────────────────────────────

class TestPredictionResponse:
    def _valid(self, **overrides: object) -> dict:
        base = {
            "unit_id": 1,
            "cycle": 50,
            "predicted_rul": 45.0,
            "risk_level": RiskLevel.MEDIUM,
            "model_version": "1.0.0",
            "prediction_timestamp": NOW,
            "confidence_interval": None,
        }
        base.update(overrides)
        return base

    def test_valid_response_accepted(self) -> None:
        resp = PredictionResponse(**self._valid())
        assert resp.predicted_rul == 45.0
        assert resp.risk_level == RiskLevel.MEDIUM

    def test_negative_rul_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PredictionResponse(**self._valid(predicted_rul=-1.0))

    def test_inconsistent_risk_level_rejected(self) -> None:
        # RUL=45 should be MEDIUM, not HIGH
        with pytest.raises(ValidationError, match="inconsistent"):
            PredictionResponse(**self._valid(
                predicted_rul=45.0,
                risk_level=RiskLevel.HIGH
            ))

    def test_high_risk_rul_accepted(self) -> None:
        resp = PredictionResponse(**self._valid(
            predicted_rul=20.0,
            risk_level=RiskLevel.HIGH,
        ))
        assert resp.risk_level == RiskLevel.HIGH

    def test_normal_risk_rul_accepted(self) -> None:
        resp = PredictionResponse(**self._valid(
            predicted_rul=100.0,
            risk_level=RiskLevel.NORMAL,
        ))
        assert resp.risk_level == RiskLevel.NORMAL

    def test_valid_confidence_interval_accepted(self) -> None:
        resp = PredictionResponse(**self._valid(
            confidence_interval=(30.0, 60.0)
        ))
        assert resp.confidence_interval == (30.0, 60.0)

    def test_inverted_confidence_interval_rejected(self) -> None:
        with pytest.raises(ValidationError, match="lower bound"):
            PredictionResponse(**self._valid(
                confidence_interval=(60.0, 30.0)
            ))

    def test_equal_confidence_bounds_accepted(self) -> None:
        # lower == upper is technically valid (degenerate interval)
        resp = PredictionResponse(**self._valid(
            confidence_interval=(45.0, 45.0)
        ))
        assert resp.confidence_interval == (45.0, 45.0)

    def test_nan_confidence_interval_rejected(self) -> None:
        with pytest.raises(ValidationError, match="finite"):
            PredictionResponse(**self._valid(
                confidence_interval=(float("nan"), 60.0)
            ))

    def test_inf_confidence_interval_rejected(self) -> None:
        with pytest.raises(ValidationError, match="finite"):
            PredictionResponse(**self._valid(
                confidence_interval=(30.0, float("inf"))
            ))

    def test_negative_confidence_lower_bound_rejected(self) -> None:
        with pytest.raises(ValidationError, match="0.0"):
            PredictionResponse(**self._valid(
                confidence_interval=(-10.0, 50.0)
            ))

    def test_zero_confidence_lower_bound_accepted(self) -> None:
        # lower == 0.0 is the boundary and must be accepted (ge=0.0, not gt)
        resp = PredictionResponse(**self._valid(
            confidence_interval=(0.0, 50.0)
        ))
        assert resp.confidence_interval == (0.0, 50.0)

    def test_json_serializable(self) -> None:
        resp = PredictionResponse(**self._valid())
        json_str = resp.model_dump_json()
        assert "predicted_rul" in json_str
        assert "risk_level" in json_str


# ── ModelMetadata ──────────────────────────────────────────────────────────

class TestModelMetadata:
    def _valid(self, **overrides: object) -> dict:
        base = {
            "model_version": "1.0.0",
            "cmapss_subset": "FD001",
            "algorithm": "XGBoost",
            "feature_count": 56,
            "max_rul": 125,
            "window_size": 10,
            "trained_at": NOW,
            "training_rows": 17731,
        }
        base.update(overrides)
        return base

    def test_valid_metadata_accepted(self) -> None:
        m = ModelMetadata(**self._valid())
        assert m.model_version == "1.0.0"
        assert m.cmapss_subset == "FD001"

    def test_invalid_version_format_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelMetadata(**self._valid(model_version="v1.0"))

    def test_invalid_subset_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelMetadata(**self._valid(cmapss_subset="FD005"))

    def test_all_valid_subsets_accepted(self) -> None:
        for subset in ["FD001", "FD002", "FD003", "FD004"]:
            m = ModelMetadata(**self._valid(cmapss_subset=subset))
            assert m.cmapss_subset == subset

    def test_zero_feature_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelMetadata(**self._valid(feature_count=0))

    def test_window_size_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelMetadata(**self._valid(window_size=1))

    def test_mlflow_run_id_optional(self) -> None:
        m = ModelMetadata(**self._valid())
        assert m.mlflow_run_id is None

    def test_mlflow_run_id_accepted(self) -> None:
        m = ModelMetadata(**self._valid(mlflow_run_id="abc123def456"))
        assert m.mlflow_run_id == "abc123def456"


# ── EvaluationResult ───────────────────────────────────────────────────────

class TestEvaluationResult:
    def _valid(self, **overrides: object) -> dict:
        base = {
            "rmse": 15.3,
            "mae": 11.2,
            "r2": 0.87,
            "f1_high_risk": 0.91,
            "precision_high_risk": 0.89,
            "recall_high_risk": 0.93,
            "evaluated_at": NOW,
            "split": "test",
        }
        base.update(overrides)
        return base

    def test_valid_result_accepted(self) -> None:
        r = EvaluationResult(**self._valid())
        assert r.rmse == 15.3
        assert r.split == "test"

    def test_negative_rmse_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationResult(**self._valid(rmse=-1.0))

    def test_negative_mae_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationResult(**self._valid(mae=-0.1))

    def test_r2_above_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationResult(**self._valid(r2=1.01))

    def test_negative_r2_accepted(self) -> None:
        # R2 can be negative (worse than mean baseline)
        r = EvaluationResult(**self._valid(r2=-0.5))
        assert r.r2 == -0.5

    def test_f1_above_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationResult(**self._valid(f1_high_risk=1.01))

    def test_invalid_split_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationResult(**self._valid(split="holdout"))

    def test_all_valid_splits_accepted(self) -> None:
        for split in ["train", "validation", "test"]:
            r = EvaluationResult(**self._valid(split=split))
            assert r.split == split

    def test_json_serializable(self) -> None:
        r = EvaluationResult(**self._valid())
        json_str = r.model_dump_json()
        assert "rmse" in json_str
        assert "f1_high_risk" in json_str