"""
Unit tests for src/models/trainer.py

Strategy:
    - MLflow is mocked entirely — no tracking server needed
    - XGBoost training uses synthetic EngineeredData (small, fast)
    - Tests cover: unit split correctness, feature extraction,
      evaluation metrics, MLflow call signatures, error handling
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.config import CMAPSSSubset
from src.data.loader import CMAPSS_COLUMNS, OP_SETTING_COLUMNS, SENSOR_COLUMNS
from src.data.feature_engineer import EngineeredData
from src.models.schemas import EvaluationResult
from src.models.trainer import (
    DEFAULT_HYPERPARAMS,
    XGBoostHyperparams,
    XGBoostTrainer,
    TrainingResult,
)


# ── Synthetic EngineeredData Factory ──────────────────────────────────────

def _make_engineered_data(
    n_units: int = 20,
    cycles_per_unit: int = 50,
    n_features: int = 10,
    window_size: int = 10,
) -> EngineeredData:
    """
    Build a minimal synthetic EngineeredData for trainer tests.

    Uses simple linear degradation: sensor value = cycle * 0.01
    RUL = max_cycle - current_cycle (simple countdown).
    """
    feature_cols = [f"feature_{i}" for i in range(n_features)]
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, cycles_per_unit + 1):
            rul = cycles_per_unit - cycle
            # Features: simple linear degradation + unit-specific offset
            feats = [cycle * 0.01 + unit * 0.001 + i * 0.1 for i in range(n_features)]
            rows.append({"unit": unit, "cycle": cycle, "rul": rul, **dict(zip(feature_cols, feats))})

    train_df = pd.DataFrame(rows)
    train_df["unit"] = train_df["unit"].astype(np.int32)
    train_df["cycle"] = train_df["cycle"].astype(np.int32)
    train_df["rul"] = train_df["rul"].astype(np.int32)

    # Test set: last 20 cycles per unit (no RUL column)
    test_rows = []
    for unit in range(1, 5):
        for cycle in range(1, 21):
            feats = [cycle * 0.01 + unit * 0.001 + i * 0.1 for i in range(n_features)]
            test_rows.append({"unit": unit, "cycle": cycle, **dict(zip(feature_cols, feats))})
    test_df = pd.DataFrame(test_rows)

    rul_series = pd.Series(
        [30 + i * 5 for i in range(4)],
        index=pd.RangeIndex(start=1, stop=5),
        dtype=np.int32,
    )

    return EngineeredData(
        train=train_df,
        test=test_df,
        rul=rul_series,
        feature_cols=tuple(feature_cols),
        window_size=window_size,
        base_feature_cols=tuple(feature_cols[:5]),
    )


@pytest.fixture()
def engineered() -> EngineeredData:
    return _make_engineered_data(n_units=20, cycles_per_unit=50)


@pytest.fixture()
def trainer() -> XGBoostTrainer:
    return XGBoostTrainer(
        hyperparams=XGBoostHyperparams(n_estimators=10),  # fast for tests
        model_version="0.0.1",
    )


# ── MLflow Mock Context ────────────────────────────────────────────────────

def _mock_mlflow_context() -> MagicMock:
    """Return a context manager that mocks all MLflow calls."""
    mock_run = MagicMock()
    mock_run.__enter__ = MagicMock(return_value=mock_run)
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_run.info.run_id = "mock-run-id-12345"
    return mock_run


# ── XGBoostHyperparams ─────────────────────────────────────────────────────

class TestXGBoostHyperparams:
    def test_default_values(self) -> None:
        h = XGBoostHyperparams()
        assert h.n_estimators == 300
        assert h.learning_rate == 0.05
        assert h.max_depth == 6

    def test_from_dict(self) -> None:
        h = XGBoostHyperparams.from_dict({"n_estimators": 100, "max_depth": 4})
        assert h.n_estimators == 100
        assert h.max_depth == 4
        assert h.learning_rate == 0.05  # default

    def test_from_dict_ignores_unknown_keys(self) -> None:
        h = XGBoostHyperparams.from_dict({"n_estimators": 50, "unknown_param": 99})
        assert h.n_estimators == 50

    def test_to_dict_has_required_keys(self) -> None:
        h = XGBoostHyperparams()
        d = h.to_dict()
        assert "n_estimators" in d
        assert "learning_rate" in d
        assert "max_depth" in d
        assert "random_state" in d

    def test_default_hyperparams_constant_exists(self) -> None:
        assert "n_estimators" in DEFAULT_HYPERPARAMS
        assert DEFAULT_HYPERPARAMS["n_estimators"] == 300


# ── Unit Split ─────────────────────────────────────────────────────────────

class TestUnitSplit:
    def test_val_units_are_last_units(self, trainer: XGBoostTrainer) -> None:
        data = _make_engineered_data(n_units=10)
        train_split, val_split, val_units = trainer._split_by_unit(data.train)
        # Last 20% of 10 units = 2 units (units 9, 10)
        assert val_units == [9, 10]

    def test_train_and_val_units_are_disjoint(self, trainer: XGBoostTrainer) -> None:
        data = _make_engineered_data(n_units=10)
        train_split, val_split, val_units = trainer._split_by_unit(data.train)
        train_units = set(train_split["unit"].unique())
        val_set = set(val_units)
        assert train_units & val_set == set()

    def test_all_units_covered(self, trainer: XGBoostTrainer) -> None:
        data = _make_engineered_data(n_units=10)
        train_split, val_split, val_units = trainer._split_by_unit(data.train)
        all_in_splits = set(train_split["unit"].unique()) | set(val_units)
        all_original = set(data.train["unit"].unique())
        assert all_in_splits == all_original

    def test_row_counts_add_up(self, trainer: XGBoostTrainer) -> None:
        data = _make_engineered_data(n_units=10)
        train_split, val_split, val_units = trainer._split_by_unit(data.train)
        assert len(train_split) + len(val_split) == len(data.train)

    def test_too_few_units_raises(self, trainer: XGBoostTrainer) -> None:
        data = _make_engineered_data(n_units=2)
        with pytest.raises(ValueError, match="Not enough units"):
            trainer._split_by_unit(data.train)

    def test_minimum_val_units_enforced(self, trainer: XGBoostTrainer) -> None:
        # 5 units → 20% = 1 unit, but minimum is 2
        data = _make_engineered_data(n_units=5)
        _, _, val_units = trainer._split_by_unit(data.train)
        assert len(val_units) >= 2


# ── Feature Extraction ─────────────────────────────────────────────────────

class TestFeatureExtraction:
    def test_returns_correct_shapes(self) -> None:
        data = _make_engineered_data(n_units=5, cycles_per_unit=10, n_features=8)
        X, y = XGBoostTrainer._extract_features(data.train, data.feature_cols)
        assert X.shape == (50, 8)  # 5 units x 10 cycles, 8 features
        assert y.shape == (50,)

    def test_x_dtype_is_float32(self) -> None:
        data = _make_engineered_data()
        X, _ = XGBoostTrainer._extract_features(data.train, data.feature_cols)
        assert X.dtype == np.float32

    def test_y_dtype_is_float32(self) -> None:
        data = _make_engineered_data()
        _, y = XGBoostTrainer._extract_features(data.train, data.feature_cols)
        assert y.dtype == np.float32

    def test_missing_feature_col_raises(self) -> None:
        data = _make_engineered_data()
        with pytest.raises(KeyError, match="missing"):
            XGBoostTrainer._extract_features(
                data.train, ["nonexistent_feature"]
            )

    def test_y_values_match_rul_column(self) -> None:
        data = _make_engineered_data(n_units=3, cycles_per_unit=5)
        _, y = XGBoostTrainer._extract_features(data.train, data.feature_cols)
        expected = data.train["rul"].to_numpy(dtype=np.float32)
        np.testing.assert_array_equal(y, expected)


# ── Evaluation ─────────────────────────────────────────────────────────────

class TestEvaluation:
    def test_returns_evaluation_result(self) -> None:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=5, verbosity=0)
        X = np.random.rand(50, 5).astype(np.float32)
        y = np.random.rand(50).astype(np.float32) * 100
        model.fit(X, y)
        result = XGBoostTrainer._evaluate(model, X, y)
        assert isinstance(result, EvaluationResult)

    def test_rmse_is_non_negative(self) -> None:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=5, verbosity=0)
        X = np.random.rand(40, 5).astype(np.float32)
        y = np.random.rand(40).astype(np.float32) * 100
        model.fit(X, y)
        result = XGBoostTrainer._evaluate(model, X, y)
        assert result.rmse >= 0.0

    def test_perfect_predictions_give_zero_rmse(self) -> None:
        from xgboost import XGBRegressor
        # Train on data it will memorize perfectly
        X = np.arange(20, dtype=np.float32).reshape(-1, 1)
        y = np.arange(20, dtype=np.float32)
        model = XGBRegressor(n_estimators=100, max_depth=10, verbosity=0)
        model.fit(X, y)
        result = XGBoostTrainer._evaluate(model, X, y)
        assert result.rmse < 1.0  # should be near-zero

    def test_split_is_validation(self) -> None:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=5, verbosity=0)
        X = np.random.rand(20, 3).astype(np.float32)
        y = np.random.rand(20).astype(np.float32) * 50
        model.fit(X, y)
        result = XGBoostTrainer._evaluate(model, X, y)
        assert result.split == "validation"

    def test_metrics_are_in_valid_ranges(self) -> None:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=5, verbosity=0)
        X = np.random.rand(50, 4).astype(np.float32)
        y = np.random.rand(50).astype(np.float32) * 100
        model.fit(X, y)
        result = XGBoostTrainer._evaluate(model, X, y)
        assert result.mae >= 0.0
        assert result.r2 <= 1.0
        assert 0.0 <= result.f1_high_risk <= 1.0
        assert 0.0 <= result.precision_high_risk <= 1.0
        assert 0.0 <= result.recall_high_risk <= 1.0


# ── Full Training Run (MLflow Mocked) ──────────────────────────────────────

class TestTrainingRun:
    def test_train_returns_training_result(
        self, trainer: XGBoostTrainer, engineered: EngineeredData
    ) -> None:
        mock_run = _mock_mlflow_context()
        with patch("src.models.trainer.mlflow.start_run", return_value=mock_run), \
             patch("src.models.trainer.mlflow.set_tracking_uri"), \
             patch("src.models.trainer.mlflow.set_experiment"), \
             patch("src.models.trainer.mlflow.log_params"), \
             patch("src.models.trainer.mlflow.log_metrics"), \
             patch("src.models.trainer.mlflow.xgboost.log_model"):
            result = trainer.train(engineered)
        assert isinstance(result, TrainingResult)

    def test_feature_cols_preserved_in_result(
        self, trainer: XGBoostTrainer, engineered: EngineeredData
    ) -> None:
        mock_run = _mock_mlflow_context()
        with patch("src.models.trainer.mlflow.start_run", return_value=mock_run), \
             patch("src.models.trainer.mlflow.set_tracking_uri"), \
             patch("src.models.trainer.mlflow.set_experiment"), \
             patch("src.models.trainer.mlflow.log_params"), \
             patch("src.models.trainer.mlflow.log_metrics"), \
             patch("src.models.trainer.mlflow.xgboost.log_model"):
            result = trainer.train(engineered)
        assert result.feature_cols == engineered.feature_cols

    def test_val_units_are_not_empty(
        self, trainer: XGBoostTrainer, engineered: EngineeredData
    ) -> None:
        mock_run = _mock_mlflow_context()
        with patch("src.models.trainer.mlflow.start_run", return_value=mock_run), \
             patch("src.models.trainer.mlflow.set_tracking_uri"), \
             patch("src.models.trainer.mlflow.set_experiment"), \
             patch("src.models.trainer.mlflow.log_params"), \
             patch("src.models.trainer.mlflow.log_metrics"), \
             patch("src.models.trainer.mlflow.xgboost.log_model"):
            result = trainer.train(engineered)
        assert len(result.val_units) >= 2

    def test_mlflow_log_params_called(
        self, trainer: XGBoostTrainer, engineered: EngineeredData
    ) -> None:
        mock_run = _mock_mlflow_context()
        with patch("src.models.trainer.mlflow.start_run", return_value=mock_run), \
             patch("src.models.trainer.mlflow.set_tracking_uri"), \
             patch("src.models.trainer.mlflow.set_experiment"), \
             patch("src.models.trainer.mlflow.log_params") as mock_log_params, \
             patch("src.models.trainer.mlflow.log_metrics"), \
             patch("src.models.trainer.mlflow.xgboost.log_model"):
            trainer.train(engineered)
        assert mock_log_params.called

    def test_mlflow_log_metrics_called(
        self, trainer: XGBoostTrainer, engineered: EngineeredData
    ) -> None:
        mock_run = _mock_mlflow_context()
        with patch("src.models.trainer.mlflow.start_run", return_value=mock_run), \
             patch("src.models.trainer.mlflow.set_tracking_uri"), \
             patch("src.models.trainer.mlflow.set_experiment"), \
             patch("src.models.trainer.mlflow.log_params"), \
             patch("src.models.trainer.mlflow.log_metrics") as mock_log_metrics, \
             patch("src.models.trainer.mlflow.xgboost.log_model"):
            trainer.train(engineered)
        assert mock_log_metrics.called
        logged = mock_log_metrics.call_args[0][0]
        assert "val_rmse" in logged
        assert "val_mae" in logged
        assert "val_r2" in logged
        assert "val_precision_high_risk" in logged

    def test_mlflow_log_model_called(
        self, trainer: XGBoostTrainer, engineered: EngineeredData
    ) -> None:
        mock_run = _mock_mlflow_context()
        with patch("src.models.trainer.mlflow.start_run", return_value=mock_run), \
             patch("src.models.trainer.mlflow.set_tracking_uri"), \
             patch("src.models.trainer.mlflow.set_experiment"), \
             patch("src.models.trainer.mlflow.log_params"), \
             patch("src.models.trainer.mlflow.log_metrics"), \
             patch("src.models.trainer.mlflow.xgboost.log_model") as mock_log_model:
            trainer.train(engineered)
        assert mock_log_model.called

    def test_metadata_version_matches(
        self, trainer: XGBoostTrainer, engineered: EngineeredData
    ) -> None:
        mock_run = _mock_mlflow_context()
        with patch("src.models.trainer.mlflow.start_run", return_value=mock_run), \
             patch("src.models.trainer.mlflow.set_tracking_uri"), \
             patch("src.models.trainer.mlflow.set_experiment"), \
             patch("src.models.trainer.mlflow.log_params"), \
             patch("src.models.trainer.mlflow.log_metrics"), \
             patch("src.models.trainer.mlflow.xgboost.log_model"):
            result = trainer.train(engineered)
        assert result.metadata.model_version == "0.0.1"

    def test_metadata_max_rul_matches_constructor_value(
        self, engineered: EngineeredData
    ) -> None:
        """
        result.metadata.max_rul must reflect the value passed to the
        trainer's constructor, not a hardcoded fallback — this is the
        regression test for the max_rul/ModelMetadata coupling bug.
        """
        custom_trainer = XGBoostTrainer(
            hyperparams=XGBoostHyperparams(n_estimators=10),
            model_version="0.0.1",
            max_rul=80,
        )
        mock_run = _mock_mlflow_context()
        with patch("src.models.trainer.mlflow.start_run", return_value=mock_run), \
             patch("src.models.trainer.mlflow.set_tracking_uri"), \
             patch("src.models.trainer.mlflow.set_experiment"), \
             patch("src.models.trainer.mlflow.log_params"), \
             patch("src.models.trainer.mlflow.log_metrics"), \
             patch("src.models.trainer.mlflow.xgboost.log_model"):
            result = custom_trainer.train(engineered)
        assert result.metadata.max_rul == 80

    def test_insufficient_units_raises(self, trainer: XGBoostTrainer) -> None:
        tiny = _make_engineered_data(n_units=2)
        mock_run = _mock_mlflow_context()
        with patch("src.models.trainer.mlflow.start_run", return_value=mock_run), \
             patch("src.models.trainer.mlflow.set_tracking_uri"), \
             patch("src.models.trainer.mlflow.set_experiment"):
            with pytest.raises(ValueError, match="Not enough units"):
                trainer.train(tiny)