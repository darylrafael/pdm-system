"""
Unit tests for src/data/preprocessor.py

Strategy:
    - Build synthetic CMAPSSData in-memory (no disk I/O required)
    - Tests cover normalization correctness, no-leakage guarantee,
      RUL clipping, constant sensor dropping, and error handling
    - All assertions are value-level, not just shape-level
"""

import logging

import numpy as np
import pandas as pd
import pytest

from src.config import CMAPSSSubset
from src.data.loader import (
    CMAPSS_COLUMNS,
    CMAPSSData,
    OP_SETTING_COLUMNS,
    SENSOR_COLUMNS,
)
from src.data.preprocessor import (
    CONSTANT_SENSORS,
    DEFAULT_MAX_RUL,
    USEFUL_SENSORS,
    CMAPSSPreprocessor,
    ProcessedData,
)


# ── Synthetic Data Factory ─────────────────────────────────────────────────

def _make_cmapss_data(
    n_train_units: int = 5,
    train_cycles: int = 40,
    n_test_units: int = 3,
    test_cycles: int = 20,
    subset: CMAPSSSubset = CMAPSSSubset.FD001,
    sensor_value: float | None = None,
) -> CMAPSSData:
    """
    Build a synthetic CMAPSSData object in memory.

    Args:
        n_train_units:  Number of training engine units.
        train_cycles:   Cycles per training unit.
        n_test_units:   Number of test engine units.
        test_cycles:    Cycles per test unit.
        subset:         CMAPSS subset label.
        sensor_value:   If set, all sensor readings are this constant value
                        (used to test zero-variance detection).

    Returns:
        CMAPSSData with realistic structure and computed RUL.
    """
    def _make_df(n_units: int, cycles: int, include_rul: bool) -> pd.DataFrame:
        rows = []
        for unit in range(1, n_units + 1):
            for cycle in range(1, cycles + 1):
                op = [float(unit + cycle) * 0.1 * i for i in range(1, 4)]
                if sensor_value is not None:
                    sensors = [float(sensor_value)] * 21
                else:
                    # Vary values by unit+cycle to create non-zero variance
                    sensors = [float(unit * 10 + cycle + i) for i in range(21)]
                row = [unit, cycle] + op + sensors
                rows.append(row)

        df = pd.DataFrame(rows, columns=CMAPSS_COLUMNS).astype({
            "unit": np.int32,
            "cycle": np.int32,
            **{col: np.float32 for col in OP_SETTING_COLUMNS},
            **{col: np.float32 for col in SENSOR_COLUMNS},
        })

        if include_rul:
            df["rul"] = (
                df.groupby("unit")["cycle"].transform("max") - df["cycle"]
            ).astype(np.int32)

        return df

    train_df = _make_df(n_train_units, train_cycles, include_rul=True)
    test_df = _make_df(n_test_units, test_cycles, include_rul=False)
    rul_series = pd.Series(
        [50 + i * 10 for i in range(n_test_units)],
        index=pd.RangeIndex(start=1, stop=n_test_units + 1),
        dtype=np.int32,
    )

    return CMAPSSData(
        subset=subset,
        train=train_df,
        test=test_df,
        rul=rul_series,
    )


@pytest.fixture()
def raw_data() -> CMAPSSData:
    """Standard synthetic CMAPSSData for most tests."""
    return _make_cmapss_data(n_train_units=5, train_cycles=40)


@pytest.fixture()
def preprocessor() -> CMAPSSPreprocessor:
    """Default preprocessor with standard max_rul."""
    return CMAPSSPreprocessor(max_rul=DEFAULT_MAX_RUL)


# ── Module-Level Constants ─────────────────────────────────────────────────

class TestModuleConstants:
    def test_constant_sensors_count(self) -> None:
        assert len(CONSTANT_SENSORS) == 7

    def test_useful_sensors_count(self) -> None:
        assert len(USEFUL_SENSORS) == 14

    def test_constant_and_useful_are_disjoint(self) -> None:
        assert set(CONSTANT_SENSORS) & set(USEFUL_SENSORS) == set()

    def test_constant_plus_useful_equals_all_sensors(self) -> None:
        assert set(CONSTANT_SENSORS) | set(USEFUL_SENSORS) == set(SENSOR_COLUMNS)

    def test_default_max_rul_is_positive(self) -> None:
        assert DEFAULT_MAX_RUL > 0


# ── Preprocessor Initialization ────────────────────────────────────────────

class TestPreprocessorInit:
    def test_default_max_rul(self) -> None:
        p = CMAPSSPreprocessor()
        assert p._max_rul == DEFAULT_MAX_RUL

    def test_custom_max_rul(self) -> None:
        p = CMAPSSPreprocessor(max_rul=80)
        assert p._max_rul == 80

    def test_zero_max_rul_raises(self) -> None:
        with pytest.raises(ValueError, match="max_rul must be positive"):
            CMAPSSPreprocessor(max_rul=0)

    def test_negative_max_rul_raises(self) -> None:
        with pytest.raises(ValueError, match="max_rul must be positive"):
            CMAPSSPreprocessor(max_rul=-10)

    def test_is_not_fitted_on_init(self) -> None:
        assert CMAPSSPreprocessor()._is_fitted is False


# ── fit_transform Output Contract ──────────────────────────────────────────

class TestFitTransformOutput:
    def test_returns_processed_data(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        assert isinstance(result, ProcessedData)

    def test_output_is_frozen(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        with pytest.raises(Exception):
            result.max_rul = 999  # type: ignore[misc]

    def test_rul_series_is_unchanged(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        pd.testing.assert_series_equal(result.rul, raw_data.rul)

    def test_train_row_count_unchanged(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        assert len(result.train) == len(raw_data.train)

    def test_test_row_count_unchanged(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        assert len(result.test) == len(raw_data.test)

    def test_max_rul_recorded_correctly(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        assert result.max_rul == DEFAULT_MAX_RUL

    def test_preprocessor_is_fitted_after_call(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        preprocessor.fit_transform(raw_data)
        assert preprocessor._is_fitted is True


# ── Constant Sensor Dropping ───────────────────────────────────────────────

class TestConstantSensorDropping:
    def test_constant_sensors_not_in_train(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        for col in CONSTANT_SENSORS:
            assert col not in result.train.columns

    def test_constant_sensors_not_in_test(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        for col in CONSTANT_SENSORS:
            assert col not in result.test.columns

    def test_dropped_sensors_recorded(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        assert set(result.dropped_sensors) == set(CONSTANT_SENSORS)

    def test_useful_sensors_retained_in_train(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        for col in USEFUL_SENSORS:
            assert col in result.train.columns

    def test_data_driven_constant_detection(self) -> None:
        """A sensor with zero variance in training data is also dropped."""
        data = _make_cmapss_data(n_train_units=3, train_cycles=20)

        # Construct a new CMAPSSData with pre-modified DataFrames instead
        # of mutating the DataFrames inside the original frozen CMAPSSData
        # (its docstring requires .copy() before in-place mutation).
        modified_train = data.train.copy()
        modified_test = data.test.copy()
        modified_train["sensor_2"] = 1.0
        modified_test["sensor_2"] = 1.0

        data_with_constant = CMAPSSData(
            subset=data.subset,
            train=modified_train,
            test=modified_test,
            rul=data.rul,
        )

        p = CMAPSSPreprocessor()
        result = p.fit_transform(data_with_constant)
        assert "sensor_2" in result.dropped_sensors


# ── RUL Clipping ───────────────────────────────────────────────────────────

class TestRULClipping:
    def test_rul_max_does_not_exceed_cap(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        assert result.train["rul"].max() <= DEFAULT_MAX_RUL

    def test_rul_min_is_zero(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        assert result.train["rul"].min() == 0

    def test_custom_rul_cap_applied(self) -> None:
        data = _make_cmapss_data(train_cycles=200)  # RUL will exceed 80
        p = CMAPSSPreprocessor(max_rul=80)
        result = p.fit_transform(data)
        assert result.train["rul"].max() <= 80

    def test_rul_not_clipped_if_below_cap(self) -> None:
        # train_cycles=10 → max RUL = 9, well below default cap of 125
        data = _make_cmapss_data(train_cycles=10)
        p = CMAPSSPreprocessor(max_rul=125)
        result = p.fit_transform(data)
        assert result.train["rul"].max() == 9  # unchanged

    def test_rul_dtype_preserved_as_int32(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        assert result.train["rul"].dtype == np.int32

    def test_test_set_has_no_rul_column(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        assert "rul" not in result.test.columns


# ── Normalization Correctness ──────────────────────────────────────────────

class TestNormalization:
    def test_normalized_values_in_range_0_1(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        for col in result.feature_cols:
            if col in result.train.columns:
                assert result.train[col].min() >= -1e-6, f"{col} below 0"
                assert result.train[col].max() <= 1 + 1e-6, f"{col} above 1"

    def test_train_min_normalized_to_zero(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        for col in result.feature_cols:
            if col in result.train.columns:
                assert abs(result.train[col].min()) < 1e-5, (
                    f"{col} min should be ~0 after normalization"
                )

    def test_train_max_normalized_to_one(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        for col in result.feature_cols:
            if col in result.train.columns:
                assert abs(result.train[col].max() - 1.0) < 1e-5, (
                    f"{col} max should be ~1 after normalization"
                )

    def test_no_data_leakage_from_test(self) -> None:
        """
        Normalization params must be fit on train only.

        If test values exceed train range, they will be outside [0, 1].
        This is expected behavior — not a bug. The test verifies params
        come from train, not train+test combined.
        """
        # Make test set with values ABOVE train range for sensor_2
        data = _make_cmapss_data(n_train_units=3, train_cycles=20)

        # Construct a new CMAPSSData with a pre-modified test DataFrame
        # instead of mutating the DataFrame inside the original frozen
        # CMAPSSData (its docstring requires .copy() before in-place
        # mutation).
        modified_test = data.test.copy()
        modified_test.loc[:, "sensor_2"] = 999999.0
        data_with_outlier = CMAPSSData(
            subset=data.subset,
            train=data.train,
            test=modified_test,
            rul=data.rul,
        )

        p = CMAPSSPreprocessor()
        result = p.fit_transform(data_with_outlier)

        # Norm params must be fit on train only — sensor_2 max from train
        train_max = data.train["sensor_2"].max()
        assert result.norm_params["sensor_2"]["max"] == pytest.approx(
            float(train_max), rel=1e-4
        )

        # Test values will be >> 1.0 because they exceed train range
        assert result.test["sensor_2"].max() > 1.0

    def test_norm_params_keys_match_feature_cols(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        assert set(result.norm_params.keys()) == set(result.feature_cols)

    def test_normalized_dtype_is_float32(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        for col in result.feature_cols:
            if col in result.train.columns:
                assert result.train[col].dtype == np.float32

    def test_zero_range_op_setting_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Zero-range op_setting column is skipped with a warning, not a raise."""
        data = _make_cmapss_data(n_train_units=3, train_cycles=20)
        modified_train = data.train.copy()
        modified_test = data.test.copy()
        modified_train["op_setting_1"] = 1.0  # constant op_setting
        modified_test["op_setting_1"] = 1.0
        data_with_const_op = CMAPSSData(
            subset=data.subset,
            train=modified_train,
            test=modified_test,
            rul=data.rul,
        )
        p = CMAPSSPreprocessor()
        with caplog.at_level(logging.WARNING, logger="src.data.preprocessor"):
            result = p.fit_transform(data_with_const_op)
        assert "Zero range" in caplog.text
        assert "op_setting_1" in caplog.text
        # Must not raise, must complete successfully
        assert isinstance(result, ProcessedData)
        # Behavioral consequence: the zero-range op_setting must actually
        # be excluded from both feature_cols and norm_params, not just
        # logged.
        assert "op_setting_1" not in result.feature_cols, (
            "Zero-range op_setting must be excluded from feature_cols"
        )
        assert "op_setting_1" not in result.norm_params, (
            "Zero-range op_setting must be excluded from norm_params"
        )


# ── Feature Columns ────────────────────────────────────────────────────────

class TestFeatureColumns:
    def test_feature_cols_contains_useful_sensors(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        for col in USEFUL_SENSORS:
            assert col in result.feature_cols

    def test_feature_cols_contains_op_settings(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        for col in OP_SETTING_COLUMNS:
            assert col in result.feature_cols

    def test_constant_sensors_not_in_feature_cols(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        for col in CONSTANT_SENSORS:
            assert col not in result.feature_cols

    def test_unit_and_cycle_not_in_feature_cols(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        result = preprocessor.fit_transform(raw_data)
        assert "unit" not in result.feature_cols
        assert "cycle" not in result.feature_cols


# ── Transform (Inference) ──────────────────────────────────────────────────

class TestTransform:
    def test_transform_before_fit_raises(self) -> None:
        p = CMAPSSPreprocessor()
        df = pd.DataFrame({"sensor_2": [1.0]})
        with pytest.raises(RuntimeError, match="fitted"):
            p.transform(df)

    def test_transform_after_fit_succeeds(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        preprocessor.fit_transform(raw_data)
        single_row = raw_data.test.iloc[[0]].copy()
        result = preprocessor.transform(single_row)
        assert isinstance(result, pd.DataFrame)
        for col in preprocessor._feature_cols:
            if col in result.columns:
                assert 0.0 - 1e-5 <= result[col].iloc[0] <= 1.0 + 1e-5, (
                    f"{col} not normalized after transform()"
                )

    def test_transform_does_not_mutate_input(
        self, preprocessor: CMAPSSPreprocessor, raw_data: CMAPSSData
    ) -> None:
        preprocessor.fit_transform(raw_data)
        single_row = raw_data.test.iloc[[0]].copy()
        original_single_row = single_row.copy()
        preprocessor.transform(single_row)
        pd.testing.assert_frame_equal(single_row, original_single_row)