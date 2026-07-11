"""
Unit tests for src/data/feature_engineer.py

Strategy:
    - Build ProcessedData in-memory using the same synthetic factory
      pattern as test_preprocessor.py
    - Tests cover feature naming, no cross-unit leakage, NaN handling,
      window size effects, row-order independence, and output container
      correctness
"""

import dataclasses

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
    DEFAULT_MAX_RUL,
    CMAPSSPreprocessor,
    ProcessedData,
)
from src.data.feature_engineer import (
    DEFAULT_WINDOW_SIZE,
    EngineeredData,
    FeatureEngineer,
)


# ── Synthetic Data Factory ─────────────────────────────────────────────────

def _make_processed_data(
    n_train_units: int = 5,
    train_cycles: int = 40,
    n_test_units: int = 3,
    test_cycles: int = 20,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> ProcessedData:
    """
    Build a synthetic ProcessedData object via the real preprocessor.
    Sensor values vary by unit+cycle to ensure non-zero variance.
    """
    rows_train, rows_test = [], []

    for unit in range(1, n_train_units + 1):
        for cycle in range(1, train_cycles + 1):
            op = [float(unit + cycle) * 0.1 * i for i in range(1, 4)]
            sensors = [float(unit * 10 + cycle + i) for i in range(21)]
            rows_train.append([unit, cycle] + op + sensors)

    for unit in range(1, n_test_units + 1):
        for cycle in range(1, test_cycles + 1):
            op = [float(unit + cycle) * 0.1 * i for i in range(1, 4)]
            sensors = [float(unit * 10 + cycle + i) for i in range(21)]
            rows_test.append([unit, cycle] + op + sensors)

    def _to_df(rows: list, include_rul: bool) -> pd.DataFrame:
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

    rul = pd.Series(
        [50 + i * 10 for i in range(n_test_units)],
        index=pd.RangeIndex(start=1, stop=n_test_units + 1),
        dtype=np.int32,
    )
    raw = CMAPSSData(
        subset=CMAPSSSubset.FD001,
        train=_to_df(rows_train, include_rul=True),
        test=_to_df(rows_test, include_rul=False),
        rul=rul,
    )
    return CMAPSSPreprocessor(max_rul=DEFAULT_MAX_RUL).fit_transform(raw)


@pytest.fixture()
def processed() -> ProcessedData:
    return _make_processed_data()


@pytest.fixture()
def engineer() -> FeatureEngineer:
    return FeatureEngineer(window_size=DEFAULT_WINDOW_SIZE)


# ── Initialization ─────────────────────────────────────────────────────────

class TestFeatureEngineerInit:
    def test_default_window_size(self) -> None:
        assert FeatureEngineer()._window_size == DEFAULT_WINDOW_SIZE

    def test_custom_window_size(self) -> None:
        assert FeatureEngineer(window_size=5)._window_size == 5

    def test_window_size_1_raises(self) -> None:
        with pytest.raises(ValueError, match="window_size must be >= 2"):
            FeatureEngineer(window_size=1)

    def test_window_size_0_raises(self) -> None:
        with pytest.raises(ValueError, match="window_size must be >= 2"):
            FeatureEngineer(window_size=0)

    def test_window_size_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="window_size must be >= 2"):
            FeatureEngineer(window_size=-5)


# ── Output Contract ────────────────────────────────────────────────────────

class TestOutputContract:
    def test_returns_engineered_data(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        assert isinstance(engineer.transform(processed), EngineeredData)

    def test_output_is_frozen(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        result = engineer.transform(processed)
        with pytest.raises(Exception):
            result.window_size = 999  # type: ignore[misc]

    def test_rul_series_unchanged(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        result = engineer.transform(processed)
        pd.testing.assert_series_equal(result.rul, processed.rul)

    def test_train_row_count_unchanged(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        result = engineer.transform(processed)
        assert len(result.train) == len(processed.train)

    def test_test_row_count_unchanged(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        result = engineer.transform(processed)
        assert len(result.test) == len(processed.test)

    def test_window_size_recorded(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        result = engineer.transform(processed)
        assert result.window_size == DEFAULT_WINDOW_SIZE

    def test_base_feature_cols_matches_input(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        result = engineer.transform(processed)
        assert result.base_feature_cols == processed.feature_cols


# ── Feature Naming ─────────────────────────────────────────────────────────

class TestFeatureNaming:
    def test_roll_mean_cols_present_in_train(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        result = engineer.transform(processed)
        w = DEFAULT_WINDOW_SIZE
        for col in processed.feature_cols:
            if "sensor" in col and "_roll" not in col and "_roc" not in col:
                assert f"{col}_roll_mean_{w}" in result.train.columns

    def test_roll_std_cols_present_in_train(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        result = engineer.transform(processed)
        w = DEFAULT_WINDOW_SIZE
        for col in processed.feature_cols:
            if "sensor" in col and "_roll" not in col and "_roc" not in col:
                assert f"{col}_roll_std_{w}" in result.train.columns

    def test_roc_cols_present_in_train(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        result = engineer.transform(processed)
        for col in processed.feature_cols:
            if "sensor" in col and "_roll" not in col and "_roc" not in col:
                assert f"{col}_roc" in result.train.columns

    def test_feature_cols_expanded(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        result = engineer.transform(processed)
        assert len(result.feature_cols) > len(processed.feature_cols)

    def test_base_features_in_feature_cols(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        result = engineer.transform(processed)
        for col in processed.feature_cols:
            assert col in result.feature_cols

    def test_same_features_in_train_and_test(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        result = engineer.transform(processed)
        train_cols = set(result.train.columns)
        test_cols = set(result.test.columns)
        # Test has no 'rul' column — check all feature cols are present
        for col in result.feature_cols:
            assert col in train_cols, f"{col} missing from train"
            assert col in test_cols, f"{col} missing from test"


# ── No Cross-Unit Leakage ──────────────────────────────────────────────────

class TestNoLeakage:
    def test_rolling_computed_per_unit(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        """
        First row of each unit must use only that unit's data.
        If rolling leaked across units, unit 2's first row would include
        unit 1's last cycles in its window — detectable by checking that
        each unit's cycle 1 roll_mean equals the raw sensor value
        (window of 1 at the start).
        """
        result = engineer.transform(processed)
        w = DEFAULT_WINDOW_SIZE

        # Pick a sensor that's present in feature cols
        sensor = next(
            col for col in result.base_feature_cols if "sensor" in col
        )
        roll_mean_col = f"{sensor}_roll_mean_{w}"

        for unit_id in result.train["unit"].unique():
            unit_rows = result.train[result.train["unit"] == unit_id]
            first_row = unit_rows.sort_values("cycle").iloc[0]
            # At cycle 1 (min_periods=1), roll_mean == raw sensor value
            assert abs(
                first_row[roll_mean_col] - first_row[sensor]
            ) < 1e-4, (
                f"Unit {unit_id}: roll_mean at cycle 1 should equal "
                f"raw sensor value — cross-unit leakage suspected"
            )

    def test_roc_first_cycle_is_nan_before_fill(
        self, processed: ProcessedData
    ) -> None:
        """
        Rate of change at the first cycle of each unit is diff=NaN
        before fill. After ffill+bfill, it should be 0.0 (since there's
        nothing to propagate forward to, bfill fills from the next value,
        but for a single-row edge case it stays 0).
        This test confirms NaN fill was applied — no NaN in output.
        """
        engineer = FeatureEngineer(window_size=2)
        result = engineer.transform(processed)

        sensor = next(
            col for col in result.base_feature_cols if "sensor" in col
        )
        roc_col = f"{sensor}_roc"

        assert not result.train[roc_col].isna().any(), (
            f"NaN found in {roc_col} after ffill+bfill"
        )


# ── Row Order Independence ─────────────────────────────────────────────────

class TestRowOrderIndependence:
    def test_shuffled_input_rows_produce_same_output(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        """
        _add_features explicitly sorts by [unit, cycle] as its first
        step, so feeding it the same underlying rows in a different
        order must not change the engineered feature values — only the
        row order of the *input* should be irrelevant, not the output.

        NOTE: reconstructed based on the review's description of this
        test (mandatory since pass 1); original file content was not
        available to copy verbatim.
        """
        shuffled_train = processed.train.sample(
            frac=1.0, random_state=42
        ).reset_index(drop=True)
        shuffled_test = processed.test.sample(
            frac=1.0, random_state=42
        ).reset_index(drop=True)

        shuffled_processed = dataclasses.replace(
            processed, train=shuffled_train, test=shuffled_test
        )

        baseline = FeatureEngineer(
            window_size=DEFAULT_WINDOW_SIZE
        ).transform(processed)
        shuffled_result = FeatureEngineer(
            window_size=DEFAULT_WINDOW_SIZE
        ).transform(shuffled_processed)

        baseline_sorted = baseline.train.sort_values(
            ["unit", "cycle"]
        ).reset_index(drop=True)
        shuffled_sorted = shuffled_result.train.sort_values(
            ["unit", "cycle"]
        ).reset_index(drop=True)

        pd.testing.assert_frame_equal(baseline_sorted, shuffled_sorted)


# ── NaN Handling ───────────────────────────────────────────────────────────

class TestNaNHandling:
    def test_no_nan_in_train_features(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        result = engineer.transform(processed)
        for col in result.feature_cols:
            if col in result.train.columns:
                assert not result.train[col].isna().any(), (
                    f"NaN found in train[{col}] after feature engineering"
                )

    def test_no_nan_in_test_features(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        result = engineer.transform(processed)
        for col in result.feature_cols:
            if col in result.test.columns:
                assert not result.test[col].isna().any(), (
                    f"NaN found in test[{col}] after feature engineering"
                )

    def test_short_unit_sequence_no_nan(self) -> None:
        """Unit with fewer cycles than window_size must still have no NaN."""
        processed = _make_processed_data(
            n_train_units=2,
            train_cycles=3,   # fewer than default window_size=10
            n_test_units=1,
            test_cycles=3,
        )
        engineer = FeatureEngineer(window_size=10)
        result = engineer.transform(processed)

        for col in result.feature_cols:
            if col in result.train.columns:
                assert not result.train[col].isna().any(), (
                    f"NaN found in train[{col}] for short unit sequence"
                )

    def test_single_cycle_unit_no_nan(self) -> None:
        """
        A unit with exactly one cycle has no window to roll over and no
        prior value to diff against. The fillna(0) safety net in
        _add_features must ensure this degenerate case still produces
        zero NaN in the engineered feature columns.

        NOTE: reconstructed based on the review's description of this
        test (mandatory since pass 1); original file content was not
        available to copy verbatim.
        """
        processed = _make_processed_data(
            n_train_units=2,
            train_cycles=1,
            n_test_units=1,
            test_cycles=1,
        )
        engineer = FeatureEngineer(window_size=DEFAULT_WINDOW_SIZE)
        result = engineer.transform(processed)

        for col in result.feature_cols:
            if col in result.train.columns:
                assert not result.train[col].isna().any(), (
                    f"NaN found in train[{col}] for single-cycle unit"
                )
            if col in result.test.columns:
                assert not result.test[col].isna().any(), (
                    f"NaN found in test[{col}] for single-cycle unit"
                )


# ── Feature Values ─────────────────────────────────────────────────────────

class TestFeatureValues:
    def test_roll_mean_dtype_is_float32(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        result = engineer.transform(processed)
        w = DEFAULT_WINDOW_SIZE
        sensor = next(
            col for col in result.base_feature_cols if "sensor" in col
        )
        assert result.train[f"{sensor}_roll_mean_{w}"].dtype == np.float32

    def test_roll_std_dtype_is_float32(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        result = engineer.transform(processed)
        w = DEFAULT_WINDOW_SIZE
        sensor = next(
            col for col in result.base_feature_cols if "sensor" in col
        )
        assert result.train[f"{sensor}_roll_std_{w}"].dtype == np.float32

    def test_roc_dtype_is_float32(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        result = engineer.transform(processed)
        sensor = next(
            col for col in result.base_feature_cols if "sensor" in col
        )
        assert result.train[f"{sensor}_roc"].dtype == np.float32

    def test_roll_std_non_negative(
        self, engineer: FeatureEngineer, processed: ProcessedData
    ) -> None:
        result = engineer.transform(processed)
        w = DEFAULT_WINDOW_SIZE
        for col in result.base_feature_cols:
            if "sensor" in col:
                std_col = f"{col}_roll_std_{w}"
                if std_col in result.train.columns:
                    assert (result.train[std_col] >= 0).all(), (
                        f"{std_col} has negative std values"
                    )

    def test_window_size_affects_smoothing(self) -> None:
        """Larger window = smoother signal = lower variance in roll_mean."""
        processed = _make_processed_data(train_cycles=50)
        sensor = next(
            col for col in processed.feature_cols if "sensor" in col
        )

        e_small = FeatureEngineer(window_size=2)
        e_large = FeatureEngineer(window_size=20)

        r_small = e_small.transform(processed)
        r_large = e_large.transform(processed)

        std_small = r_small.train[f"{sensor}_roll_mean_2"].std()
        std_large = r_large.train[f"{sensor}_roll_mean_20"].std()

        assert std_large < std_small, (
            "Larger window should produce smoother (lower std) roll_mean"
        )