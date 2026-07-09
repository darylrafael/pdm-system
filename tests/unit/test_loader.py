"""
Unit tests for src/data/loader.py

Strategy:
    - No real files required — all I/O mocked via tmp_path + monkeypatch
    - Tests cover happy path, error handling, edge cases, and validation
    - Fixtures generate synthetic CMAPSS-format data
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from src.config import CMAPSSSubset
from src.data.loader import (
    CMAPSS_COLUMNS,
    SENSOR_COLUMNS,
    OP_SETTING_COLUMNS,
    CMAPSSData,
    CMAPSSLoader,
)


# ── Synthetic Data Generators ──────────────────────────────────────────────

def _make_sequence_content(n_units: int = 3, cycles_per_unit: int = 10) -> str:
    """
    Generate synthetic CMAPSS sequence file content.

    Args:
        n_units: Number of simulated engine units.
        cycles_per_unit: Number of cycles per unit.

    Returns:
        String content mimicking a real CMAPSS .txt file.
    """
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, cycles_per_unit + 1):
            op = [round(0.1 * i, 4) for i in range(1, 4)]
            sensors = [round(float(unit + cycle + i) * 0.1, 4) for i in range(21)]
            row = [unit, cycle] + op + sensors
            rows.append(" ".join(str(v) for v in row))
    return "\n".join(rows) + "\n"


def _make_rul_content(n_units: int = 3) -> str:
    """Generate synthetic RUL file content."""
    return "\n".join(str(50 + i * 10) for i in range(n_units)) + "\n"


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture()
def cmapss_dir(tmp_path: Path) -> Path:
    """
    Create a temporary directory with synthetic CMAPSS FD001 files.

    Returns:
        Path to directory containing train/test/RUL files.
    """
    subset = "FD001"
    (tmp_path / f"train_{subset}.txt").write_text(_make_sequence_content(n_units=5, cycles_per_unit=20))
    (tmp_path / f"test_{subset}.txt").write_text(_make_sequence_content(n_units=3, cycles_per_unit=15))
    (tmp_path / f"RUL_{subset}.txt").write_text(_make_rul_content(n_units=3))
    return tmp_path


@pytest.fixture()
def loader(cmapss_dir: Path) -> CMAPSSLoader:
    """CMAPSSLoader pointed at synthetic data directory."""
    with patch("src.data.loader.get_settings") as mock_settings:
        mock_settings.return_value.cmapss_subset = CMAPSSSubset.FD001
        mock_settings.return_value.data_raw_path = str(cmapss_dir)
        return CMAPSSLoader(subset=CMAPSSSubset.FD001)


# ── Column Definitions ─────────────────────────────────────────────────────

class TestColumnDefinitions:
    def test_cmapss_columns_count(self) -> None:
        assert len(CMAPSS_COLUMNS) == 26

    def test_sensor_columns_count(self) -> None:
        assert len(SENSOR_COLUMNS) == 21

    def test_op_setting_columns_count(self) -> None:
        assert len(OP_SETTING_COLUMNS) == 3

    def test_column_names_are_unique(self) -> None:
        assert len(CMAPSS_COLUMNS) == len(set(CMAPSS_COLUMNS))

    def test_first_two_columns_are_unit_and_cycle(self) -> None:
        assert CMAPSS_COLUMNS[0] == "unit"
        assert CMAPSS_COLUMNS[1] == "cycle"


# ── Happy Path ─────────────────────────────────────────────────────────────

class TestCMAPSSLoaderHappyPath:
    def test_load_returns_cmapss_data_instance(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        assert isinstance(result, CMAPSSData)

    def test_train_has_correct_columns(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        expected = set(CMAPSS_COLUMNS) | {"rul"}
        assert set(result.train.columns) == expected

    def test_test_has_correct_columns(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        # Test set has no RUL column — that's the prediction target
        assert set(result.test.columns) == set(CMAPSS_COLUMNS)
        assert "rul" not in result.test.columns

    def test_train_has_correct_row_count(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        # 5 units × 20 cycles = 100 rows
        assert len(result.train) == 100

    def test_test_has_correct_row_count(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        # 3 units × 15 cycles = 45 rows
        assert len(result.test) == 45

    def test_rul_series_length_matches_test_units(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        assert len(result.rul) == result.test["unit"].nunique()

    def test_rul_index_starts_at_one(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        assert result.rul.index[0] == 1

    def test_subset_is_preserved(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        assert result.subset == CMAPSSSubset.FD001


# ── RUL Calculation ────────────────────────────────────────────────────────

class TestRULCalculation:
    def test_rul_at_last_cycle_is_zero(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        last_cycles = result.train.groupby("unit")["cycle"].max()
        last_rows = result.train.merge(
            last_cycles.rename("max_cycle"),
            left_on="unit",
            right_index=True,
        )
        last_rows = last_rows[last_rows["cycle"] == last_rows["max_cycle"]]
        assert (last_rows["rul"] == 0).all()

    def test_rul_is_non_negative(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        assert (result.train["rul"] >= 0).all()

    def test_rul_decreases_within_unit(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        unit_1 = result.train[result.train["unit"] == 1].sort_values("cycle")
        rul_diff = unit_1["rul"].diff().dropna()
        assert (rul_diff <= 0).all()  # RUL must be monotonically decreasing

    def test_rul_dtype_is_integer(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        assert result.train["rul"].dtype == np.int32


# ── Data Types ─────────────────────────────────────────────────────────────

class TestDataTypes:
    def test_unit_column_is_integer(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        assert result.train["unit"].dtype == np.int32

    def test_cycle_column_is_integer(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        assert result.train["cycle"].dtype == np.int32

    def test_sensor_columns_are_float32(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        for col in SENSOR_COLUMNS:
            assert result.train[col].dtype == np.float32, f"{col} should be float32"


# ── Error Handling ─────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_missing_train_file_raises_file_not_found(
        self, cmapss_dir: Path
    ) -> None:
        (cmapss_dir / "train_FD001.txt").unlink()
        with patch("src.data.loader.get_settings") as mock_settings:
            mock_settings.return_value.cmapss_subset = CMAPSSSubset.FD001
            mock_settings.return_value.data_raw_path = str(cmapss_dir)
            loader = CMAPSSLoader(subset=CMAPSSSubset.FD001)

        with pytest.raises(FileNotFoundError, match="train_FD001.txt"):
            loader.load()

    def test_missing_rul_file_raises_file_not_found(
        self, cmapss_dir: Path
    ) -> None:
        (cmapss_dir / "RUL_FD001.txt").unlink()
        with patch("src.data.loader.get_settings") as mock_settings:
            mock_settings.return_value.cmapss_subset = CMAPSSSubset.FD001
            mock_settings.return_value.data_raw_path = str(cmapss_dir)
            loader = CMAPSSLoader(subset=CMAPSSSubset.FD001)

        with pytest.raises(FileNotFoundError, match="RUL_FD001.txt"):
            loader.load()

    def test_rul_mismatch_raises_value_error(self, cmapss_dir: Path) -> None:
        # Write RUL file with wrong number of entries (2 instead of 3)
        (cmapss_dir / "RUL_FD001.txt").write_text("50\n60\n")
        with patch("src.data.loader.get_settings") as mock_settings:
            mock_settings.return_value.cmapss_subset = CMAPSSSubset.FD001
            mock_settings.return_value.data_raw_path = str(cmapss_dir)
            loader = CMAPSSLoader(subset=CMAPSSSubset.FD001)

        with pytest.raises(ValueError, match="RUL file has"):
            loader.load()

    def test_negative_cycles_raises_value_error(self, cmapss_dir: Path) -> None:
        # Corrupt train file with a negative cycle
        content = _make_sequence_content(n_units=3, cycles_per_unit=5)
        corrupted = content.replace("1 1 ", "1 -1 ", 1)
        (cmapss_dir / "train_FD001.txt").write_text(corrupted)

        with patch("src.data.loader.get_settings") as mock_settings:
            mock_settings.return_value.cmapss_subset = CMAPSSSubset.FD001
            mock_settings.return_value.data_raw_path = str(cmapss_dir)
            loader = CMAPSSLoader(subset=CMAPSSSubset.FD001)

        with pytest.raises(ValueError, match="Non-positive cycle"):
            loader.load()


# ── Immutability ───────────────────────────────────────────────────────────

class TestImmutability:
    def test_cmapss_data_is_frozen(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        with pytest.raises(Exception):
            result.subset = CMAPSSSubset.FD002  # type: ignore[misc]