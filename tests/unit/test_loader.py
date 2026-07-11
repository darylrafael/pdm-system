"""
Unit tests for src/data/loader.py

Strategy:
    - No real files required — all I/O mocked via tmp_path + monkeypatch
    - Tests cover happy path, error handling, edge cases, and validation
    - Fixtures generate synthetic CMAPSS-format data deterministically
"""

import logging
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.config import CMAPSSSubset
from src.data.loader import (
    CMAPSS_COLUMNS,
    SENSOR_COLUMNS,
    OP_SETTING_COLUMNS,
    CMAPSSData,
    CMAPSSLoader,
)


# ── Synthetic Data Generators ──────────────────────────────────────────────

def _make_sequence_df(n_units: int = 3, cycles_per_unit: int = 10) -> pd.DataFrame:
    """
    Build a synthetic CMAPSS sequence DataFrame.

    Produces small positive floats (~0.1-4.5 range) to mimic sensor scale.
    Using a DataFrame-first approach avoids fragile string formatting.
    """
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, cycles_per_unit + 1):
            op = [round(0.1 * i, 4) for i in range(1, 4)]
            # sensors: small positive floats, values vary by unit+cycle+sensor
            sensors = [round(float(unit + cycle + i) * 0.1, 4) for i in range(21)]
            rows.append([unit, cycle] + op + sensors)
    return pd.DataFrame(rows, columns=CMAPSS_COLUMNS)


def _write_sequence_file(path: Path, n_units: int = 3, cycles_per_unit: int = 10) -> None:
    """Write a synthetic CMAPSS sequence file to disk."""
    df = _make_sequence_df(n_units=n_units, cycles_per_unit=cycles_per_unit)
    df.to_csv(path, sep=" ", header=False, index=False)


def _write_rul_file(path: Path, n_units: int = 3) -> None:
    """Write a synthetic RUL file with known values: [50, 60, 70, ...]."""
    values = [50 + i * 10 for i in range(n_units)]
    path.write_text("\n".join(str(v) for v in values) + "\n")


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture()
def cmapss_dir(tmp_path: Path) -> Path:
    """
    Temporary directory with synthetic FD001 files.
    5 train units x 20 cycles, 3 test units x 15 cycles.
    """
    _write_sequence_file(tmp_path / "train_FD001.txt", n_units=5, cycles_per_unit=20)
    _write_sequence_file(tmp_path / "test_FD001.txt",  n_units=3, cycles_per_unit=15)
    _write_rul_file(tmp_path / "RUL_FD001.txt", n_units=3)
    return tmp_path


@pytest.fixture()
def loader(cmapss_dir: Path) -> CMAPSSLoader:
    """CMAPSSLoader pointed at the synthetic data directory."""
    with patch("src.data.loader.get_settings") as mock:
        mock.return_value.cmapss_subset = CMAPSSSubset.FD001
        mock.return_value.data_raw_path = str(cmapss_dir)
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
        assert isinstance(loader.load(), CMAPSSData)

    def test_train_has_correct_columns(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        assert set(result.train.columns) == set(CMAPSS_COLUMNS) | {"rul"}

    def test_test_does_not_have_rul_column(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        assert "rul" not in result.test.columns
        assert set(result.test.columns) == set(CMAPSS_COLUMNS)

    def test_train_row_count(self, loader: CMAPSSLoader) -> None:
        # 5 units x 20 cycles = 100
        assert len(loader.load().train) == 100

    def test_test_row_count(self, loader: CMAPSSLoader) -> None:
        # 3 units x 15 cycles = 45
        assert len(loader.load().test) == 45

    def test_rul_length_matches_test_units(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        assert len(result.rul) == result.test["unit"].nunique()

    def test_rul_index_starts_at_one(self, loader: CMAPSSLoader) -> None:
        assert loader.load().rul.index[0] == 1

    def test_rul_values_match_written_content(self, loader: CMAPSSLoader) -> None:
        # _write_rul_file writes [50, 60, 70] for n_units=3
        expected = [50, 60, 70]
        assert loader.load().rul.tolist() == expected

    def test_subset_is_preserved(self, loader: CMAPSSLoader) -> None:
        assert loader.load().subset == CMAPSSSubset.FD001


# ── RUL Calculation ────────────────────────────────────────────────────────

class TestRULCalculation:
    def test_rul_at_last_cycle_is_zero(self, loader: CMAPSSLoader) -> None:
        train = loader.load().train
        last_cycles = train.groupby("unit")["cycle"].max()
        for unit, max_cycle in last_cycles.items():
            last_row = train[(train["unit"] == unit) & (train["cycle"] == max_cycle)]
            assert last_row["rul"].iloc[0] == 0

    def test_rul_is_non_negative(self, loader: CMAPSSLoader) -> None:
        assert (loader.load().train["rul"] >= 0).all()

    def test_rul_decreases_monotonically_within_unit(self, loader: CMAPSSLoader) -> None:
        train = loader.load().train
        unit_1 = train[train["unit"] == 1].sort_values("cycle")
        rul_diff = unit_1["rul"].diff().dropna()
        assert (rul_diff <= 0).all()

    def test_rul_dtype_is_int32(self, loader: CMAPSSLoader) -> None:
        assert loader.load().train["rul"].dtype == np.int32


# ── Data Types ─────────────────────────────────────────────────────────────

class TestDataTypes:
    def test_unit_is_int32(self, loader: CMAPSSLoader) -> None:
        assert loader.load().train["unit"].dtype == np.int32

    def test_cycle_is_int32(self, loader: CMAPSSLoader) -> None:
        assert loader.load().train["cycle"].dtype == np.int32

    def test_sensor_columns_are_float32(self, loader: CMAPSSLoader) -> None:
        train = loader.load().train
        for col in SENSOR_COLUMNS:
            assert train[col].dtype == np.float32, f"{col} should be float32"

    def test_op_setting_columns_are_float32(self, loader: CMAPSSLoader) -> None:
        train = loader.load().train
        for col in OP_SETTING_COLUMNS:
            assert train[col].dtype == np.float32, f"{col} should be float32"


# ── Error Handling ─────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_missing_train_file_raises(self, cmapss_dir: Path) -> None:
        (cmapss_dir / "train_FD001.txt").unlink()
        with patch("src.data.loader.get_settings") as mock:
            mock.return_value.cmapss_subset = CMAPSSSubset.FD001
            mock.return_value.data_raw_path = str(cmapss_dir)
            loader = CMAPSSLoader(subset=CMAPSSSubset.FD001)
        with pytest.raises(FileNotFoundError, match="train_FD001.txt"):
            loader.load()

    def test_missing_test_file_raises(self, cmapss_dir: Path) -> None:
        (cmapss_dir / "test_FD001.txt").unlink()
        with patch("src.data.loader.get_settings") as mock:
            mock.return_value.cmapss_subset = CMAPSSSubset.FD001
            mock.return_value.data_raw_path = str(cmapss_dir)
            loader = CMAPSSLoader(subset=CMAPSSSubset.FD001)
        with pytest.raises(FileNotFoundError, match="test_FD001.txt"):
            loader.load()

    def test_missing_rul_file_raises(self, cmapss_dir: Path) -> None:
        (cmapss_dir / "RUL_FD001.txt").unlink()
        with patch("src.data.loader.get_settings") as mock:
            mock.return_value.cmapss_subset = CMAPSSSubset.FD001
            mock.return_value.data_raw_path = str(cmapss_dir)
            loader = CMAPSSLoader(subset=CMAPSSSubset.FD001)
        with pytest.raises(FileNotFoundError, match="RUL_FD001.txt"):
            loader.load()

    def test_rul_count_mismatch_raises(self, cmapss_dir: Path) -> None:
        # Write 2 RUL entries for 3 test units
        (cmapss_dir / "RUL_FD001.txt").write_text("50\n60\n")
        with patch("src.data.loader.get_settings") as mock:
            mock.return_value.cmapss_subset = CMAPSSSubset.FD001
            mock.return_value.data_raw_path = str(cmapss_dir)
            loader = CMAPSSLoader(subset=CMAPSSSubset.FD001)
        with pytest.raises(ValueError, match="RUL file has"):
            loader.load()

    def test_negative_cycles_raises(self, cmapss_dir: Path) -> None:
        # Build a corrupt file explicitly: unit=1, cycle=-1 guaranteed
        bad_df = _make_sequence_df(n_units=2, cycles_per_unit=5)
        bad_df.loc[0, "cycle"] = -1  # explicitly corrupt the cycle column
        bad_df.to_csv(cmapss_dir / "train_FD001.txt", sep=" ", header=False, index=False, na_rep='NaN')

        with patch("src.data.loader.get_settings") as mock:
            mock.return_value.cmapss_subset = CMAPSSSubset.FD001
            mock.return_value.data_raw_path = str(cmapss_dir)
            loader = CMAPSSLoader(subset=CMAPSSSubset.FD001)
        with pytest.raises(ValueError, match="Non-positive cycle"):
            loader.load()

    def test_empty_train_file_raises(self, cmapss_dir: Path) -> None:
        (cmapss_dir / "train_FD001.txt").write_text("")
        with patch("src.data.loader.get_settings") as mock:
            mock.return_value.cmapss_subset = CMAPSSSubset.FD001
            mock.return_value.data_raw_path = str(cmapss_dir)
            loader = CMAPSSLoader(subset=CMAPSSSubset.FD001)
        with pytest.raises(ValueError, match="empty"):
            loader.load()

    def test_wrong_column_count_raises(self, cmapss_dir: Path) -> None:
        # Write a file with only 10 columns (simulates all-null sensor dropout)
        bad_df = _make_sequence_df(n_units=2, cycles_per_unit=5).iloc[:, :10]
        bad_df.to_csv(cmapss_dir / "train_FD001.txt", sep=" ", header=False, index=False, na_rep='NaN')

        with patch("src.data.loader.get_settings") as mock:
            mock.return_value.cmapss_subset = CMAPSSSubset.FD001
            mock.return_value.data_raw_path = str(cmapss_dir)
            loader = CMAPSSLoader(subset=CMAPSSSubset.FD001)
        with pytest.raises(ValueError, match="Expected 26 columns"):
            loader.load()

    def test_non_sequential_unit_ids_raises(self, cmapss_dir: Path) -> None:
        # Test file with unit IDs {1, 2, 4} - gap at 3
        bad_df = _make_sequence_df(n_units=3, cycles_per_unit=5)
        bad_df.loc[bad_df["unit"] == 3, "unit"] = 4  # create gap
        bad_df.to_csv(cmapss_dir / "test_FD001.txt", sep=" ", header=False, index=False)

        with patch("src.data.loader.get_settings") as mock:
            mock.return_value.cmapss_subset = CMAPSSSubset.FD001
            mock.return_value.data_raw_path = str(cmapss_dir)
            loader = CMAPSSLoader(subset=CMAPSSSubset.FD001)
        with pytest.raises(ValueError, match="not sequential"):
            loader.load()


# ── Immutability ───────────────────────────────────────────────────────────

class TestImmutability:
    def test_cmapss_data_is_frozen(self, loader: CMAPSSLoader) -> None:
        result = loader.load()
        with pytest.raises(Exception):
            result.subset = CMAPSSSubset.FD002  # type: ignore[misc]


# ── Multi-Subset Smoke Test ────────────────────────────────────────────────

class TestMultiSubset:
    def test_fd002_subset_loads_correctly(self, tmp_path: Path) -> None:
        """Smoke test: subset parameterization wires FD002 files correctly."""
        _write_sequence_file(tmp_path / "train_FD002.txt", n_units=3, cycles_per_unit=10)
        _write_sequence_file(tmp_path / "test_FD002.txt",  n_units=2, cycles_per_unit=8)
        _write_rul_file(tmp_path / "RUL_FD002.txt", n_units=2)

        with patch("src.data.loader.get_settings") as mock:
            mock.return_value.cmapss_subset = CMAPSSSubset.FD002
            mock.return_value.data_raw_path = str(tmp_path)
            loader = CMAPSSLoader(subset=CMAPSSSubset.FD002)

        result = loader.load()
        assert result.subset == CMAPSSSubset.FD002
        assert len(result.train) == 30   # 3 units x 10 cycles
        assert len(result.rul) == 2


# ── Format Edge Cases ──────────────────────────────────────────────────────

class TestFormatEdgeCases:
    def test_trailing_null_column_is_dropped(self, cmapss_dir: Path) -> None:
        """27-column file (CMAPSS trailing-space quirk) loads correctly as 26 columns."""
        df_with_phantom = _make_sequence_df(n_units=2, cycles_per_unit=5)
        df_with_phantom["_phantom"] = np.nan  # simulate all-null trailing column
        df_with_phantom.to_csv(
            cmapss_dir / "train_FD001.txt", sep=" ", header=False, index=False,
            na_rep='NaN'
        )
        _write_sequence_file(cmapss_dir / "test_FD001.txt", n_units=2, cycles_per_unit=5)
        _write_rul_file(cmapss_dir / "RUL_FD001.txt", n_units=2)

        with patch("src.data.loader.get_settings") as mock:
            mock.return_value.cmapss_subset = CMAPSSSubset.FD001
            mock.return_value.data_raw_path = str(cmapss_dir)
            loader = CMAPSSLoader(subset=CMAPSSSubset.FD001)

        result = loader.load()
        assert set(result.train.columns) == set(CMAPSS_COLUMNS) | {"rul"}

    def test_high_nan_sensor_triggers_warning(
        self, cmapss_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Sensor column with >5% NaN ratio triggers a warning log."""

        df = _make_sequence_df(n_units=2, cycles_per_unit=10)
        df.loc[df.index[1:], "sensor_1"] = np.nan  # >5% NaN on sensor_1
        df.to_csv(cmapss_dir / "train_FD001.txt", sep=" ", header=False, index=False, na_rep='NaN')
        _write_sequence_file(cmapss_dir / "test_FD001.txt", n_units=2, cycles_per_unit=5)
        _write_rul_file(cmapss_dir / "RUL_FD001.txt", n_units=2)

        with patch("src.data.loader.get_settings") as mock:
            mock.return_value.cmapss_subset = CMAPSSSubset.FD001
            mock.return_value.data_raw_path = str(cmapss_dir)
            loader = CMAPSSLoader(subset=CMAPSSSubset.FD001)

        with caplog.at_level(logging.WARNING, logger="src.data.loader"):
            loader.load()

        assert "High NaN ratio" in caplog.text
        assert "sensor_1" in caplog.text