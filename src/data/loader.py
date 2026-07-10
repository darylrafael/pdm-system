"""
Raw data loader for NASA CMAPSS (C-MAPSS) dataset.

Handles loading of train, test, and RUL files for all four subsets
(FD001-FD004). This module is responsible ONLY for raw ingestion -
no preprocessing, normalization, or feature engineering occurs here.

Dataset structure:
    - Space-separated .txt files, no header row
    - 26 columns: unit, cycle, 3 operational settings, 21 sensor readings
    - Train file: full run-to-failure sequences
    - Test file:  truncated sequences (RUL prediction target)
    - RUL file:   true Remaining Useful Life for each unit in test set

Reference:
    Saxena, A. et al. (2008). Damage propagation modeling for aircraft
    engine run-to-failure simulation. PHM Conference.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.config import CMAPSSSubset, get_settings

__all__ = [
    "CMAPSSData",
    "CMAPSSLoader",
    "CMAPSS_COLUMNS",
    "SENSOR_COLUMNS",
    "OP_SETTING_COLUMNS",
]

logger = logging.getLogger(__name__)

# ── Column Definitions ─────────────────────────────────────────────────────

#: All 26 column names in the order they appear in CMAPSS files
CMAPSS_COLUMNS: list[str] = (
    ["unit", "cycle"]
    + [f"op_setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

#: Sensor columns only (used frequently in downstream modules)
SENSOR_COLUMNS: list[str] = [f"sensor_{i}" for i in range(1, 22)]

#: Operational setting columns
OP_SETTING_COLUMNS: list[str] = [f"op_setting_{i}" for i in range(1, 4)]

#: Valid split literals for internal use
_SplitLiteral = Literal["train", "test", "RUL"]

#: Threshold for high-NaN ratio warning in sensor columns
_HIGH_NAN_THRESHOLD: float = 0.05


# ── Data Containers ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CMAPSSData:
    """
    Immutable container for a loaded CMAPSS subset.

    Note: frozen=True prevents attribute reassignment, but the
    pd.DataFrame objects held by train and test are themselves mutable.
    Downstream consumers must call .copy() before in-place mutation.

    Attributes:
        subset: Dataset subset identifier (e.g. FD001)
        train:  Training DataFrame with RUL column appended.
                shape: (N, 27) - 26 CMAPSS columns + rul
        test:   Test DataFrame (truncated sequences, no RUL).
                shape: (M, 26) - 26 CMAPSS columns only
        rul:    True RUL values for each unit in test set.
                1-indexed to match CMAPSS unit numbering convention.
    """

    subset: CMAPSSSubset
    train: pd.DataFrame
    test: pd.DataFrame
    rul: pd.Series


# ── Loader ─────────────────────────────────────────────────────────────────


class CMAPSSLoader:
    """
    Loads raw NASA CMAPSS dataset files into typed DataFrames.

    Responsibilities:
        - Locate files based on configured data path and subset
        - Parse space-separated .txt files with correct column names
        - Calculate RUL for training data (max_cycle - current_cycle)
        - Validate loaded data for basic sanity checks

    Does NOT:
        - Normalize or scale any values  -> preprocessor.py
        - Engineer features              -> feature_engineer.py
        - Split train/validation         -> trainer.py

    Usage:
        loader = CMAPSSLoader()
        data = loader.load()
        print(data.train.head())
    """

    def __init__(self, subset: CMAPSSSubset | None = None) -> None:
        """
        Initialize loader with optional subset override.

        Args:
            subset: Override the subset from settings.
                    Useful for testing with different subsets.
        """
        self._settings = get_settings()
        self._subset = subset or self._settings.cmapss_subset
        self._raw_path = Path(self._settings.data_raw_path)

    # ── Public Interface ───────────────────────────────────────────────────

    def load(self) -> CMAPSSData:
        """
        Load all three CMAPSS files for the configured subset.

        Synchronous. In async contexts, wrap with:
            await asyncio.to_thread(loader.load)

        Returns:
            CMAPSSData: Immutable container with train, test, and RUL data.

        Raises:
            FileNotFoundError: If any required file is missing.
            ValueError: If loaded data fails sanity checks.
        """
        logger.info(
            "Loading CMAPSS subset %s from %s",
            self._subset.value,
            self._raw_path,
        )

        train_df = self._load_sequences(split="train")
        test_df = self._load_sequences(split="test")
        rul_series = self._load_rul()

        train_df = self._append_rul(train_df)

        self._validate(train_df, test_df, rul_series)

        logger.info(
            "Loaded %s - train: %d rows (%d units) | test: %d rows (%d units) | RUL: %d values",
            self._subset.value,
            len(train_df),
            train_df["unit"].nunique(),
            len(test_df),
            test_df["unit"].nunique(),
            len(rul_series),
        )

        return CMAPSSData(
            subset=self._subset,
            train=train_df,
            test=test_df,
            rul=rul_series,
        )

    # ── Private Helpers ────────────────────────────────────────────────────

    def _resolve_path(self, split: _SplitLiteral) -> Path:
        """
        Resolve the file path for a given split.

        Args:
            split: One of 'train', 'test', or 'RUL'

        Returns:
            Resolved absolute path.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        filename = f"{split}_{self._subset.value}.txt"
        path = self._raw_path / filename

        if not path.exists():
            raise FileNotFoundError(
                f"CMAPSS file not found: {path}\n"
                f"Download from: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps\n"
                f"Place files in: {self._raw_path.resolve()}"
            )

        return path

    def _load_sequences(self, split: Literal["train", "test"]) -> pd.DataFrame:
        """
        Load train or test sequence file.

        Args:
            split: 'train' or 'test'

        Returns:
            DataFrame with exactly 26 named columns, dtypes enforced.

        Raises:
            ValueError: If the file is empty or has unexpected column count.
        """
        path = self._resolve_path(split)

        df = pd.read_csv(
            path,
            sep=r"\s+",  # handles variable whitespace + trailing spaces
            header=None,
            names=CMAPSS_COLUMNS,
            dtype={
                "unit": np.int32,
                "cycle": np.int32,
                **{col: np.float32 for col in OP_SETTING_COLUMNS},
                **{col: np.float32 for col in SENSOR_COLUMNS},
            },
        )

        # Guard: empty file produces a zero-row DataFrame with misleading state
        if df.empty:
            raise ValueError(
                f"CMAPSS file is empty: {path}\n"
                "Download a fresh copy from Kaggle."
            )

        # CMAPSS files sometimes have a trailing empty column - drop it
        df = df.dropna(axis=1, how="all")

        # Assert exactly 26 columns remain after dropping phantom columns.
        # A real sensor column that is entirely NaN would be silently dropped
        # here; catching it now prevents a cryptic KeyError in preprocessing.
        if len(df.columns) != len(CMAPSS_COLUMNS):
            raise ValueError(
                f"Expected {len(CMAPSS_COLUMNS)} columns after parsing, "
                f"got {len(df.columns)}. File may be malformed: {path}"
            )

        # This is also the ONLY place all-null columns are caught — see
        # _validate(), which relies on this guard already having fired.

        logger.debug("Loaded %s split: %d rows, %d columns", split, len(df), len(df.columns))
        return df

    def _load_rul(self) -> pd.Series:
        """
        Load the RUL ground truth file for the test set.

        Returns:
            Series of RUL values, 1-indexed to match unit numbers.

        Raises:
            FileNotFoundError: If RUL file is missing.
        """
        path = self._resolve_path("RUL")

        rul = pd.read_csv(
            path,
            header=None,
            names=["rul"],
            dtype={"rul": np.int32},
        )

        # Index from 1 to match CMAPSS unit numbering convention
        rul.index = pd.RangeIndex(start=1, stop=len(rul) + 1, step=1)

        return rul["rul"]

    @staticmethod
    def _append_rul(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate and append RUL column to training DataFrame.

        RUL at each row = max_cycle_for_that_unit - current_cycle

        Args:
            df: Training DataFrame without RUL column.

        Returns:
            DataFrame with 'rul' column appended.

        Raises:
            ValueError: If the input DataFrame is empty.
        """
        if df.empty:
            raise ValueError("Training DataFrame is empty — cannot compute RUL.")

        max_cycles = df.groupby("unit")["cycle"].transform("max")
        df = df.copy()
        df["rul"] = (max_cycles - df["cycle"]).astype(np.int32)
        return df

    def _validate(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        rul: pd.Series,
    ) -> None:
        """
        Run sanity checks on loaded data.

        Args:
            train: Training DataFrame with RUL appended.
            test:  Test DataFrame.
            rul:   RUL Series for test units.

        Raises:
            ValueError: If any critical check fails.
        """
        # RUL count must match number of test units
        n_test_units = test["unit"].nunique()
        if len(rul) != n_test_units:
            raise ValueError(
                f"RUL file has {len(rul)} entries but test set has "
                f"{n_test_units} units. Data may be corrupted."
            )

        # Test unit IDs must be sequential from 1 to n
        # (guards against gaps from file corruption e.g. {1, 2, 4})
        expected_units = set(range(1, n_test_units + 1))
        actual_units = set(test["unit"].unique())
        if expected_units != actual_units:
            raise ValueError(
                f"Test unit IDs are not sequential from 1: {sorted(actual_units)}"
            )

        # Training RUL must be non-negative
        if (train["rul"] < 0).any():
            raise ValueError("Negative RUL values detected in training data.")

        # Ground truth RUL must be non-negative
        if (rul < 0).any():
            raise ValueError("Negative values detected in RUL ground truth file.")

        # Cycle numbers must be positive
        if (train["cycle"] <= 0).any() or (test["cycle"] <= 0).any():
            raise ValueError("Non-positive cycle numbers detected.")

        # All-null columns are caught upstream in _load_sequences — see
        # the dropna + column count assertion there. No check is needed
        # (and none is reachable) here.

        # Scoped to SENSOR_COLUMNS only: OP_SETTING_COLUMNS represent
        # discrete flight condition clusters in CMAPSS, where partial NaN
        # has a different interpretation and isn't a data-quality signal.
        # Warn on high-NaN sensor columns (non-fatal: log only)
        high_nan = [
            col for col in SENSOR_COLUMNS
            if train[col].isnull().mean() > _HIGH_NAN_THRESHOLD
        ]
        if high_nan:
            logger.warning(
                "High NaN ratio (>%.0f%%) detected in columns: %s — "
                "review before feature engineering.",
                _HIGH_NAN_THRESHOLD * 100,
                high_nan,
            )

        logger.debug("Data validation passed.")