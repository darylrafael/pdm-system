"""
Data preprocessor for NASA CMAPSS dataset.

Receives raw CMAPSSData from the loader and produces clean, normalized
DataFrames ready for feature engineering. Enforces strict no-leakage
discipline: normalization parameters are fit on train set only, then
applied to the test set.

Responsibilities:
    - Drop constant/near-zero-variance sensor columns (uninformative)
    - Clip RUL to a maximum value (piecewise linear target — standard
      practice in CMAPSS literature to reduce noise from healthy cycles)
    - Fit Min-Max normalization on train set
    - Apply fitted normalization to test set (no leakage)
    - Return immutable ProcessedData container

Does NOT:
    - Compute rolling statistics        -> feature_engineer.py
    - Split train into train/validation -> trainer.py
    - Perform any model training        -> trainer.py

Reference for constant sensor list:
    Heimes, F.O. (2008). Recurrent neural networks for remaining useful
    life estimation. PHM Conference.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.loader import (
    CMAPSSData,
    OP_SETTING_COLUMNS,
    SENSOR_COLUMNS,
)

__all__ = [
    "CMAPSSPreprocessor",
    "ProcessedData",
    "CONSTANT_SENSORS",
    "USEFUL_SENSORS",
    "DEFAULT_MAX_RUL",
]

logger = logging.getLogger(__name__)

# ── Sensor Definitions ─────────────────────────────────────────────────────

#: Sensors with near-zero variance in FD001 — carry no predictive signal.
#: Per Heimes (2008) and Saxena (2008) analysis of CMAPSS FD001.
#: Note: FD002/FD003/FD004 have multiple operating conditions — some of
#: these sensors may become informative there. Revisit if expanding subsets.
CONSTANT_SENSORS: list[str] = [
    "sensor_1",   # fan inlet temperature — constant in FD001
    "sensor_5",   # physical fan speed — constant in FD001
    "sensor_6",   # static pressure — constant in FD001
    "sensor_10",  # corrected fan speed — constant in FD001
    "sensor_16",  # bypass ratio — constant in FD001
    "sensor_18",  # bleed enthalpy — constant in FD001
    "sensor_19",  # demanded fan speed — constant in FD001
]

#: Sensors retained after dropping constants
USEFUL_SENSORS: list[str] = [
    s for s in SENSOR_COLUMNS if s not in CONSTANT_SENSORS
]

#: Maximum RUL cap for piecewise linear target (standard in CMAPSS literature)
#: Cycles beyond this are equally "healthy" — capping reduces noise from
#: early degradation data where sensor readings are indistinguishable.
DEFAULT_MAX_RUL: int = 125

#: Variance threshold below which a sensor is considered constant
_VARIANCE_THRESHOLD: float = 1e-6


# ── Output Container ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProcessedData:
    """
    Immutable container for preprocessed CMAPSS data.

    All DataFrames retain original columns minus constant sensors.
    Sensor values are Min-Max normalized to [0, 1].
    Training RUL is clipped to max_rul.

    Note: frozen=True prevents attribute reassignment, but DataFrames
    are mutable. Downstream consumers must .copy() before in-place ops.

    Invariant: norm_params.keys() always equals set(feature_cols) exactly.
    Any op_setting column found to have zero range (min == max) in the
    training data is excluded from both feature_cols and norm_params,
    since it carries no normalizable signal. Downstream code may safely
    assume every column in feature_cols has a corresponding entry in
    norm_params.

    Attributes:
        train:           Normalized training DataFrame with clipped RUL.
        test:            Normalized test DataFrame (no RUL column).
        rul:             Ground truth RUL Series for test units (unchanged).
        feature_cols:    Ordered list of columns used as model input
                          features. Always matches norm_params.keys()
                          exactly (see norm_params).
        norm_params:     {col: {"min": float, "max": float}} fit on train
                          only. Keys always match feature_cols exactly;
                          a zero-range op_setting column is excluded from
                          both rather than included at raw scale.
        max_rul:         RUL cap value applied during preprocessing.
        dropped_sensors: Sensor columns removed due to near-zero variance.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    rul: pd.Series
    feature_cols: list[str]
    norm_params: dict[str, dict[str, float]]
    max_rul: int
    dropped_sensors: list[str]


# ── Preprocessor ───────────────────────────────────────────────────────────


class CMAPSSPreprocessor:
    """
    Cleans and normalizes raw CMAPSSData for downstream modeling.

    Usage:
        loader = CMAPSSLoader()
        raw = loader.load()

        preprocessor = CMAPSSPreprocessor(max_rul=125)
        processed = preprocessor.fit_transform(raw)

        # Access results
        X_train = processed.train[processed.feature_cols]
        y_train = processed.train["rul"]
        X_test  = processed.test[processed.feature_cols]
    """

    def __init__(self, max_rul: int = DEFAULT_MAX_RUL) -> None:
        """
        Initialize preprocessor.

        Args:
            max_rul: RUL cap for piecewise linear target. Must be positive.

        Raises:
            ValueError: If max_rul is not a positive integer.
        """
        if max_rul <= 0:
            raise ValueError(f"max_rul must be positive, got {max_rul}")

        self._max_rul = max_rul
        self._norm_params: dict[str, dict[str, float]] = {}
        self._dropped_sensors: list[str] = []
        self._feature_cols: list[str] = []
        self._is_fitted: bool = False

    # ── Public Interface ───────────────────────────────────────────────────

    def fit_transform(self, data: CMAPSSData) -> ProcessedData:
        """
        Fit normalization on training data and transform both splits.

        Steps:
            1. Drop constant/near-zero-variance sensor columns
            2. Clip training RUL to max_rul
            3. Define feature columns (post drop)
            4. Fit Min-Max params on train only  (no leakage) — any
               zero-range op_setting column is excluded from feature_cols
               here as well, keeping feature_cols and norm_params in sync
            5. Apply normalization to both train and test

        Synchronous. In async contexts, wrap with:
            await asyncio.to_thread(preprocessor.fit_transform, data)

        Args:
            data: Raw CMAPSSData from CMAPSSLoader.

        Returns:
            ProcessedData: Normalized, feature-ready container.

        Raises:
            ValueError: If data is malformed or normalization fails.
        """
        logger.info(
            "Preprocessing CMAPSS %s (max_rul=%d)",
            data.subset.value,
            self._max_rul,
        )

        train = data.train.copy()
        test = data.test.copy()

        # Step 1 — Drop uninformative sensors
        train, test, self._dropped_sensors = self._drop_constant_sensors(
            train, test
        )

        # Step 2 — Clip RUL to piecewise linear cap
        train = self._clip_rul(train)

        # Step 3 — Define feature columns (post drop)
        self._feature_cols = [
            col for col in USEFUL_SENSORS
            if col not in self._dropped_sensors
        ] + OP_SETTING_COLUMNS

        # Step 4 — Fit normalization on train only
        self._fit_normalization(train)

        # Zero-range op_setting columns are excluded from norm_params by
        # _fit_normalization; keep feature_cols consistent with
        # norm_params so the two always share the same key set.
        self._feature_cols = [
            col for col in self._feature_cols if col in self._norm_params
        ]

        # Step 5 — Apply to both splits
        train = self._apply_normalization(train)
        test = self._apply_normalization(test)

        self._is_fitted = True

        logger.info(
            "Preprocessing complete — %d features | dropped: %s | RUL cap: %d",
            len(self._feature_cols),
            self._dropped_sensors or "none",
            self._max_rul,
        )

        return ProcessedData(
            train=train,
            test=test,
            rul=data.rul,
            feature_cols=list(self._feature_cols),
            norm_params={k: dict(v) for k, v in self._norm_params.items()},
            max_rul=self._max_rul,
            dropped_sensors=list(self._dropped_sensors),
        )

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted normalization to new data at inference time.

        Must call fit_transform() before this method.

        Args:
            df: Raw DataFrame with same schema as training data.

        Returns:
            Normalized DataFrame.

        Raises:
            RuntimeError: If called before fit_transform().
        """
        if not self._is_fitted:
            raise RuntimeError(
                "CMAPSSPreprocessor must be fitted before calling transform(). "
                "Call fit_transform() first."
            )
        return self._apply_normalization(df.copy())

    # ── Private Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _drop_constant_sensors(
        train: pd.DataFrame,
        test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Remove sensor columns with near-zero variance.

        Two-pass strategy:
            Pass 1: Drop known CMAPSS constants (literature-based)
            Pass 2: Drop any remaining data-driven zero-variance columns
                    (robustness for non-FD001 subsets or corrupted data)

        Args:
            train: Training DataFrame.
            test:  Test DataFrame.

        Returns:
            (cleaned_train, cleaned_test, dropped_column_names)
        """
        dropped: list[str] = []

        # Pass 1 — Known constants from CMAPSS literature
        known_present = [
            col for col in CONSTANT_SENSORS if col in train.columns
        ]
        train = train.drop(columns=known_present)
        test = test.drop(columns=known_present, errors="ignore")
        dropped.extend(known_present)

        # Pass 2 — Data-driven detection of remaining near-zero-variance cols
        remaining = [col for col in SENSOR_COLUMNS if col in train.columns]
        data_driven = [
            col for col in remaining
            if train[col].var() < _VARIANCE_THRESHOLD
        ]

        if data_driven:
            logger.warning(
                "Data-driven variance check found additional constant "
                "columns: %s — dropping.",
                data_driven,
            )
            train = train.drop(columns=data_driven)
            test = test.drop(columns=data_driven, errors="ignore")
            dropped.extend(data_driven)

        logger.debug("Dropped %d constant sensor(s): %s", len(dropped), dropped)
        return train, test, dropped

    def _clip_rul(self, train: pd.DataFrame) -> pd.DataFrame:
        """
        Clip RUL values to self._max_rul (piecewise linear target).

        Args:
            train: Training DataFrame with 'rul' column.

        Returns:
            DataFrame with RUL clipped to [0, max_rul].
        """
        train = train.copy()
        original_max = int(train["rul"].max())
        train["rul"] = train["rul"].clip(upper=self._max_rul).astype(np.int32)
        logger.debug(
            "RUL clipped: original max=%d → capped at %d",
            original_max,
            self._max_rul,
        )
        return train

    def _fit_normalization(self, train: pd.DataFrame) -> None:
        """
        Compute Min-Max normalization params from training data.

        Stores params in self._norm_params as:
            {col_name: {"min": float, "max": float}}

        Only normalizes sensor + op_setting columns.
        unit, cycle, and rul are left in their original scale.

        Sensor columns are expected to have already been screened for
        zero-range (constant) values by _drop_constant_sensors(); a
        zero-range sensor column here indicates that screening was
        bypassed and is treated as an error. Op_setting columns are NOT
        screened by _drop_constant_sensors() (which is sensor-only), so
        a zero-range op_setting column is logged and skipped instead of
        raising — it is excluded from norm_params entirely (see the
        feature_cols/norm_params invariant note in fit_transform, which
        subsequently removes any such column from self._feature_cols so
        the two stay in sync).

        Args:
            train: Training DataFrame (post constant-sensor removal).

        Raises:
            ValueError: If any sensor column has min == max (undropped
                constant sensor).
        """
        cols_to_normalize = [
            col for col in self._feature_cols
            if col in train.columns
        ]

        self._norm_params = {}
        zero_range: list[str] = []

        for col in cols_to_normalize:
            col_min = float(train[col].min())
            col_max = float(train[col].max())

            if col_min == col_max:
                zero_range.append(col)
                continue

            self._norm_params[col] = {"min": col_min, "max": col_max}

        if zero_range:
            sensor_zero_range = [c for c in zero_range if c in SENSOR_COLUMNS]
            op_setting_zero_range = [
                c for c in zero_range if c in OP_SETTING_COLUMNS
            ]

            if sensor_zero_range:
                raise ValueError(
                    f"Cannot normalize — zero range (min == max) in sensor "
                    f"columns: {sensor_zero_range}. These should have been "
                    f"caught by _drop_constant_sensors()."
                )

            if op_setting_zero_range:
                logger.warning(
                    "Zero range (min == max) in op_setting columns: %s — "
                    "_drop_constant_sensors() only screens sensor columns, "
                    "not op_settings. Excluding these columns from "
                    "norm_params and feature_cols entirely.",
                    op_setting_zero_range,
                )

        logger.debug(
            "Fitted norm params for %d columns.", len(self._norm_params)
        )

    def _apply_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted Min-Max normalization to a DataFrame.

        Columns not in norm_params (unit, cycle, rul) are preserved as-is.
        Missing columns in df are silently skipped, with a warning logged.

        Args:
            df: DataFrame to normalize (modified copy is returned).

        Returns:
            DataFrame with normalized sensor + op_setting columns (float32).
        """
        df = df.copy()

        for col, params in self._norm_params.items():
            if col not in df.columns:
                logger.warning(
                    "Column '%s' expected by norm_params but missing from "
                    "input — skipping normalization. Model predictions may "
                    "be corrupted.",
                    col,
                )
                continue
            col_range = params["max"] - params["min"]
            df[col] = (
                (df[col] - params["min"]) / col_range
            ).astype(np.float32)

        return df