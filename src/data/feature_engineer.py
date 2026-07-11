"""
Feature engineering for NASA CMAPSS predictive maintenance data.

Receives ProcessedData (normalized, constant-sensor-dropped) and adds
time-series features that capture degradation trends over time.

Feature families added:
    1. Rolling mean   — smoothed sensor signal over a sliding window
    2. Rolling std    — local variability (increases near failure)
    3. Rate of change — first-order diff per sensor (degradation velocity)

All rolling operations are computed PER UNIT to prevent information
leakage across engine boundaries. NaN values introduced at the start of
each unit's window are handled via forward-fill then backward-fill.

Does NOT:
    - Normalize features          -> already done in preprocessor.py
    - Select features             -> trainer.py handles this
    - Split train/validation      -> trainer.py

Reference:
    Li, X. et al. (2018). Remaining useful life estimation in prognostics
    using deep convolution neural networks. Reliability Engineering &
    System Safety.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.preprocessor import ProcessedData, USEFUL_SENSORS

__all__ = [
    "FeatureEngineer",
    "EngineeredData",
    "DEFAULT_WINDOW_SIZE",
]

logger = logging.getLogger(__name__)

#: Default rolling window size in cycles
DEFAULT_WINDOW_SIZE: int = 10

#: Shared column-name constants to avoid literal-string drift
UNIT_COL: str = "unit"
CYCLE_COL: str = "cycle"


# ── Output Container ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class EngineeredData:
    """
    Immutable container for feature-engineered CMAPSS data.

    Extends ProcessedData with rolling and rate-of-change features.
    Original normalized sensor columns are retained alongside new features.

    Note: frozen=True prevents attribute reassignment. DataFrames are
    mutable — downstream consumers must .copy() before in-place ops.
    feature_cols and base_feature_cols are tuples (not lists) so the
    "immutable container" guarantee holds for the whole object, not just
    its top-level attributes.

    Attributes:
        train:          Training DataFrame with all engineered features.
        test:           Test DataFrame with all engineered features.
        rul:            Ground truth RUL Series (unchanged from input).
        feature_cols:   All columns to use as model inputs — original
                        normalized sensors + all engineered features.
        window_size:    Rolling window size used during engineering.
        base_feature_cols: Original feature columns from ProcessedData
                           (before engineering).
    """

    train: pd.DataFrame
    test: pd.DataFrame
    rul: pd.Series
    feature_cols: tuple[str, ...]
    window_size: int
    base_feature_cols: tuple[str, ...]


# ── Feature Engineer ───────────────────────────────────────────────────────


class FeatureEngineer:
    """
    Adds rolling and rate-of-change features to normalized CMAPSS data.

    Usage:
        engineer = FeatureEngineer(window_size=10)
        engineered = engineer.transform(processed_data)

        X_train = engineered.train[engineered.feature_cols]
        y_train = engineered.train["rul"]
    """

    def __init__(self, window_size: int = DEFAULT_WINDOW_SIZE) -> None:
        """
        Initialize feature engineer.

        Args:
            window_size: Number of cycles in rolling window.
                         Must be >= 2 (std requires at least 2 points).

        Raises:
            ValueError: If window_size < 2.
        """
        if window_size < 2:
            raise ValueError(
                f"window_size must be >= 2 (std requires at least 2 points), "
                f"got {window_size}"
            )
        self._window_size = window_size

    # ── Public Interface ───────────────────────────────────────────────────

    def transform(self, data: ProcessedData) -> EngineeredData:
        """
        Add rolling and rate-of-change features to both train and test.

        Feature naming convention:
            {sensor}_roll_mean_{w}  — rolling mean, window w
            {sensor}_roll_std_{w}   — rolling std,  window w
            {sensor}_roc            — rate of change (first diff)

        Rolling is computed per unit group to prevent cross-unit leakage.
        NaN from rolling start of each unit is filled: ffill then bfill.

        Synchronous. In async contexts, wrap with:
            await asyncio.to_thread(engineer.transform, data)

        Args:
            data: ProcessedData from CMAPSSPreprocessor.fit_transform().

        Returns:
            EngineeredData with expanded feature set.
        """
        logger.info(
            "Engineering features — window_size=%d | base_features=%d",
            self._window_size,
            len(data.feature_cols),
        )

        # Sensors to engineer — only USEFUL_SENSORS that survived preprocessing
        sensors_to_engineer = [
            col for col in data.feature_cols
            if col in USEFUL_SENSORS
        ]

        train = self._add_features(data.train.copy(), sensors_to_engineer)
        test = self._add_features(data.test.copy(), sensors_to_engineer)

        # Build complete feature column list
        engineered_cols = self._get_engineered_col_names(sensors_to_engineer)
        all_feature_cols = tuple(data.feature_cols) + tuple(engineered_cols)

        logger.info(
            "Feature engineering complete — total features: %d "
            "(base: %d + engineered: %d)",
            len(all_feature_cols),
            len(data.feature_cols),
            len(engineered_cols),
        )

        return EngineeredData(
            train=train,
            test=test,
            rul=data.rul,
            feature_cols=all_feature_cols,
            window_size=self._window_size,
            base_feature_cols=tuple(data.feature_cols),
        )

    # ── Private Helpers ────────────────────────────────────────────────────

    def _add_features(
        self,
        df: pd.DataFrame,
        sensors: list[str],
    ) -> pd.DataFrame:
        """
        Add rolling mean, rolling std, and rate-of-change features.

        All rolling operations are grouped by 'unit' to prevent
        information leakage across engine boundaries.

        Ordering precondition:
            Rolling/diff operations are order-dependent. This method does
            not assume the incoming df is pre-sorted by the caller — it
            explicitly sorts by [UNIT_COL, CYCLE_COL] as its first step,
            so correctness does not depend on upstream row order.

        Performance note:
            Rolling/std/diff are computed via direct grouped operations
            (groupby(...).rolling(...), groupby(...).diff()) rather than
            groupby(...).transform(lambda x: ...), which avoids a
            Python-level lambda call and redundant group re-splitting per
            sensor per feature family. Rolling results carry a
            (group, index) MultiIndex and are realigned back to df's
            index via reset_index(level=0, drop=True); diff() does not
            need this since SeriesGroupBy.diff() already returns a result
            aligned to the original index.

        NaN handling:
            - Rolling operations produce NaN for the first (window_size - 1)
              rows of each unit
            - ffill propagates the first valid value backward within unit
            - bfill handles the edge case where the entire unit window
              is smaller than window_size (very short sequences)
            - A final fillna(0) is applied as a safety net for units with
              too few rows for ffill/bfill to have anything to propagate
              from (e.g. a single-cycle unit), guaranteeing no NaN survives
              in the output regardless of unit length. This 0 encodes a
              modeling assumption (zero variability / zero rate-of-change
              for degenerate short units) — see comment at the call site.

        Args:
            df:      DataFrame to add features to (modified in-place copy).
            sensors: Sensor column names to engineer features from.

        Returns:
            DataFrame with new feature columns appended, sorted by
            [UNIT_COL, CYCLE_COL].
        """
        w = self._window_size

        # Enforce the ordering precondition explicitly rather than relying
        # on the upstream preprocessor to preserve row order.
        df = df.sort_values([UNIT_COL, CYCLE_COL]).reset_index(drop=True)

        for sensor in sensors:
            if sensor not in df.columns:
                logger.warning(
                    "Sensor '%s' expected but not found in DataFrame — "
                    "skipping feature engineering for this column.",
                    sensor,
                )
                continue

            grouped = df.groupby(UNIT_COL, sort=False)[sensor]

            # Rolling mean — captures smoothed degradation trend
            roll_mean_col = f"{sensor}_roll_mean_{w}"
            df[roll_mean_col] = (
                grouped.rolling(w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
                .astype(np.float32)
            )

            # Rolling std — captures local variability (rises near failure)
            roll_std_col = f"{sensor}_roll_std_{w}"
            df[roll_std_col] = (
                grouped.rolling(w, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
                .astype(np.float32)
            )

            # Rate of change — first diff per unit (degradation velocity)
            roc_col = f"{sensor}_roc"
            df[roc_col] = grouped.diff().astype(np.float32)

        # Fill NaN introduced by rolling start and diff at unit boundaries
        # ffill: propagate first valid value forward within each unit
        # bfill: handle units shorter than window_size
        new_cols = self._get_engineered_col_names(sensors)
        present_new_cols = [c for c in new_cols if c in df.columns]

        if present_new_cols:
            df[present_new_cols] = (
                df.groupby(UNIT_COL, sort=False)[present_new_cols]
                .transform(lambda x: x.ffill().bfill())
            )
            # Safety net: units too short for ffill/bfill to fill from
            # (e.g. a single-cycle unit) still have NaN at this point.
            # 0 = zero-variability / zero-rate-of-change assumption for
            # these degenerate units — defensible for CMAPSS since
            # single-cycle units aren't expected in real data, but it is
            # a modeling choice, not a mathematical fact.
            df[present_new_cols] = df[present_new_cols].fillna(0)

        return df

    def _get_engineered_col_names(self, sensors: list[str]) -> list[str]:
        """
        Return the ordered list of engineered column names for given sensors.

        Args:
            sensors: Sensor column names.

        Returns:
            List of engineered feature column names in a consistent order:
            [roll_mean cols..., roll_std cols..., roc cols...]
        """
        w = self._window_size
        roll_mean = [f"{s}_roll_mean_{w}" for s in sensors]
        roll_std  = [f"{s}_roll_std_{w}"  for s in sensors]
        roc       = [f"{s}_roc"           for s in sensors]
        return roll_mean + roll_std + roc