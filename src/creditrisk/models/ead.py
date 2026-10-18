"""Exposure at Default estimation via Credit Conversion Factor.

EAD captures how much of a committed-but-undrawn line a borrower will draw
down before default. The LEQ (Loan Equivalent) factor approach estimates the
fraction of the undrawn commitment that converts to exposure.

Under the Basel advanced IRB approach, EAD for revolving facilities is the
current drawn amount plus the CCF applied to the undrawn portion. Getting
the CCF wrong on large revolvers is one of the fastest ways to blow through
a capital buffer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


@dataclass
class EADResult:
    """Container for a fitted EAD model."""

    ccf_mean: float
    ccf_median: float
    ccf_p90: float
    model_coefs: dict[str, float] | None = None

    def summary(self) -> str:
        lines = [
            "─" * 37,
            f"  Mean CCF       {self.ccf_mean:.3f}",
            f"  Median CCF     {self.ccf_median:.3f}",
            f"  P90 CCF        {self.ccf_p90:.3f}",
            "─" * 37,
        ]
        return "\n".join(lines)


class ExposureAtDefault:
    """EAD estimation using the LEQ / CCF approach.

    Two modes:
    - ``method="simple"``: empirical CCF (drawn-at-default / undrawn-limit).
    - ``method="regression"``: CCF modelled as a function of obligor
      characteristics.
    """

    def __init__(self, method: str = "simple") -> None:
        if method not in ("simple", "regression"):
            raise ValueError(f"Unknown method {method!r}")
        self.method = method
        self._model: LinearRegression | None = None
        self._fitted = False

    def fit(
        self,
        features: pd.DataFrame,
        drawn_column: str,
        limit_column: str,
        drawn_at_default_column: str,
    ) -> EADResult:
        """Fit the EAD model.

        Parameters
        ----------
        features : DataFrame
            Must contain the drawn, limit and drawn-at-default columns.
        drawn_column : str
            Current drawn amount.
        limit_column : str
            Total committed limit.
        drawn_at_default_column : str
            Drawn amount at default (for calculating actual CCF).
        """
        drawn = features[drawn_column].to_numpy()
        limit = features[limit_column].to_numpy()
        dad = features[drawn_at_default_column].to_numpy()

        undrawn = np.maximum(limit - drawn, 0)

        # CCF = additional draw / undrawn
        additional = np.maximum(dad - drawn, 0)
        mask = undrawn > 0
        ccf = np.zeros_like(additional)
        ccf[mask] = additional[mask] / undrawn[mask]
        ccf = np.clip(ccf, 0, 1)

        model_coefs = None
        if self.method == "regression":
            X_cols = [
                c
                for c in features.columns
                if c not in (drawn_column, limit_column, drawn_at_default_column)
            ]
            if X_cols:
                X = features[X_cols].to_numpy()
                self._model = LinearRegression()
                self._model.fit(X, ccf)
                model_coefs = dict(zip(X_cols, self._model.coef_))

        self._fitted = True

        return EADResult(
            ccf_mean=float(np.mean(ccf)),
            ccf_median=float(np.median(ccf)),
            ccf_p90=float(np.percentile(ccf, 90)),
            model_coefs=model_coefs,
        )

    def predict(
        self,
        features: pd.DataFrame,
        drawn_column: str = "",
        limit_column: str = "",
        drawn_at_default_column: str = "",
    ) -> np.ndarray:
        """Predict EAD for new observations.

        Returns the expected drawn-at-default amount.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        drawn = features[drawn_column].to_numpy()
        limit = features[limit_column].to_numpy()
        undrawn = np.maximum(limit - drawn, 0)

        if self.method == "regression" and self._model is not None:
            X_cols = [
                c
                for c in features.columns
                if c not in (drawn_column, limit_column, drawn_at_default_column)
            ]
            ccf_pred = np.clip(self._model.predict(features[X_cols].to_numpy()), 0, 1)
            return drawn + ccf_pred * undrawn

        return drawn + self.fit(
            features, drawn_column, limit_column, drawn_at_default_column
        ).ccf_mean * undrawn
