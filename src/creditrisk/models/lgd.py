"""Loss Given Default estimation with downturn calibration.

LGD is the fraction of exposure lost when a borrower defaults. It is bounded
in [0, 1] and typically modelled via beta regression, which respects the
support constraint and handles the heteroskedasticity that makes OLS
inappropriate.

The downturn LGD is the regulatory requirement under Basel: it must reflect
adverse economic conditions, not the through-the-cycle average. The
``downturn_calibration`` function applies a scalar uplift derived from the
worst realised LGD in the historical window, which is the simple approach
accepted by most regulators when a full stress-test linkage is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import digamma, polygamma
from scipy.stats import beta


@dataclass
class LGDResult:
    """Container for a fitted LGD model."""

    params: dict[str, float]
    intercept: float
    precision: float  # phi parameter of the beta distribution
    mean_lgd: float
    downturn_lgd: float | None = None

    def summary(self) -> str:
        lines = [
            "─" * 37,
            f"  Mean LGD       {self.mean_lgd:.3f}",
            f"  Precision (φ)  {self.precision:.2f}",
        ]
        if self.downturn_lgd is not None:
            lines.append(f"  Downturn LGD   {self.downturn_lgd:.3f}")
        lines.append("─" * 37)
        return "\n".join(lines)


def _beta_loglik(
    theta: np.ndarray, X: np.ndarray, y: np.ndarray, weights: np.ndarray | None
) -> float:
    """Negative log-likelihood for beta regression with logit link."""
    k = X.shape[1]
    beta_coef = theta[:k]
    phi = np.exp(theta[k])  # precision, constrained positive via exp

    eta = X @ beta_coef
    mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -50, 50)))  # logit link
    mu = np.clip(mu, 1e-8, 1 - 1e-8)

    a = mu * phi
    b = (1 - mu) * phi

    ll = beta.logpdf(y, a, b)
    if weights is not None:
        ll = ll * weights
    return float(-np.sum(ll))


class LossGivenDefault:
    """Beta regression LGD model.

    Parameters
    ----------
    downturn_window : int
        Number of quarters to look back for the downturn calibration
        reference point. Default 12 (three years of quarterly data).
    """

    def __init__(self, downturn_window: int = 12) -> None:
        self.downturn_window = downturn_window
        self._params: np.ndarray | None = None
        self._fitted = False

    def fit(
        self,
        features: pd.DataFrame,
        lgd_column: str,
        *,
        weights: str | None = None,
    ) -> LGDResult:
        """Fit beta regression via maximum likelihood.

        Parameters
        ----------
        features : DataFrame
            Feature matrix including the LGD column.
        lgd_column : str
            Name of the LGD column (values in [0, 1]).
        weights : str or None
            Optional column name for observation weights.
        """
        y = features[lgd_column].to_numpy().astype(float)
        if np.any((y <= 0) | (y >= 1)):
            raise ValueError(
                "LGD values must be in (0, 1). Apply a squeeze transform "
                "for boundary values: y' = (y*(n-1) + 0.5) / n"
            )

        X_cols = [c for c in features.columns if c != lgd_column]
        if weights is not None:
            X_cols = [c for c in X_cols if c != weights]
        X = features[X_cols].to_numpy()

        X_design = np.column_stack([np.ones(X.shape[0]), X])
        w = features[weights].to_numpy() if weights else None

        init = np.zeros(X_design.shape[1] + 1)  # +1 for log(phi)
        init[-1] = 1.0

        result = minimize(
            _beta_loglik,
            init,
            args=(X_design, y, w),
            method="L-BFGS-B",
            options={"maxiter": 5000},
        )

        self._params = result.x
        self._fitted = True

        coef = self._params[: X_design.shape[1]]
        log_phi = self._params[-1]
        phi = np.exp(log_phi)

        param_names = ["intercept"] + X_cols
        params_dict = dict(zip(param_names, coef))

        mu_pred = 1.0 / (1.0 + np.exp(-np.clip(X_design @ coef, -50, 50)))
        mean_lgd = float(np.mean(mu_pred))

        downturn_lgd = _downturn_calibration(y, self.downturn_window)

        return LGDResult(
            params=params_dict,
            intercept=float(coef[0]),
            precision=phi,
            mean_lgd=mean_lgd,
            downturn_lgd=downturn_lgd,
        )

    def predict(self, features: pd.DataFrame, lgd_column: str = "") -> np.ndarray:
        """Return LGD predictions for new observations."""
        if not self._fitted or self._params is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        X_cols = [c for c in features.columns if c != lgd_column]
        X = features[X_cols].to_numpy()
        X_design = np.column_stack([np.ones(X.shape[0]), X])
        k = X_design.shape[1]
        eta = X_design @ self._params[:k]
        eta = np.clip(eta, -50, 50)
        return 1.0 / (1.0 + np.exp(-eta))


def _downturn_calibration(lgd_samples: np.ndarray, window: int) -> float:
    """Simple downturn calibration: the worst observed average LGD over a
    rolling window of the given length.

    For a proper regulatory filing you would link this to a macro variable
    (e.g. GDP growth) via a satellite model. This implementation provides the
    reference point a model validation team expects to see before signing off
    on a full stress-test linkage.
    """
    if len(lgd_samples) < window:
        return float(np.max(lgd_samples))
    series = pd.Series(lgd_samples)
    rolling = series.rolling(window=window).mean()
    return float(rolling.max())
