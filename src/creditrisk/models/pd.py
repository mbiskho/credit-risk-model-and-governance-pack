"""Probability of Default estimation with calibration diagnostics.

PD models estimate the likelihood a borrower defaults within a given horizon.
The logistic regression estimator is the regulatory baseline; the XGBoost
variant is for challenger models that must still pass the same calibration
and stability tests as the incumbent.

Every estimator exposes the same interface so a model validation team can
slot any implementation into the same validation harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2


@dataclass
class PDResult:
    """Container for a fitted PD model and its diagnostics."""

    params: dict[str, float]
    intercept: float
    auroc: float
    brier_score: float
    hosmer_lemeshow_pvalue: float
    population_stability_index: float | None = None
    calibration_curve: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        lines = [
            "─" * 37,
            f"  AUROC          {self.auroc:.3f}",
            f"  Brier Score    {self.brier_score:.3f}",
            f"  H-L p-value    {self.hosmer_lemeshow_pvalue:.2f}",
        ]
        if self.population_stability_index is not None:
            lines.append(f"  PSI            {self.population_stability_index:.4f}")
        lines.append("─" * 37)
        return "\n".join(lines)


def _hosmer_lemeshow(y_true: np.ndarray, y_pred: np.ndarray, g: int = 10) -> float:
    """Hosmer-Lemeshow goodness-of-fit test.

    The null hypothesis is that the model is well calibrated — a high p-value
    means you cannot reject that the predicted probabilities match observed
    default rates across deciles. A low p-value (typically < 0.05) is a red
    flag for a model validation team.

    Returns the p-value from the chi-squared distribution with g-2 degrees
    of freedom.
    """
    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    n = len(y_true)
    edges = np.linspace(0, n, g + 1, dtype=int)

    stat = 0.0
    for i in range(g):
        mask = slice(edges[i], edges[i + 1])
        obs = y_true_sorted[mask]
        exp = y_pred_sorted[mask]
        o = obs.sum()
        e = exp.sum()
        n_g = len(obs)
        if e == 0 or e == n_g:
            continue
        stat += (o - e) ** 2 / (e * (1 - e / n_g))
    return float(1 - chi2.cdf(stat, g - 2))


class ProbabilityOfDefault:
    """Logistic regression PD model with calibration diagnostics.

    Parameters
    ----------
    method : str
        ``"logistic"`` (default) for sklearn LogisticRegression.
        ``"xgb"`` reserved for future challenger-model support.
    """

    def __init__(self, method: str = "logistic") -> None:
        if method != "logistic":
            raise ValueError(f"Unsupported method {method!r}; use 'logistic'")
        self.method = method
        self._scaler = StandardScaler()
        self._model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self._fitted = False
        self._feature_cols: list[str] = []

    def fit(
        self,
        features: pd.DataFrame,
        default_flag: str,
        *,
        weights: str | None = None,
    ) -> PDResult:
        """Fit the model and return diagnostics.

        Parameters
        ----------
        features : DataFrame
            Feature matrix. Must not contain the default flag column.
        default_flag : str
            Name of the binary (0/1) default indicator column in ``features``.
        weights : str or None
            Optional column name for observation weights (e.g. exposure).
        """
        y = features[default_flag].to_numpy().astype(float)
        X_cols = [c for c in features.columns if c != default_flag]
        if weights is not None:
            X_cols = [c for c in X_cols if c != weights]
        X = features[X_cols].to_numpy()

        sample_weight = features[weights].to_numpy() if weights else None

        self._feature_cols = X_cols
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y, sample_weight=sample_weight)
        self._fitted = True

        y_prob = self._model.predict_proba(X_scaled)[:, 1]
        y_pred = self._model.predict(X_scaled)

        params = dict(zip(X_cols, self._model.coef_[0]))
        auroc = roc_auc_score(y, y_prob)
        brier = float(np.mean((y_prob - y) ** 2))
        hl_pval = _hosmer_lemeshow(y, y_prob)

        return PDResult(
            params=params,
            intercept=float(self._model.intercept_[0]),
            auroc=auroc,
            brier_score=brier,
            hosmer_lemeshow_pvalue=hl_pval,
        )

    def predict_proba(self, features: pd.DataFrame, default_flag: str = "") -> np.ndarray:
        """Return default probabilities for new observations."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        X = features[self._feature_cols].to_numpy()
        X_scaled = self._scaler.transform(X)
        return self._model.predict_proba(X_scaled)[:, 1]


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, bins: int = 10
) -> float:
    """Population Stability Index between two score distributions.

    PSI measures how much the score distribution has shifted from development
    to production. A value above 0.25 is a model-risk flag; above 0.10
    warrants investigation. The bins are equal-width across the combined
    range so the comparison is not sensitive to the choice of cutpoints.

    Parameters
    ----------
    expected : ndarray
        Scores from the development sample.
    actual : ndarray
        Scores from the production or out-of-time sample.
    bins : int
        Number of bins (default 10).
    """
    combined = np.concatenate([expected, actual])
    edges = np.linspace(combined.min(), combined.max(), bins + 1)
    e_hist, _ = np.histogram(expected, bins=edges)
    a_hist, _ = np.histogram(actual, bins=edges)

    e_prop = e_hist / e_hist.sum()
    a_prop = a_hist / a_hist.sum()

    epsilon = 1e-10
    psi = 0.0
    for e, a in zip(e_prop, a_prop):
        e = max(e, epsilon)
        a = max(a, epsilon)
        psi += (a - e) * np.log(a / e)
    return float(psi)
