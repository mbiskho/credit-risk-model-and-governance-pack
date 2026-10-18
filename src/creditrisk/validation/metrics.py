"""Model validation suite for credit risk models.

The validation framework checks three dimensions every model risk function
expects before signing off on a model:

1. **Discrimination** — can the model separate defaulters from non-defaulters?
   Measured by AUROC and Gini coefficient.

2. **Calibration** — do the predicted probabilities match observed default
   rates? Measured by the Hosmer-Lemeshow test and binomial test.

3. **Stability** — has the score distribution shifted since development?
   Measured by the Population Stability Index (PSI) and Characteristic
   Stability Index (CSI).

Each check returns a structured result so a validation report can be
generated automatically from the output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

from creditrisk.models.pd import _hosmer_lemeshow, population_stability_index


@dataclass
class ValidationReport:
    """Structured validation output for a PD model."""

    auroc: float
    gini: float
    hosmer_lemeshow_pvalue: float
    calibration_pass: bool
    psi: float | None = None
    psi_flag: str = "N/A"
    binomial_pvalue: float | None = None
    binomial_pass: bool | None = None

    def summary(self) -> str:
        lines = [
            "╔" + "═" * 43 + "╗",
            "║  Model Validation Report" + " " * 20 + "║",
            "╠" + "═" * 43 + "╣",
            f"║  AUROC              {self.auroc:>12.3f}   ║",
            f"║  Gini               {self.gini:>12.3f}   ║",
            f"║  H-L p-value        {self.hosmer_lemeshow_pvalue:>12.4f}   ║",
            f"║  Calibration        {_pass_fail(self.calibration_pass):>12}   ║",
        ]
        if self.psi is not None:
            lines.append(f"║  PSI                {self.psi:>12.4f}   ║")
            lines.append(f"║  PSI Flag           {self.psi_flag:>12}   ║")
        if self.binomial_pvalue is not None:
            lines.append(f"║  Binomial p-value   {self.binomial_pvalue:>12.4f}   ║")
            lines.append(f"║  Binomial test      {_pass_fail(self.binomial_pass):>12}   ║")
        lines.append("╚" + "═" * 43 + "╝")
        return "\n".join(lines)


def _pass_fail(condition: bool | None) -> str:
    if condition is True:
        return "PASS"
    if condition is False:
        return "FAIL"
    return "N/A"


def validate_pd_model(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_prob_dev: np.ndarray | None = None,
) -> ValidationReport:
    """Run the full PD validation suite.

    Parameters
    ----------
    y_true : ndarray
        Binary default indicators (0/1) for the validation sample.
    y_prob : ndarray
        Predicted default probabilities for the validation sample.
    y_prob_dev : ndarray or None
        Predicted default probabilities for the development sample, used
        to compute the Population Stability Index. If None, PSI is skipped.
    """
    auroc = roc_auc_score(y_true, y_prob)
    gini = 2 * auroc - 1
    hl_pval = _hosmer_lemeshow(y_true, y_prob)
    calibration_pass = hl_pval >= 0.05

    psi = None
    psi_flag = "N/A"
    if y_prob_dev is not None:
        psi = population_stability_index(y_prob_dev, y_prob)
        if psi < 0.10:
            psi_flag = "OK"
        elif psi < 0.25:
            psi_flag = "WARN"
        else:
            psi_flag = "FLAG"

    # Binomial test: expected defaults vs observed at the mean PD
    n = len(y_true)
    expected_pd = float(np.mean(y_prob))
    observed_defaults = int(np.sum(y_true))
    expected_defaults = n * expected_pd
    binomial_pval = None
    binomial_pass = None
    if expected_defaults > 0 and observed_defaults > 0:
        result = binomtest(observed_defaults, n=n, p=expected_pd)
        binomial_pval = float(result.pvalue)
        binomial_pass = binomial_pval >= 0.05

    return ValidationReport(
        auroc=auroc,
        gini=gini,
        hosmer_lemeshow_pvalue=hl_pval,
        calibration_pass=calibration_pass,
        psi=psi,
        psi_flag=psi_flag,
        binomial_pvalue=binomial_pval,
        binomial_pass=binomial_pass,
    )


def characteristic_stability_index(
    feature_dev: np.ndarray,
    feature_val: np.ndarray,
    bins: int = 10,
) -> float:
    """Characteristic Stability Index for a single input feature.

    CSI is PSI applied to an input variable rather than the model score.
    A high CSI on a key feature signals that the population has shifted
    in a way the model may not handle — it is a leading indicator of
    model decay before the score distribution itself has drifted.
    """
    return population_stability_index(feature_dev, feature_val, bins=bins)
