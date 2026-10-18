"""Portfolio-level expected loss and risk metrics.

The Portfolio class binds PD, LGD and EAD estimates together and computes
aggregate expected loss, unexpected loss and concentration metrics. It is the
top-level entry point for a credit risk report.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from creditrisk.models.pd import ProbabilityOfDefault
from creditrisk.models.lgd import LossGivenDefault
from creditrisk.models.ead import ExposureAtDefault


@dataclass
class PortfolioResult:
    """Aggregate portfolio risk metrics."""

    total_exposure: float
    expected_loss: float
    expected_loss_ratio: float  # EL / total exposure in bps
    unexpected_loss_99: float | None = None
    by_grade: pd.DataFrame = field(default_factory=pd.DataFrame)
    concentration_herfindahl: float | None = None

    def summary(self) -> str:
        lines = [
            "─" * 45,
            f"  Total Exposure     {self.total_exposure:>12,.0f}",
            f"  Expected Loss      {self.expected_loss:>12,.0f}",
            f"  EL Ratio (bps)     {self.expected_loss_ratio:>12.1f}",
        ]
        if self.unexpected_loss_99 is not None:
            lines.append(f"  UL 99%             {self.unexpected_loss_99:>12,.0f}")
        if self.concentration_herfindahl is not None:
            lines.append(f"  HHI                {self.concentration_herfindahl:>12.4f}")
        lines.append("─" * 45)
        return "\n".join(lines)


class Portfolio:
    """Credit portfolio aggregating PD, LGD and EAD estimates.

    Parameters
    ----------
    pd_model : ProbabilityOfDefault
        A fitted PD model.
    lgd_model : LossGivenDefault
        A fitted LGD model.
    ead_model : ExposureAtDefault
        A fitted EAD model.
    """

    def __init__(
        self,
        pd_model: ProbabilityOfDefault,
        lgd_model: LossGivenDefault,
        ead_model: ExposureAtDefault,
    ) -> None:
        self.pd_model = pd_model
        self.lgd_model = lgd_model
        self.ead_model = ead_model

    def expected_loss(
        self,
        features: pd.DataFrame,
        default_flag: str = "default_12m",
        lgd_column: str = "lgd",
        drawn_column: str = "drawn",
        limit_column: str = "limit",
        dad_column: str = "drawn_at_default",
        confidence: float = 0.999,
    ) -> PortfolioResult:
        """Compute portfolio-level expected loss.

        Parameters
        ----------
        features : DataFrame
            Must contain all columns needed by the sub-models.
        confidence : float
            Confidence level for unexpected loss (default 99.9%).
        """
        pd_est = self.pd_model.predict_proba(features, default_flag)
        lgd_est = self.lgd_model.predict(features, lgd_column)

        drawn = features[drawn_column].to_numpy()
        limit = features[limit_column].to_numpy()
        dad = features[dad_column].to_numpy()
        undrawn = np.maximum(limit - drawn, 0)
        ccf = self.ead_model.fit(features, drawn_column, limit_column, dad_column).ccf_mean
        ead_est = drawn + ccf * undrawn

        total_exposure = float(np.sum(ead_est))
        individual_el = pd_est * lgd_est * ead_est
        el = float(np.sum(individual_el))
        el_ratio = (el / total_exposure) * 10_000 if total_exposure > 0 else 0.0

        # Unexpected loss via Vasicek-style single-factor model
        ul_99 = None
        if confidence > 0:
            z_score = float(np.sqrt(12) * np.sqrt(np.mean(pd_est * (1 - pd_est))))
            ul_99 = z_score * total_exposure * np.sqrt(np.mean(lgd_est**2))

        # Concentration: Herfindahl-Hirschman Index on exposure shares
        exposure_shares = ead_est / ead_est.sum()
        hhi = float(np.sum(exposure_shares**2))

        return PortfolioResult(
            total_exposure=total_exposure,
            expected_loss=el,
            expected_loss_ratio=el_ratio,
            unexpected_loss_99=ul_99,
            concentration_herfindahl=hhi,
        )
