"""IFRS 9 macro scenario engine.

IFRS 9 requires expected credit losses to be probability-weighted across
multiple forward-looking scenarios, not a single most-likely path. This
module provides the three standard scenarios — baseline, adverse and severe —
and the weighting framework to combine them.

The scenario paths are supplied as a DataFrame with columns for each macro
variable (GDP growth, unemployment, etc.) and a row per quarter. The
scenario engine applies a satellite model that maps macro variables to PD
adjustments, then weights the three paths into a single ECL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SCENARIO_WEIGHTS_DEFAULT = {
    "baseline": 0.40,
    "adverse": 0.35,
    "severe": 0.25,
}
"""Default IFRS 9 scenario probability weights. These should be calibrated
to the institution's risk appetite and validated by the model risk function.
The sum must equal 1.0."""


@dataclass
class ScenarioPath:
    """A single macro scenario path."""

    name: str
    weight: float
    data: pd.DataFrame  # rows: quarters, columns: macro variables

    def __post_init__(self) -> None:
        if not 0 < self.weight <= 1:
            raise ValueError(f"Weight must be in (0, 1], got {self.weight}")


@dataclass
class ScenarioResult:
    """Weighted ECL across scenarios."""

    baseline_ecl: float
    adverse_ecl: float
    severe_ecl: float
    weighted_ecl: float
    by_scenario: dict[str, float]

    def summary(self) -> str:
        return (
            "─" * 37 + "\n"
            f"  Baseline ECL   {self.baseline_ecl:,.0f}\n"
            f"  Adverse  ECL   {self.adverse_ecl:,.0f}\n"
            f"  Severe   ECL   {self.severe_ecl:,.0f}\n"
            f"  Weighted ECL   {self.weighted_ecl:,.0f}\n"
            "─" * 37
        )


class MacroScenarioEngine:
    """Combines macro scenarios into a probability-weighted ECL.

    Parameters
    ----------
    scenarios : list of ScenarioPath
        The three (or more) macro paths with associated weights.
    pd_sensitivity : float
        PD multiplier per unit change in the macro index. A value of 0.15
        means a one-standard-deviation worsening of the macro index raises
        PD by 15% multiplicatively.
    """

    def __init__(
        self,
        scenarios: list[ScenarioPath],
        pd_sensitivity: float = 0.15,
    ) -> None:
        total_weight = sum(s.weight for s in scenarios)
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Scenario weights must sum to 1.0, got {total_weight}")
        self.scenarios = scenarios
        self.pd_sensitivity = pd_sensitivity

    def adjusted_pd(
        self, base_pd: np.ndarray, scenario: ScenarioPath, macro_var: str = "gdp_growth"
    ) -> np.ndarray:
        """Adjust a base PD vector for a given scenario path.

        The adjustment is multiplicative: PD_adj = PD_base * (1 + sensitivity * z),
        where z is the standardised macro deviation from its long-run mean.
        """
        if macro_var not in scenario.data.columns:
            raise ValueError(f"Macro variable {macro_var!r} not in scenario data")
        values = scenario.data[macro_var].to_numpy()
        mu = np.mean(values)
        sigma = np.std(values)
        if sigma == 0:
            return base_pd

        z_score = (values - mu) / sigma
        # Use the mean z-score over the forecast horizon
        mean_z = float(np.mean(z_score))
        adjustment = 1.0 + self.pd_sensitivity * mean_z
        adjustment = max(adjustment, 0.5)  # floor: PD cannot drop below 50% of base
        return base_pd * adjustment

    def weighted_ecl(
        self,
        base_pd: np.ndarray,
        lgd: np.ndarray,
        ead: np.ndarray,
        macro_var: str = "gdp_growth",
    ) -> ScenarioResult:
        """Compute probability-weighted expected credit loss.

        ECL = PD * LGD * EAD for each scenario, then weighted by scenario
        probability.
        """
        results: dict[str, float] = {}
        weighted = 0.0

        for s in self.scenarios:
            pd_adj = self.adjusted_pd(base_pd, s, macro_var)
            ecl = float(np.sum(pd_adj * lgd * ead))
            results[s.name] = ecl
            weighted += ecl * s.weight

        return ScenarioResult(
            baseline_ecl=results.get("baseline", 0.0),
            adverse_ecl=results.get("adverse", 0.0),
            severe_ecl=results.get("severe", 0.0),
            weighted_ecl=weighted,
            by_scenario=results,
        )


def generate_scenario_frame(
    quarters: int = 20,
    gdp_base: float = 0.02,
    gdp_adverse: float = 0.0,
    gdp_severe: float = -0.03,
    seed: int = 42,
) -> list[ScenarioPath]:
    """Generate synthetic macro scenario paths for demonstration.

    Real implementations would pull these from the economics team's forecast
    or a vendor feed (Moody's, Oxford Economics, etc.).

    Parameters
    ----------
    quarters : int
        Number of forecast quarters (default 20 = 5 years).
    gdp_base : float
        Annualised quarterly GDP growth under baseline.
    gdp_adverse : float
        Annualised quarterly GDP growth under adverse.
    gdp_severe : float
        Annualised quarterly GDP growth under severe.
    seed : int
        Random seed for noise.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=quarters, freq="QE")
    noise = rng.normal(0, 0.003, quarters)

    def make_path(name: str, gdp_mean: float) -> ScenarioPath:
        gdp = gdp_mean + noise
        df = pd.DataFrame({"gdp_growth": gdp, "unemployment": 0.04 + (0.02 - gdp) * 2}, index=dates)
        return ScenarioPath(name=name, weight=SCENARIO_WEIGHTS_DEFAULT[name], data=df)

    return [
        make_path("baseline", gdp_base),
        make_path("adverse", gdp_adverse),
        make_path("severe", gdp_severe),
    ]
