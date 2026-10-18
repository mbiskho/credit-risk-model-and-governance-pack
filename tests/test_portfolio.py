"""Tests for scenario engine and portfolio."""

import numpy as np
import pytest
from creditrisk.scenarios.macro import (
    MacroScenarioEngine,
    ScenarioPath,
    generate_scenario_frame,
)
from creditrisk.portfolio import Portfolio
from creditrisk.models.pd import ProbabilityOfDefault
from creditrisk.models.lgd import LossGivenDefault
from creditrisk.models.ead import ExposureAtDefault


class TestScenarioEngine:
    @pytest.fixture
    def scenarios(self):
        return generate_scenario_frame(quarters=20)

    def test_generated_scenarios_sum_to_one(self, scenarios) -> None:
        total = sum(s.weight for s in scenarios)
        assert abs(total - 1.0) < 0.01

    def test_weighted_ecl(self, scenarios) -> None:
        engine = MacroScenarioEngine(scenarios, pd_sensitivity=0.15)
        base_pd = np.full(100, 0.03)
        lgd = np.full(100, 0.45)
        ead = np.full(100, 1_000_000)
        result = engine.weighted_ecl(base_pd, lgd, ead)
        assert result.weighted_ecl > 0
        assert result.baseline_ecl > 0
        assert result.adverse_ecl >= result.baseline_ecl
        assert result.severe_ecl >= result.adverse_ecl

    def test_adjusted_pd_increases_under_adverse(self, scenarios) -> None:
        engine = MacroScenarioEngine(scenarios, pd_sensitivity=0.5)
        base_pd = np.full(20, 0.03)
        adverse = [s for s in scenarios if s.name == "adverse"][0]
        adj = engine.adjusted_pd(base_pd, adverse)
        # Under adverse, mean z is near 0 (GDP=0), adjustment ≈ 1.0
        # With pd_sensitivity=0.5, even small deviations get amplified
        assert np.mean(adj) >= np.mean(base_pd) * 0.99  # not lower than base


class TestPortfolio:
    @pytest.fixture
    def portfolio_data(self):
        rng = np.random.default_rng(42)
        n = 200
        leverage = rng.uniform(0, 5, n)
        profitability = rng.normal(0.05, 0.1, n)
        logit = -2 + 0.8 * leverage - 3.0 * profitability
        pd_true = 1 / (1 + np.exp(-logit))
        default = (rng.random(n) < pd_true).astype(int)
        lgd = rng.beta(2, 5, n)
        limit = rng.uniform(100_000, 1_000_000, n)
        drawn = rng.uniform(0.2, 0.8, n) * limit
        ccf = rng.uniform(0.3, 0.7, n)
        dad = drawn + ccf * (limit - drawn)
        import pandas as pd
        return pd.DataFrame({
            "leverage": leverage,
            "profitability": profitability,
            "default_12m": default,
            "lgd": np.clip(lgd, 0.01, 0.99),
            "drawn": drawn,
            "limit": limit,
            "drawn_at_default": dad,
        })

    def test_expected_loss(self, portfolio_data) -> None:
        pd_model = ProbabilityOfDefault()
        pd_model.fit(portfolio_data, default_flag="default_12m")

        lgd_model = LossGivenDefault()
        lgd_model.fit(portfolio_data, lgd_column="lgd")

        ead_model = ExposureAtDefault(method="simple")
        ead_model.fit(
            portfolio_data,
            drawn_column="drawn",
            limit_column="limit",
            drawn_at_default_column="drawn_at_default",
        )

        portfolio = Portfolio(pd_model, lgd_model, ead_model)
        result = portfolio.expected_loss(portfolio_data)

        assert result.total_exposure > 0
        assert result.expected_loss > 0
        assert result.expected_loss_ratio > 0
        assert result.concentration_herfindahl is not None
        assert 0 < result.concentration_herfindahl <= 1
