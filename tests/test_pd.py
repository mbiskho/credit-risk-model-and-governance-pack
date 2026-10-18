"""Tests for the PD model module."""

import numpy as np
import pandas as pd
import pytest
from creditrisk.models.pd import (
    ProbabilityOfDefault,
    _hosmer_lemeshow,
    population_stability_index,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    leverage = rng.uniform(0, 5, n)
    profitability = rng.normal(0.05, 0.1, n)
    logit = -2 + 0.8 * leverage - 3.0 * profitability
    pd_true = 1 / (1 + np.exp(-logit))
    default = (rng.random(n) < pd_true).astype(int)
    return pd.DataFrame(
        {
            "leverage": leverage,
            "profitability": profitability,
            "default_12m": default,
            "exposure": rng.uniform(100_000, 10_000_000, n),
        }
    )


class TestProbabilityOfDefault:
    def test_fit_returns_diagnostics(self, sample_data: pd.DataFrame) -> None:
        model = ProbabilityOfDefault()
        result = model.fit(sample_data, default_flag="default_12m")

        assert 0.5 < result.auroc <= 1.0
        assert 0.0 <= result.brier_score <= 0.25
        assert 0.0 <= result.hosmer_lemeshow_pvalue <= 1.0
        assert "leverage" in result.params
        assert "profitability" in result.params

    def test_fit_with_weights(self, sample_data: pd.DataFrame) -> None:
        model = ProbabilityOfDefault()
        result = model.fit(
            sample_data, default_flag="default_12m", weights="exposure"
        )
        assert result.auroc > 0.5

    def test_predict_proba_returns_probabilities(
        self, sample_data: pd.DataFrame
    ) -> None:
        model = ProbabilityOfDefault()
        model.fit(sample_data, default_flag="default_12m")
        probs = model.predict_proba(sample_data)

        assert len(probs) == len(sample_data)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_predict_before_fit_raises(self, sample_data: pd.DataFrame) -> None:
        model = ProbabilityOfDefault()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(sample_data)

    def test_summary_output(self, sample_data: pd.DataFrame) -> None:
        model = ProbabilityOfDefault()
        result = model.fit(sample_data, default_flag="default_12m")
        summary = result.summary()
        assert "AUROC" in summary
        assert "Brier" in summary
        assert "H-L" in summary


class TestHosmerLemeshow:
    def test_perfect_calibration_gives_high_pvalue(self) -> None:
        rng = np.random.default_rng(123)
        y_pred = rng.uniform(0.1, 0.5, 1000)
        y_true = (rng.random(1000) < y_pred).astype(float)
        pval = _hosmer_lemeshow(y_true, y_pred)
        assert pval > 0.01  # well-calibrated model should not be rejected

    def test_poor_calibration_gives_low_pvalue(self) -> None:
        rng = np.random.default_rng(456)
        y_pred = rng.uniform(0.01, 0.99, 1000)
        # True rate is fixed at 0.5, predictions vary — this is miscalibrated
        y_true = (rng.random(1000) < 0.5).astype(float)
        pval = _hosmer_lemeshow(y_true, y_pred)
        # With this much miscalibration, H-L should flag it
        assert pval < 0.05


class TestPopulationStabilityIndex:
    def test_identical_distributions_give_zero(self) -> None:
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 1000)
        psi = population_stability_index(scores, scores)
        assert psi < 0.01

    def test_shifted_distribution_gives_positive(self) -> None:
        rng = np.random.default_rng(42)
        dev = rng.normal(0.3, 0.1, 1000)
        val = rng.normal(0.7, 0.1, 1000)
        psi = population_stability_index(dev, val)
        assert psi > 0.5  # large shift should produce high PSI
