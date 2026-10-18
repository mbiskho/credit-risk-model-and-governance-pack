"""Tests for LGD and EAD models."""

import numpy as np
import pandas as pd
import pytest
from creditrisk.models.lgd import LossGivenDefault
from creditrisk.models.ead import ExposureAtDefault


@pytest.fixture
def lgd_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 300
    collateral = rng.uniform(0, 1, n)
    seniority = rng.integers(0, 3, n)
    eta = 0.5 - 1.5 * collateral - 0.3 * seniority
    mu = 1 / (1 + np.exp(-eta))
    phi = 15
    a = mu * phi
    b = (1 - mu) * phi
    lgd = rng.beta(a, b)
    lgd = np.clip(lgd, 0.01, 0.99)
    return pd.DataFrame({"collateral": collateral, "seniority": seniority, "lgd": lgd})


@pytest.fixture
def ead_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 300
    limit = rng.uniform(100_000, 1_000_000, n)
    drawn = rng.uniform(0.2, 0.8, n) * limit
    ccf_true = rng.uniform(0.3, 0.7, n)
    dad = drawn + ccf_true * (limit - drawn)
    return pd.DataFrame(
        {
            "drawn": drawn,
            "limit": limit,
            "drawn_at_default": dad,
            "rating_grade": rng.integers(1, 8, n),
        }
    )


class TestLossGivenDefault:
    def test_fit_returns_diagnostics(self, lgd_data: pd.DataFrame) -> None:
        model = LossGivenDefault()
        result = model.fit(lgd_data, lgd_column="lgd")

        assert 0 < result.mean_lgd < 1
        assert result.precision > 0
        assert "collateral" in result.params
        assert result.downturn_lgd is not None

    def test_boundary_values_raise(self, lgd_data: pd.DataFrame) -> None:
        data = lgd_data.copy()
        data.loc[0, "lgd"] = 0.0
        model = LossGivenDefault()
        with pytest.raises(ValueError, match="squeeze transform"):
            model.fit(data, lgd_column="lgd")

    def test_predict_returns_valid_lgd(self, lgd_data: pd.DataFrame) -> None:
        model = LossGivenDefault()
        model.fit(lgd_data, lgd_column="lgd")
        pred = model.predict(lgd_data)

        assert len(pred) == len(lgd_data)
        assert np.all((pred > 0) & (pred < 1))

    def test_downturn_lgd_exceeds_mean(self, lgd_data: pd.DataFrame) -> None:
        model = LossGivenDefault(downturn_window=10)
        result = model.fit(lgd_data, lgd_column="lgd")
        assert result.downturn_lgd >= result.mean_lgd


class TestExposureAtDefault:
    def test_fit_simple_returns_ccf_stats(self, ead_data: pd.DataFrame) -> None:
        model = ExposureAtDefault(method="simple")
        result = model.fit(
            ead_data,
            drawn_column="drawn",
            limit_column="limit",
            drawn_at_default_column="drawn_at_default",
        )
        assert 0 < result.ccf_mean < 1
        assert 0 < result.ccf_median < 1
        assert result.ccf_p90 >= result.ccf_median

    def test_fit_regression(self, ead_data: pd.DataFrame) -> None:
        model = ExposureAtDefault(method="regression")
        result = model.fit(
            ead_data,
            drawn_column="drawn",
            limit_column="limit",
            drawn_at_default_column="drawn_at_default",
        )
        assert result.model_coefs is not None
        assert "rating_grade" in result.model_coefs
