"""Tests for validation suite and governance module."""

import json
import numpy as np
import pytest
from creditrisk.validation.metrics import (
    ValidationReport,
    validate_pd_model,
    characteristic_stability_index,
)
from creditrisk.governance.register import (
    ModelRegister,
    generate_validation_report,
)


class TestValidatePDModel:
    def test_perfect_model_passes(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        y_prob = rng.uniform(0.01, 0.3, n)
        y_true = (rng.random(n) < y_prob).astype(float)
        report = validate_pd_model(y_true, y_prob)
        assert report.auroc > 0.5
        assert report.gini > 0
        assert report.calibration_pass

    def test_with_psi(self) -> None:
        rng = np.random.default_rng(42)
        n = 500
        y_prob_dev = rng.normal(0.1, 0.05, n)
        y_prob_val = rng.normal(0.15, 0.05, n)
        y_true = (rng.random(n) < y_prob_val).astype(float)
        report = validate_pd_model(y_true, y_prob_val, y_prob_dev)
        assert report.psi is not None
        assert report.psi_flag in ("OK", "WARN", "FLAG")

    def test_report_summary(self) -> None:
        rng = np.random.default_rng(42)
        y_prob = rng.uniform(0.01, 0.3, 200)
        y_true = (rng.random(200) < y_prob).astype(float)
        report = validate_pd_model(y_true, y_prob)
        summary = report.summary()
        assert "AUROC" in summary
        assert "Gini" in summary
        assert "PASS" in summary or "FAIL" in summary


class TestCharacteristicStabilityIndex:
    def test_same_distribution_gives_low_csi(self) -> None:
        rng = np.random.default_rng(42)
        feat = rng.normal(0, 1, 1000)
        csi = characteristic_stability_index(feat, feat)
        assert csi < 0.01


class TestModelRegister:
    @pytest.fixture
    def register(self, tmp_path) -> ModelRegister:
        db = tmp_path / "test_register.db"
        return ModelRegister(str(db))

    def test_register_and_log(self, register: ModelRegister) -> None:
        register.register_version("PD_SME_v3", "3.2.1", "Q3 recalibration", "data_ref")
        record = register.log_approval(
            "PD_SME_v3", "3.2.1", "development", "Alice", "evidence_pkg"
        )
        assert record.stage == "development"
        assert record.approved_by == "Alice"
        assert len(record.evidence_hash) == 16

    def test_approval_status_tracks_stages(self, register: ModelRegister) -> None:
        register.register_version("PD_SME_v3", "3.2.1", "Q3", "data")
        assert not register.is_approved("PD_SME_v3", "3.2.1")

        register.log_approval("PD_SME_v3", "3.2.1", "development", "Alice", "ev")
        assert not register.is_approved("PD_SME_v3", "3.2.1")

        register.log_approval("PD_SME_v3", "3.2.1", "validation", "Bob", "ev")
        register.log_approval("PD_SME_v3", "3.2.1", "committee", "MRC", "ev")
        assert register.is_approved("PD_SME_v3", "3.2.1")

    def test_audit_trail(self, register: ModelRegister) -> None:
        register.register_version("PD_SME_v3", "3.2.1", "Q3", "data")
        register.log_approval("PD_SME_v3", "3.2.1", "development", "Alice", "ev")
        register.log_approval("PD_SME_v3", "3.2.1", "validation", "Bob", "ev")
        trail = register.audit_trail("PD_SME_v3", "3.2.1")
        assert len(trail) == 2
        assert trail[0].approved_by == "Alice"
        assert trail[1].approved_by == "Bob"

    def test_invalid_stage_raises(self, register: ModelRegister) -> None:
        register.register_version("PD_SME_v3", "3.2.1", "Q3", "data")
        with pytest.raises(ValueError, match="Stage must be"):
            register.log_approval("PD_SME_v3", "3.2.1", "board", "Alice", "ev")

    def test_generate_validation_report(self, register: ModelRegister) -> None:
        register.register_version("PD_SME_v3", "3.2.1", "Q3", "data")
        register.log_approval("PD_SME_v3", "3.2.1", "development", "Alice", "ev")
        report = generate_validation_report(
            "PD_SME_v3", "3.2.1", register, {"auroc": 0.84, "gini": 0.68}
        )
        parsed = json.loads(report)
        assert parsed["model_id"] == "PD_SME_v3"
        assert parsed["metrics"]["auroc"] == 0.84
        assert parsed["approval_status"]["development"] is not None
        assert parsed["approval_status"]["validation"] is None
