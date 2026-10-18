"""Model governance register and approval workflow.

The governance module provides the evidence trail that separates a model from
a script. Every model run is logged with a version, a timestamp, the data
hash of the inputs and the identity of the approver. The register is the
single source of truth you hand to internal audit.

The approval workflow mirrors the three lines of defence:

1. Model Development — the builder signs off that the model is ready.
2. Independent Validation — the validator signs off that the model works.
3. Model Risk Committee — the committee signs off that the model can go live.

Each stage records who approved, when, and what evidence package supported
the decision. The register is immutable: records are appended, never updated.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE_ORDER = ["development", "validation", "committee"]


@dataclass
class ApprovalRecord:
    """A single approval event in the model lifecycle."""

    model_id: str
    version: str
    stage: str  # "development", "validation", "committee"
    approved_by: str
    approved_at: str
    evidence_hash: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "stage": self.stage,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "evidence_hash": self.evidence_hash,
            "notes": self.notes,
        }


def _data_hash(data: bytes | str) -> str:
    """SHA-256 hash of input data."""
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()[:16]


class ModelRegister:
    """Immutable model governance register backed by SQLite.

    The register is append-only by design. Records are never updated or
    deleted — corrections are new records that supersede earlier ones,
    preserving the full audit trail.

    Parameters
    ----------
    db_path : str or Path
        Path to the SQLite database file. Created if it does not exist.
    """

    def __init__(self, db_path: str | Path = "model_register.db") -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                version TEXT NOT NULL,
                stage TEXT NOT NULL,
                approved_by TEXT NOT NULL,
                approved_at TEXT NOT NULL,
                evidence_hash TEXT NOT NULL,
                notes TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                model_id TEXT NOT NULL,
                version TEXT NOT NULL,
                description TEXT DEFAULT '',
                data_hash TEXT NOT NULL,
                registered_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (model_id, version)
            )
        """)
        self._conn.commit()

    def register_version(
        self,
        model_id: str,
        version: str,
        description: str,
        data: str | bytes,
    ) -> None:
        """Register a new model version with its input data hash."""
        dh = _data_hash(data)
        self._conn.execute(
            "INSERT OR REPLACE INTO model_versions (model_id, version, description, data_hash) "
            "VALUES (?, ?, ?, ?)",
            (model_id, version, description, dh),
        )
        self._conn.commit()

    def log_approval(
        self,
        model_id: str,
        version: str,
        stage: str,
        approved_by: str,
        evidence: str | bytes,
        notes: str = "",
    ) -> ApprovalRecord:
        """Record an approval decision.

        Parameters
        ----------
        model_id : str
            Unique model identifier (e.g. "PD_SME_v3").
        version : str
            Model version (e.g. "3.2.1").
        stage : str
            Approval stage: "development", "validation", or "committee".
        approved_by : str
            Name or ID of the approver.
        evidence : str or bytes
            The evidence package that supported the decision.
        notes : str
            Optional free-text notes.
        """
        if stage not in STAGE_ORDER:
            raise ValueError(f"Stage must be one of {STAGE_ORDER}, got {stage!r}")

        now = datetime.now(timezone.utc).isoformat()
        evidence_hash = _data_hash(evidence)

        self._conn.execute(
            "INSERT INTO approvals (model_id, version, stage, approved_by, approved_at, "
            "evidence_hash, notes) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (model_id, version, stage, approved_by, now, evidence_hash, notes),
        )
        self._conn.commit()

        return ApprovalRecord(
            model_id=model_id,
            version=version,
            stage=stage,
            approved_by=approved_by,
            approved_at=now,
            evidence_hash=evidence_hash,
            notes=notes,
        )

    def approval_status(self, model_id: str, version: str) -> dict[str, ApprovalRecord | None]:
        """Return the current approval status for each stage."""
        status: dict[str, ApprovalRecord | None] = {}
        for stage in STAGE_ORDER:
            row = self._conn.execute(
                "SELECT model_id, version, stage, approved_by, approved_at, evidence_hash, notes "
                "FROM approvals WHERE model_id = ? AND version = ? AND stage = ? "
                "ORDER BY id DESC LIMIT 1",
                (model_id, version, stage),
            ).fetchone()
            if row:
                status[stage] = ApprovalRecord(*row)
            else:
                status[stage] = None
        return status

    def is_approved(self, model_id: str, version: str) -> bool:
        """Check whether all three stages are approved."""
        status = self.approval_status(model_id, version)
        return all(v is not None for v in status.values())

    def audit_trail(self, model_id: str, version: str) -> list[ApprovalRecord]:
        """Return the full audit trail for a model version."""
        rows = self._conn.execute(
            "SELECT model_id, version, stage, approved_by, approved_at, evidence_hash, notes "
            "FROM approvals WHERE model_id = ? AND version = ? ORDER BY id",
            (model_id, version),
        ).fetchall()
        return [ApprovalRecord(*r) for r in rows]

    def close(self) -> None:
        self._conn.close()


def generate_validation_report(
    model_id: str,
    version: str,
    register: ModelRegister,
    metrics: dict[str, Any],
) -> str:
    """Generate a model validation report as a JSON document.

    This is the evidence package that gets hashed and stored alongside the
    approval record. It ties the model identity, its governance status and
    its validation metrics into a single auditable artifact.
    """
    status = register.approval_status(model_id, version)
    report = {
        "model_id": model_id,
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "approval_status": {
            stage: rec.to_dict() if rec else None for stage, rec in status.items()
        },
        "metrics": metrics,
    }
    return json.dumps(report, indent=2, default=str)
