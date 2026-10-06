from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from kernaut.models import (
    CandidateBundle,
    CandidateDraft,
    EvaluationRecord,
    EvidenceRecord,
    FormulationRecord,
    ParameterTrial,
    RunManifest,
)


class FrontierItem(BaseModel):
    candidate_id: str
    name: str
    contract: str
    origin: str = "discovered"
    score: float
    runtime_seconds: float
    tier: int


class CandidateStore:
    """SQLite-backed source of truth for candidates, evidence, scores, and lineage."""

    def __init__(self, path: str | Path) -> None:
        # Worker execution and external model clients may temporarily alter process context. Keep
        # the durable source of truth anchored to one canonical absolute path for the full run.
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    contract TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
                    tier INTEGER,
                    accepted INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
                    score REAL NOT NULL,
                    runtime_seconds REAL NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(run_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS formulations (
                    formulation_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidate_drafts (
                    draft_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS parameter_trials (
                    trial_id TEXT PRIMARY KEY,
                    draft_id TEXT NOT NULL REFERENCES candidate_drafts(draft_id),
                    score REAL NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_evidence_candidate ON evidence(candidate_id, id);
                CREATE INDEX IF NOT EXISTS idx_evaluations_candidate
                    ON evaluations(candidate_id, id);
                """
            )

    def add_formulation(self, formulation: FormulationRecord) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO formulations VALUES (?, ?, ?)",
                (
                    formulation.formulation_id,
                    formulation.model_dump_json(),
                    formulation.created_at.isoformat(),
                ),
            )

    def get_formulation(self, formulation_id: str) -> FormulationRecord | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT payload FROM formulations WHERE formulation_id = ?", (formulation_id,)
            ).fetchone()
        return FormulationRecord.model_validate_json(row["payload"]) if row else None

    def add_draft(self, draft: CandidateDraft) -> str:
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO candidate_drafts VALUES (?, ?, ?)",
                (draft.draft_id, draft.model_dump_json(), draft.created_at.isoformat()),
            )
        return draft.draft_id

    def get_draft(self, draft_id: str) -> CandidateDraft | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT payload FROM candidate_drafts WHERE draft_id = ?", (draft_id,)
            ).fetchone()
        return CandidateDraft.model_validate_json(row["payload"]) if row else None

    def add_parameter_trial(self, trial: ParameterTrial) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO parameter_trials VALUES (?, ?, ?, ?, ?)",
                (
                    trial.trial_id,
                    trial.draft_id,
                    trial.score,
                    trial.model_dump_json(),
                    trial.created_at.isoformat(),
                ),
            )

    def get_parameter_trial(self, trial_id: str) -> ParameterTrial | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT payload FROM parameter_trials WHERE trial_id = ?", (trial_id,)
            ).fetchone()
        return ParameterTrial.model_validate_json(row["payload"]) if row else None

    def parameter_trials(self, draft_id: str) -> list[ParameterTrial]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT payload FROM parameter_trials WHERE draft_id = ? ORDER BY score DESC",
                (draft_id,),
            ).fetchall()
        trials: list[ParameterTrial] = []
        for row in rows:
            payload = json.loads(row["payload"])
            # Pydantic serializes non-finite failure sentinels as JSON null. Keep
            # those earlier failed trials queryable without treating them as
            # competitive scores.
            if payload.get("score") is None:
                payload["score"] = float("-inf")
            trials.append(ParameterTrial.model_validate(payload))
        return trials

    def add_candidate(self, candidate: CandidateBundle) -> str:
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO candidates VALUES (?, ?, ?, ?, ?)",
                (
                    candidate.candidate_id,
                    candidate.name,
                    candidate.contract.value,
                    candidate.model_dump_json(),
                    candidate.created_at.isoformat(),
                ),
            )
        return candidate.candidate_id

    def get_candidate(self, candidate_id: str) -> CandidateBundle | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT payload FROM candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
        return CandidateBundle.model_validate_json(row["payload"]) if row else None

    def add_evidence(self, evidence: EvidenceRecord) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO evidence(candidate_id, tier, accepted, payload, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    evidence.candidate_id,
                    int(evidence.tier) if evidence.tier is not None else None,
                    evidence.accepted,
                    evidence.model_dump_json(),
                    evidence.created_at.isoformat(),
                ),
            )

    def latest_evidence(self, candidate_id: str) -> EvidenceRecord | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT payload FROM evidence WHERE candidate_id = ? ORDER BY id DESC LIMIT 1",
                (candidate_id,),
            ).fetchone()
        return EvidenceRecord.model_validate_json(row["payload"]) if row else None

    def add_evaluation(self, evaluation: EvaluationRecord) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO evaluations(
                       candidate_id, score, runtime_seconds, payload, created_at
                   )
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    evaluation.candidate_id,
                    evaluation.score,
                    evaluation.runtime_seconds,
                    evaluation.model_dump_json(),
                    evaluation.created_at.isoformat(),
                ),
            )

    def latest_evaluation(self, candidate_id: str) -> EvaluationRecord | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT payload FROM evaluations WHERE candidate_id = ? ORDER BY id DESC LIMIT 1",
                (candidate_id,),
            ).fetchone()
        return EvaluationRecord.model_validate_json(row["payload"]) if row else None

    def best_selection_score_since(self, marker: str) -> float | None:
        """Best selection score evaluated at/after `marker` (UTC isoformat).

        Attribution uses the evaluation timestamp so re-evaluating an older
        candidate still counts toward the campaign that produced it.
        """
        with self._connect() as db:
            row = db.execute(
                """
                SELECT MAX(
                           COALESCE(
                               json_extract(v.payload, '$.metadata.selection_score'),
                               v.score
                           )
                       ) AS best
                FROM evaluations v
                JOIN candidates c ON c.candidate_id = v.candidate_id
                WHERE COALESCE(json_extract(c.payload, '$.origin'), 'discovered') = 'discovered'
                  AND v.created_at >= ?
                """,
                (marker,),
            ).fetchone()
        best = row["best"] if row else None
        if best is None:
            return None
        score = float(best)
        return score if math.isfinite(score) else None

    def add_run(self, manifest: RunManifest) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?)",
                (manifest.run_id, manifest.model_dump_json(), manifest.started_at.isoformat()),
            )

    def append_event(self, run_id: str, sequence: int, payload: dict[str, Any]) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO events(run_id, sequence, payload) VALUES (?, ?, ?)",
                (run_id, sequence, json.dumps(payload, default=str)),
            )

    def load_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT payload FROM events WHERE run_id = ? ORDER BY sequence", (run_id,)
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT c.candidate_id, c.name, c.contract,
                       COALESCE(json_extract(c.payload, '$.origin'), 'discovered') AS origin,
                       e.tier, e.accepted, v.score, v.runtime_seconds
                FROM candidates c
                LEFT JOIN evidence e ON e.id = (
                    SELECT id FROM evidence
                    WHERE candidate_id = c.candidate_id ORDER BY id DESC LIMIT 1
                )
                LEFT JOIN evaluations v ON v.id = (
                    SELECT id FROM evaluations
                    WHERE candidate_id = c.candidate_id ORDER BY id DESC LIMIT 1
                )
                ORDER BY c.created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def pareto_frontier(self) -> list[FrontierItem]:
        items = [
            FrontierItem.model_validate(row)
            for row in self.recent(limit=100_000)
            if row["accepted"] and row["score"] is not None
        ]
        frontier: list[FrontierItem] = []
        for item in items:
            dominated = any(
                other.candidate_id != item.candidate_id
                and other.score >= item.score
                and other.runtime_seconds <= item.runtime_seconds
                and (other.score > item.score or other.runtime_seconds < item.runtime_seconds)
                for other in items
            )
            if not dominated:
                frontier.append(item)
        return sorted(frontier, key=lambda item: (-item.score, item.runtime_seconds))
