import sqlite3

import pytest

from kernaut.archive import CandidateStore
from kernaut.models import (
    CandidateBundle,
    EvaluationRecord,
    EvidenceRecord,
    EvidenceTier,
    RunManifest,
)


def test_archive_and_pareto_frontier(tmp_path) -> None:
    store = CandidateStore(tmp_path / "archive.sqlite")
    candidates = [
        CandidateBundle(
            name="first",
            contract="feature_map",
            source="def feature_point(x, parameters): return x",
        ),
        CandidateBundle(
            name="second",
            contract="input_transform",
            source="def transform_point(x, parameters): return x",
        ),
    ]
    for index, candidate in enumerate(candidates):
        store.add_candidate(candidate)
        store.add_evidence(
            EvidenceRecord(
                candidate_id=candidate.candidate_id,
                tier=EvidenceTier.CONTRACT_CERTIFIED,
                accepted=True,
                checks=[],
                contract=candidate.contract,
            )
        )
        store.add_evaluation(
            EvaluationRecord(
                candidate_id=candidate.candidate_id,
                score=float(index),
                negative_log_likelihood=float(-index),
                runtime_seconds=float(2 - index),
                jitter=1e-6,
                condition_number=2.0,
            )
        )
    frontier = store.pareto_frontier()
    assert [item.name for item in frontier] == ["second"]
    assert store.get_candidate(candidates[0].candidate_id) == candidates[0]
    assert store.path.is_absolute()


def test_archive_path_remains_valid_after_working_directory_change(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    store = CandidateStore("relative/archive.sqlite")
    store.add_run(
        RunManifest(
            run_id="durable-run",
            seed=0,
            config={},
            python_version="test",
            platform="test",
        )
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    store.append_event("durable-run", 0, {"role": "user", "content": "still durable"})
    assert store.load_events("durable-run")[0]["content"] == "still durable"


def test_archive_connection_is_closed_after_each_operation(tmp_path) -> None:
    store = CandidateStore(tmp_path / "archive.sqlite")

    with store._connect() as db:
        assert db.execute("SELECT 1").fetchone()[0] == 1

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        db.execute("SELECT 1")


def test_parameter_trials_load_earlier_failed_scores(tmp_path) -> None:
    store = CandidateStore(tmp_path / "archive.sqlite")
    with store._connect() as db:
        db.execute(
            "INSERT INTO candidate_drafts VALUES (?, ?, ?)",
            ("draft", "{}", "2026-01-01T00:00:00+00:00"),
        )
        db.execute(
            "INSERT INTO parameter_trials VALUES (?, ?, ?, ?, ?)",
            (
                "trial",
                "draft",
                float("-inf"),
                '{"trial_id":"trial","draft_id":"draft",'
                '"parameters":{},"score":null,"evaluation":{},'
                '"created_at":"2026-01-01T00:00:00+00:00"}',
                "2026-01-01T00:00:00+00:00",
            ),
        )

    trials = store.parameter_trials("draft")

    assert len(trials) == 1
    assert trials[0].score == float("-inf")


def test_failed_evaluation_with_infinite_condition_is_readable(tmp_path):
    import math

    store = CandidateStore(tmp_path / "archive.sqlite")
    candidate = CandidateBundle(
        name="failed_fit",
        contract="feature_map",
        source="def feature_point(x, parameters): return x",
    )
    store.add_candidate(candidate)
    record = EvaluationRecord(
        candidate_id=candidate.candidate_id,
        score=-2.0,
        negative_log_likelihood=2.0,
        runtime_seconds=0.1,
        jitter=1.0,
        condition_number=float("inf"),
        metadata={"failed_tasks": 1},
    )
    store.add_evaluation(record)
    loaded = store.latest_evaluation(candidate.candidate_id)
    assert loaded.score == -2.0
    assert math.isinf(loaded.condition_number)
