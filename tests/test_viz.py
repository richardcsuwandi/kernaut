import json
import math
import platform
import sys
from importlib.resources import files

from kernaut.archive import CandidateStore
from kernaut.models import (
    CandidateBundle,
    EvaluationRecord,
    EvidenceRecord,
    EvidenceTier,
    RunManifest,
)
from kernaut.viz import load_snapshot
from kernaut.viz.server import _encode_json


def test_snapshot_exposes_source_metrics_evidence_and_trace(tmp_path) -> None:
    archive = tmp_path / "archive.sqlite"
    store = CandidateStore(archive)
    candidate = CandidateBundle(
        name="viewer_candidate",
        contract="feature_map",
        source="def feature_point(x, parameters): return x",
        rationale="Visible in the research ledger.",
    )
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
            score=2.5,
            negative_log_likelihood=-2.5,
            runtime_seconds=0.25,
            jitter=1e-6,
            condition_number=3.0,
        )
    )
    store.add_run(
        RunManifest(
            run_id="run-viz",
            seed=7,
            config={},
            python_version=sys.version,
            platform=platform.platform(),
        )
    )
    store.append_event(
        "run-viz",
        0,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "call_id": "call-1",
                    "name": "submit_candidate",
                    "arguments": {"name": candidate.name},
                }
            ],
        },
    )
    store.append_event(
        "run-viz",
        1,
        {
            "role": "tool",
            "content": json.dumps({"ok": True, "candidate_id": candidate.candidate_id}),
            "tool_call_id": "call-1",
        },
    )

    snapshot = load_snapshot(archive)
    assert snapshot["schema_version"] == 3
    assert snapshot["summary"] == {
        "runs": 1,
        "candidates": 1,
        "verified": 1,
        "evaluated": 1,
        "frontier": 1,
        "baselines": 0,
    }
    item = snapshot["candidates"][0]
    assert item["source"] == candidate.source
    assert item["score"] == 2.5
    assert item["tier_label"] == "Contract certified"
    assert item["on_frontier"]
    assert item["origin"] == "discovered"
    assert item["baseline_delta"] is None
    assert snapshot["baseline_reference"] is None
    assert item["run_ids"] == ["run-viz"]
    assert snapshot["events"][1]["tool_name"] == "submit_candidate"
    assert snapshot["events"][0]["candidate_ids"] == [candidate.candidate_id]
    assert snapshot["events"][1]["candidate_ids"] == [candidate.candidate_id]


def test_viewer_assets_are_packaged() -> None:
    static = files("kernaut.viz.static")
    app = static.joinpath("app.js").read_text()
    styles = static.joinpath("style.css").read_text()
    assert static.joinpath("index.html").is_file()
    assert static.joinpath("style.css").is_file()
    assert static.joinpath("app.js").is_file()
    assert 'data-tab="verification"' in static.joinpath("index.html").read_text()
    assert "function renderVerification()" in app
    assert "campaign events" in app
    assert "event.run_id === state.selectedRun" in app
    assert ".assurance-ladder" in styles
    assert ".event-body { max-height" not in styles


def test_viewer_json_replaces_non_finite_search_scores() -> None:
    payload = {
        "trials": [
            {"score": -math.inf},
            {"score": math.inf},
            {"score": math.nan},
            {"score": -0.5},
        ]
    }

    encoded = _encode_json(payload)

    assert json.loads(
        encoded, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token))
    ) == {
        "trials": [
            {"score": None},
            {"score": None},
            {"score": None},
            {"score": -0.5},
        ]
    }
