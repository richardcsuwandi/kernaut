from __future__ import annotations

import argparse
import json
import math
import mimetypes
import sqlite3
import webbrowser
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

STATIC_ROOT = files("kernaut.viz.static")
TIER_LABELS = {
    None: "Rejected",
    0: "Executable",
    1: "Empirical",
    2: "Contract certified",
}


def _connect_readonly(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(f"archive not found: {path}")
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True, timeout=2)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _decode(raw: str | None) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _json_safe(value: Any) -> Any:
    """Convert a value to JSON that a browser can read.

    Python's JSON encoder can emit NaN and Infinity. Those values are not valid JSON,
    and a browser's ``Response.json()`` method rejects them. Failed search trials can
    record these values to indicate failure. Represent them as missing values in
    the response sent to the viewer.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _encode_json(payload: Any) -> bytes:
    return json.dumps(_json_safe(payload), default=str, allow_nan=False).encode()


def _frontier_ids(candidates: list[dict[str, Any]]) -> set[str]:
    eligible = [
        candidate
        for candidate in candidates
        if candidate["accepted"] and candidate["score"] is not None
    ]
    frontier: set[str] = set()
    for item in eligible:
        dominated = any(
            other["candidate_id"] != item["candidate_id"]
            and other["score"] >= item["score"]
            and other["runtime_seconds"] <= item["runtime_seconds"]
            and (
                other["score"] > item["score"] or other["runtime_seconds"] < item["runtime_seconds"]
            )
            for other in eligible
        )
        if not dominated:
            frontier.add(item["candidate_id"])
    return frontier


def _progress_timeline(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return latest candidate scores in submission order and track the best score."""
    names = {candidate["candidate_id"]: candidate["name"] for candidate in candidates}
    scored = [
        candidate
        for candidate in candidates
        if candidate["score"] is not None
        and isinstance(candidate.get("evaluation"), dict)
        and candidate["evaluation"].get("created_at")
    ]
    # Order candidates by submission time. Baselines can be evaluated again in later runs.
    # Ordering by those later evaluations would change the displayed discovery sequence.
    scored.sort(key=lambda candidate: str(candidate.get("created_at") or ""))
    timeline: list[dict[str, Any]] = []
    running_best: float | None = None
    for index, candidate in enumerate(scored):
        score = float(candidate["score"])
        improves = running_best is None or score > running_best
        evaluation = candidate.get("evaluation") or {}
        metadata = evaluation.get("metadata") if isinstance(evaluation, dict) else {}
        parents = [names.get(parent_id, parent_id) for parent_id in candidate.get("parents", [])]
        origin = candidate["origin"]
        if origin == "baseline":
            label = "baseline"
        elif parents:
            label = f"revision of {parents[0]}"
        else:
            label = "new root"
        change_note = (
            candidate.get("mathematical_form") or candidate.get("optimized_parameters") or ""
        )
        timeline.append(
            {
                "index": index,
                "candidate_id": candidate["candidate_id"],
                "name": candidate["name"],
                "origin": origin,
                "accepted": bool(candidate["accepted"]),
                "tier": candidate["tier"],
                "score": score,
                "created_at": evaluation["created_at"],
                "run_ids": candidate.get("run_ids", []),
                "parents": parents,
                "label": label,
                "change_note": change_note[:160],
                "niche": candidate.get("construction_niche", ""),
                "novelty_distance": (
                    metadata.get("functional_novelty") if isinstance(metadata, dict) else None
                ),
                "running_best": running_best,
                "improves_best": improves,
            }
        )
        if improves:
            running_best = score
    return timeline


def load_snapshot(path: str | Path) -> dict[str, Any]:
    archive = Path(path)
    with _connect_readonly(archive) as db:
        run_rows = db.execute(
            """
            SELECT r.run_id, r.payload, r.created_at, COUNT(e.id) AS event_count
            FROM runs r LEFT JOIN events e ON e.run_id = r.run_id
            GROUP BY r.run_id ORDER BY r.created_at DESC
            """
        ).fetchall()
        event_rows = db.execute(
            "SELECT run_id, sequence, payload, created_at FROM events ORDER BY run_id, sequence"
        ).fetchall()
        candidate_rows = db.execute(
            """
            SELECT c.candidate_id, c.payload,
                   e.payload AS evidence_payload,
                   v.payload AS evaluation_payload
            FROM candidates c
            LEFT JOIN evidence e ON e.id = (
                SELECT id FROM evidence
                WHERE candidate_id = c.candidate_id ORDER BY id DESC LIMIT 1
            )
            LEFT JOIN evaluations v ON v.id = (
                SELECT id FROM evaluations
                WHERE candidate_id = c.candidate_id ORDER BY id DESC LIMIT 1
            )
            ORDER BY c.created_at DESC
            """
        ).fetchall()

    runs: list[dict[str, Any]] = []
    for row in run_rows:
        manifest = _decode(row["payload"]) or {}
        runs.append({**manifest, "event_count": row["event_count"]})

    decoded_events = [(row, _decode(row["payload"]) or {}) for row in event_rows]
    call_names: dict[str, str] = {}
    call_candidates: dict[str, str] = {}
    candidate_runs: dict[str, set[str]] = {}
    for row, payload in decoded_events:
        for call in payload.get("tool_calls") or []:
            call_names[call.get("call_id", "")] = call.get("name", "unknown")
        content_payload = _decode(payload.get("content")) if payload.get("role") == "tool" else None
        if isinstance(content_payload, dict) and content_payload.get("candidate_id"):
            candidate_id = content_payload["candidate_id"]
            candidate_runs.setdefault(candidate_id, set()).add(row["run_id"])
            if payload.get("tool_call_id"):
                call_candidates[payload["tool_call_id"]] = candidate_id

    events: list[dict[str, Any]] = []
    for row, payload in decoded_events:
        tool_name = call_names.get(payload.get("tool_call_id", ""))
        content_payload = _decode(payload.get("content")) if payload.get("role") == "tool" else None
        candidate_ids: set[str] = set()
        if isinstance(content_payload, dict) and content_payload.get("candidate_id"):
            candidate_ids.add(content_payload["candidate_id"])
        for call in payload.get("tool_calls") or []:
            arguments = call.get("arguments") or {}
            candidate_id = arguments.get("candidate_id") or call_candidates.get(
                call.get("call_id", "")
            )
            if candidate_id:
                candidate_ids.add(candidate_id)
        events.append(
            {
                "run_id": row["run_id"],
                "sequence": row["sequence"],
                "created_at": row["created_at"],
                "role": payload.get("role", "unknown"),
                "content": payload.get("content") or "",
                "tool_calls": payload.get("tool_calls") or [],
                "tool_name": tool_name,
                "tool_result": content_payload,
                "candidate_ids": sorted(candidate_ids),
            }
        )

    candidates: list[dict[str, Any]] = []
    for row in candidate_rows:
        candidate = _decode(row["payload"]) or {}
        evidence = _decode(row["evidence_payload"])
        evaluation = _decode(row["evaluation_payload"])
        tier = evidence.get("tier") if isinstance(evidence, dict) else None
        accepted = bool(evidence and evidence.get("accepted"))
        candidates.append(
            {
                **candidate,
                "origin": candidate.get("origin", "discovered"),
                "candidate_id": row["candidate_id"],
                "evidence": evidence,
                "evaluation": evaluation,
                "tier": tier,
                "tier_label": TIER_LABELS.get(tier, f"Tier {tier}"),
                "accepted": accepted,
                "score": evaluation.get("score") if isinstance(evaluation, dict) else None,
                "runtime_seconds": (
                    evaluation.get("runtime_seconds") if isinstance(evaluation, dict) else None
                ),
                "run_ids": sorted(candidate_runs.get(row["candidate_id"], set())),
            }
        )
    frontier = _frontier_ids(candidates)
    progress = _progress_timeline(candidates)
    baseline_candidates = [
        candidate
        for candidate in candidates
        if candidate["origin"] == "baseline" and candidate["score"] is not None
    ]
    best_baseline = (
        max(baseline_candidates, key=lambda candidate: candidate["score"])
        if baseline_candidates
        else None
    )
    for candidate in candidates:
        candidate["on_frontier"] = candidate["candidate_id"] in frontier
        candidate["baseline_delta"] = (
            candidate["score"] - best_baseline["score"]
            if best_baseline is not None and candidate["score"] is not None
            else None
        )

    return {
        "schema_version": 3,
        "archive": str(archive.resolve()),
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "runs": len(runs),
            "candidates": len(candidates),
            "verified": sum(candidate["accepted"] for candidate in candidates),
            "evaluated": sum(candidate["evaluation"] is not None for candidate in candidates),
            "frontier": len(frontier),
            "baselines": len(baseline_candidates),
        },
        "baseline_reference": (
            {
                "candidate_id": best_baseline["candidate_id"],
                "name": best_baseline["name"],
                "score": best_baseline["score"],
                "negative_log_likelihood": best_baseline["evaluation"].get(
                    "negative_log_likelihood"
                ),
                "runtime_seconds": best_baseline["runtime_seconds"],
                "jitter": best_baseline["evaluation"].get("jitter"),
                "condition_number": best_baseline["evaluation"].get("condition_number"),
            }
            if best_baseline is not None
            else None
        ),
        "runs": runs,
        "candidates": candidates,
        "progress": progress,
        "events": events,
    }


class ViewerHandler(BaseHTTPRequestHandler):
    archive_path: Path

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route == "/api/snapshot":
            try:
                self._send_json(load_snapshot(self.archive_path))
            except Exception as error:
                self._send_json({"error": f"{type(error).__name__}: {error}"}, status=500)
            return
        name = "index.html" if route in {"", "/"} else route.lstrip("/")
        if "/" in name or name not in {"index.html", "style.css", "app.js"}:
            self.send_error(404)
            return
        resource = STATIC_ROOT.joinpath(name)
        try:
            body = resource.read_bytes()
        except FileNotFoundError:
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = _encode_json(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve(
    archive: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
) -> None:
    archive_path = Path(archive).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive not found: {archive_path}")
    handler = type("ArchiveViewerHandler", (ViewerHandler,), {"archive_path": archive_path})
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}"
    print(f"Kernaut viewer: {url}")
    print(f"Archive: {archive_path}")
    print("Press Ctrl-C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped.")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local Kernaut archive viewer")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args(argv)
    serve(args.archive, host=args.host, port=args.port, open_browser=args.open_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
