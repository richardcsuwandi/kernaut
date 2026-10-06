from __future__ import annotations

import argparse
import json
import sys
import uuid
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv

from kernaut.agent import EvolutionarySynthesisController, HarnessTools, SynthesisController
from kernaut.archive import CandidateStore
from kernaut.config import AppConfig, ExecutionConfig
from kernaut.evaluation import (
    CHEM_INPUT_DIMENSION,
    GLUCOSE_INPUT_DIMENSION,
    ChemBenchKernelEvaluator,
    Dataset,
    FunctionalNoveltyEvaluator,
    GaussianProcessEvaluator,
    GlucoseKernelEvaluator,
    GreenhouseKernelEvaluator,
    MetaBOBenchmark,
    MetaKernelEvaluator,
    MetaSearchEvaluator,
    NoveltyPolicy,
    StandardBaselineRunner,
    chembench_baseline_candidates,
    glucose_baseline_candidates,
    greenhouse_baseline_candidates,
    meta_baseline_candidates,
)
from kernaut.evaluation.meta_bo import MetaSplit
from kernaut.execution import SubprocessExecutor
from kernaut.llm import build_model
from kernaut.models import CandidateBundle, EvaluationRecord
from kernaut.verification import VerificationPolicy, Verifier


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kernaut", description="Kernaut research harness")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Create an experiment workspace")
    init.add_argument("directory", type=Path)
    verify = sub.add_parser("verify", help="Verify a candidate bundle JSON file")
    verify.add_argument("candidate", type=Path)
    verify.add_argument("--config", type=Path)
    run = sub.add_parser("run", help="Run an agent synthesis campaign")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--context", type=Path, required=True)
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--archive", type=Path, required=True)
    run.add_argument("--run-id")
    meta_run = sub.add_parser(
        "meta-run", help="Evolve kernels across the procedural meta-BO training tasks"
    )
    meta_run.add_argument("--config", type=Path, required=True)
    meta_run.add_argument("--context", type=Path, required=True)
    meta_run.add_argument("--archive", type=Path, required=True)
    meta_run.add_argument("--run-id")
    meta_run.add_argument(
        "--strategy",
        choices=["conversational", "evolutionary"],
        default="conversational",
        help="Use one continuous agent conversation or quality-diverse evolutionary campaigns.",
    )
    meta_run.add_argument(
        "--iterations", type=_positive_int, default=12, help="Evolutionary campaign count."
    )
    meta_run.add_argument(
        "--rounds-per-iteration",
        type=_positive_int,
        default=10,
        help="Maximum model turns in each evolutionary campaign.",
    )
    meta_run.add_argument("--episodes-per-family", type=int, default=2)
    meta_run.add_argument(
        "--max-rounds", type=_positive_int, help="Override campaign.max_rounds from the config file"
    )
    meta_run.add_argument(
        "--max-tool-calls",
        type=_positive_int,
        help="Override campaign.max_tool_calls from the config file",
    )
    meta_run.add_argument(
        "--novelty-threshold",
        type=float,
        default=0.08,
        help="Minimum mean normalized centered-Gram distance from familiar kernels",
    )
    meta_benchmark = sub.add_parser(
        "meta-benchmark",
        help="Evaluate frozen archived kernels on meta-train, validation, and held-out BO tasks",
    )
    meta_benchmark.add_argument("--archive", type=Path, required=True)
    meta_benchmark.add_argument("--config", type=Path)
    meta_benchmark.add_argument(
        "--split", choices=["all", "train", "validation", "test"], default="all"
    )
    meta_benchmark.add_argument("--episodes-per-family", type=int, default=1)
    meta_benchmark.add_argument("--bo-steps", type=int, default=8)
    meta_benchmark.add_argument("--candidate-pool-size", type=int, default=128)
    meta_benchmark.add_argument("--limit", type=int, default=100)
    meta_benchmark.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help=(
            "Evaluate this frozen archived candidate (repeatable). Reference baselines are always "
            "included. Without this option, recent accepted candidates are evaluated."
        ),
    )
    baselines = sub.add_parser(
        "baselines", help="Tune and archive the standard kernel reference baselines"
    )
    baselines.add_argument("--data", type=Path, required=True)
    baselines.add_argument("--archive", type=Path, required=True)
    baselines.add_argument("--config", type=Path)
    ts_run = sub.add_parser(
        "ts-run", help="Evolve kernels across greenhouse-gas forecasting episodes"
    )
    ts_run.add_argument("--config", type=Path, required=True)
    ts_run.add_argument("--context", type=Path, required=True)
    ts_run.add_argument("--archive", type=Path, required=True)
    ts_run.add_argument("--run-id")
    ts_run.add_argument(
        "--strategy",
        choices=["conversational", "evolutionary"],
        default="conversational",
        help="Use one continuous agent conversation or quality-diverse evolutionary campaigns.",
    )
    ts_run.add_argument(
        "--iterations", type=_positive_int, default=12, help="Evolutionary campaign count."
    )
    ts_run.add_argument(
        "--rounds-per-iteration",
        type=_positive_int,
        default=10,
        help="Maximum model turns in each evolutionary campaign.",
    )
    ts_run.add_argument("--episodes-per-family", type=int, default=3)
    ts_run.add_argument(
        "--max-rounds", type=_positive_int, help="Override campaign.max_rounds from the config file"
    )
    ts_run.add_argument(
        "--max-tool-calls",
        type=_positive_int,
        help="Override campaign.max_tool_calls from the config file",
    )
    ts_run.add_argument(
        "--novelty-threshold",
        type=float,
        default=0.08,
        help="Minimum mean normalized centered-Gram distance from familiar kernels",
    )
    ts_benchmark = sub.add_parser(
        "ts-benchmark",
        help="Evaluate frozen archived kernels on train, validation, and held-out gas series",
    )
    ts_benchmark.add_argument("--archive", type=Path, required=True)
    ts_benchmark.add_argument("--config", type=Path)
    ts_benchmark.add_argument(
        "--split", choices=["all", "train", "validation", "test"], default="all"
    )
    ts_benchmark.add_argument("--episodes-per-family", type=int, default=1)
    ts_benchmark.add_argument(
        "--horizon-months",
        type=int,
        default=48,
        help=(
            "Forecast horizon length. N2O's 304-month record caps this at 88 "
            "with the default 216-month minimum training window."
        ),
    )
    ts_benchmark.add_argument("--limit", type=int, default=100)
    ts_benchmark.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help=(
            "Evaluate this frozen archived candidate (repeatable). Reference baselines are always "
            "included. Without this option, recent accepted candidates are evaluated."
        ),
    )
    chem_run = sub.add_parser(
        "chem-run", help="Evolve kernels on ChemBench enzyme-kinetics episodes"
    )
    chem_run.add_argument("--chembench-root", type=Path, required=True)
    chem_run.add_argument("--config", type=Path, required=True)
    chem_run.add_argument("--context", type=Path, required=True)
    chem_run.add_argument("--archive", type=Path, required=True)
    chem_run.add_argument("--run-id")
    chem_run.add_argument(
        "--strategy", choices=["conversational", "evolutionary"], default="conversational"
    )
    chem_run.add_argument("--iterations", type=_positive_int, default=12)
    chem_run.add_argument("--rounds-per-iteration", type=_positive_int, default=10)
    chem_run.add_argument("--episodes-per-domain", type=_positive_int, default=1)
    chem_run.add_argument("--train-points", type=_positive_int, default=20)
    chem_run.add_argument("--test-points", type=_positive_int, default=20)
    chem_run.add_argument("--noise-level", type=float, default=0.01)
    chem_run.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    chem_run.add_argument("--max-rounds", type=_positive_int, help="Override campaign.max_rounds")
    chem_run.add_argument(
        "--max-tool-calls", type=_positive_int, help="Override campaign.max_tool_calls"
    )
    chem_run.add_argument("--novelty-threshold", type=float, default=0.08)
    chem_run.add_argument(
        "--worst-domain-weight",
        type=float,
        default=0.0,
        help="Penalty weight on the worst per-domain CRPS, added to the mean in the fitness score",
    )
    chem_benchmark = sub.add_parser(
        "chem-benchmark",
        help="Evaluate frozen kernels on ChemBench train/validation/test domains",
    )
    chem_benchmark.add_argument("--chembench-root", type=Path, required=True)
    chem_benchmark.add_argument("--archive", type=Path, required=True)
    chem_benchmark.add_argument("--config", type=Path)
    chem_benchmark.add_argument(
        "--split", choices=["all", "train", "validation", "test"], default="all"
    )
    chem_benchmark.add_argument("--episodes-per-domain", type=_positive_int, default=1)
    chem_benchmark.add_argument("--train-points", type=_positive_int, default=20)
    chem_benchmark.add_argument("--test-points", type=_positive_int, default=20)
    chem_benchmark.add_argument("--noise-level", type=float, default=0.01)
    chem_benchmark.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    chem_benchmark.add_argument(
        "--worst-domain-weight",
        type=float,
        default=0.0,
        help="Penalty weight on the worst per-domain CRPS, added to the mean in the reported score",
    )
    chem_benchmark.add_argument("--limit", type=_positive_int, default=100)
    chem_benchmark.add_argument("--candidate-id", action="append", default=[])
    chem_benchmark.add_argument(
        "--candidate-file",
        type=Path,
        action="append",
        default=[],
        help="Evaluate a candidate JSON directly (repeatable).",
    )
    glucose_run = sub.add_parser(
        "glucose-run", help="Evolve kernels on GlucoseBench CGM forecasting (children)"
    )
    glucose_run.add_argument(
        "--data", type=Path, required=True, help="Output of examples/glucose/export_training.py"
    )
    glucose_run.add_argument("--config", type=Path, required=True)
    glucose_run.add_argument("--context", type=Path, required=True)
    glucose_run.add_argument("--archive", type=Path, required=True)
    glucose_run.add_argument("--run-id")
    glucose_run.add_argument(
        "--strategy", choices=["conversational", "evolutionary"], default="conversational"
    )
    glucose_run.add_argument("--iterations", type=_positive_int, default=12)
    glucose_run.add_argument("--rounds-per-iteration", type=_positive_int, default=10)
    glucose_run.add_argument(
        "--max-rounds", type=_positive_int, help="Override campaign.max_rounds"
    )
    glucose_run.add_argument(
        "--max-tool-calls", type=_positive_int, help="Override campaign.max_tool_calls"
    )
    glucose_run.add_argument("--novelty-threshold", type=float, default=0.08)
    glucose_benchmark = sub.add_parser(
        "glucose-benchmark",
        help="Score kernels on GlucoseBench training/validation leave-one-episode-out folds",
    )
    glucose_benchmark.add_argument("--data", type=Path, required=True)
    glucose_benchmark.add_argument("--archive", type=Path, required=True)
    glucose_benchmark.add_argument("--config", type=Path)
    glucose_benchmark.add_argument("--split", choices=["all", "train", "validation"], default="all")
    glucose_benchmark.add_argument("--limit", type=_positive_int, default=100)
    glucose_benchmark.add_argument("--candidate-id", action="append", default=[])
    inspect = sub.add_parser("inspect", help="Inspect candidates or the Pareto frontier")
    inspect.add_argument("--archive", type=Path, required=True)
    inspect.add_argument("--frontier", action="store_true")
    inspect.add_argument("--limit", type=int, default=20)
    extensions = sub.add_parser("extensions", help="List installed extension names")
    extensions.add_argument("--group", choices=["tasks", "models", "baselines"])
    task_run = sub.add_parser("task-run", help="Run a task from an installed extension")
    task_run.add_argument("--task", required=True)
    task_run.add_argument("--options", type=Path, help="Task options as a JSON object")
    task_run.add_argument("--config", type=Path, required=True)
    task_run.add_argument("--archive", type=Path, required=True)
    task_run.add_argument("--context", type=Path, help="Override the task's proposer context")
    task_run.add_argument("--baseline", action="append", default=[])
    task_run.add_argument("--run-id")
    viz = sub.add_parser("viz", help="Open the local archive viewer")
    viz.add_argument("--archive", type=Path, required=True)
    viz.add_argument("--host", default="127.0.0.1")
    viz.add_argument("--port", type=int, default=8765)
    viz.add_argument("--open", action="store_true", dest="open_browser")
    return parser


def _load_candidate(path: Path) -> CandidateBundle:
    return CandidateBundle.model_validate_json(path.read_text())


def _baseline_winners(
    store: CandidateStore,
    verifier: Verifier,
    evaluator: Any,
    *,
    archive: bool,
    final_evaluator: MetaSearchEvaluator | None = None,
    candidates: list[CandidateBundle] | None = None,
) -> list[CandidateBundle]:
    if candidates is None:
        candidates = meta_baseline_candidates()
    winners: dict[str, tuple[CandidateBundle, EvaluationRecord]] = {}
    for candidate in candidates:
        evaluation = evaluator.evaluate(candidate)
        kind = str(candidate.parameters.get("tree", {}).get("kind", candidate.name))
        previous = winners.get(kind)
        if previous is None or evaluation.score > previous[1].score:
            winners[kind] = (candidate, evaluation)
    selected = [item[0] for item in winners.values()]
    if archive:
        for candidate, evaluation in winners.values():
            if final_evaluator is not None:
                evaluation = final_evaluator.evaluate(candidate)
            store.add_candidate(candidate)
            evidence = verifier.verify(candidate)
            store.add_evidence(evidence)
            if not evidence.accepted:
                raise RuntimeError(f"baseline {candidate.name} failed verification")
            store.add_evaluation(evaluation)
            print(
                f"[baseline] {candidate.name}: score={evaluation.score:.6f}",
                file=sys.stderr,
                flush=True,
            )
    return selected


def _collect_frozen_candidates(
    store: CandidateStore,
    requested_ids: list[str],
    limit: int,
    seen: set[str],
) -> list[CandidateBundle]:
    selected: list[CandidateBundle] = []
    if requested_ids:
        for candidate_id in requested_ids:
            candidate = store.get_candidate(candidate_id)
            frozen_evidence = store.latest_evidence(candidate_id)
            if candidate is None:
                raise ValueError(f"frozen candidate {candidate_id!r} was not found")
            if frozen_evidence is None or not frozen_evidence.accepted:
                raise ValueError(f"frozen candidate {candidate_id!r} is not accepted")
            if candidate.candidate_id not in seen:
                selected.append(candidate)
                seen.add(candidate.candidate_id)
    else:
        for row in store.recent(limit=limit):
            if not row["accepted"] or row["candidate_id"] in seen:
                continue
            candidate = store.get_candidate(row["candidate_id"])
            if candidate is not None:
                selected.append(candidate)
                seen.add(candidate.candidate_id)
    return selected


def _archived_discoveries(store: CandidateStore) -> list[CandidateBundle]:
    return [
        candidate
        for row in store.recent(limit=100_000)
        if row["origin"] == "discovered"
        and row["accepted"]
        and (candidate := store.get_candidate(row["candidate_id"])) is not None
    ]


def _launch_campaign(
    config: AppConfig,
    store: CandidateStore,
    tools: HarnessTools,
    context_text: str,
    *,
    strategy: str,
    iterations: int,
    rounds_per_iteration: int,
    max_rounds: int | None,
    max_tool_calls: int | None,
    run_id: str | None,
) -> str:
    model = build_model(config)
    effective_tool_calls = (
        max_tool_calls if max_tool_calls is not None else config.campaign.max_tool_calls
    )
    if strategy == "evolutionary":
        result = EvolutionarySynthesisController(
            model,
            tools,
            store,
            iterations=iterations,
            rounds_per_iteration=rounds_per_iteration,
            tool_calls_per_iteration=effective_tool_calls,
            seed=config.campaign.seed,
        ).run(context_text, run_id=run_id or uuid.uuid4().hex[:12])
    else:
        result = SynthesisController(
            model,
            tools,
            store,
            max_rounds=max_rounds if max_rounds is not None else config.campaign.max_rounds,
            max_tool_calls=effective_tool_calls,
            seed=config.campaign.seed,
        ).run(context_text, run_id=run_id)
    return result.model_dump_json(indent=2)


def _forecast_execution_config(execution: ExecutionConfig) -> ExecutionConfig:
    """Allow larger worker responses for forecasting episodes.

    Forecasting evaluates up to roughly 570 monthly points at once, so the trusted
    worker must transport Gram matrices far larger than the interactive default. The
    candidate stdout/stderr caps inside the worker are unchanged.
    """
    minimum_bytes = 8 * 1024 * 1024
    if execution.max_output_bytes >= minimum_bytes:
        return execution
    return execution.model_copy(update={"max_output_bytes": minimum_bytes})


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    load_dotenv(Path.cwd() / ".env")
    if args.command == "extensions":
        from kernaut.extensions import GROUPS, extension_names

        groups = [f"kernaut.{args.group}"] if args.group else GROUPS
        print(json.dumps({group: extension_names(group) for group in groups}, indent=2))
        return 0
    if args.command == "task-run":
        from dataclasses import replace

        from kernaut.tasks import load_baselines, load_task, run_task

        config = AppConfig.from_toml(args.config)
        options = json.loads(args.options.read_text()) if args.options else {}
        if not isinstance(options, dict):
            raise ValueError("Task options must be a JSON object")
        executor = SubprocessExecutor(config.execution)
        task = load_task(args.task, executor, options)
        if args.context:
            task = replace(task, context=args.context.read_text())
        baselines = [c for name in args.baseline for c in load_baselines(name, task)]
        result = run_task(
            task,
            build_model(config),
            CandidateStore(args.archive),
            executor,
            campaign=config.campaign,
            baselines=baselines,
            run_id=args.run_id,
        )
        print(result.model_dump_json(indent=2))
        return 0
    if args.command == "viz":
        from kernaut.viz import serve

        serve(args.archive, host=args.host, port=args.port, open_browser=args.open_browser)
        return 0
    if args.command == "init":
        targets = [args.directory / name for name in ("campaign.toml", "context.md", "data.json")]
        if any(path.exists() for path in targets):
            raise FileExistsError("Experiment files already exist. Choose a new directory.")
        args.directory.mkdir(parents=True, exist_ok=True)
        template = files("kernaut.templates").joinpath("modelscope-qwen.toml")
        (args.directory / "campaign.toml").write_text(template.read_text())
        (args.directory / "context.md").write_text(
            "# Kernel synthesis task\n\nDescribe the input domain and modelling goals.\n"
        )
        (args.directory / "data.json").write_text(
            json.dumps({"x": [[-1.0], [0.0], [1.0]], "y": [1.0, 0.0, 1.0]}, indent=2) + "\n"
        )
        print(f"Initialized {args.directory}")
        return 0
    if args.command == "verify":
        execution = AppConfig.from_toml(args.config).execution if args.config else ExecutionConfig()
        evidence = Verifier(SubprocessExecutor(execution)).verify(_load_candidate(args.candidate))
        print(evidence.model_dump_json(indent=2))
        return 0 if evidence.accepted else 1
    if args.command == "inspect":
        store = CandidateStore(args.archive)
        if args.frontier:
            payload = [item.model_dump(mode="json") for item in store.pareto_frontier()]
        else:
            payload = store.recent(args.limit)
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if args.command == "chem-benchmark":
        execution = AppConfig.from_toml(args.config).execution if args.config else ExecutionConfig()
        store = CandidateStore(args.archive)
        executor = SubprocessExecutor(execution)
        verifier = Verifier(
            executor,
            VerificationPolicy(
                trials=2,
                contract_trials=1,
                input_dimension=CHEM_INPUT_DIMENSION,
                input_dimensions=(CHEM_INPUT_DIMENSION,),
            ),
        )
        chem_common = {
            "episodes_per_domain": args.episodes_per_domain,
            "train_points": args.train_points,
            "test_points": args.test_points,
            "noise_level": args.noise_level,
            "difficulty": args.difficulty,
            "worst_domain_weight": args.worst_domain_weight,
        }
        chem_baseline_fitness = ChemBenchKernelEvaluator(
            executor,
            args.chembench_root,
            split="train",
            **chem_common,
        )
        chem_candidates = _baseline_winners(
            store,
            verifier,
            chem_baseline_fitness,
            archive=False,
            candidates=chembench_baseline_candidates(),
        )
        chem_seen = {candidate.candidate_id for candidate in chem_candidates}
        chem_candidates.extend(
            _collect_frozen_candidates(store, args.candidate_id, args.limit, chem_seen)
        )
        for path in args.candidate_file:
            file_candidate = _load_candidate(path)
            file_evidence = verifier.verify(file_candidate)
            if not file_evidence.accepted:
                raise ValueError(f"candidate file {path} did not pass verification")
            if file_candidate.candidate_id not in chem_seen:
                chem_candidates.append(file_candidate)
                chem_seen.add(file_candidate.candidate_id)

        chem_splits: tuple[MetaSplit, ...] = (
            ("train", "validation", "test")
            if args.split == "all"
            else (cast(MetaSplit, args.split),)
        )
        chem_evaluators = {
            split: ChemBenchKernelEvaluator(
                executor,
                args.chembench_root,
                split=split,
                **chem_common,
            )
            for split in chem_splits
        }
        chem_report: list[dict[str, Any]] = []
        for chem_candidate in chem_candidates:
            chem_split_results: dict[str, Any] = {}
            for chem_split, chem_split_evaluator in chem_evaluators.items():
                chem_result = chem_split_evaluator.evaluate(chem_candidate)
                chem_split_results[chem_split] = {
                    "score": chem_result.score,
                    "mean_crps": chem_result.metadata["mean_crps"],
                    "mean_nlpd": chem_result.metadata["mean_nlpd"],
                    "mean_rmse": chem_result.metadata["mean_rmse"],
                    "worst_domain_crps": chem_result.metadata["worst_domain_crps"],
                    "successful_tasks": chem_result.metadata["successful_tasks"],
                    "failed_tasks": chem_result.metadata["failed_tasks"],
                    "tasks": chem_result.metadata["tasks"],
                }
            chem_report.append(
                {
                    "candidate_id": chem_candidate.candidate_id,
                    "name": chem_candidate.name,
                    "origin": chem_candidate.origin.value,
                    "splits": chem_split_results,
                }
            )
            print(
                f"[chem-benchmark] finished {chem_candidate.name}",
                file=sys.stderr,
                flush=True,
            )
        chem_sort_split = "test" if "test" in chem_splits else chem_splits[-1]
        chem_report.sort(key=lambda item: item["splits"][chem_sort_split]["mean_crps"])
        print(json.dumps(chem_report, indent=2, default=str))
        return 0
    if args.command == "glucose-benchmark":
        execution = AppConfig.from_toml(args.config).execution if args.config else ExecutionConfig()
        store = CandidateStore(args.archive)
        executor = SubprocessExecutor(_forecast_execution_config(execution))
        verifier = Verifier(
            executor,
            VerificationPolicy(
                trials=2,
                contract_trials=1,
                input_dimension=GLUCOSE_INPUT_DIMENSION,
                input_dimensions=(GLUCOSE_INPUT_DIMENSION,),
            ),
        )
        glucose_candidates = _baseline_winners(
            store,
            verifier,
            GlucoseKernelEvaluator(executor, args.data, split="train"),
            archive=False,
            candidates=glucose_baseline_candidates(),
        )
        glucose_seen = {candidate.candidate_id for candidate in glucose_candidates}
        glucose_candidates.extend(
            _collect_frozen_candidates(store, args.candidate_id, args.limit, glucose_seen)
        )
        glucose_splits: tuple[MetaSplit, ...] = (
            ("train", "validation") if args.split == "all" else (cast(MetaSplit, args.split),)
        )
        glucose_evaluators = {
            split: GlucoseKernelEvaluator(executor, args.data, split=split)
            for split in glucose_splits
        }
        glucose_report: list[dict[str, Any]] = []
        for glucose_candidate in glucose_candidates:
            glucose_split_results: dict[str, Any] = {}
            for glucose_split, glucose_evaluator in glucose_evaluators.items():
                glucose_result = glucose_evaluator.evaluate(glucose_candidate)
                glucose_split_results[glucose_split] = {
                    key: glucose_result.metadata[key]
                    for key in (
                        "mean_crps",
                        "geomean_cgm_nmse",
                        "mean_nlpd",
                        "mean_rmse",
                        "successful_tasks",
                        "failed_tasks",
                    )
                }
            glucose_report.append(
                {
                    "candidate_id": glucose_candidate.candidate_id,
                    "name": glucose_candidate.name,
                    "origin": glucose_candidate.origin.value,
                    "splits": glucose_split_results,
                }
            )
            print(
                f"[glucose-benchmark] finished {glucose_candidate.name}",
                file=sys.stderr,
                flush=True,
            )
        glucose_report.sort(key=lambda item: item["splits"][glucose_splits[-1]]["mean_crps"])
        print(json.dumps(glucose_report, indent=2, default=str))
        return 0
    if args.command == "meta-benchmark":
        execution = AppConfig.from_toml(args.config).execution if args.config else ExecutionConfig()
        store = CandidateStore(args.archive)
        executor = SubprocessExecutor(execution)
        verifier = Verifier(
            executor,
            VerificationPolicy(
                trials=2,
                contract_trials=1,
                input_dimension=8,
                input_dimensions=(2, 5, 6, 8),
            ),
        )
        baseline_fitness = MetaKernelEvaluator(executor, split="train", episodes_per_family=1)
        candidates = _baseline_winners(store, verifier, baseline_fitness, archive=False)
        seen = {candidate.candidate_id for candidate in candidates}
        candidates.extend(_collect_frozen_candidates(store, args.candidate_id, args.limit, seen))

        splits: tuple[MetaSplit, ...] = (
            ("train", "validation", "test")
            if args.split == "all"
            else (cast(MetaSplit, args.split),)
        )
        bo_benchmark = MetaBOBenchmark(
            executor,
            episodes_per_family=args.episodes_per_family,
            bo_steps=args.bo_steps,
            candidate_pool_size=args.candidate_pool_size,
        )
        report: list[dict[str, Any]] = []
        for candidate in candidates:
            split_results: dict[str, Any] = {}
            for split in splits:
                predictive = MetaKernelEvaluator(
                    executor,
                    split=split,
                    episodes_per_family=args.episodes_per_family,
                ).evaluate(candidate)
                split_results[split] = {
                    "predictive": {
                        "score": predictive.score,
                        "mean_crps": predictive.metadata["mean_crps"],
                        "mean_nlpd": predictive.metadata["mean_nlpd"],
                        "mean_rmse": predictive.metadata["mean_rmse"],
                        "successful_tasks": predictive.metadata["successful_tasks"],
                        "failed_tasks": predictive.metadata["failed_tasks"],
                    },
                    "bo": bo_benchmark.evaluate(candidate, split),
                }
            report.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "name": candidate.name,
                    "origin": candidate.origin.value,
                    "splits": split_results,
                }
            )
            print(f"[meta-benchmark] finished {candidate.name}", file=sys.stderr, flush=True)
        if "test" in splits:
            report.sort(key=lambda item: item["splits"]["test"]["bo"]["regret_auc"])
        print(json.dumps(report, indent=2, default=str))
        return 0
    if args.command == "baselines":
        execution = AppConfig.from_toml(args.config).execution if args.config else ExecutionConfig()
        store = CandidateStore(args.archive)
        executor = SubprocessExecutor(execution)
        dataset = Dataset.model_validate_json(args.data.read_text())
        verifier = Verifier(
            executor,
            VerificationPolicy(input_dimension=len(dataset.x[0])),
        )
        runner = StandardBaselineRunner(
            store,
            verifier,
            GaussianProcessEvaluator(executor),
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
        winners = runner.run(dataset)
        print(
            json.dumps(
                [
                    {
                        "candidate_id": candidate.candidate_id,
                        "name": candidate.name,
                        "score": evaluation.score,
                        "negative_log_likelihood": evaluation.negative_log_likelihood,
                        "runtime_seconds": evaluation.runtime_seconds,
                        "jitter": evaluation.jitter,
                        "condition_number": evaluation.condition_number,
                    }
                    for candidate, evaluation in winners
                ],
                indent=2,
            )
        )
        return 0
    if args.command == "meta-run":
        config = AppConfig.from_toml(args.config)
        store = CandidateStore(args.archive)
        executor = SubprocessExecutor(config.execution)
        verifier = Verifier(
            executor,
            VerificationPolicy(
                trials=2,
                contract_trials=1,
                input_dimension=8,
                input_dimensions=(2, 5, 6, 8),
                seed=config.campaign.seed,
            ),
        )
        novelty_evaluator = FunctionalNoveltyEvaluator(
            executor,
            archive_candidates=lambda: _archived_discoveries(store),
            minimum_distance=args.novelty_threshold,
        )
        meta_evaluator = MetaKernelEvaluator(
            executor,
            split="train",
            episodes_per_family=args.episodes_per_family,
            novelty_evaluator=novelty_evaluator,
        )
        search_evaluator = MetaSearchEvaluator(
            meta_evaluator,
            MetaBOBenchmark(
                executor,
                episodes_per_family=args.episodes_per_family,
                bo_steps=3,
                candidate_pool_size=64,
            ),
        )
        tuning_evaluator = MetaKernelEvaluator(
            executor,
            split="train",
            episodes_per_family=1,
            train_points_base=4,
            test_points=16,
            novelty_evaluator=novelty_evaluator,
        )
        _baseline_winners(
            store,
            verifier,
            meta_evaluator,
            archive=True,
            final_evaluator=search_evaluator,
        )
        anchor = Dataset(x=[[0.0, 0.0], [1.0, 1.0]], y=[0.0, 1.0])
        tools = HarnessTools(
            store,
            verifier,
            search_evaluator,
            anchor,
            novelty_policy=NoveltyPolicy(),
            parameter_evaluator=tuning_evaluator,
        )
        print(
            _launch_campaign(
                config,
                store,
                tools,
                args.context.read_text(),
                strategy=args.strategy,
                iterations=args.iterations,
                rounds_per_iteration=args.rounds_per_iteration,
                max_rounds=args.max_rounds,
                max_tool_calls=args.max_tool_calls,
                run_id=args.run_id,
            )
        )
        return 0
    if args.command == "chem-run":
        config = AppConfig.from_toml(args.config)
        store = CandidateStore(args.archive)
        executor = SubprocessExecutor(config.execution)
        verifier = Verifier(
            executor,
            VerificationPolicy(
                trials=2,
                contract_trials=1,
                input_dimension=CHEM_INPUT_DIMENSION,
                input_dimensions=(CHEM_INPUT_DIMENSION,),
                seed=config.campaign.seed,
            ),
        )
        novelty_evaluator = FunctionalNoveltyEvaluator(
            executor,
            archive_candidates=lambda: _archived_discoveries(store),
            minimum_distance=args.novelty_threshold,
            probe_dimensions=(CHEM_INPUT_DIMENSION,),
        )
        chem_search_evaluator = ChemBenchKernelEvaluator(
            executor,
            args.chembench_root,
            split="train",
            episodes_per_domain=args.episodes_per_domain,
            train_points=args.train_points,
            test_points=args.test_points,
            noise_level=args.noise_level,
            difficulty=args.difficulty,
            worst_domain_weight=args.worst_domain_weight,
            novelty_evaluator=novelty_evaluator,
        )
        _baseline_winners(
            store,
            verifier,
            chem_search_evaluator,
            archive=True,
            candidates=chembench_baseline_candidates(),
        )
        anchor = Dataset(
            x=[[0.0] * CHEM_INPUT_DIMENSION, [1.0] * CHEM_INPUT_DIMENSION],
            y=[0.0, 1.0],
        )
        tools = HarnessTools(
            store,
            verifier,
            chem_search_evaluator,
            anchor,
            novelty_policy=NoveltyPolicy(),
            parameter_evaluator=chem_search_evaluator,
        )
        print(
            _launch_campaign(
                config,
                store,
                tools,
                args.context.read_text(),
                strategy=args.strategy,
                iterations=args.iterations,
                rounds_per_iteration=args.rounds_per_iteration,
                max_rounds=args.max_rounds,
                max_tool_calls=args.max_tool_calls,
                run_id=args.run_id,
            )
        )
        return 0
    if args.command == "glucose-run":
        config = AppConfig.from_toml(args.config)
        store = CandidateStore(args.archive)
        executor = SubprocessExecutor(_forecast_execution_config(config.execution))
        verifier = Verifier(
            executor,
            VerificationPolicy(
                trials=2,
                contract_trials=1,
                input_dimension=GLUCOSE_INPUT_DIMENSION,
                input_dimensions=(GLUCOSE_INPUT_DIMENSION,),
                seed=config.campaign.seed,
            ),
        )
        novelty_evaluator = FunctionalNoveltyEvaluator(
            executor,
            archive_candidates=lambda: _archived_discoveries(store),
            minimum_distance=args.novelty_threshold,
            probe_dimensions=(GLUCOSE_INPUT_DIMENSION,),
        )
        glucose_search_evaluator = GlucoseKernelEvaluator(
            executor, args.data, split="train", novelty_evaluator=novelty_evaluator
        )
        _baseline_winners(
            store,
            verifier,
            glucose_search_evaluator,
            archive=True,
            candidates=glucose_baseline_candidates(),
        )
        anchor = Dataset(
            x=[[0.0] * GLUCOSE_INPUT_DIMENSION, [1.0] * GLUCOSE_INPUT_DIMENSION],
            y=[0.0, 1.0],
        )
        tools = HarnessTools(
            store,
            verifier,
            glucose_search_evaluator,
            anchor,
            novelty_policy=NoveltyPolicy(),
            parameter_evaluator=glucose_search_evaluator,
        )
        print(
            _launch_campaign(
                config,
                store,
                tools,
                args.context.read_text(),
                strategy=args.strategy,
                iterations=args.iterations,
                rounds_per_iteration=args.rounds_per_iteration,
                max_rounds=args.max_rounds,
                max_tool_calls=args.max_tool_calls,
                run_id=args.run_id,
            )
        )
        return 0
    if args.command == "ts-run":
        config = AppConfig.from_toml(args.config)
        store = CandidateStore(args.archive)
        executor = SubprocessExecutor(_forecast_execution_config(config.execution))
        verifier = Verifier(
            executor,
            VerificationPolicy(
                trials=2,
                contract_trials=1,
                input_dimension=1,
                input_dimensions=(1, 2, 4, 8),
                seed=config.campaign.seed,
            ),
        )
        novelty_evaluator = FunctionalNoveltyEvaluator(
            executor,
            archive_candidates=lambda: _archived_discoveries(store),
            minimum_distance=args.novelty_threshold,
        )
        gas_evaluator = GreenhouseKernelEvaluator(
            executor,
            split="train",
            episodes_per_family=args.episodes_per_family,
            novelty_evaluator=novelty_evaluator,
        )
        gas_tuning_evaluator = GreenhouseKernelEvaluator(
            executor,
            split="train",
            episodes_per_family=1,
            horizon_months=24,
            min_train_months=192,
            novelty_evaluator=novelty_evaluator,
        )
        _baseline_winners(
            store,
            verifier,
            gas_evaluator,
            archive=True,
            candidates=greenhouse_baseline_candidates(),
        )
        anchor = Dataset(x=[[0.0], [1.0]], y=[0.0, 1.0])
        tools = HarnessTools(
            store,
            verifier,
            gas_evaluator,
            anchor,
            novelty_policy=NoveltyPolicy(),
            parameter_evaluator=gas_tuning_evaluator,
        )
        print(
            _launch_campaign(
                config,
                store,
                tools,
                args.context.read_text(),
                strategy=args.strategy,
                iterations=args.iterations,
                rounds_per_iteration=args.rounds_per_iteration,
                max_rounds=args.max_rounds,
                max_tool_calls=args.max_tool_calls,
                run_id=args.run_id,
            )
        )
        return 0
    if args.command == "ts-benchmark":
        execution = _forecast_execution_config(
            AppConfig.from_toml(args.config).execution if args.config else ExecutionConfig()
        )
        store = CandidateStore(args.archive)
        executor = SubprocessExecutor(execution)
        verifier = Verifier(
            executor,
            VerificationPolicy(
                trials=2,
                contract_trials=1,
                input_dimension=1,
                input_dimensions=(1, 2, 4, 8),
            ),
        )
        gas_baseline_fitness = GreenhouseKernelEvaluator(
            executor,
            split="train",
            episodes_per_family=args.episodes_per_family,
            horizon_months=args.horizon_months,
        )
        candidates = _baseline_winners(
            store,
            verifier,
            gas_baseline_fitness,
            archive=False,
            candidates=greenhouse_baseline_candidates(),
        )
        seen = {candidate.candidate_id for candidate in candidates}
        candidates.extend(_collect_frozen_candidates(store, args.candidate_id, args.limit, seen))

        ts_splits: tuple[MetaSplit, ...] = (
            ("train", "validation", "test")
            if args.split == "all"
            else (cast(MetaSplit, args.split),)
        )
        ts_report: list[dict[str, Any]] = []
        for candidate in candidates:
            ts_split_results: dict[str, Any] = {}
            for split in ts_splits:
                predictive = GreenhouseKernelEvaluator(
                    executor,
                    split=split,
                    episodes_per_family=args.episodes_per_family,
                    horizon_months=args.horizon_months,
                ).evaluate(candidate)
                ts_split_results[split] = {
                    "score": predictive.score,
                    "mean_crps": predictive.metadata["mean_crps"],
                    "mean_nlpd": predictive.metadata["mean_nlpd"],
                    "mean_rmse": predictive.metadata["mean_rmse"],
                    "successful_tasks": predictive.metadata["successful_tasks"],
                    "failed_tasks": predictive.metadata["failed_tasks"],
                }
            ts_report.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "name": candidate.name,
                    "origin": candidate.origin.value,
                    "splits": ts_split_results,
                }
            )
            print(f"[ts-benchmark] finished {candidate.name}", file=sys.stderr, flush=True)
        if "test" in ts_splits:
            ts_report.sort(key=lambda item: item["splits"]["test"]["mean_crps"])
        print(json.dumps(ts_report, indent=2, default=str))
        return 0
    if args.command == "run":
        config = AppConfig.from_toml(args.config)
        store = CandidateStore(args.archive)
        executor = SubprocessExecutor(config.execution)
        dataset = Dataset.model_validate_json(args.data.read_text())
        verifier = Verifier(
            executor,
            VerificationPolicy(seed=config.campaign.seed, input_dimension=len(dataset.x[0])),
        )
        standard_evaluator = GaussianProcessEvaluator(executor)
        StandardBaselineRunner(
            store,
            verifier,
            standard_evaluator,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        ).run(dataset)
        tools = HarnessTools(store, verifier, standard_evaluator, dataset)
        standard_controller = SynthesisController(
            build_model(config),
            tools,
            store,
            max_rounds=config.campaign.max_rounds,
            max_tool_calls=config.campaign.max_tool_calls,
            seed=config.campaign.seed,
        )
        standard_result = standard_controller.run(args.context.read_text(), run_id=args.run_id)
        print(standard_result.model_dump_json(indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
