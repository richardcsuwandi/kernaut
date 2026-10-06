from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from copy import deepcopy
from typing import Any, Protocol

from kernaut.archive import CandidateStore
from kernaut.evaluation import Dataset
from kernaut.evaluation.novelty import NoveltyPolicy, formulation_id
from kernaut.models import (
    CandidateBundle,
    CandidateDraft,
    ContractKind,
    EvaluationRecord,
    FormulationRecord,
    ParameterSpec,
    ParameterTrial,
)
from kernaut.verification import Verifier


class CandidateEvaluator(Protocol):
    def evaluate(self, candidate: CandidateBundle, dataset: Dataset) -> EvaluationRecord: ...


class HarnessTools:
    def __init__(
        self,
        store: CandidateStore,
        verifier: Verifier,
        evaluator: CandidateEvaluator,
        dataset: Dataset,
        novelty_policy: NoveltyPolicy | None = None,
        parameter_evaluator: CandidateEvaluator | None = None,
    ) -> None:
        self.store = store
        self.verifier = verifier
        self.evaluator = evaluator
        self.parameter_evaluator = parameter_evaluator or evaluator
        self.dataset = dataset
        self.novelty_policy = novelty_policy

    @property
    def schemas(self) -> list[dict[str, Any]]:
        schemas = [
            {
                "name": "submit_candidate",
                "description": (
                    "Archive a new kernel program proposal without verifying or scoring it."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "contract": {
                            "type": "string",
                            "enum": [item.value for item in ContractKind],
                        },
                        "source": {"type": "string"},
                        "parameters": {"type": "object"},
                        "parents": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Parent candidate IDs for a genuine revision. Omit or pass an "
                                "empty array for an independent root idea."
                            ),
                        },
                        "rationale": {"type": "string"},
                        "formulation_id": {"type": "string"},
                    },
                    "required": ["name", "contract", "source"],
                    "additionalProperties": False,
                },
            },
            {
                "name": "verify_candidate",
                "description": (
                    "Run deterministic T0-T2 verification and return structured evidence."
                ),
                "input_schema": self._id_schema(),
            },
            {
                "name": "evaluate_candidate",
                "description": (
                    "Fit and score an accepted kernel with the deterministic GP backend."
                ),
                "input_schema": self._id_schema(),
            },
            {
                "name": "inspect_candidate",
                "description": ("Inspect a candidate, its latest evidence, and latest evaluation."),
                "input_schema": self._id_schema(),
            },
            {
                "name": "query_archive",
                "description": (
                    "List recent candidates or the accepted quality-cost Pareto frontier."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "view": {"type": "string", "enum": ["recent", "frontier"]},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["view"],
                    "additionalProperties": False,
                },
            },
        ]
        if self.novelty_policy is not None:
            schemas[0:0] = [
                {
                    "name": "propose_formulation",
                    "description": (
                        "Record the mathematical kernel idea before writing code. Returns the "
                        "formulation_id required to stage or directly submit a candidate."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            field: (
                                {
                                    "type": "string",
                                    "enum": [
                                        "input_geometry",
                                        "spectral_periodic",
                                        "feature_projection",
                                        "additive_interaction",
                                        "multiresolution_local",
                                        "residual_composite",
                                        "open_ended",
                                    ],
                                }
                                if field == "construction_niche"
                                else {"type": "string"}
                            )
                            for field in self.novelty_policy.required_fields
                        },
                        "required": list(self.novelty_policy.required_fields),
                        "additionalProperties": False,
                    },
                },
                {
                    "name": "stage_candidate",
                    "description": (
                        "Persist an unsubmitted kernel implementation and parameter search space. "
                        "Use this before backend tuning; drafts are not candidate lineage nodes."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "contract": {
                                "type": "string",
                                "enum": [item.value for item in ContractKind],
                            },
                            "source": {"type": "string"},
                            "initial_parameters": {"type": "object"},
                            "parameter_space": {
                                "type": "object",
                                "additionalProperties": {
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "type": "string",
                                            "enum": ["float", "int", "categorical"],
                                        },
                                        "scale": {
                                            "type": "string",
                                            "enum": ["linear", "log"],
                                        },
                                        "lower": {"type": "number"},
                                        "upper": {"type": "number"},
                                        "choices": {"type": "array"},
                                    },
                                    "additionalProperties": False,
                                },
                            },
                            "parents": {"type": "array", "items": {"type": "string"}},
                            "rationale": {"type": "string"},
                            "formulation_id": {"type": "string"},
                        },
                        "required": ["name", "contract", "source", "formulation_id"],
                        "additionalProperties": False,
                    },
                },
                {
                    "name": "verify_staged_candidate",
                    "description": "Verify a staged implementation using its initial parameters.",
                    "input_schema": self._draft_id_schema(),
                },
                {
                    "name": "evaluate_parameters",
                    "description": (
                        "Evaluate one manual parameter setting for a staged implementation."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "draft_id": {"type": "string"},
                            "parameters": {"type": "object"},
                        },
                        "required": ["draft_id", "parameters"],
                        "additionalProperties": False,
                    },
                },
                {
                    "name": "optimize_parameters",
                    "description": (
                        "Deterministically tune a staged candidate before final submission."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "draft_id": {"type": "string"},
                            "budget": {"type": "integer", "minimum": 1, "maximum": 64},
                        },
                        "required": ["draft_id"],
                        "additionalProperties": False,
                    },
                },
                {
                    "name": "submit_staged_candidate",
                    "description": (
                        "Archive a verified staged candidate using a selected parameter trial."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "draft_id": {"type": "string"},
                            "trial_id": {"type": "string"},
                        },
                        "required": ["draft_id", "trial_id"],
                        "additionalProperties": False,
                    },
                },
            ]
        return schemas

    @staticmethod
    def _id_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"candidate_id": {"type": "string"}},
            "required": ["candidate_id"],
            "additionalProperties": False,
        }

    @staticmethod
    def _draft_id_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"draft_id": {"type": "string"}},
            "required": ["draft_id"],
            "additionalProperties": False,
        }

    @property
    def handlers(self) -> dict[str, Callable[..., str]]:
        handlers: dict[str, Callable[..., str]] = {
            "submit_candidate": self.submit_candidate,
            "verify_candidate": self.verify_candidate,
            "evaluate_candidate": self.evaluate_candidate,
            "inspect_candidate": self.inspect_candidate,
            "query_archive": self.query_archive,
        }
        if self.novelty_policy is not None:
            handlers["propose_formulation"] = self.propose_formulation
            handlers["stage_candidate"] = self.stage_candidate
            handlers["verify_staged_candidate"] = self.verify_staged_candidate
            handlers["evaluate_parameters"] = self.evaluate_parameters
            handlers["optimize_parameters"] = self.optimize_parameters
            handlers["submit_staged_candidate"] = self.submit_staged_candidate
        return handlers

    def propose_formulation(self, **arguments: Any) -> str:
        if self.novelty_policy is None:
            return self._json({"ok": False, "error": "novelty mode is not enabled"})
        payload = {
            field: str(arguments.get(field, "")).strip()
            for field in self.novelty_policy.required_fields
        }
        missing = [field for field, value in payload.items() if not value]
        if missing:
            return self._json({"ok": False, "error": "missing: " + ", ".join(missing)})
        if payload["construction_niche"] not in self.novelty_policy.construction_niches:
            return self._json(
                {
                    "ok": False,
                    "error": "construction_niche must be one of: "
                    + ", ".join(sorted(self.novelty_policy.construction_niches)),
                }
            )
        identifier = formulation_id(payload)
        self.store.add_formulation(FormulationRecord(formulation_id=identifier, payload=payload))
        return self._json(
            {
                "ok": True,
                "formulation_id": identifier,
                "next": "Stage an implementation and tune it before final submission.",
            }
        )

    def submit_candidate(self, **arguments: Any) -> str:
        try:
            identifier = arguments.pop("formulation_id", None)
            if self.novelty_policy is not None:
                record = self.store.get_formulation(str(identifier))
                if record is None:
                    raise ValueError(
                        "call propose_formulation first and pass its returned formulation_id"
                    )
                arguments.update(record.payload)
            candidate = CandidateBundle.model_validate(arguments)
            if self.novelty_policy is not None:
                self.novelty_policy.validate_submission(candidate, self.store)
            self.store.add_candidate(candidate)
            return self._json(
                {
                    "ok": True,
                    "candidate_id": candidate.candidate_id,
                    "next": "Call verify_candidate before evaluation.",
                }
            )
        except Exception as error:
            return self._error(error)

    def stage_candidate(self, **arguments: Any) -> str:
        try:
            identifier = str(arguments.get("formulation_id", ""))
            record = self.store.get_formulation(identifier)
            if record is None:
                raise ValueError("formulation not found; call propose_formulation first")
            draft = CandidateDraft.model_validate(arguments)
            candidate = self._candidate_from_draft(draft, draft.initial_parameters)
            if self.novelty_policy is not None:
                self.novelty_policy.validate_submission(candidate, self.store)
            self.store.add_draft(draft)
            return self._json(
                {
                    "ok": True,
                    "draft_id": draft.draft_id,
                    "next": "Verify the draft, then optimize or manually evaluate parameters.",
                }
            )
        except Exception as error:
            return self._error(error)

    def verify_staged_candidate(self, draft_id: str) -> str:
        draft = self.store.get_draft(draft_id)
        if draft is None:
            return self._json({"ok": False, "error": "draft not found"})
        evidence = self.verifier.verify(self._candidate_from_draft(draft, draft.initial_parameters))
        return evidence.model_dump_json(indent=2)

    def evaluate_parameters(self, draft_id: str, parameters: dict[str, Any]) -> str:
        draft = self.store.get_draft(draft_id)
        if draft is None:
            return self._json({"ok": False, "error": "draft not found"})
        try:
            merged = deepcopy(draft.initial_parameters)
            for path, value in parameters.items():
                self._set_parameter(merged, path, value)
            trial = self._run_parameter_trial(draft, merged)
            return trial.model_dump_json(indent=2)
        except Exception as error:
            return self._error(error)

    def optimize_parameters(self, draft_id: str, budget: int = 16) -> str:
        draft = self.store.get_draft(draft_id)
        if draft is None:
            return self._json({"ok": False, "error": "draft not found"})
        if budget < 1 or budget > 64:
            return self._json({"ok": False, "error": "budget must be between 1 and 64"})
        trials: list[ParameterTrial] = []
        configurations = [deepcopy(draft.initial_parameters)]
        configurations.extend(self._parameter_configurations(draft, budget - 1))
        for parameters in configurations:
            try:
                trials.append(self._run_parameter_trial(draft, parameters))
            except Exception as error:
                failure = ParameterTrial(
                    trial_id=self._trial_id(draft.draft_id, parameters),
                    draft_id=draft.draft_id,
                    parameters=parameters,
                    score=float("-inf"),
                    evaluation={"error": f"{type(error).__name__}: {error}"},
                )
                self.store.add_parameter_trial(failure)
                trials.append(failure)
        best = max(trials, key=lambda item: item.score)
        return self._json(
            {
                "ok": math.isfinite(best.score),
                "draft_id": draft_id,
                "best_trial_id": best.trial_id,
                "best_score": best.score,
                "best_parameters": best.parameters,
                "trials": [
                    {"trial_id": item.trial_id, "score": item.score, "parameters": item.parameters}
                    for item in sorted(trials, key=lambda item: item.score, reverse=True)
                ],
                "next": "Submit the selected parameter trial as an immutable candidate.",
            }
        )

    def submit_staged_candidate(self, draft_id: str, trial_id: str) -> str:
        draft = self.store.get_draft(draft_id)
        trial = self.store.get_parameter_trial(trial_id)
        if draft is None or trial is None or trial.draft_id != draft_id:
            return self._json({"ok": False, "error": "draft or matching trial not found"})
        try:
            candidate = self._candidate_from_draft(draft, trial.parameters)
            evidence = self.verifier.verify(candidate)
            if not evidence.accepted:
                return self._json(
                    {
                        "ok": False,
                        "error": "selected parameters failed final Tier-2 verification",
                        "evidence": evidence.model_dump(mode="json"),
                    }
                )
            self.store.add_candidate(candidate)
            self.store.add_evidence(evidence)
            return self._json(
                {
                    "ok": True,
                    "candidate_id": candidate.candidate_id,
                    "selected_trial_id": trial_id,
                    "next": "Run full candidate evaluation.",
                }
            )
        except Exception as error:
            return self._error(error)

    def verify_candidate(self, candidate_id: str) -> str:
        candidate = self.store.get_candidate(candidate_id)
        if candidate is None:
            return self._json({"ok": False, "error": "candidate not found"})
        evidence = self.verifier.verify(candidate)
        self.store.add_evidence(evidence)
        return evidence.model_dump_json(indent=2)

    def evaluate_candidate(self, candidate_id: str) -> str:
        candidate = self.store.get_candidate(candidate_id)
        if candidate is None:
            return self._json({"ok": False, "error": "candidate not found"})
        evidence = self.store.latest_evidence(candidate_id)
        if evidence is None or evidence.tier is None or int(evidence.tier) < 1:
            return self._json(
                {"ok": False, "error": "candidate must have at least Tier 1 evidence"}
            )
        try:
            evaluation = self.evaluator.evaluate(candidate, self.dataset)
            evaluation = self._enrich_evaluation(candidate, evaluation)
            self.store.add_evaluation(evaluation)
            return evaluation.model_dump_json(indent=2)
        except Exception as error:
            return self._error(error)

    def inspect_candidate(self, candidate_id: str) -> str:
        candidate = self.store.get_candidate(candidate_id)
        if candidate is None:
            return self._json({"ok": False, "error": "candidate not found"})
        evidence = self.store.latest_evidence(candidate_id)
        evaluation = self.store.latest_evaluation(candidate_id)
        return self._json(
            {
                "candidate": candidate.model_dump(mode="json"),
                "evidence": evidence.model_dump(mode="json") if evidence else None,
                "evaluation": evaluation.model_dump(mode="json") if evaluation else None,
            }
        )

    def query_archive(self, view: str, limit: int = 20) -> str:
        if view == "frontier":
            return self._json([item.model_dump() for item in self.store.pareto_frontier()])
        return self._json(self.store.recent(limit=limit))

    def _candidate_from_draft(
        self, draft: CandidateDraft, parameters: dict[str, Any]
    ) -> CandidateBundle:
        formulation = self.store.get_formulation(draft.formulation_id)
        if formulation is None:
            raise ValueError("draft formulation not found")
        return CandidateBundle.model_validate(
            {
                "name": draft.name,
                "contract": draft.contract,
                "source": draft.source,
                "parameters": deepcopy(parameters),
                "parents": draft.parents,
                "rationale": draft.rationale,
                **formulation.payload,
            }
        )

    def _run_parameter_trial(
        self, draft: CandidateDraft, parameters: dict[str, Any]
    ) -> ParameterTrial:
        candidate = self._candidate_from_draft(draft, parameters)
        evaluation = self._enrich_evaluation(
            candidate, self.parameter_evaluator.evaluate(candidate, self.dataset)
        )
        complexity_penalty = 0.001 * math.log1p(len(draft.parameter_space))
        base_selection_score = float(evaluation.metadata.get("selection_score", evaluation.score))
        tuned_score = base_selection_score - complexity_penalty
        payload = evaluation.model_dump(mode="json")
        payload["raw_score"] = evaluation.score
        payload["complexity_penalty"] = complexity_penalty
        trial = ParameterTrial(
            trial_id=self._trial_id(draft.draft_id, parameters),
            draft_id=draft.draft_id,
            parameters=deepcopy(parameters),
            score=tuned_score,
            evaluation=payload,
        )
        self.store.add_parameter_trial(trial)
        return trial

    def _enrich_evaluation(
        self, candidate: CandidateBundle, evaluation: EvaluationRecord
    ) -> EvaluationRecord:
        metadata = deepcopy(evaluation.metadata)
        tasks = metadata.get("tasks")
        if not isinstance(tasks, list):
            metadata.setdefault("selection_score", evaluation.score)
            return evaluation.model_copy(update={"metadata": metadata})

        baseline_by_task: dict[str, float] = {}
        for row in self.store.recent(limit=100_000):
            if row.get("origin") != "baseline" or row.get("score") is None:
                continue
            reference = self.store.latest_evaluation(str(row["candidate_id"]))
            if reference is None:
                continue
            reference_tasks = reference.metadata.get("tasks", [])
            if not isinstance(reference_tasks, list):
                continue
            for item in reference_tasks:
                if not isinstance(item, dict) or "task" not in item or "crps" not in item:
                    continue
                name = str(item["task"])
                crps = float(item["crps"])
                baseline_by_task[name] = min(crps, baseline_by_task.get(name, float("inf")))

        deltas: list[float] = []
        family_deltas: dict[str, list[float]] = {}
        wins = 0
        conditions: list[float] = []
        for item in tasks:
            if not isinstance(item, dict) or "task" not in item or "crps" not in item:
                continue
            task_name = str(item["task"])
            baseline = baseline_by_task.get(task_name)
            if baseline is not None and baseline > 0:
                relative = (baseline - float(item["crps"])) / baseline
                deltas.append(relative)
                family_deltas.setdefault(task_name.split(":", 1)[0], []).append(relative)
                wins += relative > 0
            condition = item.get("condition_number")
            if isinstance(condition, (int, float)) and math.isfinite(condition):
                conditions.append(float(condition))

        mean_improvement = sum(deltas) / len(deltas) if deltas else 0.0
        worst_family = min(
            ((sum(values) / len(values), family) for family, values in family_deltas.items()),
            default=(0.0, "unknown"),
        )
        max_condition = max(conditions, default=float(evaluation.condition_number))
        stability_penalty = max(0.0, math.log10(max(max_condition, 1.0)) - 4.0) * 0.01
        metadata["baseline_relative"] = {
            "mean_crps_improvement": mean_improvement,
            "episode_wins": wins,
            "episode_count": len(deltas),
            "per_family": {
                family: sum(values) / len(values) for family, values in family_deltas.items()
            },
            "worst_family": worst_family[1],
            "worst_family_improvement": worst_family[0],
        }
        metadata["stability"] = {
            "condition_p50": self._percentile(conditions, 0.5),
            "condition_p95": self._percentile(conditions, 0.95),
            "maximum_condition": max_condition,
        }
        novelty = metadata.get("functional_novelty", 0.0)
        novelty_penalty = float(metadata.get("novelty_penalty", 0.0))
        # A substantial quality improvement can justify a candidate close to a reference kernel.
        # A weak duplicate still receives the full soft penalty. This avoids a -2 discontinuity.
        novelty_relief = min(max(mean_improvement, 0.0) * 5.0, 1.0)
        adjusted_novelty_penalty = novelty_penalty * (1.0 - novelty_relief)
        metadata["quality_descriptors"] = {
            "construction_niche": candidate.construction_niche or candidate.contract.value,
            "feature_growth": candidate.feature_growth or "unknown",
            "novelty_band": self._novelty_band(float(novelty)),
        }
        bo = metadata.get("bo")
        bo_auc_improvement = 0.0
        final_regret_improvement = 0.0
        if isinstance(bo, dict):
            baseline_auc: list[float] = []
            baseline_final: list[float] = []
            for row in self.store.recent(limit=100_000):
                if row.get("origin") != "baseline":
                    continue
                reference = self.store.latest_evaluation(str(row["candidate_id"]))
                reference_bo = reference.metadata.get("bo") if reference is not None else None
                if isinstance(reference_bo, dict):
                    baseline_auc.append(float(reference_bo["regret_auc"]))
                    baseline_final.append(float(reference_bo["final_regret"]))
            if baseline_auc:
                best_auc = min(baseline_auc)
                bo_auc_improvement = (best_auc - float(bo["regret_auc"])) / max(best_auc, 1e-12)
            if baseline_final:
                best_final = min(baseline_final)
                final_regret_improvement = (best_final - float(bo["final_regret"])) / max(
                    best_final, 1e-12
                )
            quality = (
                0.55 * mean_improvement
                + 0.30 * bo_auc_improvement
                + 0.10 * final_regret_improvement
                + 0.05 * worst_family[0]
            )
        else:
            quality = mean_improvement
        metadata["selection_score"] = quality - stability_penalty - adjusted_novelty_penalty
        metadata["selection_components"] = {
            "mean_crps_improvement": mean_improvement,
            "bo_auc_improvement": bo_auc_improvement,
            "final_regret_improvement": final_regret_improvement,
            "worst_family_improvement": worst_family[0],
            "stability_penalty": stability_penalty,
            "novelty_penalty": adjusted_novelty_penalty,
        }
        return evaluation.model_copy(update={"metadata": metadata})

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        position = quantile * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    @staticmethod
    def _novelty_band(distance: float) -> str:
        if distance < 0.03:
            return "duplicate"
        if distance < 0.08:
            return "near_reference"
        if distance < 0.15:
            return "distinct"
        return "strongly_distinct"

    def _parameter_configurations(self, draft: CandidateDraft, count: int) -> list[dict[str, Any]]:
        paths = sorted(draft.parameter_space)
        primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
        if len(paths) > len(primes):
            raise ValueError(f"at most {len(primes)} tunable parameters are supported")
        configurations: list[dict[str, Any]] = []
        for sample_index in range(1, count + 1):
            parameters = deepcopy(draft.initial_parameters)
            for dimension, path in enumerate(paths):
                unit = self._radical_inverse(sample_index, primes[dimension])
                value = self._sample_parameter(
                    draft.parameter_space[path],
                    unit,
                    self._parameter_value(parameters, path),
                )
                self._set_parameter(parameters, path, value)
            configurations.append(parameters)
        return configurations

    @staticmethod
    def _radical_inverse(index: int, base: int) -> float:
        result = 0.0
        factor = 1.0 / base
        while index:
            result += factor * (index % base)
            index //= base
            factor /= base
        return result

    @staticmethod
    def _sample_parameter(spec: ParameterSpec, unit: float, template: Any = None) -> Any:
        """Sample a parameter while preserving the shape of its initial value.

        Each parameter specification defines a scalar domain. Kernel programs may expose
        vectors or matrices, such as one frequency per spectral component. Apply scalar
        sampling recursively to those values. Do not replace an array with a scalar.
        """
        if isinstance(template, dict):
            return {
                key: HarnessTools._sample_parameter(spec, unit, value)
                for key, value in template.items()
            }
        if isinstance(template, (list, tuple)):
            values = [
                HarnessTools._sample_parameter(
                    spec,
                    (unit + (index + 1) * 0.6180339887498949) % 1.0,
                    value,
                )
                for index, value in enumerate(template)
            ]
            return tuple(values) if isinstance(template, tuple) else values
        if spec.type == "categorical":
            return deepcopy(spec.choices[min(int(unit * len(spec.choices)), len(spec.choices) - 1)])
        assert spec.lower is not None and spec.upper is not None
        if spec.scale == "log":
            log_value = math.log(spec.lower) + unit * (math.log(spec.upper) - math.log(spec.lower))
            value = math.exp(log_value)
        else:
            value = spec.lower + unit * (spec.upper - spec.lower)
        return int(round(value)) if spec.type == "int" else value

    @staticmethod
    def _set_parameter(parameters: dict[str, Any], path: str, value: Any) -> None:
        parts = path.split(".")
        target = parameters
        for part in parts[:-1]:
            child = target.get(part)
            if not isinstance(child, dict):
                child = {}
                target[part] = child
            target = child
        target[parts[-1]] = deepcopy(value)

    @staticmethod
    def _parameter_value(parameters: dict[str, Any], path: str) -> Any:
        target: Any = parameters
        for part in path.split("."):
            if not isinstance(target, dict):
                return None
            target = target.get(part)
        return target

    @staticmethod
    def _trial_id(draft_id: str, parameters: dict[str, Any]) -> str:
        encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(f"{draft_id}:{encoded}".encode()).hexdigest()[:16]

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, indent=2, default=str)

    def _error(self, error: Exception) -> str:
        return self._json({"ok": False, "error": f"{type(error).__name__}: {error}"})
