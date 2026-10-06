from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import IntEnum, StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class ContractKind(StrEnum):
    UNVERIFIED = "unverified"
    FEATURE_MAP = "feature_map"
    RESIDUAL_FEATURE_MAP = "residual_feature_map"
    ADDITIVE_FEATURE_MAP = "additive_feature_map"
    SPECTRAL = "spectral"
    INPUT_TRANSFORM = "input_transform"
    RESIDUAL_INPUT_TRANSFORM = "residual_input_transform"
    CLOSURE = "closure"


class CandidateOrigin(StrEnum):
    DISCOVERED = "discovered"
    BASELINE = "baseline"


class EvidenceTier(IntEnum):
    EXECUTABLE = 0
    EMPIRICAL = 1
    CONTRACT_CERTIFIED = 2


class CandidateBundle(BaseModel):
    """Immutable, content-addressed proposal emitted by the synthesis agent."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    contract: ContractKind
    origin: CandidateOrigin = CandidateOrigin.DISCOVERED
    source: str = Field(min_length=1, max_length=100_000)
    parameters: dict[str, Any] = Field(default_factory=dict)
    parents: tuple[str, ...] = ()
    rationale: str = Field(default="", max_length=10_000)
    mathematical_form: str = Field(default="", max_length=20_000)
    psd_argument: str = Field(default="", max_length=10_000)
    novelty_claim: str = Field(default="", max_length=10_000)
    closest_known_kernel: str = Field(default="", max_length=2_000)
    expected_bo_behavior: str = Field(default="", max_length=5_000)
    construction_niche: str = Field(default="", max_length=200)
    equivalence_analysis: str = Field(default="", max_length=5_000)
    optimized_parameters: str = Field(default="", max_length=5_000)
    domain_scale_analysis: str = Field(default="", max_length=5_000)
    irrelevant_coordinate_mechanism: str = Field(default="", max_length=5_000)
    cross_coordinate_mechanism: str = Field(default="", max_length=5_000)
    falsification_test: str = Field(default="", max_length=5_000)
    feature_growth: str = Field(default="", max_length=1_000)
    expected_conditioning: str = Field(default="", max_length=2_000)
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def candidate_id(self) -> str:
        canonical = json.dumps(
            {
                "name": self.name,
                "contract": self.contract.value,
                "source": self.source,
                "parameters": self.parameters,
                "parents": self.parents,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class ParameterSpec(BaseModel):
    type: Literal["float", "int", "categorical"] = "float"
    scale: Literal["linear", "log"] = "linear"
    lower: float | None = None
    upper: float | None = None
    choices: list[Any] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_domain(self) -> ParameterSpec:
        if self.type == "categorical":
            if not self.choices:
                raise ValueError("categorical parameter requires nonempty choices")
            return self
        if self.lower is None or self.upper is None or self.lower >= self.upper:
            raise ValueError("numeric parameter requires lower < upper")
        if self.scale == "log" and self.lower <= 0:
            raise ValueError("log-scaled parameter requires a positive lower bound")
        return self


class FormulationRecord(BaseModel):
    formulation_id: str
    payload: dict[str, str]
    created_at: datetime = Field(default_factory=utc_now)


class CandidateDraft(BaseModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    contract: ContractKind
    source: str = Field(min_length=1, max_length=100_000)
    initial_parameters: dict[str, Any] = Field(default_factory=dict)
    parameter_space: dict[str, ParameterSpec] = Field(default_factory=dict)
    formulation_id: str
    parents: tuple[str, ...] = ()
    rationale: str = Field(default="", max_length=10_000)
    created_at: datetime = Field(default_factory=utc_now)

    @property
    def draft_id(self) -> str:
        canonical = json.dumps(
            {
                "name": self.name,
                "contract": self.contract.value,
                "source": self.source,
                "initial_parameters": self.initial_parameters,
                "parameter_space": {
                    key: value.model_dump(mode="json")
                    for key, value in self.parameter_space.items()
                },
                "formulation_id": self.formulation_id,
                "parents": self.parents,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class ParameterTrial(BaseModel):
    trial_id: str
    draft_id: str
    parameters: dict[str, Any]
    score: float
    evaluation: dict[str, Any]
    created_at: datetime = Field(default_factory=utc_now)


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str
    metrics: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


class EvidenceRecord(BaseModel):
    candidate_id: str
    tier: EvidenceTier | None
    accepted: bool
    checks: list[CheckResult]
    contract: ContractKind
    implementation_digest: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class EvaluationRecord(BaseModel):
    candidate_id: str
    score: float
    negative_log_likelihood: float
    runtime_seconds: float
    jitter: float
    condition_number: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class RunManifest(BaseModel):
    run_id: str
    seed: int
    config: dict[str, Any]
    python_version: str
    platform: str
    started_at: datetime = Field(default_factory=utc_now)
