from __future__ import annotations

import math
import random
import sys
from collections import deque
from typing import Any

from kernaut.config import BanditConfig, ModelConfig

from .base import AssistantReply, ConversationMessage, LanguageModel


class ThompsonSampler:
    """Select model indices with a Beta-Bernoulli bandit over a sliding window.

    Each bandit arm represents one model. Rewards can be fractional. An update
    ``update(arm, r)`` with ``r in [0, 1]`` adds ``r`` to the arm's alpha mass and
    ``1 - r`` to its beta mass. An observation therefore provides fractional evidence
    instead of a binary success or failure.
    """

    def __init__(self, n_arms: int, *, bandit: BanditConfig) -> None:
        self.n_arms = n_arms
        self.bandit = bandit
        self.history: list[deque[float]] = [deque(maxlen=bandit.window_size) for _ in range(n_arms)]

    def select(self, rng: random.Random) -> int:
        """Use round-robin sampling until each arm has ``min_samples``, then Thompson sampling."""
        counts = [len(window) for window in self.history]
        if min(counts) < self.bandit.min_samples:
            return counts.index(min(counts))
        best_arm, best_draw = 0, -math.inf
        for arm in range(self.n_arms):
            alpha = self.bandit.alpha_prior + sum(self.history[arm])
            beta = self.bandit.beta_prior + sum(1.0 - reward for reward in self.history[arm])
            draw = rng.betavariate(alpha, beta)
            if draw > best_draw:
                best_arm, best_draw = arm, draw
        return best_arm

    def update(self, arm: int, reward: float) -> None:
        self.history[arm].append(min(1.0, max(0.0, reward)))

    def posterior_mean(self, arm: int) -> float:
        alpha = self.bandit.alpha_prior + sum(self.history[arm])
        beta = self.bandit.beta_prior + sum(1.0 - reward for reward in self.history[arm])
        return alpha / (alpha + beta)

    def has_data(self) -> bool:
        return any(self.history)


class EnsembleModel(LanguageModel):
    """Select models from a weighted ensemble using feedback from search campaigns.

    The evolutionary loop calls :meth:`begin_campaign` and :meth:`end_campaign`
    around each campaign. One selected model handles all requests in that campaign.
    Its campaign score provides the reward for the Thompson bandit.

    Outside a campaign, including conversational searches, each ``complete()`` call
    samples a model in proportion to its posterior mean. Before observations are
    available, sampling uses the configured prior weights.
    """

    def __init__(
        self,
        children: list[LanguageModel],
        configs: list[ModelConfig],
        *,
        seed: int,
        bandit: BanditConfig,
    ) -> None:
        if not children:
            raise ValueError("EnsembleModel requires at least one child model")
        self.children = list(children)
        self.labels = [config.model.rsplit("/", 1)[-1] for config in configs]
        self.prior_weights = [config.weight for config in configs]
        self.sampler = ThompsonSampler(len(children), bandit=bandit)
        self.rng = random.Random(seed)
        self.active: int | None = None
        self.picks = [0] * len(children)
        self._score_min: float | None = None
        self._score_max: float | None = None

    def begin_campaign(self) -> tuple[int, LanguageModel]:
        index = self.sampler.select(self.rng)
        self.active = index
        self.picks[index] += 1
        means = ", ".join(
            f"{label}={self.sampler.posterior_mean(arm):.2f}"
            for arm, label in enumerate(self.labels)
        )
        print(
            f"[ensemble] campaign assigned to {self.labels[index]} | posterior means: {means}",
            file=sys.stderr,
            flush=True,
        )
        return index, self.children[index]

    def end_campaign(self, index: int, score: float | None) -> None:
        self.active = None
        reward = self._normalize(score)
        self.sampler.update(index, reward)
        raw = "none" if score is None or not math.isfinite(score) else f"{score:.4f}"
        print(
            f"[ensemble] campaign finished | model={self.labels[index]} "
            f"best_score={raw} normalized_reward={reward:.3f}",
            file=sys.stderr,
            flush=True,
        )

    def complete(
        self,
        messages: list[ConversationMessage],
        tools: list[dict[str, Any]],
        system_prompt: str,
    ) -> AssistantReply:
        if self.active is not None:
            return self.children[self.active].complete(messages, tools, system_prompt)
        index = self._sample_by_posterior()
        return self.children[index].complete(messages, tools, system_prompt)

    def _normalize(self, score: float | None) -> float:
        """Map a campaign score to [0, 1] using the minimum and maximum observed across
        campaigns.
        """
        if score is None or not math.isfinite(score):
            return 0.0
        if self._score_min is None or score < self._score_min:
            self._score_min = score
        if self._score_max is None or score > self._score_max:
            self._score_max = score
        span = self._score_max - self._score_min
        if span <= 0.0:
            return 0.5
        return (score - self._score_min) / span

    def _sample_by_posterior(self) -> int:
        weights = (
            [self.sampler.posterior_mean(arm) for arm in range(len(self.children))]
            if self.sampler.has_data()
            else self.prior_weights
        )
        return self.rng.choices(range(len(self.children)), weights=weights, k=1)[0]
