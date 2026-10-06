"""Lazy discovery of installed extensions using Python package entry points."""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any

GROUPS = ("kernaut.tasks", "kernaut.models", "kernaut.baselines")


def extension_names(group: str) -> tuple[str, ...]:
    """List installed names without importing or executing extension code."""
    if group not in GROUPS:
        raise ValueError(f"Unknown extension group: {group}")
    return tuple(sorted({entry.name for entry in entry_points(group=group)}))


def load_extension(group: str, name: str) -> Any:
    """Load exactly one selected factory, rejecting ambiguous registrations."""
    extension_names(group)
    matches = list(entry_points(group=group, name=name))
    if not matches:
        raise ValueError(f"Unknown {group} extension {name!r}. Install its package first.")
    if len(matches) != 1:
        raise ValueError(f"Multiple packages register {group} extension {name!r}")
    factory = matches[0].load()
    if not callable(factory):
        raise TypeError(f"{group} extension {name!r} must be a callable factory")
    return factory
