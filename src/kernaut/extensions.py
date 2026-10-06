"""Find installed extensions from the names registered in Python package metadata.

An extension package maps each name to a function that creates a task, model,
or baseline. These mappings are called entry points.
"""

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
    """Load the function registered for one selected extension name.

    Report an error if no package or multiple packages register that name.
    The returned function creates the selected task, model, or baseline.
    """
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
