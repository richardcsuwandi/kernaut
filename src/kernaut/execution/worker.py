from __future__ import annotations

import builtins
import contextlib
import io
import json
import sys
import time
import traceback
import types
from typing import Any

import numpy as np

from kernaut.contracts.runtime import (
    ALLOWED_IMPORT_ROOTS,
    DISALLOWED_CALLS,
    build_gram_matrix,
)
from kernaut.models import CandidateBundle


def _install_resource_limits(limits: dict[str, int]) -> dict[str, str]:
    if sys.platform == "win32":
        return {"resource": "resource limits unavailable on Windows"}
    import resource

    requested = {
        resource.RLIMIT_CPU: limits["cpu_seconds"],
        resource.RLIMIT_AS: limits["memory_bytes"],
        resource.RLIMIT_FSIZE: limits["file_bytes"],
        resource.RLIMIT_NOFILE: limits["open_files"],
    }
    warnings: dict[str, str] = {}
    for resource_id, requested_soft in requested.items():
        try:
            _, hard = resource.getrlimit(resource_id)
            soft = requested_soft if hard == resource.RLIM_INFINITY else min(requested_soft, hard)
            resource.setrlimit(resource_id, (soft, hard))
        except (OSError, ValueError) as error:
            warnings[str(resource_id)] = str(error)
    return warnings


def _load_candidate(source: str, module_name: str) -> types.ModuleType:
    module = types.ModuleType(module_name)
    safe_builtins = dict(vars(builtins))
    original_import = builtins.__import__

    def restricted_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level or name.split(".")[0] not in ALLOWED_IMPORT_ROOTS:
            raise ImportError(f"candidate import is not allowed: {name}")
        return original_import(name, globals, locals, fromlist, level)

    for name in DISALLOWED_CALLS:
        safe_builtins.pop(name, None)
    # Python's import statement itself needs __import__. Direct calls are rejected by the
    # static source validator, while this hook restricts statements to approved modules.
    safe_builtins["__import__"] = restricted_import
    namespace = module.__dict__
    namespace["__builtins__"] = safe_builtins
    code = compile(source, f"<{module_name}>", "exec")
    exec(code, namespace, namespace)
    return module


def main() -> None:
    started = time.monotonic()
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        payload = json.load(sys.stdin)
        candidate = CandidateBundle.model_validate(payload["candidate"])
        x = np.asarray(payload["x"], dtype=np.float64)
        limit_warnings = _install_resource_limits(payload["limits"])
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            module = _load_candidate(candidate.source, f"candidate_{candidate.candidate_id}")
            gram = build_gram_matrix(module, candidate.contract, x, candidate.parameters)
        if gram.shape != (len(x), len(x)):
            raise ValueError(f"Gram matrix has shape {gram.shape}; expected {(len(x), len(x))}")
        if not np.all(np.isfinite(gram)):
            raise ValueError("Gram matrix contains non-finite values")
        limit = int(payload["max_output_bytes"])
        response = {
            "ok": True,
            "gram": gram.tolist(),
            "candidate_stdout": stdout.getvalue()[:limit],
            "candidate_stderr": stderr.getvalue()[:limit],
            "worker_seconds": time.monotonic() - started,
            "limit_warnings": limit_warnings,
        }
    except BaseException as error:
        response = {
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(limit=8),
        }
    json.dump(response, sys.stdout)


if __name__ == "__main__":
    main()
