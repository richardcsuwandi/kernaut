# Contributing to Kernaut

Contributions can add tasks, baseline kernels, model adapters, tutorials, or corrections.
See the [contribution requirements](docs/contributing.md) and [custom benchmark protocol](docs/custom-benchmarks.md).
You can discuss an idea in an issue or open a pull request directly for a small fix.
Start with a small change and describe the behavior that the change enables or fixes.
For a new research method, discuss the evaluation protocol before running large experiments.

## Development

Use Python 3.11 or later on macOS or Linux. In the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install -e examples/extension
ruff check .
ruff format --check .
mypy src/kernaut
pytest
```

Keep generated runs, secrets, caches, model weights, and personal server configurations out of
commits. Tests must run without API keys or paid model calls. Add focused behavior tests for
new extension interfaces or fixes. Include the exact command and observed result in your pull
request.

## Add a task, model, or baseline

Follow [docs/extensions.md](docs/extensions.md). Prefer a separate extension package when the
integration requires substantial optional dependencies. Keep training, validation, and test data
separate. State whether higher or lower scores are better. Document seeds, preprocessing, resource
limits, and data sources. Never send held-out test targets to the model that proposes kernels.
Do not use those targets to select a kernel.

Baseline extensions must use the same evaluator and data budget as the comparison method.
The `task-run` command evaluates the supplied parameters. It does not automatically tune them.
A new learned baseline needs an explicit training and tuning protocol.

Model adapters must preserve tool call identifiers, return structured tool requests, and keep
credentials in environment variables. Document required dependencies and timeout behavior.

## Protect the construction contracts

A new construction contract requires:

- An implementation in the trusted interpreter.
- An explicit mathematical argument for kernel validity.
- A registration entry that describes the contract.
- Tests for valid and invalid candidates.

Never repair or reinterpret a failed candidate to make it pass. Numerical PSD checks are diagnostic evidence,
not a proof for arbitrary inputs.

## Review expectations

Keep the scope focused. Explain changes to public interfaces and add a runnable example when
useful. Distinguish unit tests and small integration checks from benchmark evidence. Do not update reported
paper results from an incomplete run. Preserve attribution and third-party license notices.
