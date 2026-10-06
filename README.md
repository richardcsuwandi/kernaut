<p align="center">
  <img src="assets/kernaut-logo.svg" alt="Kernaut: Kernel Autoresearch" width="640">
</p>

# Kernel Autoresearch (Kernaut)

**Discover kernel programs with coding agents and PSD-preserving construction contracts.**

Kernaut lets language models propose kernel components while a deterministic backend verifies,
evaluates, and archives them. Bring your own dataset or extend the package with new tasks,
model providers, and baseline kernels.

Accompanies **Kernel Autoresearch for Open-Ended Model Discovery**, by
**Richard Cornelius Suwandi, Feng Yin, and Kevin Murphy**.

## Install

Use Python 3.11 or later on macOS or Linux. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For development, install `.[dev]`. For the Anthropic API, install `.[anthropic]`.
Windows has not been tested and lacks the worker's Unix resource limits.

## Quick start

Verify an example kernel without an API key:

```bash
kernaut verify examples/candidates/dual_warp_fold.json
```

To search with an LLM, copy `.env.example` to `.env`, set your provider key, and review the
model name in the configuration. LLM-backed searches can incur provider charges.

```bash
kernaut run --config configs/openai.toml --context examples/task.md \
  --data examples/data.json --archive runs/search/archive.sqlite
kernaut inspect --archive runs/search/archive.sqlite
kernaut viz --archive runs/search/archive.sqlite --open
```

The dataset is JSON with a matrix `x` and a vector `y`. The Markdown context describes your
inputs and modeling goals. Use training data for search and reserve held-out data for evaluation.

Configurations in [`configs/`](configs) cover OpenAI, Anthropic, OpenRouter, OpenAI-compatible
servers, Ollama, Claude Code, and Codex. Use `mixed-ensemble.toml` to combine model providers.

## Extend Kernaut

Install a separate Python package that registers one or more extension groups:

| Group | Adds |
| --- | --- |
| `kernaut.tasks` | Training data, domain context, and an evaluator |
| `kernaut.models` | An LLM provider or agent adapter |
| `kernaut.baselines` | Reference kernel programs |

See the [extension guide](docs/extensions.md) and [installable example](examples/extension).
Try a complete offline campaign with a fixed example model:

```bash
pip install -e examples/extension
kernaut task-run --task sine --config examples/extension/offline.toml \
  --baseline linear-demo --archive runs/offline/archive.sqlite
```

The [task guide](docs/tasks.md) covers the built-in BBO, forecasting, enzyme-kinetics, and glucose
adapters. Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and checks.
The extension API is experimental and may change before version 1.0.

## Verification

Only candidates that pass Tier 2 construction-contract checks enter the accepted archive.
PSD validity follows from supported construction rules, conditional on a correct trusted
interpreter. Numerical checks alone do not prove validity across all inputs.

Candidate subprocesses contain research failures but are not a hardened sandbox for hostile code.
Installed extensions execute as trusted Python code. See [SECURITY.md](SECURITY.md).

## Citation

```bibtex
@misc{suwandi2026kernaut,
  title = {Kernel Autoresearch for Open-Ended Model Discovery},
  author = {Suwandi, Richard Cornelius and Yin, Feng and Murphy, Kevin},
  year = {2026},
  note = {Preprint}
}
```

Machine-readable metadata: [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE). Bundled greenhouse data retain their [source attribution](src/kernaut/data/greenhouse/SOURCE.md).
External benchmark packages retain their own licenses.
