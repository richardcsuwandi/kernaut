<p align="center">
  <img src="assets/kernaut-logo.svg" alt="Kernaut: Kernel Autoresearch" width="720">
</p>

# Kernaut: Kernel Autoresearch

Kernels encode inductive biases: they determine which inputs a model treats as similar and which patterns it can learn.
Kernaut treats kernel design as open-ended model discovery.
Language models propose kernel programs, construction contracts govern their validity, and task evaluators measure their usefulness.
An archive records each program, its parameters, verification evidence, and results.

The framework targets methods that use positive semidefinite kernels, including kernel ridge regression, support vector machines, and Gaussian processes.
The supplied evaluators use Gaussian processes. Other kernel methods require a task evaluator for their fitting and scoring rules.
Meta-training, meta-validation, and meta-test splits assess whether discovered inductive biases transfer to unseen tasks.

[Installation](docs/getting-started.md) | [Benchmarks](docs/tasks.md) |
[Meta-evaluation protocol](docs/meta-evaluation.md) | [Custom benchmarks](docs/custom-benchmarks.md) |
[Archive visualization](docs/visualize.md)

## Interactive archive example

The documentation includes a [guided archive viewer](https://richardcsuwandi.github.io/kernaut/visualize/).
Select candidates and chart markers, inspect verification evidence, and follow the recorded discovery history.
The example contains 19 evaluated candidates from a historical meta-training run, including the dual warp–fold (DWF) kernel.
It displays fixed results without running candidate code or calling a model provider.

See the [visualization guide](docs/visualize.md) for local preview instructions and interpretation.

## Installation and offline example

Use Python 3.11 or later on macOS or Linux:

```bash
git clone https://github.com/richardcsuwandi/kernaut.git
cd kernaut
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e examples/extension
```

Run the example and inspect its archive:

```bash
kernaut task-run --task sine --config examples/extension/offline.toml \
  --baseline linear-demo --archive runs/demo/archive.sqlite
kernaut viz --archive runs/demo/archive.sqlite --open
```

The example uses predefined model replies to verify and evaluate one candidate without an API key.
For model-generated proposals, [configure a model provider](docs/getting-started.md#choose-a-model).

## Benchmarks and evaluation tasks

| Benchmark or task | Evaluation scope | Interface |
| --- | --- | --- |
| Black-box optimization | Predictive accuracy and Bayesian optimization on transformed analytic functions, with six held-out function families | [`meta-run`, `meta-benchmark`](docs/tasks.md#black-box-optimization) |
| Greenhouse-gas forecasting | Transfer from CO2, CH4, and N2O to held-out records. The paper tests SF6, CFC-12, and CFC-11. The package includes SF6 testing | [`ts-run`, `ts-benchmark`](docs/tasks.md#greenhouse-gas-forecasting) |
| ChemBench enzyme kinetics | Reaction-rate prediction across ten training mechanism domains and five held-out domains | [`chem-run`, `chem-benchmark`](docs/tasks.md#chembench-enzyme-kinetics) |
| GlucoseBench forecasting | Transfer across simulated patient groups. Local search uses children and validation uses adolescents. Hidden adult testing requires the external evaluator | [`glucose-run`, `glucose-benchmark`](docs/tasks.md#glucosebench-forecasting) |
| User-supplied regression | Gaussian process marginal likelihood on a JSON dataset. Users define separate validation and test evaluations | [`run`](docs/your-problem.md) |
| Offline sine-wave example | A small integration example for tasks, model adapters, baselines, and verification | [`task-run`](docs/getting-started.md#run-without-an-api-key) |

The [benchmark guide](docs/tasks.md) lists data requirements, metrics, task families, and commands.
The [meta-evaluation guide](docs/meta-evaluation.md) explains split membership, episode generation, configuration, and candidate selection.
This separation follows the evaluation framework discussed by [Goldie et al. (2026)](https://arxiv.org/abs/2603.17863).

## Custom tasks and reference methods

Provide training data as JSON and a task description in Markdown.
After configuring a model provider, run:

```bash
kernaut run --config configs/openai.toml --context my-task.md \
  --data my-data.json --archive runs/my-task/archive.sqlite
```

The default regression workflow tunes eight reference kernels before searching for new programs.
Use a [task extension](docs/extensions.md#add-a-task-and-its-context) for another data loader or scoring rule.
For evaluation across episodes, follow the [custom benchmark protocol](docs/custom-benchmarks.md).
The [baseline guide](docs/baselines.md) explains how to add reference kernels and report comparable evaluation budgets.

## Verification and inspection

Kernaut accepts candidates that pass its Tier 2 construction-contract checks.
The construction rules preserve positive semidefiniteness, the property required of a valid kernel.
This guarantee depends on the trusted interpreter being correct. Numerical checks alone are not a proof for all inputs.

The [visualizer](docs/visualize.md) connects candidate code to scores, verification evidence, and recorded agent conversations.
Candidate processes limit execution failures but do not securely isolate hostile code.
See the [verification guide](docs/verification.md) and [security policy](SECURITY.md).

## Contributions

Contributions can add benchmark tasks, reference kernels, model adapters, documentation, or corrections.
Include a reproducible example, an explicit evaluation protocol, and tests that do not require provider credentials.
See the [contribution guide](CONTRIBUTING.md).

## Paper and citation

Accompanies **Kernel Autoresearch for Open-Ended Model Discovery**, by
**Richard Cornelius Suwandi, Feng Yin, and Kevin Murphy**.

```bibtex
@misc{suwandi2026kernaut,
  title = {Kernel Autoresearch for Open-Ended Model Discovery},
  author = {Suwandi, Richard Cornelius and Yin, Feng and Murphy, Kevin},
  year = {2026},
  note = {Preprint}
}
```

See [references and benchmark attribution](docs/references.md) for DiscoGen, CRPS, and the benchmark sources.

[Citation metadata](CITATION.cff) | [MIT license](LICENSE) |
[Greenhouse data attribution](src/kernaut/data/greenhouse/SOURCE.md)
