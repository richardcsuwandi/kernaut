# Kernel Autoresearch for Open-Ended Model Discovery

Kernels encode the inductive biases of a wide range of machine learning models, and the choice of kernel largely determines what a model can learn from limited data.

**Kernel Autoresearch (Kernaut)** treats kernel design as open-ended program synthesis. Coding agents write kernels as programs, and construction contracts ensure that every accepted kernel is valid.
A quality-diversity archive keeps strong kernels with distinct behaviors, and meta-evaluators test whether the discoveries generalize to tasks that the search never saw.

<p align="center">
  <img src="assets/kernaut-logo.svg" alt="Kernel Autoresearch (Kernaut)" width="720">
</p>

> The name also combines *kernel* and *astronaut*: Kernaut explores unfamiliar spaces of kernels, much as an astronaut navigates unknown territory.

Kernaut applies to any method that needs a positive semidefinite kernel. Examples include Gaussian process surrogates for Bayesian optimization ([Wistuba and Grabocka, 2021](https://arxiv.org/abs/2101.07667)), scientific modeling in chemistry ([Griffiths et al., 2023](https://arxiv.org/abs/2212.04450)) and for differential equations ([Chen et al., 2021](https://arxiv.org/abs/2103.12959)), and kernel-based uncertainty estimates for language models ([Nikitin et al., 2024](https://arxiv.org/abs/2405.20003)).
The supplied meta-evaluators use Gaussian processes. For another kernel method, write a meta-evaluator with its fitting and scoring rules.

- **Documentation**: <https://richardcsuwandi.github.io/kernaut/>
- **Interactive archive demo**: <https://richardcsuwandi.github.io/kernaut/visualize/>
- **GitHub repository**: <https://github.com/richardcsuwandi/kernaut>

## How It Works

<p align="center">
  <img src="assets/method-overview.png" alt="Kernaut framework overview" width="100%">
</p>

Kernaut searches for reusable kernels through four steps:

1. **Propose.** A coding agent writes kernel components, such as feature maps or input transforms.
2. **Verify.** A trusted backend assembles the components using construction rules that preserve kernel validity under stated assumptions.
3. **Evaluate and refine.** The system scores candidates and keeps strong kernels with distinct behaviors in an archive. Agents use these results to guide further proposals.
4. **Test transfer.** Validation selects a frozen kernel, which is then evaluated on tasks the search never saw.

See [how Kernaut works](docs/how-it-works.md) for the example and links to the verification and evaluation guides.

## Quick Start

Use Python 3.11 or later on macOS or Linux. Install Kernaut:

```bash
git clone https://github.com/richardcsuwandi/kernaut.git
cd kernaut
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e examples/extension
```

Run the offline example, which needs no API key:

```bash
kernaut task-run --task sine --config examples/extension/offline.toml \
  --baseline linear-demo --archive runs/demo/archive.sqlite
```

Inspect the archive in the visualizer:

```bash
kernaut viz --archive runs/demo/archive.sqlite --open
```

The example replays predefined model replies to verify and evaluate one candidate.
To generate proposals with a language model, [configure a model provider](docs/getting-started.md#choose-a-model).
See the [full documentation](https://richardcsuwandi.github.io/kernaut/) for detailed usage.

## Interactive Archive Demo

<p align="center">
  <a href="https://richardcsuwandi.github.io/kernaut/visualize/"><img src="assets/archive-visualizer.png" alt="Kernaut archive visualizer" width="100%"></a>
</p>

The documentation includes a [guided archive viewer](https://richardcsuwandi.github.io/kernaut/visualize/).
Select candidates, inspect their verification evidence, and follow the recorded discovery history.
The demo shows 19 evaluated candidates from a historical meta-training run, including the dual warp-fold (DWF) kernel.
It displays fixed results, so it never runs candidate code or calls a model provider.
The [visualization guide](docs/visualize.md) explains how to preview your own archives.

## Example Discovery: Dual Warp-Fold (DWF) Kernel

An agent discovered the **dual warp-fold (DWF)** kernel on the black-box optimization benchmark.
DWF combines a gentle warp with a triangular fold of each input coordinate. The fold maps mirrored inputs to the same feature value, while the warp keeps them distinguishable.

<p align="center">
  <img src="assets/dwf-geometry.png" alt="The warp and fold of the discovered DWF kernel" width="100%">
</p>

*The gentle warp and triangular fold used by DWF.*

DWF applies a Matérn-5/2 kernel to this discovered representation, so similarity depends on input location as well as distance. This makes it nonstationary, unlike a standard Matérn kernel. Sums and products of standard stationary kernels cannot recover this structure.

Because the discovered kernel programs are short and interpretable, they invite human–AI collaboration: researchers can understand the agent's proposal and refine its assumptions. For DWF, we separated the warp and fold into additive kernel components, strengthening the connection between mirrored inputs. This human-refined version reduced held-out predictive error by a further 5.7%, showing how an agent's discovery can become a starting point for further model design.

<p align="center">
  <img src="assets/dwf-prior-samples.png" alt="Prior samples from Matérn-5/2, DWF, and the additive refinement" width="100%">
</p>

*Functions sampled before fitting data: (a) Matérn-5/2, (b) DWF, and (c) the human refinement. DWF samples have visible corners at the fold.*

Try the [black-box optimization benchmark](docs/tasks.md#black-box-optimization), or see the [paper](#citation) for the full analysis.

## Benchmarks

| Benchmark or task | Evaluation scope | Interface |
| --- | --- | --- |
| Black-box optimization | Predictive accuracy and Bayesian optimization on transformed analytic functions, with six held-out function families | [`meta-run`, `meta-benchmark`](docs/tasks.md#black-box-optimization) |
| Greenhouse-gas forecasting | Transfer from CO2, CH4, and N2O to the held-out records SF6, CFC-12, and CFC-11 | [`ts-run`, `ts-benchmark`](docs/tasks.md#greenhouse-gas-forecasting) |
| ChemBench enzyme kinetics | Reaction-rate prediction across ten training mechanism domains and five held-out domains | [`chem-run`, `chem-benchmark`](docs/tasks.md#chembench-enzyme-kinetics) |
| GlucoseBench forecasting | Transfer across simulated patient groups. Local search uses children and validation uses adolescents. Hidden adult testing requires the external evaluator | [`glucose-run`, `glucose-benchmark`](docs/tasks.md#glucosebench-forecasting) |
| User-supplied regression | Gaussian process marginal likelihood on a JSON dataset. Users define separate validation and test evaluations | [`run`](docs/your-problem.md) |
| Offline sine-wave example | A small integration example for tasks, model adapters, baselines, and verification | [`task-run`](docs/getting-started.md#run-without-an-api-key) |

The [benchmark guide](docs/tasks.md) lists data requirements, metrics, task families, and commands.
Each benchmark splits tasks into meta-training, meta-validation, and meta-test sets, so you can check whether discovered inductive biases transfer to unseen tasks.
The [meta-evaluation guide](docs/meta-evaluation.md) explains the splits, episode generation, and candidate selection. We follow the evaluation framework of [DiscoGen](https://arxiv.org/abs/2603.17863) (Goldie et al., 2026).

## Your Own Problem

Provide your training data as JSON and a task description in Markdown.
Then configure a model provider and run:

```bash
kernaut run --config configs/openai.toml --context my-task.md \
  --data my-data.json --archive runs/my-task/archive.sqlite
```

The default regression workflow tunes eight reference kernels before it searches for new programs.
To use another data loader or scoring rule, write a [task extension](docs/extensions.md#add-a-task-and-its-context).
To evaluate across episodes, follow the [custom benchmark protocol](docs/custom-benchmarks.md).
To add reference kernels and report comparable evaluation budgets, see the [baseline guide](docs/baselines.md).

## Verification

Kernaut accepts only candidates that pass its Tier 2 construction-contract checks.
The construction rules preserve positive semidefiniteness, the property that makes a kernel valid.
This guarantee assumes that the trusted interpreter is correct. Numerical checks alone do not prove validity for all inputs.

The [visualizer](docs/visualize.md) links candidate code to scores, verification evidence, and recorded agent conversations.
Candidate processes limit execution failures, but they do not securely isolate hostile code.
See the [verification guide](docs/verification.md) and the [security policy](SECURITY.md).

## Contributing

We welcome contributions that help Kernaut apply to more tasks and domains.

- **Found a bug?** [Open an issue](https://github.com/richardcsuwandi/kernaut/issues).
- **Want to add a task or domain?** Follow the [custom benchmark protocol](docs/custom-benchmarks.md).
- **Want to add a reference kernel or model adapter?** See the [baseline guide](docs/baselines.md) and the [extension guide](docs/extensions.md).
- **Want to add a construction contract?** A new contract widens the kernels that agents can write, for example state-space kernels from linear ODEs or kernels for structured inputs. It needs an implementation in the trusted interpreter, a proof of kernel validity, and tests for valid and invalid candidates. See [Add a construction contract](CONTRIBUTING.md#add-a-construction-contract).

Include a reproducible example, an explicit evaluation protocol, and tests that need no provider credentials.
See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Citation

If you use Kernaut in your research, please cite our paper **Kernel Autoresearch for Open-Ended Model Discovery** by Richard Cornelius Suwandi, Feng Yin, and Kevin Murphy:

```bibtex
@misc{suwandi2026kernaut,
  title = {Kernel Autoresearch for Open-Ended Model Discovery},
  author = {Suwandi, Richard Cornelius and Yin, Feng and Murphy, Kevin},
  year = {2026},
  note = {Preprint}
}
```

See [references and benchmark attribution](docs/references.md) for DiscoGen, CRPS, and the benchmark sources.
See also the [citation metadata](CITATION.cff) and the [greenhouse data attribution](src/kernaut/data/greenhouse/SOURCE.md).

## License

Kernaut is released under the [MIT License](LICENSE).
