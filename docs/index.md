---
hide:
  - toc
---
# Kernel Autoresearch for Open-Ended Model Discovery

Kernels encode the inductive biases of a wide range of machine learning models, and the choice of kernel largely determines what a model can learn from limited data.
**Kernel Autoresearch (Kernaut)** treats kernel design as open-ended program synthesis. Coding agents write kernels as programs, and construction contracts ensure that every accepted kernel is valid.
A quality-diversity archive keeps strong kernels with distinct behaviors, and meta-evaluators test whether the discoveries generalize to tasks that the search never saw.
{ .lead }

<img class="hero-logo" src="assets/kernaut-logo.svg" alt="Kernel Autoresearch (Kernaut)">

> The name combines *kernel* and *astronaut*: Kernaut explores unfamiliar spaces of kernels, much as an astronaut navigates unknown territory.

[Installation and examples](getting-started.md){ .md-button .md-button--primary }
[Meta-evaluation protocol](meta-evaluation.md){ .md-button }

## Kernel design as model discovery

A fixed library of kernels limits which structures automated search can express.
Kernaut searches over programs instead, and it keeps explicit rules for constructing positive semidefinite kernels.
These kernels apply to any method that needs one. Examples include Gaussian process surrogates for Bayesian optimization ([Wistuba and Grabocka, 2021](https://arxiv.org/abs/2101.07667)), scientific modeling in chemistry ([Griffiths et al., 2023](https://arxiv.org/abs/2212.04450)) and for differential equations ([Chen et al., 2021](https://arxiv.org/abs/2103.12959)), and kernel-based uncertainty estimates for language models ([Nikitin et al., 2024](https://arxiv.org/abs/2405.20003)).

The supplied meta-evaluators use Gaussian processes. To use another kernel method, provide its fitting and scoring rules through a [meta-evaluator](extensions.md#add-a-task-and-its-context).
The [meta-evaluation protocol](meta-evaluation.md) tests whether discovered inductive biases transfer to unseen tasks. It follows [DiscoGen](https://arxiv.org/abs/2603.17863) (Goldie et al., 2026).

## How Kernaut works

Agents propose kernel components, a trusted backend verifies their construction, and evaluation guides further proposals. A frozen kernel is then tested on tasks the search never saw.
Read [how Kernaut works](how-it-works.md) for the framework overview, or follow the [DWF example](how-it-works.md#example-discovery-the-dwf-kernel) to see a discovered kernel.

## Benchmarks and tasks

| Benchmark or task | Purpose |
| --- | --- |
| [Black-box optimization](tasks.md#black-box-optimization) | Evaluate prediction and optimization on transformed analytic functions |
| [Greenhouse-gas forecasting](tasks.md#greenhouse-gas-forecasting) | Evaluate transfer across NOAA gas records. The paper tests SF6, CFC-12, and CFC-11. The package includes SF6 testing |
| [ChemBench enzyme kinetics](tasks.md#chembench-enzyme-kinetics) | Evaluate transfer from canonical rate-law mechanisms to distinct held-out mechanisms |
| [GlucoseBench forecasting](tasks.md#glucosebench-forecasting) | Evaluate kernel transfer across simulated patient groups |
| [User-supplied regression](your-problem.md) | Search on observations supplied as an input matrix and target vector |
| [Offline sine-wave example](getting-started.md#run-without-an-api-key) | Check the complete task and verification workflow without a model provider |

The [benchmark guide](tasks.md) describes the data, metrics, and available commands.
The [meta-evaluation protocol](meta-evaluation.md) specifies which tasks are available during search, selection, and final evaluation.

## Methods and extensions

Use [custom regression tasks](your-problem.md) for a single dataset, or define a [custom benchmark](custom-benchmarks.md) for evaluation across task families.
[Baseline extensions](baselines.md) support comparisons under a shared evaluator.
[Model adapters](extensions.md#add-a-model-provider) connect additional providers to the search interface.

The [archive visualizer](visualize.md) supports inspection during a search and analysis of completed runs.
The [contribution guide](contributing.md) specifies the requirements for new tasks, reference methods, and integrations.

## Research publication

**Kernel Autoresearch for Open-Ended Model Discovery**  
Richard Cornelius Suwandi, Feng Yin, and Kevin Murphy.

See the [project citation](https://github.com/richardcsuwandi/kernaut#citation),
[related references](references.md), and [verification guarantees](verification.md).
