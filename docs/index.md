---
hide:
  - toc
---

<img class="hero-logo" src="assets/kernaut-logo.svg" alt="Kernaut: Kernel Autoresearch">

# Kernaut: Kernel Autoresearch

Kernels encode inductive biases by defining which inputs a model treats as similar.
Kernaut discovers these biases as executable kernel programs.
Language models propose candidates, construction contracts govern their validity, and task evaluators measure their usefulness.
The archive links each program to its parameters, verification evidence, and scores.
{ .lead }

[Installation and examples](getting-started.md){ .md-button .md-button--primary }
[Meta-evaluation protocol](meta-evaluation.md){ .md-button }

## Kernel design as model discovery

A fixed library of kernels limits which structures automated search can express.
Kernaut searches over programs while retaining explicit rules for constructing positive semidefinite kernels.
These kernels can be used in kernel ridge regression, support vector machines, Gaussian processes, and other methods that accept such kernels.

The supplied evaluators use Gaussian processes. To use another kernel method, provide its fitting and scoring rules through a [task evaluator](extensions.md#add-a-task-and-its-context).
The [meta-evaluation protocol](meta-evaluation.md) tests whether discovered inductive biases transfer to unseen tasks, following [Goldie et al. (2026)](https://arxiv.org/abs/2603.17863).

## Benchmarks and tasks

| Benchmark or task | Purpose |
| --- | --- |
| [Black-box optimization](tasks.md#black-box-optimization) | Evaluate prediction and optimization on transformed analytic functions |
| [Greenhouse-gas forecasting](tasks.md#greenhouse-gas-forecasting) | Evaluate forecasts across NOAA gas records, including held-out SF6 |
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

See the [project citation](https://github.com/richardcsuwandi/kernaut#paper-and-citation),
[related references](references.md), and [verification guarantees](verification.md).
