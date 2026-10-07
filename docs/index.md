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

Kernaut runs a propose, verify, and evaluate loop. Each part addresses one challenge of open-ended kernel discovery:

- **Construction contracts.** Agents write kernel components under one of four contracts: feature maps, spectral representations, input transformations of a library kernel, and closures (sums, products, scalings, and pullbacks of library or accepted kernels). A trusted interpreter assembles each kernel with positive semidefinite (PSD) preserving rules. Validity then follows from the construction, under the stated assumptions of each contract.
- **Verification tiers.** Tier 0 checks execution and output shape. Tier 1 adds numerical PSD tests on sampled inputs. Tier 2 adds contract assembly and consistency checks. Only Tier 2 programs enter the archive. Numerical tests alone are not enough: in our stress tests, 22 to 58% of unrestricted LLM-generated kernels that passed the initial screen failed broader tests.
- **Quality-diversity archive.** A MAP-Elites archive keeps the best program in each cell. A cell combines the kernel's declared niche, its feature growth, and its novelty band. Archived kernels and structured feedback guide later proposals.
- **Novelty screening.** Before an agent submits code, it registers the kernel's mathematical form, its PSD argument, and its closest known kernel. Kernaut then compares a behavioral fingerprint (the normalized Gram matrix on fixed probe points) with reference kernels. It penalizes near-duplicates of known kernels, unless they predict clearly better.
- **Meta-evaluation.** Search and parameter tuning use meta-training tasks, and meta-validation tasks select one frozen program. Meta-test tasks use families, records, mechanisms, or patient groups that the search never saw. Because search never uses the validation tasks, selection is a finite model-selection problem. Its guarantee depends on the number of frozen candidates, not on the size of the program space.

Read the [verification guide](verification.md) for the evidence tiers and the [meta-evaluation protocol](meta-evaluation.md) for the splits.

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

See the [project citation](https://github.com/richardcsuwandi/kernaut#paper-and-citation),
[related references](references.md), and [verification guarantees](verification.md).
