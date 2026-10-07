# References and attribution

These references describe the evaluation framework, statistical methods, and external data used by Kernaut.
Cite the relevant benchmark sources alongside the Kernaut preprint when reporting results.
The package adapts these tasks to kernel evaluation. Its metrics and protocols can differ from those of the original projects.

## DiscoGen and meta-evaluation

Goldie, Alexander D., et al. (2026).
*DiscoGen: Procedural Generation of Algorithm Discovery Tasks in Machine Learning.*
[arXiv:2603.17863](https://arxiv.org/abs/2603.17863).

DiscoGen provides procedural tasks for algorithm discovery and an evaluation framework that separates discovery from generalization assessment.
Kernaut adopts this meta-evaluation perspective. Its adapters define the specific partitions documented in the [protocol guide](meta-evaluation.md).
See the [DiscoGen repository](https://github.com/AlexGoldie/discogen) for its task-generation code.

```bibtex
@misc{goldie2026discogen,
  title = {{DiscoGen}: Procedural Generation of Algorithm Discovery Tasks in Machine Learning},
  author = {Goldie, Alexander D. and Wang, Zilin and Hayler, Adrian and
    Nathani, Deepak and Toledo, Edan and Thampiratwong, Ken and
    Kalisz, Aleksandra and Beukman, Michael and Erlebach, Hannah and
    Letcher, Alistair and Reddy, Shashank and Wibault, Clarisse and
    Wolf, Theo and O'Neill, Charles and Berdica, Uljad and
    Roberts, Nicholas and Rahmani, Saeed and Raileanu, Roberta and
    Whiteson, Shimon and Foerster, Jakob N.},
  year = {2026},
  eprint = {2603.17863},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2603.17863}
}
```

## Probabilistic scoring

Gneiting, Tilmann, and Raftery, Adrian E. (2007).
“Strictly Proper Scoring Rules, Prediction, and Estimation.”
*Journal of the American Statistical Association*, 102(477), 359–378.
[DOI: 10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437).

The continuous ranked probability score (CRPS) measures predictive-distribution error. Lower values are better.
Kernaut uses negative CRPS in predictive search scores.

## ChemBench

Kabra, Sanchit, Abhyankar, Nikhil, Desai, Saaketh, Iyer, Prasad, and Reddy, Chandan K. (2026).
*LLM-AutoSciLab: Closed-Loop Scientific Discovery via Active Experimentation with LLMs.*
[arXiv:2605.24043](https://arxiv.org/abs/2605.24043).

Kernaut uses the project's ChemBench oracle for reaction-rate observations.
The adapter targets the [documented source revision](tasks.md#chembench-enzyme-kinetics), which determines the available mechanism identifiers.
See the [LLM-AutoSciLab repository](https://github.com/scientific-discovery/LLM-AutoSciLab).

## GlucoseBench

Murphy, Kevin (2026).
*Model Discovery Agent: LLM-assisted Bayesian experiment design for data-efficient discovery of mechanistic world models.*
[arXiv:2608.09696](https://arxiv.org/abs/2608.09696).

This work describes GlucoseBench, the simulated glucose forecasting task used by the Kernaut adapter.
The local adapter uses public training episodes for search and validation.
Hidden-test evaluation remains the responsibility of the external GlucoseBench evaluator.

## NOAA greenhouse-gas records

NOAA Global Monitoring Laboratory. *Trends in Atmospheric Greenhouse Gases.*
[Data portal](https://gml.noaa.gov/ccgg/trends/).

The package includes monthly global mean records for CO2, CH4, N2O, and SF6.
It also includes CFC-12 and CFC-11 from the [NOAA HATS program](https://gml.noaa.gov/hats/).
Consult the bundled [source and processing notes](https://github.com/richardcsuwandi/kernaut/blob/main/src/kernaut/data/greenhouse/SOURCE.md)
and the source-specific citation instructions when using these observations.
