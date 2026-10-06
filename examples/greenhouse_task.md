# Time-series forecasting kernel discovery

Design a single dimension-agnostic Gaussian-process kernel that acts as a transferable prior for
forecasting long monthly time series. Each episode standardizes a training window of one record
and forecasts a fixed horizon of future months. Inputs are 1-D values in [0, 1] (the month index
scaled to the record), and observation noise is small. The kernel should also remain numerically
stable when evaluated at input dimensions 1 through 8.

This is a mathematical-novelty campaign. Discovered closure trees are rejected because ordinary
sums and products of standard kernels belong in the baseline bank. Start every candidate with
`propose_formulation`, giving its exact mathematical form, PSD argument, closest known kernel,
behavioral distinction, and a falsification test. Then stage, verify, tune, and submit it through the
normal immutable-candidate workflow.

Prefer `input_transform`, `feature_map`, `residual_input_transform`, or genuinely new `spectral`
constructions. The trusted interpreter constructs the Gram matrix; candidate code should only
return the certified pointwise transform, features, or spectral components required by its
contract. The kernel must accept any input dimension. Independent ideas have no
parents, while a real revision must cite the candidate it changes.

Develop at least two structurally different proposals before refining the strongest one.
