# Enzyme-kinetics kernel discovery

Design one certified Gaussian-process kernel that acts as a transferable prior for learning
enzyme-catalyzed reaction rates from scattered experiments. Each dataset records the initial
reaction rate `r_0` measured under a set of experimental conditions. The underlying rate law
differs between datasets and is not given.

Every input is a seven-dimensional vector in `[0, 1]`, with coordinates in this order:

1. log substrate concentration `C_A`.
2. inhibitor concentration `C_I` (linear, can be zero).
3. log second-substrate concentration `C_B`.
4. product concentration `C_P` (linear, can be zero).
5. log enzyme loading `Enz`.
6. temperature `T` (linear, 278-368 K).
7. pH (linear, 4-10).

This is a mathematical-novelty campaign. Discovered closure trees are rejected because ordinary
sums and products of standard kernels belong in the baseline bank. Start every candidate with
`propose_formulation`, giving its exact mathematical form, PSD argument, closest known kernel,
behavioral distinction, and a falsification test. Then stage, verify, tune, and submit it through the
normal immutable-candidate workflow.

Prefer `input_transform`, `feature_map`, `residual_input_transform`, or genuinely new `spectral`
constructions. The trusted interpreter constructs the Gram matrix. Candidate code should only return the certified pointwise transform, features, or spectral components required by its
contract. The input dimension is fixed at seven for this benchmark. Independent ideas have no
parents, while a real revision must cite the candidate it changes.

Develop at least two structurally different proposals before refining the strongest one. Do not
claim that a covariance is itself a recovered rate law.
