# What verification means

A construction contract defines the code interface and the rules used to build a kernel.
For example, a feature-map candidate supplies a function for one input point.
The trusted interpreter evaluates that function and forms the kernel from inner products.

## Evidence tiers

| Tier | Evidence |
| --- | --- |
| 0: Executable | The program runs on the checked inputs |
| 1: Empirical | Numerical kernel checks pass on the sampled inputs |
| 2: Contract certified | The candidate passes the construction-contract checks |

Only Tier 2 candidates enter the accepted archive.
Rejected and unscored proposals can still appear in the archive so you can inspect their history.

The supported construction rules preserve positive semidefiniteness, the property required of a valid kernel.
This guarantee depends on the correctness of the trusted interpreter.
Numerical checks alone cannot prove validity for all inputs.

Verification and predictive evaluation answer different questions.
A verified candidate can still score poorly on your task.
Inspect both the verification evidence and the evaluation results before selecting a kernel.

## Execution and trust

Kernaut runs generated candidates in separate processes with resource limits and restricted imports.
These controls limit execution failures. They do not provide secure isolation for hostile code.
Use an external sandbox when you do not trust the candidate author.

Installed task, model, and baseline extensions run as ordinary Python code in the main process.
Install only packages that you trust.
See the [security policy](https://github.com/richardcsuwandi/kernaut/blob/main/SECURITY.md).
