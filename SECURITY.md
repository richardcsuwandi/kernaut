# Security

Kernaut runs generated Python code in a separate process. The process has resource limits,
restricted imports, a filtered environment, and access to a limited set of built-in functions.
These controls limit the effect of failures during research runs. They do not provide secure
isolation for hostile code.

If you do not trust the candidate author, use an external sandbox designed to isolate hostile code.
Task, model, and baseline extensions run as ordinary Python code in the main process.
Install only extensions that you trust.

Store API keys in environment variables. Keep keys out of candidate code, parameters,
configuration files, archives, and prompts. Archives may contain model messages and user context.
Review archives before sharing them. The local dashboard listens on `127.0.0.1` by default.

Report security problems privately to richardsuwandi@link.cuhk.edu.cn. Include the affected
version, a small example that reproduces the problem, and the expected impact.
Do not publish credentials or sensitive run data in an issue.
