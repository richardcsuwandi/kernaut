# Security

Kernaut executes generated Python in a separate process with resource limits, restricted
imports, a scrubbed environment, and a reduced built-in namespace. These controls contain
research failures. They are not a hardened security boundary for hostile code.

Use an external hardened sandbox when candidate authors are outside your trust boundary.
Installed task, model, and baseline plugins execute as ordinary trusted Python code in the
main process. Install only extensions you trust.

Keep API keys out of candidate source, parameters, configuration files, archives, and prompts.
Use environment variables. Archives may contain model messages and user context, so review
archives before sharing them. The local dashboard binds to `127.0.0.1` by default.

Report security problems privately to richardsuwandi@link.cuhk.edu.cn. Include the affected
version, a minimal reproduction, and the expected impact. Avoid publishing credentials or
sensitive run data in an issue.
