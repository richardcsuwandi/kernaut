---
hide:
  - toc
---

# Archive visualization

The interactive example below uses the same visualizer as `kernaut viz`.
Select a numbered step for guidance, or explore candidates and charts directly.
Text and plots render in the browser, so they remain clear when enlarged.

<iframe class="archive-demo" src="../demo/index.html#candidate" title="Guided Kernaut archive walkthrough" loading="lazy"></iframe>

[Open the full-width viewer](demo/index.html#candidate){ target="_blank" rel="noopener" }
[Read the verification guarantees](verification.md)
{ .demo-links }

## About this example

Explore 19 evaluated kernels: nine discoveries and ten baselines. DWF is selected initially.
**Run trace** shows each discovery's complete recorded conversation, including proposals, tool calls, results, and the final report.
Personal names and usernames are anonymized where present.

Scores come from a historical meta-training run, not held-out testing.
The example loads saved results and does not execute code or call a model.

## Viewer sections

| View | Interpretation and interaction |
| --- | --- |
| Candidate | Select a sidebar entry to inspect its parameters, source, rationale, and evaluation summary. `BASE` identifies a reference kernel |
| Verification | Inspect execution, numerical checks, and construction-contract evidence. Tier 2 concerns kernel validity under the trusted interpreter, not predictive quality |
| Landscape | Compare score and runtime. Select a marker, or focus it with the keyboard and press Enter or Space, to inspect its candidate |
| Progress | Inspect each candidate's latest stored score in submission order. The curve records improvements in the running best. It is not a complete history of every reevaluation |
| Run trace | Read recorded messages and tool results. The documentation example includes the complete recorded campaign for each discovery |

Search by candidate name or identifier. **★ Best** restricts the sidebar to candidates on the quality–cost frontier.
A candidate is on this frontier if no accepted candidate matches or improves both its score and runtime while improving at least one.
Use **Previous**, **Next**, or the numbered steps to navigate the walkthrough. **Restart** returns to DWF.
The full-width viewer provides more space for plots. On small screens, plots can be scrolled horizontally without shrinking their labels.

Compare scores only under the same task and evaluation protocol.
Runtime depends on the machine and workload, so small timing differences require repeated measurements.
The recorded rationale is the agent's explanation, not independent evidence of novelty or correctness.
Archived comments are preserved. In DWF, the fold is a triangular wave, continuous but not globally smooth.
Despite its name, `fold_phase` controls an amplitude multiplier through `sin(fold_phase)`.
See the reviewed [DWF example](https://github.com/richardcsuwandi/kernaut/blob/main/examples/candidates/dual_warp_fold.json) for clarified comments.

## Open a local archive

After completing the [offline example](getting-started.md#run-without-an-api-key), run:

```bash
kernaut viz --archive runs/demo/archive.sqlite --open
```

For another search, replace the path with its `.sqlite` archive.
The viewer opens at `http://127.0.0.1:8765` and refreshes while the search runs.
Press `Ctrl-C` in the terminal to stop it. Use `--port 8766` if the default port is occupied.
Local archives include their recorded agent conversations and tool results.
Review context and messages before sharing an archive.

## Preview the documentation locally

From the repository root, with the Python environment active:

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Open the address printed by MkDocs, normally `http://127.0.0.1:8000/kernaut/` for this project.
The interactive example requires no archive server or model credentials.
