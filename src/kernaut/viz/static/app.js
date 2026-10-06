const snapshotUrl = document.querySelector('meta[name="kernaut-snapshot"]')?.content;

const state = {
  snapshot: null,
  selectedId: null,
  selectedRun: "all",
  bestOnly: false,
  query: "",
  activeTab: "candidate",
  backendOutdated: false,
  timer: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function currentTheme() {
  return document.documentElement.dataset.theme || "light";
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("kernaut-theme", theme);
  $("#theme-toggle").setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} mode`);
  if (state.snapshot) renderLandscape();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fmt(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  const number = Number(value);
  if (number !== 0 && (Math.abs(number) >= 1e4 || Math.abs(number) < 1e-3)) {
    return number.toExponential(2);
  }
  return number.toFixed(digits).replace(/\.?0+$/, "");
}

function shortId(value) {
  return value ? String(value).slice(0, 8) : "—";
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.classList.add("show");
  setTimeout(() => node.classList.remove("show"), 1400);
}

async function load() {
  try {
    const response = await fetch(snapshotUrl || "/api/snapshot", { cache: "no-store" });
    const payload = await response.json();
    if (!response.ok || payload.error) throw new Error(payload.error || `HTTP ${response.status}`);
    const wasOutdated = state.backendOutdated;
    state.backendOutdated = payload.schema_version !== 3;
    state.snapshot = payload;
    if (!state.selectedId || !payload.candidates.some((c) => c.candidate_id === state.selectedId)) {
      state.selectedId = payload.candidates.find((c) => c.candidate_id === payload.demo?.initial_candidate_id)?.candidate_id || payload.candidates[0]?.candidate_id || null;
    }
    $("#live-dot").className = state.backendOutdated ? "live-dot error" : "live-dot ok";
    $("#sync-label").textContent = state.backendOutdated ? "Restart the viewer" : snapshotUrl ? "Fixed research example" : `Synced ${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}`;
    render();
    document.dispatchEvent(new CustomEvent("kernaut:loaded", { detail: payload }));
    if (state.backendOutdated && !wasOutdated) toast("Restart kernaut viz to load the updated archive API");
  } catch (error) {
    $("#live-dot").className = "live-dot error";
    $("#sync-label").textContent = "Archive unavailable";
    toast(error.message);
    document.dispatchEvent(new CustomEvent("kernaut:load-error"));
  }
}

function render() {
  const data = state.snapshot;
  if (!data) return;
  $("#archive-path").textContent = data.archive;
  renderList();
  renderCandidate();
  renderVerification();
  renderLandscape();
  renderProgress();
  renderTrace();
}

function filteredCandidates() {
  const query = state.query.trim().toLowerCase();
  return state.snapshot.candidates.filter((candidate) => {
    if (state.bestOnly && !candidate.on_frontier) return false;
    if (!query) return true;
    return [candidate.name, candidate.candidate_id, candidate.contract, candidate.rationale]
      .join(" ").toLowerCase().includes(query);
  });
}

function renderList() {
  const candidates = filteredCandidates();
  $("#candidate-count").textContent = `${candidates.length} shown`;
  $("#candidate-list").innerHTML = candidates.length ? candidates.map((candidate) => `
    <button class="candidate-row ${candidate.candidate_id === state.selectedId ? "active" : ""}"
            data-candidate="${escapeHtml(candidate.candidate_id)}" type="button">
      <span class="candidate-name">${candidate.origin === "baseline" ? `<span class="baseline-tag">BASE</span>` : ""}${escapeHtml(candidate.name)}</span>
      <span class="best-mark" ${candidate.on_frontier ? 'title="Best candidate"' : 'aria-hidden="true"'}>${candidate.on_frontier ? "★" : ""}</span>
      <span class="candidate-id">${shortId(candidate.candidate_id)}, ${escapeHtml(candidate.contract)}</span>
      <span class="score">${candidate.score === null ? "unscored" : fmt(candidate.score, 3)}${candidate.baseline_delta == null ? "" : `, Δ ${candidate.baseline_delta >= 0 ? "+" : ""}${fmt(candidate.baseline_delta, 3)}`}</span>
    </button>
  `).join("") : `<div class="empty">No candidates match.</div>`;
  $$("[data-candidate]").forEach((button) => button.addEventListener("click", () => {
    selectCandidate(button.dataset.candidate);
  }));
}

function selectCandidate(candidateId, tabName = null) {
  if (!state.snapshot?.candidates.some((c) => c.candidate_id === candidateId)) return false;
  state.selectedId = candidateId;
  state.selectedRun = "all";
  renderList();
  renderCandidate();
  renderVerification();
  renderTrace();
  if (tabName) activateTab(tabName);
  return true;
}

function bindPlotInteractions(selector) {
  $$(selector).forEach((node) => {
    const candidate = state.snapshot.candidates.find((c) => c.candidate_id === node.dataset.plotId);
    node.setAttribute("role", "button");
    node.setAttribute("tabindex", "0");
    node.setAttribute("aria-label", `Inspect ${candidate.name}, score ${fmt(candidate.score)}`);
    const inspect = () => selectCandidate(node.dataset.plotId, "candidate");
    node.addEventListener("click", inspect);
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        inspect();
      }
    });
  });
}

function selectedCandidate() {
  return state.snapshot?.candidates.find((candidate) => candidate.candidate_id === state.selectedId);
}

function metric(label, value) {
  return `<div class="metric"><strong>${escapeHtml(value)}</strong><span>${escapeHtml(label)}</span></div>`;
}

function renderCandidate() {
  const candidate = selectedCandidate();
  const panel = $("#candidate-panel");
  if (!candidate) {
    panel.innerHTML = `<div class="empty">Waiting for the first generated program.</div>`;
    return;
  }
  const evaluation = candidate.evaluation || {};
  const baseline = state.snapshot.baseline_reference;
  const metadata = evaluation.metadata || {};
  panel.innerHTML = `
    <div class="candidate-title">
      <div>
        <p class="eyebrow">${escapeHtml(candidate.contract)}, ${escapeHtml(candidate.candidate_id)}</p>
        <h2>${escapeHtml(candidate.name)}</h2>
        <div class="badges">
          <span class="badge ${candidate.accepted ? "verified" : ""}">${escapeHtml(candidate.tier_label)}</span>
          ${candidate.origin === "baseline" ? `<span class="badge baseline">baseline</span>` : `<span class="badge">discovered</span>`}
          ${candidate.accepted ? `<span class="badge verified">accepted</span>` : `<span class="badge">not accepted</span>`}
          ${candidate.on_frontier ? `<span class="badge best">★ Best candidate</span>` : ""}
        </div>
      </div>
    </div>
    <div class="metric-grid">
      ${metric("Score ↑", fmt(evaluation.score))}
      ${metric("Negative log likelihood ↓", fmt(evaluation.negative_log_likelihood))}
      ${metric("Vs best baseline ↑", candidate.baseline_delta == null ? "—" : `${candidate.baseline_delta >= 0 ? "+" : ""}${fmt(candidate.baseline_delta)}`)}
      ${metric("Reference baseline", baseline ? `${baseline.name}, ${fmt(baseline.score, 3)}` : "—")}
      ${metric("Runtime", evaluation.runtime_seconds == null ? "—" : `${fmt(evaluation.runtime_seconds, 3)} s`)}
      ${metric("Condition number", fmt(evaluation.condition_number, 3))}
      ${metric("Evidence tier", candidate.tier == null ? "—" : `T${candidate.tier}`)}
      ${metric("Jitter", fmt(evaluation.jitter, 6))}
      ${metric("Parents", candidate.parents?.length || 0)}
      ${metric("Run", candidate.run_ids?.map(shortId).join(", ") || "—")}
      ${metric("Observations", metadata.n ?? "—")}
      ${metric("Input dimension", metadata.dimension ?? "—")}
    </div>
    <h3 class="section-title">Design rationale</h3>
    <div class="prose">${escapeHtml(candidate.rationale || "No rationale recorded.")}</div>
    <h3 class="section-title">Parameters</h3>
    <pre>${escapeHtml(JSON.stringify(candidate.parameters || {}, null, 2))}</pre>
    <h3 class="section-title">Generated source</h3>
    <div class="code-wrap">
      <button id="copy-source" class="quiet-button copy" type="button">Copy</button>
      <pre><code>${escapeHtml(candidate.source)}</code></pre>
    </div>
  `;
  $("#copy-source")?.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(candidate.source);
      toast("Source copied");
    } catch {
      toast("Copy is unavailable. Select the source text to copy it.");
    }
  });
}

const assuranceTiers = [
  { tier: 0, label: "Executable", note: "Isolated program runs" },
  { tier: 1, label: "Empirical", note: "Randomized PSD checks" },
  { tier: 2, label: "Certified", note: "Contract interpreter" },
];

function checkState(check) {
  if (check.passed) return "pass";
  return "fail";
}

function readableMetric(name, value) {
  const label = name.replaceAll("_", " ");
  if (value === null || value === undefined) return [label, "—"];
  if (name === "runtime_seconds") return [label, `${fmt(value, 3)} s`];
  if (typeof value === "number") return [label, fmt(value, 3)];
  return [label, String(value)];
}

function evidenceGroup(check) {
  if (["source_shape", "execution", "randomized_psd"].includes(check.name)) return "Runtime";
  return "Contract";
}

function renderEvidenceCheck(check) {
  const status = checkState(check);
  const metrics = Object.entries(check.metrics || {})
    .filter(([, value]) => value !== null && value !== undefined)
    .map(([name, value]) => readableMetric(name, value));
  return `
    <article class="evidence-row ${status}">
      <span class="evidence-mark" aria-hidden="true">${status === "pass" ? "✓" : status === "pending" ? "○" : "×"}</span>
      <div class="evidence-copy">
        <div class="evidence-name">${escapeHtml(check.name.replaceAll("_", " "))}</div>
        <p>${escapeHtml(check.detail)}</p>
        ${metrics.length ? `<dl class="evidence-metrics">${metrics.map(([label, value]) => `
          <div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl>` : ""}
      </div>
    </article>`;
}

function renderVerification() {
  const panel = $("#verification-panel");
  const candidate = selectedCandidate();
  if (!candidate) {
    panel.innerHTML = `<div class="empty">Select a candidate to inspect its verification.</div>`;
    return;
  }
  const evidence = candidate.evidence || {};
  // Older archives may contain the retired formal-template placeholder. The
  // implemented evidence model ends at contract-verified Tier 2.
  const checks = (evidence.checks || []).filter((check) => check.name !== "formal_template");
  if (!checks.length) {
    panel.innerHTML = `
      <div class="verification-empty">
        <span>UNVERIFIED</span>
        <strong>No evidence recorded</strong>
        <p>This candidate has not passed through the verification harness yet.</p>
      </div>`;
    return;
  }
  const achievedTier = Number.isInteger(candidate.tier) ? candidate.tier : -1;
  const contractCheck = checks.find((check) => check.name === "contract_conformance");
  const predatesContractChecks = achievedTier >= 2 && !contractCheck;
  const grouped = ["Runtime", "Contract"].map((name) => ({
    name,
    checks: checks.filter((check) => evidenceGroup(check) === name),
  })).filter((group) => group.checks.length);
  const tierName = assuranceTiers.find((item) => item.tier === achievedTier)?.label || "No tier";

  panel.innerHTML = `
    <header class="verification-head">
      <div>
        <p class="eyebrow">${escapeHtml(candidate.contract)}, ${shortId(candidate.candidate_id)}</p>
        <h2>${escapeHtml(candidate.name)}</h2>
        <p>Evidence for the kernel constructor, its runtime behavior, and its formal proof boundary.</p>
      </div>
      <div class="assurance-result ${candidate.accepted ? "accepted" : "rejected"}">
        <span>${candidate.accepted ? "Accepted" : "Not accepted"}</span>
        <strong>${achievedTier >= 0 ? `T${achievedTier}, ${tierName}` : "No evidence tier"}</strong>
      </div>
    </header>

    <ol class="assurance-ladder" aria-label="Verification assurance tiers">
      ${assuranceTiers.map((item) => {
        const stateName = item.tier <= achievedTier ? "complete" : "pending";
        return `<li class="${stateName} ${item.tier === achievedTier ? "current" : ""}">
          <span class="tier-index">T${item.tier}</span>
          <strong>${item.label}</strong>
          <small>${item.note}</small>
        </li>`;
      }).join("")}
    </ol>

    ${predatesContractChecks ? `<div class="archive-notice"><strong>Legacy evidence</strong><span>This record predates the strengthened Tier 2 consistency checks. Reverify the candidate to populate them.</span></div>` : ""}

    <div class="evidence-summary">
      <span><b>${checks.filter((check) => check.passed).length}</b> passed</span>
      <span><b>${checks.filter((check) => checkState(check) === "fail").length}</b> failed</span>
    </div>

    <div class="evidence-groups">
      ${grouped.map((group) => `
        <section class="evidence-group">
          <div class="evidence-group-title"><span>${group.name}</span><small>${group.checks.length} ${group.checks.length === 1 ? "check" : "checks"}</small></div>
          <div class="evidence-list">${group.checks.map(renderEvidenceCheck).join("")}</div>
        </section>`).join("")}
    </div>

    <footer class="verification-footnote">
      <strong>How to read this</strong>
      <p>A Tier 2 candidate uses a restricted certificate interface. The trusted interpreter applies construction rules that preserve positive semidefiniteness (PSD). Randomized checks provide diagnostics. Kernel validity follows from the construction rules and depends on the correctness of the interpreter.</p>
    </footer>`;
}

function renderLandscape() {
  const panel = $("#landscape-panel");
  const points = state.snapshot.candidates.filter((c) => c.score !== null && c.runtime_seconds !== null);
  if (!points.length) {
    panel.innerHTML = `<div class="empty">The quality–cost landscape appears after evaluations.</div>`;
    return;
  }
  // Match SVG user units to the rendered container so text remains a true
  // 11px instead of being enlarged on wide screens.
  const width = Math.max(640, (panel.clientWidth || 960) - 72);
  const height = Math.max(380, Math.min(560, Math.round(width * 0.46)));
  const margin = { left: 64, right: 24, top: 26, bottom: 48 };
  const xs = points.map((p) => Number(p.runtime_seconds));
  const ys = points.map((p) => Number(p.score));
  const baseline = state.snapshot.baseline_reference;
  let [xmin, xmax] = [Math.min(...xs), Math.max(...xs)];
  let [ymin, ymax] = [Math.min(...ys), Math.max(...ys)];
  if (xmin === xmax) [xmin, xmax] = [xmin * 0.9, xmax * 1.1 || 1];
  if (ymin === ymax) [ymin, ymax] = [ymin - 1, ymax + 1];
  const x = (v) => margin.left + (v - xmin) / (xmax - xmin) * (width - margin.left - margin.right);
  const y = (v) => height - margin.bottom - (v - ymin) / (ymax - ymin) * (height - margin.top - margin.bottom);
  const ticks = Array.from({ length: 5 }, (_, i) => i / 4);
  const grid = ticks.map((t) => {
    const gx = margin.left + t * (width - margin.left - margin.right);
    const gy = margin.top + t * (height - margin.top - margin.bottom);
    return `<line class="plot-grid" x1="${gx}" y1="${margin.top}" x2="${gx}" y2="${height - margin.bottom}" />
      <line class="plot-grid" x1="${margin.left}" y1="${gy}" x2="${width - margin.right}" y2="${gy}" />
      <text class="axis-label" x="${margin.left - 9}" y="${gy + 4}" text-anchor="end">${fmt(ymax - t * (ymax - ymin), 2)}</text>`;
  }).join("");
  const referenceLine = baseline ? `
    <line class="baseline-reference" x1="${margin.left}" y1="${y(baseline.score)}" x2="${width - margin.right}" y2="${y(baseline.score)}" />
    <text class="baseline-label" x="${width - margin.right - 4}" y="${y(baseline.score) - 7}" text-anchor="end">Best baseline, ${fmt(baseline.score, 3)}</text>` : "";
  const dots = points.map((point) => {
    const px = x(point.runtime_seconds);
    const py = y(point.score);
    const alignRight = px > width - margin.right - 220;
    const labelX = px + (alignRight ? -10 : 10);
    const labelY = Math.max(margin.top + 12, py - 8);
    const radius = 4 + (point.tier || 0);
    const color = point.origin === "baseline" ? "#d99a3d" : point.on_frontier ? "#4a97ec" : point.accepted ? "#31b487" : "#8b9098";
    const marker = point.origin === "baseline"
      ? `<rect class="point baseline-point" x="${px - radius}" y="${py - radius}" width="${radius * 2}" height="${radius * 2}" rx="2" fill="${color}">`
      : `<circle class="point" cx="${px}" cy="${py}" r="${radius}" fill="${color}">`;
    const markerClose = point.origin === "baseline" ? "</rect>" : "</circle>";
    return `
    <g data-plot-id="${escapeHtml(point.candidate_id)}">
      ${marker}
        <title>${escapeHtml(point.name)}, score ${fmt(point.score)}, ${fmt(point.runtime_seconds, 3)}s</title>
      ${markerClose}
      <text class="plot-label" x="${labelX}" y="${labelY}" text-anchor="${alignRight ? "end" : "start"}">${escapeHtml(point.name)}</text>
    </g>`;
  }).join("");
  panel.innerHTML = `
    <div class="plot-heading">
      <strong>Candidate performance</strong>
      <span>Discoveries and baseline kernels on the same score and runtime axes</span>
    </div>
    <div class="plot-card">
      <svg id="pareto-plot" viewBox="0 0 ${width} ${height}" role="img" aria-label="Candidate score versus runtime">
        ${grid}
        ${referenceLine}
        <line class="plot-axis" x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" />
        <line class="plot-axis" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" />
        <text class="axis-title" x="${width / 2}" y="${height - 10}" text-anchor="middle">Runtime (seconds), lower is better</text>
        <text class="axis-title" transform="translate(16 ${height / 2}) rotate(-90)" text-anchor="middle">Score, higher is better</text>
        <text class="axis-label" x="${margin.left}" y="${height - margin.bottom + 20}">${fmt(xmin, 3)}</text>
        <text class="axis-label" x="${width - margin.right}" y="${height - margin.bottom + 20}" text-anchor="end">${fmt(xmax, 3)}</text>
        ${dots}
      </svg>
      <div class="legend"><span><b>★</b> best</span><span><i class="baseline-key"></i>baseline</span><span><i style="background:#31b487"></i>discovered</span><span>Dot size indicates evidence strength</span></div>
    </div>`;
  bindPlotInteractions("#pareto-plot [data-plot-id]");
}

function renderProgress() {
  const panel = $("#progress-panel");
  if (state.backendOutdated) {
    panel.innerHTML = `<div class="trace-empty"><span class="trace-empty-icon backend">RESTART</span><strong>Updated viewer backend required</strong><p>The progress timeline needs schema v3. Stop the server and start <code>kernaut viz</code> again.</p></div>`;
    return;
  }
  const timeline = state.snapshot.progress || [];
  const scored = timeline.filter((step) => step.score !== null);
  if (!scored.length) {
    panel.innerHTML = `<div class="empty">The discovery timeline appears after the first evaluation.</div>`;
    return;
  }
  const kept = scored.filter((step) => step.improves_best);
  const width = Math.max(720, (panel.clientWidth || 960) - 72);
  const height = Math.max(420, Math.min(600, Math.round(width * 0.5)));
  const margin = { left: 68, right: 30, top: 30, bottom: 52 };
  const ys = scored.map((step) => Number(step.score));
  let [ymin, ymax] = [Math.min(...ys), Math.max(...ys)];
  const pad = (ymax - ymin) * 0.08 || 0.01;
  ymin -= pad; ymax += pad;
  const x = (index) => margin.left + (scored.length === 1 ? 0.5 : index / (scored.length - 1)) * (width - margin.left - margin.right);
  const y = (value) => height - margin.bottom - (value - ymin) / (ymax - ymin) * (height - margin.top - margin.bottom);
  const ticks = Array.from({ length: 5 }, (_, i) => i / 4);

  // Step path through running best values.
  const steps = [];
  kept.forEach((step, position) => {
    const px = x(step.index);
    const py = y(step.score);
    const nextX = position + 1 < kept.length ? x(kept[position + 1].index) : width - margin.right;
    if (!steps.length) {
      steps.push(`M ${px} ${py}`);
    } else {
      steps.push(`L ${px} ${py}`);
    }
    steps.push(`L ${nextX} ${py}`);
  });
  const bestPath = `<path class="progress-best" d="${steps.join(" ")}" fill="none" />`;

  const grid = ticks.map((t) => {
    const gy = margin.top + t * (height - margin.top - margin.bottom);
    return `<line class="plot-grid" x1="${margin.left}" y1="${gy}" x2="${width - margin.right}" y2="${gy}" />
      <text class="axis-label" x="${margin.left - 9}" y="${gy + 4}" text-anchor="end">${fmt(ymax - t * (ymax - ymin), 2)}</text>`;
  }).join("");

  const points = scored.map((step) => {
    const px = x(step.index);
    const py = y(Number(step.score));
    const isKept = step.improves_best;
    const color = step.origin === "baseline"
      ? "#d99a3d"
      : isKept ? "#31b487" : "var(--muted-dot, #9aa1a9)";
    const shape = step.origin === "baseline"
      ? `<rect class="point baseline-point" x="${px - 4}" y="${py - 4}" width="8" height="8" rx="2" fill="${color}"></rect>`
      : `<circle class="point ${isKept ? "kept-point" : "discarded-point"}" cx="${px}" cy="${py}" r="${isKept ? 6 : 3.2}" fill="${color}"></circle>`;
    const tooltip = [
      `${step.name}`,
      `score ${fmt(step.score)}, #${step.index + 1}${isKept && step.origin !== "baseline" ? ", new best" : ""}`,
      `${step.label}${step.niche ? `, ${step.niche}` : ""}`,
      step.novelty_distance != null ? `novelty distance ${fmt(step.novelty_distance, 3)}` : "",
      step.change_note,
    ].filter(Boolean).join("\n");
    let annotation = "";
    if (isKept) {
      const anchorRight = px > width - margin.right - 200;
      const labelX = px + (anchorRight ? -10 : 10);
      const labelY = py - 12;
      const label = step.name.length > 34 ? `${step.name.slice(0, 33)}…` : step.name;
      annotation = `<text class="progress-annotation" x="${labelX}" y="${labelY}"
        text-anchor="${anchorRight ? "end" : "start"}">${escapeHtml(label)}</text>`;
    }
    return `<g data-plot-id="${escapeHtml(step.candidate_id)}">${shape}<title>${escapeHtml(tooltip)}</title></g>${annotation}`;
  }).join("");

  panel.innerHTML = `
    <div class="plot-heading">
      <strong>Discovery progress: ${scored.length} scored candidates</strong>
      <span>Latest scores in candidate submission order</span>
    </div>
    <div class="plot-card">
      <svg id="progress-plot" viewBox="0 0 ${width} ${height}" role="img" aria-label="Latest candidate scores in submission order">
        ${grid}
        ${bestPath}
        <line class="plot-axis" x1="${margin.left}" y1="${height - margin.bottom}" x2="${width - margin.right}" y2="${height - margin.bottom}" />
        <line class="plot-axis" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${height - margin.bottom}" />
        <text class="axis-title" x="${width / 2}" y="${height - 12}" text-anchor="middle">Candidate #, submission order</text>
        <text class="axis-title" transform="translate(16 ${height / 2}) rotate(-90)" text-anchor="middle">Score, higher is better</text>
        <text class="axis-label" x="${margin.left}" y="${height - margin.bottom + 20}">1</text>
        <text class="axis-label" x="${width - margin.right}" y="${height - margin.bottom + 20}" text-anchor="end">${scored.length}</text>
        ${points}
      </svg>
      <div class="legend">
        <span><i style="background:#31b487"></i>new best</span>
        <span><i style="background:#9aa1a9"></i>other candidate</span>
        <span><i class="baseline-key"></i>baseline</span>
        <span>Click a marker to open the candidate</span>
      </div>
    </div>`;
  bindPlotInteractions("#progress-plot [data-plot-id]");
  layoutProgressAnnotations();
}

// After the SVG is in the DOM, measure the real rendered bounding boxes of
// annotation labels (rotation included) and push colliding ones downward.
function layoutProgressAnnotations() {
  const svg = $("#progress-plot");
  if (!svg) return;
  const labels = [...svg.querySelectorAll("text.progress-annotation")];
  const placedRects = [];
  labels.forEach((textNode) => {
    const x = parseFloat(textNode.getAttribute("x"));
    let y = parseFloat(textNode.getAttribute("y"));
    let dy = 0;
    let guard = 0;
    // Measure unrotated, shift until clear of everything already placed,
    // then commit the rotation and record the true rendered rect.
    textNode.removeAttribute("transform");
    for (; guard < 8; guard++) {
      textNode.setAttribute("y", y);
      const rect = textNode.getBoundingClientRect();
      const collides = placedRects.some((other) =>
        rect.left < other.right + 4 &&
        other.left < rect.right + 4 &&
        Math.abs((rect.top + rect.bottom) / 2 - other.centerY) < 15
      );
      if (!collides) break;
      y += 16;
      dy += 16;
    }
    textNode.setAttribute("transform", `rotate(-18 ${x} ${y})`);
    const finalRect = textNode.getBoundingClientRect();
    placedRects.push({
      left: finalRect.left,
      right: finalRect.right,
      centerY: (finalRect.top + finalRect.bottom) / 2,
    });
  });
}

function eventText(event) {
  if (event.role === "tool") {
    const result = event.tool_result;
    if (typeof result === "object" && result !== null) return JSON.stringify(result, null, 2);
    return event.content;
  }
  const calls = (event.tool_calls || []).map((call) => `${call.name}(${JSON.stringify(call.arguments, null, 2)})`).join("\n");
  return [event.content, calls].filter(Boolean).join("\n");
}

function renderTrace() {
  const panel = $("#trace-panel");
  const candidate = selectedCandidate();
  if (!candidate) {
    panel.innerHTML = `<div class="empty">Select a candidate to inspect its trace.</div>`;
    return;
  }
  if (candidate.origin === "baseline") {
    panel.innerHTML = `<div class="trace-empty"><span class="trace-empty-icon">BASE</span><strong>No agent trace</strong><p>Baseline kernels are evaluated as reference methods. They do not have agent conversations.</p></div>`;
    return;
  }
  if (state.backendOutdated) {
    panel.innerHTML = `<div class="trace-empty"><span class="trace-empty-icon backend">RESTART</span><strong>Updated viewer backend required</strong><p>This server version cannot link run events to candidates. Stop the server and start <code>kernaut viz</code> again.</p></div>`;
    return;
  }
  const runIds = new Set(candidate.run_ids || []);
  const runs = state.snapshot.runs.filter((run) => runIds.has(run.run_id));
  if (!runs.length) {
    panel.innerHTML = `<div class="empty">${snapshotUrl ? "No recorded conversation is linked to this candidate." : "No trace events are linked to this candidate."}</div>`;
    return;
  }
  if (state.selectedRun === "all" || !runs.some((run) => run.run_id === state.selectedRun)) {
    state.selectedRun = runs[0].run_id;
  }
  const events = state.snapshot.events.filter((event) => event.run_id === state.selectedRun);
  const linkedEventCount = events.filter((event) =>
    (event.candidate_ids || []).includes(candidate.candidate_id)
  ).length;
  panel.innerHTML = `
    ${snapshotUrl ? `<p class="prose">${escapeHtml(state.snapshot.demo?.trace_note || "Recorded campaign conversation.")}</p>` : ""}
    <div class="trace-heading"><div><strong>${escapeHtml(candidate.name)}</strong><span>${events.length} campaign events, ${linkedEventCount} linked to candidate</span></div></div>
    <div class="run-select"><span class="eyebrow" style="margin:0">Campaign</span>
      <select id="run-select">${runs.map((run) => {
        const count = state.snapshot.events.filter((event) => event.run_id === run.run_id).length;
        return `<option value="${escapeHtml(run.run_id)}" ${run.run_id === state.selectedRun ? "selected" : ""}>${shortId(run.run_id)}, ${count} campaign events, seed ${run.seed}</option>`;
      }).join("")}</select>
    </div>
    <div class="timeline">${events.map((event) => `
      <div class="event ${event.role} ${(event.candidate_ids || []).includes(candidate.candidate_id) ? "candidate-linked" : ""}">
        <div class="event-head"><span class="event-role">${escapeHtml(event.tool_name || event.role)}</span><span class="event-sequence">#${event.sequence}</span><span class="muted">${escapeHtml(event.created_at)}</span></div>
        <div class="event-body">${escapeHtml(eventText(event) || "(empty response)")}</div>
      </div>`).join("")}</div>`;
  $("#run-select")?.addEventListener("change", (event) => {
    state.selectedRun = event.target.value;
    renderTrace();
  });
}

function activateTab(name) {
  state.activeTab = name;
  window.history.replaceState(null, "", `#${name}`);
  $$(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === name));
  ["candidate", "verification", "landscape", "progress", "trace"].forEach((panel) => {
    $(`#${panel}-panel`).classList.toggle("hidden", panel !== name);
  });
  if (name === "landscape" && state.snapshot) renderLandscape();
  if (name === "progress" && state.snapshot) renderProgress();
  if (name === "verification" && state.snapshot) renderVerification();
  if (name === "trace" && state.snapshot) renderTrace();
  document.dispatchEvent(new CustomEvent("kernaut:tab-changed", { detail: name }));
}

$("#search").addEventListener("input", (event) => {
  state.query = event.target.value;
  renderList();
});
$("#best-filter").addEventListener("click", (event) => {
  state.bestOnly = !state.bestOnly;
  event.currentTarget.classList.toggle("active", state.bestOnly);
  renderList();
});
$("#refresh").addEventListener("click", load);
$("#theme-toggle").addEventListener("click", () => setTheme(currentTheme() === "dark" ? "light" : "dark"));
$$(".tab").forEach((tab) => tab.addEventListener("click", () => activateTab(tab.dataset.tab)));

window.KernautViewer = Object.freeze({ selectCandidate, showTab: activateTab });
setTheme(currentTheme());
const initialTab = window.location.hash.replace("#", "");
activateTab(["candidate", "verification", "landscape", "progress", "trace"].includes(initialTab) ? initialTab : "candidate");
load();
if (!snapshotUrl) state.timer = setInterval(load, 3000);

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (state.activeTab === "landscape" && state.snapshot) renderLandscape();
    if (state.activeTab === "progress" && state.snapshot) renderProgress();
  }, 120);
});
