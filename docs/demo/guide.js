(() => {
  const steps = [
    { tab: "candidate", label: "Candidate", title: "Inspect a kernel program",
      text: "DWF is selected initially. Choose another candidate in the sidebar to inspect its score, parameters, source, and recorded rationale." },
    { tab: "verification", label: "Verification", title: "Examine the verification evidence",
      text: "Inspect the selected candidate’s runtime and construction-contract checks. Tier 2 supports kernel validity under the trusted interpreter, not predictive superiority." },
    { tab: "landscape", label: "Comparison", title: "Compare predictive score and runtime",
      text: "Each marker represents a candidate. Squares are baselines. Higher scores and lower runtimes are preferable. Select a marker, or focus it and press Enter, to inspect its program." },
    { tab: "progress", label: "History", title: "Follow the recorded candidate history",
      text: "The curve tracks the best score in submission order, using each candidate’s latest evaluation. Select a marker to inspect the corresponding program." },
    { tab: "trace", label: "Conversation", title: "Follow the recorded discovery conversation",
      text: "Read the task prompt, agent proposals, tool calls, results, and final response. This step opens DWF. Select another discovery to read its complete recorded campaign." },
  ];
  let current = 0;
  let dwfId = null;
  let ready = false;
  const guide = document.querySelector("#demo-guide");
  guide.innerHTML = `
    <div class="guide-top">
      <span class="guide-label">Interactive archive <span>Historical research example</span></span>
      <div class="guide-controls">
        <button id="guide-previous" type="button" disabled aria-label="Previous walkthrough step">Previous</button>
        <button id="guide-next" type="button" disabled aria-label="Next walkthrough step">Next</button>
      </div>
    </div>
    <nav class="guide-steps" aria-label="Walkthrough steps">
      ${steps.map((step, i) => `<button type="button" data-step="${i}" disabled><span>${i + 1}</span> ${step.label}</button>`).join("")}
    </nav>
    <div class="guide-copy" aria-live="polite">
      <h2 id="guide-title">Loading the example</h2>
      <p id="guide-description">The viewer reads a fixed archive excerpt. It does not run candidate code or contact a model provider.</p>
    </div>`;

  function update(index) {
    current = index;
    guide.querySelectorAll("[data-step]").forEach((button, i) => {
      if (i === index) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });
    document.querySelector("#guide-title").textContent = steps[index].title;
    document.querySelector("#guide-description").textContent = steps[index].text;
    document.querySelector("#guide-previous").disabled = !ready || index === 0;
    const next = document.querySelector("#guide-next");
    next.disabled = !ready;
    next.textContent = index === steps.length - 1 ? "Restart" : "Next";
    next.setAttribute("aria-label", index === steps.length - 1 ? "Restart walkthrough" : "Next walkthrough step");
  }

  function go(index) {
    if (!ready) return;
    if (index === 0 || index === 4) {
      const search = document.querySelector("#search");
      search.value = "";
      search.dispatchEvent(new Event("input", { bubbles: true }));
      const best = document.querySelector("#best-filter");
      if (best.classList.contains("active")) best.click();
      window.KernautViewer.selectCandidate(dwfId);
    }
    window.KernautViewer.showTab(steps[index].tab);
    document.querySelector(".workspace").scrollTo({ top: 0, behavior: "instant" });
  }

  guide.querySelectorAll("[data-step]").forEach((button) => {
    button.addEventListener("click", () => go(Number(button.dataset.step)));
  });
  document.querySelector("#guide-previous").addEventListener("click", () => go(Math.max(0, current - 1)));
  document.querySelector("#guide-next").addEventListener("click", () => go((current + 1) % steps.length));
  document.addEventListener("kernaut:tab-changed", (event) => {
    const index = steps.findIndex((step) => step.tab === event.detail);
    if (index >= 0) update(index);
  });
  document.addEventListener("kernaut:loaded", (event) => {
    dwfId = event.detail.demo.initial_candidate_id;
    ready = true;
    guide.querySelectorAll("[data-step]").forEach((button) => { button.disabled = false; });
    document.querySelector("#refresh").textContent = "Reload example";
    update(Math.max(0, steps.findIndex((step) => `#${step.tab}` === window.location.hash)));
  });
  document.addEventListener("kernaut:load-error", () => {
    document.querySelector("#guide-title").textContent = "The example could not load";
    document.querySelector("#guide-description").textContent = "Use Reload example to retry, or return to the documentation for instructions to open a local archive.";
  });
})();
