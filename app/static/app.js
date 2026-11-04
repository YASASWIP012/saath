const LEVEL_SHORT = ["Not related", "Wish dead", "Thoughts", "Method", "Intent", "Plan", "Act"];
const HIGH = 4;
const $ = (s) => document.querySelector(s);

async function api(path, body) {
  const res = await fetch(path, body ? {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body),
  } : undefined);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

// ---------- tabs ----------
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", x === b));
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${b.dataset.tab}`));
  if (b.dataset.tab === "evidence") loadEvidence();
}));

// ---------- shared rendering ----------
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function bandLabel(band, reason) {
  if (band === "review" && reason === "language") return "Review · language";
  if (band === "urgent" && reason === "critical") return "Critical";
  return { urgent: "Urgent", review: "Review", routine: "Standard" }[band];
}

function reasons(s, meta) {
  const [lo, hi] = s.level_range;
  const out = [];
  const pct = Math.round((s.risk_calibrated ?? s.risk) * 100);
  out.push(`Chance this message is at level 4 or above (intent, plan or act): <b>${pct}%</b>.`);
  if (s.band === "urgent" && s.band_reason === "critical") {
    out.push(`Critical: in testing, 92% of messages scored this high were truly at level 4 or above. About one message in eight lands here.`);
  } else if (s.band === "urgent") {
    out.push(`That clears the urgent cut-off. The cut-off is set so that at most
      ${Math.round(meta.alpha_urgent * 100)}% of genuinely high-risk messages fall below it.`);
  } else if (s.band === "review" && s.band_reason === "language") {
    out.push(`Written in romanised Hindi or Telugu. Saath reads these poorly, so it will not put them in the standard queue. Please read it yourself.`);
  } else if (s.band === "review") {
    out.push(`Below the urgent cut-off, but the plausible range reaches level ${hi}, so it sits above the standard queue.`);
  } else {
    out.push(`Plausible range stays below level 4.`);
  }
  out.push(lo === hi
    ? `The model is confident about the level: ${lo} (${LEVEL_SHORT[lo]}).`
    : `Plausible levels: ${lo} to ${hi}. The true level is inside this range for ${Math.round((1 - meta.alpha_level) * 100)}% of messages.`);
  out.push(`Waiting time also counts. Nobody is left at the bottom indefinitely.`);
  return out;
}

function renderDetail(el, s, meta, title, text) {
  const maxW = Math.max(0.0001, ...s.evidence.map((e) => e.weight));
  const highlighted = s.evidence.length
    ? s.evidence.map((e) => {
        const a = Math.min(0.55, (e.weight / maxW) * 0.55);
        return `<span style="background: rgba(var(--hl), ${e.weight > 0.01 ? a : 0})" title="${Math.round(e.weight * 100)} pts">${escapeHtml(e.text)}</span>`;
      }).join(" ")
    : escapeHtml(text);
  const [lo, hi] = s.level_range;
  const bars = s.proba.map((p, i) =>
    `<div class="bar ${i >= HIGH ? "hr" : ""} ${i >= lo && i <= hi ? "in" : ""}" style="height:${Math.max(2, p * 100)}%" title="${(p * 100).toFixed(1)}%"></div>`).join("");
  el.classList.remove("empty");
  el.innerHTML = `
    <div class="row" style="display:flex;justify-content:space-between;align-items:center;gap:8px">
      <h2>${escapeHtml(title)}</h2><span class="chip ${s.band}">${bandLabel(s.band, s.band_reason)}</span>
    </div>
    <p class="msg">${highlighted}</p>
    <h3>Severity on the Columbia scale</h3>
    <div class="levels">${bars}</div>
    <div class="level-names">${LEVEL_SHORT.map((n, i) => `<div>${i}<br>${n}</div>`).join("")}</div>
    <h3>Why it is here</h3>
    <ul class="why">${reasons(s, meta).map((r) => `<li>${r}</li>`).join("")}</ul>
    <p class="muted" style="font-size:12px;margin-top:16px">Highlighted lines moved the score most. Scored in ${s.latency_ms} ms on CPU.</p>`;
}

// ---------- live queue ----------
let META = null;
let DEMO = [];
let order = "risk";
let queue = [];
let clock = 0; // simulated minutes
let timer = null;
let nextIdx = 0;
let served = { high: 0, highWaitSum: 0 };
let selected = null;
let pinCount = 0;

function priority(item) {
  // A counsellor's manual move beats any automatic order.
  if (item.pinned !== undefined) return 1e6 - item.pinned;
  if (order === "fifo") return -item.arrived;
  // Same rule as the replay: risk percentile plus waiting time. Messages the
  // model cannot read well are never sorted below the middle of the queue.
  const p = item.score.priority ?? item.score.risk;
  const floor = item.score.band === "review" ? 0.5 : 0;
  return Math.max(p, floor) + (clock - item.arrived) / 45;
}

function renderQueue() {
  const list = $("#queue");
  const sorted = [...queue].sort((a, b) => priority(b) - priority(a) || a.arrived - b.arrived);
  // FLIP: remember positions, re-render, animate from old to new.
  const before = new Map([...list.children].map((li) => [li.dataset.id, li.getBoundingClientRect().top]));
  list.innerHTML = sorted.map((it) => `
    <li class="card ${it.score.band} ${it.score.band_reason === "critical" ? "critical" : ""} ${it.fresh ? "new" : ""} ${selected === it.id ? "sel" : ""}" data-id="${it.id}">
      <div class="row"><span class="who">${escapeHtml(it.name)}</span><span class="chip ${it.score.band}">${bandLabel(it.score.band, it.score.band_reason)}</span></div>
      <p class="snip">${escapeHtml(it.text)}</p>
      <div class="row"><span class="wait">waiting ${Math.round(clock - it.arrived)} min</span><span class="wait">level ${it.score.level_range[0]}–${it.score.level_range[1]}</span></div>
      ${selected === it.id && it.pinned === undefined ? `<button class="pin" data-pin="${it.id}">Move to top</button>` : ""}${it.pinned !== undefined ? `<span class="wait">moved to top by counsellor</span>` : ""}
    </li>`).join("") || `<li class="muted">Queue is empty.</li>`;
  queue.forEach((it) => (it.fresh = false));
  [...list.children].forEach((li) => {
    const old = before.get(li.dataset.id);
    if (old === undefined) return;
    const dy = old - li.getBoundingClientRect().top;
    if (dy) li.animate([{ transform: `translateY(${dy}px)` }, { transform: "none" }], { duration: 350, easing: "ease-out" });
  });
  list.querySelectorAll(".card").forEach((li) => li.addEventListener("click", () => select(li.dataset.id)));
  list.querySelectorAll(".pin").forEach((btn) => btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const it = queue.find((q) => q.id === btn.dataset.pin);
    if (it) { it.pinned = pinCount++; renderQueue(); }
  }));
  const waitingHigh = queue.filter((q) => q.score.band === "urgent");
  const longest = waitingHigh.length ? Math.max(...waitingHigh.map((q) => clock - q.arrived)) : 0;
  $("#stats").innerHTML = `
    <span>Clock <b>${String(Math.floor(clock / 60)).padStart(2, "0")}:${String(Math.round(clock % 60)).padStart(2, "0")}</b></span>
    <span>Urgent waiting <b>${waitingHigh.length}</b></span>
    <span>Longest urgent wait <b>${Math.round(longest)} min</b></span>`;
}

async function select(id) {
  selected = id;
  const it = queue.find((q) => q.id === id);
  renderQueue();
  if (!it) return;
  const s = it.full || (it.full = await api("/api/score", { text: it.text }));
  renderDetail($("#detail"), s, META, it.name, it.text);
}

function arrive() {
  if (nextIdx >= DEMO.length) return;
  const d = DEMO[nextIdx++];
  queue.push({ ...d, id: `m${nextIdx}`, arrived: clock, fresh: true });
}

function serveOne() {
  if (!queue.length) return;
  const sorted = [...queue].sort((a, b) => priority(b) - priority(a) || a.arrived - b.arrived);
  const top = sorted[0];
  queue = queue.filter((q) => q !== top);
  if (top.score.band === "urgent") { served.high++; served.highWaitSum += clock - top.arrived; }
}

function tick() {
  clock += 1;
  if (clock % 2 === 0) arrive();
  if (clock % 7 === 0) serveOne();
  renderQueue();
  if (nextIdx >= DEMO.length && !queue.length) stopShift();
}

function stopShift() { clearInterval(timer); timer = null; $("#shift").textContent = "Start a shift"; }

$("#shift").addEventListener("click", () => {
  if (timer) return stopShift();
  queue = []; clock = 0; nextIdx = 0; selected = null;
  for (let i = 0; i < 6; i++) arrive();
  timer = setInterval(tick, 900);
  $("#shift").textContent = "Pause";
  renderQueue();
});

document.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".seg button").forEach((x) => x.classList.toggle("on", x === b));
  order = b.dataset.order;
  renderQueue();
}));

// ---------- try a message ----------
$("#try-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("#try-text").value.trim();
  if (!text) return;
  $("#try-result").innerHTML = `<p class="muted">Scoring…</p>`;
  const s = await api("/api/score", { text });
  renderDetail($("#try-result"), s, META, "Your message", text);
});

// ---------- evidence ----------
let evidenceLoaded = false;
function fmt(v, d = 3) { return v == null ? "–" : Number(v).toFixed(d); }
function ci(arr, d = 3) { return Array.isArray(arr) ? `${fmt(arr[0], d)}<br><span class="muted ci">${fmt(arr[1], 2)}–${fmt(arr[2], 2)}</span>` : fmt(arr, d); }

function lineChart(series, xs, { yLabel, xLabel, pct = true }) {
  const W = 560, H = 260, P = { l: 48, r: 12, t: 12, b: 36 };
  const all = Object.values(series).flatMap((s) => [...s.hi, ...s.lo]);
  const yMax = Math.max(...all) * 1.08, yMin = 0;
  const x = (i) => P.l + (i / (xs.length - 1)) * (W - P.l - P.r);
  const y = (v) => H - P.b - ((v - yMin) / (yMax - yMin)) * (H - P.t - P.b);
  const colors = { "First come": "var(--muted)", Keyword: "var(--review)", Saath: "var(--urgent)", "Frontier LLM": "var(--routine)", Oracle: "var(--accent)" };
  const paths = Object.entries(series).map(([name, s]) => {
    const band = s.hi.map((v, i) => `${x(i)},${y(v)}`).concat(s.lo.map((v, i) => `${x(i)},${y(v)}`).reverse()).join(" ");
    const line = s.mean.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join("");
    const c = colors[name] || "var(--ink)";
    return `<polygon points="${band}" fill="${c}" opacity=".12"/><path d="${line}" fill="none" stroke="${c}" stroke-width="2.2" ${name === "Oracle" ? 'stroke-dasharray="4 4"' : ""}/>`;
  }).join("");
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => yMin + f * (yMax - yMin));
  const legend = Object.keys(series).map((n, i) => `<g transform="translate(${P.l + 8 + i * 96},${P.t + 6})"><rect width="12" height="3" y="4" fill="${colors[n]}"/><text x="18" y="10">${n}</text></g>`).join("");
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="${yLabel} by ${xLabel}">
    ${ticks.map((t) => `<line x1="${P.l}" x2="${W - P.r}" y1="${y(t)}" y2="${y(t)}" stroke="var(--line)"/><text x="${P.l - 6}" y="${y(t) + 4}" text-anchor="end">${pct ? Math.round(t * 100) + "%" : t.toFixed(0)}</text>`).join("")}
    ${xs.map((v, i) => `<text x="${x(i)}" y="${H - P.b + 16}" text-anchor="middle">${v}</text>`).join("")}
    <text x="${(W + P.l) / 2}" y="${H - 4}" text-anchor="middle">${xLabel}</text>
    ${paths}${legend}</svg>`;
}

async function loadEvidence() {
  if (evidenceLoaded) return;
  const el = $("#evidence-body");
  try {
    const r = await api("/api/results");
    evidenceLoaded = true;
    const h = r.queue.headline;
    const rr = r.queue_robustness?.post_bootstrap?.relative_reduction_vs_fifo;
    const cells = r.queue_robustness?.sensitivity || [];
    const red = cells.map((c) => c.relative_reduction);
    const sens = red.length ? `Across ${red.length} alternative settings: ${Math.round(Math.min(...red) * 100)}–${Math.round(Math.max(...red) * 100)}%.` : "";
    el.innerHTML = `
      ${r.night ? `<div class="detail night">
        <div class="night-head"><h3>One night, two queues</h3><button class="primary" id="replay">Replay the night</button></div>
        <p class="muted" style="margin:0 0 12px;font-size:13px">The same 12-hour shift at 1.1× load: identical arrivals, identical patience, 4 counsellors. Only the order of the queue differs.</p>
        <div class="race">
          <div class="lane"><div class="lane-name">First come, first served</div><div class="count" id="c-fifo">0</div><div class="lane-sub">high-risk texters gave up</div></div>
          <div class="lane ours"><div class="lane-name">Saath</div><div class="count" id="c-saath">0</div><div class="lane-sub">high-risk texters gave up</div></div>
          <div class="lane clock"><div class="lane-name">Shift clock</div><div class="count" id="c-clock">00:00</div><div class="lane-sub" id="c-arrived">0 high-risk texters so far</div></div>
        </div>
        <div id="race-chart"></div>
      </div>` : ""}
      <div class="kpis">
        <div class="kpi"><div class="v">${Math.round(h.fifo * 100)}% → ${Math.round(h.saath * 100)}%</div><div class="l">High-risk texters who give up before anyone answers, at ${h.load}× load (first come vs Saath)</div></div>
        ${rr ? `<div class="kpi"><div class="v">${Math.round(rr.mean * 100)}% fewer</div><div class="l">High-risk people lost, 95% CI ${Math.round(rr.ci[0] * 100)}–${Math.round(rr.ci[1] * 100)}% from resampling the posts. ${sens}</div></div>` : ""}
        <div class="kpi"><div class="v">${fmt(r.urgent.miss_rate * 100, 1)}%</div><div class="l">High-risk messages that missed the urgent band in testing (promised ≤ ${Math.round(r.urgent.alpha * 100)}%)</div></div>
        <div class="kpi"><div class="v">${fmt(r.models.find((m) => m.ours).hr_auroc[0], 3)}</div><div class="l">AUROC for level 4+ on held-out posts</div></div>
        <div class="kpi"><div class="v">${r.latency_ms} ms</div><div class="l">Per message on a laptop CPU. No text leaves the machine.</div></div>
      </div>
      <div class="grid2">
        <div class="detail"><h3>High-risk texters who give up waiting</h3>
          ${lineChart(r.queue.series, r.queue.loads, { yLabel: "share abandoned", xLabel: "offered load (arrivals ÷ counsellor capacity)" })}
          <p class="muted" style="font-size:13px">${r.queue.note}</p></div>
        <div class="detail"><h3>Model comparison on ${r.n_posts} C-SSRS-labelled posts</h3>
          <table><thead><tr><th>Model</th><th>Weighted κ</th><th>Recall L4+</th><th>Precision L4+</th></tr></thead>
          <tbody>${r.models.map((m) => `<tr class="${m.ours ? "ours" : ""}"><td>${m.name}</td><td class="num">${ci(m.qwk)}</td><td class="num">${ci(m.hr_recall)}</td><td class="num">${ci(m.hr_precision)}</td></tr>`).join("")}</tbody></table>
          <p class="muted" style="font-size:13px">${r.models_note}</p></div>
      </div>`;
    if (r.night) {
      $("#replay").addEventListener("click", () => playNight(r.night));
      playNight(r.night);
    }
  } catch (err) {
    el.innerHTML = `<p class="muted">Results are not available yet (${escapeHtml(String(err))}).</p>`;
  }
}

let nightTimer = null;
function playNight(night) {
  clearInterval(nightTimer);
  const { t } = night.fifo;
  const W = 900, H = 160, P = { l: 36, r: 10, t: 10, b: 24 };
  const yMax = Math.max(...night.fifo.hr_gave_up, 1) * 1.1;
  const x = (i) => P.l + (i / (t.length - 1)) * (W - P.l - P.r);
  const y = (v) => H - P.b - (v / yMax) * (H - P.t - P.b);
  const path = (arr, n) => arr.slice(0, n + 1).map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join("");
  let i = 0;
  const draw = () => {
    $("#c-fifo").textContent = night.fifo.hr_gave_up[i];
    $("#c-saath").textContent = night.saath.hr_gave_up[i];
    const m = t[i];
    $("#c-clock").textContent = `${String(Math.floor(m / 60)).padStart(2, "0")}:${String(m % 60).padStart(2, "0")}`;
    $("#c-arrived").textContent = `${night.fifo.hr_arrived[i]} high-risk texters so far`;
    $("#race-chart").innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Cumulative high-risk give-ups over the shift">
      ${[0, 0.5, 1].map((f) => `<line x1="${P.l}" x2="${W - P.r}" y1="${y(f * yMax)}" y2="${y(f * yMax)}" stroke="var(--line)"/><text x="${P.l - 6}" y="${y(f * yMax) + 4}" text-anchor="end">${Math.round(f * yMax)}</text>`).join("")}
      ${[0, 3, 6, 9, 12].map((h) => `<text x="${x((h * 60) / (t[t.length - 1]) * (t.length - 1))}" y="${H - 6}" text-anchor="middle">${h}h</text>`).join("")}
      <path d="${path(night.fifo.hr_gave_up, i)}" fill="none" stroke="var(--muted)" stroke-width="2.5"/>
      <path d="${path(night.saath.hr_gave_up, i)}" fill="none" stroke="var(--urgent)" stroke-width="2.5"/></svg>`;
  };
  draw();
  nightTimer = setInterval(() => {
    i += 1;
    if (i >= t.length) { clearInterval(nightTimer); return; }
    draw();
  }, 60);
}

// ---------- boot ----------
(async function boot() {
  META = await api("/api/meta");
  DEMO = await api("/api/demo-queue");
  const scores = await api("/api/score-batch", { texts: DEMO.map((d) => d.text) });
  DEMO = DEMO.map((d, i) => ({ ...d, score: scores[i] }));
  renderQueue();
})();
