// The whole Saath ensemble, run in the visitor's browser.
//
// The two sentence encoders are downloaded once (int8 ONNX builds, about
// 390 MB together) and cached by the browser. Their logistic heads, the
// TF-IDF member and every cut-off come from the deployed model, so the
// result follows the same rules as saath/service.py. The int8 encoders are
// close to, but not bit-identical with, the float encoders the cut-offs were
// calibrated on; validate() measures how close.

import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2";

const HIGH = 4;
const BROWSER_MODELS = {
  "intfloat/multilingual-e5-base": "Xenova/multilingual-e5-base",
  "sentence-transformers/all-mpnet-base-v2": "Xenova/all-mpnet-base-v2",
};
const PREFIX = (m) => (m.startsWith("intfloat/") ? "query: " : "");

let SPEC = null;
const PIPES = new Map();

export async function load(onProgress = () => {}) {
  SPEC = SPEC || (await (await fetch("data/full.json")).json());
  const loaded = {};
  for (const m of SPEC.members.filter((x) => x.kind === "embed")) {
    if (PIPES.has(m.model)) continue;
    PIPES.set(
      m.model,
      await pipeline("feature-extraction", BROWSER_MODELS[m.model], {
        dtype: "q8",
        progress_callback: (p) => {
          if (p.status === "progress" && p.total) {
            loaded[`${m.model}/${p.file}`] = [p.loaded, p.total];
            const [a, t] = Object.values(loaded).reduce(([x, y], [l, tt]) => [x + l, y + tt], [0, 0]);
            onProgress(a / t);
          }
        },
      }),
    );
  }
  return SPEC;
}

const sigmoid = (z) => 1 / (1 + Math.exp(-z));

async function memberExceed(member, texts, tfidfRaw) {
  if (member.kind === "tfidf") return Promise.all(texts.map((t) => tfidfRaw(t)));
  const pipe = PIPES.get(member.model);
  const out = await pipe(texts.map((t) => PREFIX(member.model) + t), { pooling: "mean", normalize: true });
  return out.tolist().map((v) => member.heads.map((h) => sigmoid(h.b + v.reduce((s, x, j) => s + x * h.coef[j], 0))));
}

// Ensemble.exceed_proba: weighted mean of each member's raw P(level > k).
async function exceedMany(texts, tfidfRaw) {
  const per = await Promise.all(SPEC.members.map((m) => memberExceed(m, texts, tfidfRaw)));
  const w = SPEC.weights || SPEC.members.map(() => 1);
  const tot = w.reduce((a, b) => a + b, 0);
  return texts.map((_, i) => per[0][i].map((_, k) => per.reduce((s, p, j) => s + (w[j] / tot) * p[i][k], 0)));
}

const clipDown = (e) => {
  const out = e.map((p) => Math.min(Math.max(p, 0), 1));
  for (let k = 1; k < out.length; k++) out[k] = Math.min(out[k], out[k - 1]);
  return out;
};

function levelProba(exceed) {
  const e = clipDown(exceed);
  const upper = [1, ...e];
  const lower = [...e, 0];
  const p = upper.map((u, i) => Math.max(u - lower[i], 0));
  const s = p.reduce((a, b) => a + b, 0) || 1;
  return p.map((x) => x / s);
}

function interp(x, xs, ys) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  let i = 1;
  while (xs[i] < x) i++;
  const t = (x - xs[i - 1]) / (xs[i] - xs[i - 1] || 1);
  return ys[i - 1] + t * (ys[i] - ys[i - 1]);
}

// Scorer.score in saath/service.py, step for step.
function finish(text, exceed, isIndic, levels) {
  const cal = SPEC.calibration;
  const proba = levelProba(exceed);
  const risk = exceed[HIGH - 1];
  const level = clipDown(exceed).filter((p) => p > 0.5).length;
  let cdf = 0;
  const tail = proba.map((p) => { cdf += p; return Math.min(cdf, 1 - cdf + p); });
  let keep = tail.map((s, k) => (s >= cal.level_qhat ? k : -1)).filter((k) => k >= 0);
  if (!keep.length) keep = [tail.indexOf(Math.max(...tail))];
  let lo = Math.min(...keep);
  let hi = Math.max(...keep);
  const urgent = risk >= cal.urgent_threshold;
  if (urgent) hi = Math.max(hi, HIGH);
  const critical = cal.critical_threshold != null && risk >= cal.critical_threshold;
  let band, reason;
  if (urgent) [band, reason] = ["urgent", critical ? "critical" : "risk"];
  else if (hi >= HIGH) [band, reason] = ["review", "range"];
  else if (isIndic(text)) [band, reason] = ["review", "language"];
  else [band, reason] = ["routine", "risk"];
  const quant = Array.from({ length: 101 }, (_, i) => i / 100);
  return {
    level,
    level_label: levels[level],
    level_range: [lo, hi],
    proba,
    risk,
    risk_calibrated: cal.risk_x ? interp(risk, cal.risk_x, cal.risk_y) : risk,
    priority: cal.risk_quantiles ? interp(risk, cal.risk_quantiles, quant) : risk,
    band,
    band_reason: reason,
  };
}

const sentences = (text) => {
  const parts = text.split(/(?<=[.!?…])\s+|\n+/).map((p) => p.trim()).filter(Boolean);
  return parts.length ? parts : [text.trim()];
};

export async function score(text, { tfidfRaw, isIndic, levels }) {
  const t0 = performance.now();
  const parts = sentences(text);
  const variants = [text, ...parts.map((_, i) => parts.filter((__, j) => j !== i).join(" ") || ".")];
  const ex = await exceedMany(variants, tfidfRaw);
  const out = finish(text, ex[0], isIndic, levels);
  out.evidence = parts.map((p, i) => ({ text: p, weight: Math.max(out.risk - ex[i + 1][HIGH - 1], 0) }));
  out.latency_ms = Math.round(performance.now() - t0);
  out.in_browser = true;
  return out;
}

// Compare against the Python service on the texts it scored ahead of time.
export async function validate({ tfidfRaw, isIndic, levels }) {
  const texts = Object.keys(SPEC.ref);
  const ex = await exceedMany(texts, tfidfRaw);
  return texts.map((t, i) => {
    const js = finish(t, ex[i], isIndic, levels);
    const py = SPEC.ref[t];
    return { risk: [py.risk, +js.risk.toFixed(4)], level: [py.level, js.level], band: [py.band, js.band], range: [py.level_range, js.level_range] };
  });
}
