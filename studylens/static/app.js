// StudyLens front end: vanilla JS, no build step.
// State lives here; the server is stateless (image + graph are sent back with each question).
const $ = (id) => document.getElementById(id);
let graph = null;      // {nodes, edges, source, image_size, warnings}
let imageBlob = null;  // the current image, kept only in the browser
let evidence = new Set(); // "src->tgt" keys to highlight
let answerNodes = new Set();
let health = null;
let curatedQs = [];   // example questions shipped with a bundled sample (none for uploads)

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const label = (id) => (graph.nodes.find((n) => n.id === id) || { label: id }).label;
const pct = (x) => (x == null ? "–" : (100 * x).toFixed(1) + "%");

// ------------------------------------------------------------------ tabs
document.querySelectorAll("nav .tab").forEach((b) => b.onclick = () => {
  document.querySelectorAll("nav .tab").forEach((x) => x.classList.toggle("active", x === b));
  $("tab-ask").hidden = b.dataset.tab !== "ask";
  $("tab-eval").hidden = b.dataset.tab !== "eval";
  if (b.dataset.tab === "eval") loadRuns();
});

// ------------------------------------------------------------------ init
async function init() {
  try {
    health = await (await fetch("/api/health")).json();
    const b = $("backend");
    if (health.extractor_is_ai) { b.textContent = `Model: ${health.extractor}`; b.className = "badge ai"; }
    else { b.textContent = "No model loaded: ground-truth annotation fallback (NOT AI)"; b.className = "badge fallback"; }
    $("limits").textContent = `Any image: ${health.formats.join(", ")} · max ${health.max_upload_mb} MB. Processed in memory, never stored on the server.`;
  } catch { $("backend").textContent = "backend unreachable"; }
  const s = await (await fetch("/api/samples")).json();
  for (const smp of s.samples) {
    const o = document.createElement("option"); o.value = smp.file; o.textContent = smp.title; $("samples").append(o);
  }
  $("samples").onchange = async (e) => {
    if (!e.target.value) return;
    const blob = await (await fetch(`/api/samples/${e.target.value}`)).blob();
    loadImage(blob, s.samples.find((x) => x.file === e.target.value).example_questions);
  };
  $("file").onchange = (e) => { const f = e.target.files[0]; if (f) loadImage(f, []); e.target.value = ""; };
}

let extractCtl = null; // AbortController of a running extraction

// Picking an image only previews it; the "Extract graph" button starts the model.
function loadImage(blob, examples) {
  if (health && blob.size > health.max_upload_mb * 1024 * 1024) { setStatus(`File is larger than ${health.max_upload_mb} MB.`); return; }
  if (extractCtl) extractCtl.abort();
  imageBlob = blob; graph = null; evidence.clear(); answerNodes.clear();
  $("img").onerror = () => { $("img").removeAttribute("src"); $("empty").hidden = false; $("empty").textContent = "Your browser cannot preview this format; the graph can still be extracted."; };
  $("img").src = URL.createObjectURL(blob); $("empty").hidden = true;
  $("rawbox").hidden = true; $("progress").hidden = true;
  ocrLines = null; if (askCtl) askCtl.abort(); buildPipes(); buildMain(); refs = []; focus = null;
  $("exAnswer").innerHTML = `<p class="hint">Extract the graph, then ask a question.</p>`; $("exRefs").innerHTML = "";
  renderGraph();
  curatedQs = examples || [];
  setChips(curatedQs, curatedQs.length ? "Try:" : "");
  $("extractBtn").disabled = false; $("extractBtn").textContent = "Extract graph";
  $("extractHint").textContent = health && health.extractor_is_ai ? "Runs the vision-language model (≈20–90 s on a laptop; dense diagrams with many boxes can take several minutes)." : "Uses the bundled ground-truth graph (no model loaded).";
  setStatus("");
}

// ---- progress bar driven by real server events (NDJSON stream from /api/extract_stream)
let progTimer = null, progT0 = 0;
function progress(pct, label, cls = "") {
  $("progress").hidden = false;
  $("pfill").style.width = Math.max(2, Math.min(100, pct)) + "%";
  $("pfill").className = "pfill " + cls;
  $("plabel").textContent = label;
}
function progressCounts(html) { $("pcounts").innerHTML = html; }
function startTimer() {
  progT0 = performance.now(); clearInterval(progTimer);
  progTimer = setInterval(() => { $("ptime").textContent = ((performance.now() - progT0) / 1000).toFixed(0) + " s"; }, 250);
}
function stopTimer() { clearInterval(progTimer); $("ptime").textContent = ((performance.now() - progT0) / 1000).toFixed(1) + " s"; }

async function extractGraph() {
  if (!imageBlob) return;
  const btn = $("extractBtn");
  btn.disabled = true; btn.textContent = "Extracting…";
  graph = null; evidence.clear(); answerNodes.clear(); renderGraph();
  $("rawbox").hidden = true; setStatus(""); buildPipes(); buildMain(); refs = []; focus = null;
  extractCtl = new AbortController();
  startTimer(); progress(3, "Uploading the image…"); progressCounts("");
  const fd = new FormData(); fd.append("file", imageBlob, "upload");
  let done = null, failed = null;
  try {
    const r = await fetch("/api/extract_stream", { method: "POST", body: fd, signal: extractCtl.signal });
    if (!r.ok) { failed = (await r.json()).detail || r.status; }
    else {
      const reader = r.body.getReader(), dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done: end } = await reader.read();
        if (end) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
          if (!line) continue;
          const ev = JSON.parse(line);
          if (ev.stage === "loading_model") progress(10, "Loading the model into memory (first extraction only, ≈20 s)…");
          else if (ev.stage === "encoding") progress(18, "Reading the image…");
          else if (ev.stage === "ocr") progress(10, "Reading the text with OCR (PaddleOCR)…", "live");
          else if (ev.stage === "ocr_done") { progress(25, "Grouping text into labelled parts…"); progressCounts(`OCR found <b>${ev.n_texts}</b> text items → <b>${ev.entities}</b> labelled parts`); }
          else if (ev.stage === "structure") progress(30, "The model is finding the arrows between the parts…", "live");
          else if (ev.stage === "structure_done") { progress(70, "Finding the region of each part…"); progressCounts(`<b>${ev.entities}</b> parts · <b>${ev.arrows}</b> arrows`); }
          else if (ev.stage === "locating") progress(75, `Locating ${ev.n} part(s) that are not inside a box (model grounding)…`, "live");
          else if (ev.stage === "generating") {
            // Output length is not known in advance, so the bar position here is an estimate;
            // the counts below it are read from what the model has actually written so far.
            progress(20 + 75 * (1 - Math.exp(-ev.tokens / 250)), "Model is writing the graph…", "live");
            progressCounts(`<b>${ev.entities}</b> entities · <b>${ev.arrows}</b> arrows found so far <span class="hint">(${ev.tokens} tokens)</span>`);
          } else if (ev.stage === "done") done = ev.graph;
          else if (ev.stage === "error") failed = ev.detail;
        }
      }
    }
  } catch (e) {
    if (e.name === "AbortError") { stopTimer(); return; }
    failed = String(e);
  }
  stopTimer(); extractCtl = null;
  btn.disabled = false; btn.textContent = "Re-extract";
  if (failed || !done) {
    progress(100, "Extraction failed", "err"); setStatus("Extraction failed: " + (failed || "no result"));
    return;
  }
  graph = done;
  // the scene-graph extraction already ran OCR: reuse it for the E1 baseline
  if (graph.texts?.length) ocrLines = graph.texts.map((t) => ({ text: t.text, box: t.box, score: 1 }));
  progress(100, `Done: ${graph.nodes.length} parts and ${graph.edges.length} arrows`, "ok");
  progressCounts(`<span class="hint">source: ${esc(graph.source)}</span>`);
  if (graph.raw_output) { $("raw").textContent = graph.raw_output; $("rawbox").hidden = false; }
  setStatus("Review the entities and arrows, fix any mistakes, then ask a question.");
  renderGraph();
  refreshSuggestions();
}
$("extractBtn").onclick = extractGraph;

function setStatus(t) { $("status").textContent = t; }
function setChips(qs, title) {
  $("chips").innerHTML = "";
  if (qs.length && title) { const t = document.createElement("span"); t.className = "hint"; t.textContent = title; $("chips").append(t); }
  for (const q of qs) { const b = document.createElement("button"); b.type = "button"; b.textContent = q; b.onclick = () => { $("q").value = q; ask(); }; $("chips").append(b); }
}

// Question suggestions built from the extracted graph, so ANY image (not only the bundled
// samples) gets suggestions. Templates only: no model call, so they appear instantly.
function suggestFromGraph(g) {
  if (!g || !g.nodes.length) return [];
  const L = (id) => label(id), ok = (id) => { const t = L(id); return t && t.length <= 40; };
  const E = g.edges.filter((e) => ok(e.source) && ok(e.target) && e.source !== e.target);
  const indeg = {}, outdeg = {}, outs = {};
  for (const e of E) {
    indeg[e.target] = (indeg[e.target] || 0) + 1; outdeg[e.source] = (outdeg[e.source] || 0) + 1;
    (outs[e.source] ||= []).push(e.target);
  }
  const top = (deg) => Object.keys(deg).sort((a, b) => deg[b] - deg[a])[0];
  const qs = [];
  if (!E.length) {
    const named = g.nodes.filter((n) => ok(n.id)).slice(0, 2);
    for (const n of named) qs.push(`What does the diagram show about ${L(n.id)}?`);
    qs.push("What is this diagram about?");
    return qs;
  }
  const hub = top(outdeg), sink = top(indeg);
  if (hub) qs.push(`What does ${L(hub)} lead to?`);
  if (sink && sink !== hub) qs.push(`Which things point to ${L(sink)}?`);
  // a two-step chain a -> b -> c makes a "what if" question
  for (const e of E) {
    const c = (outs[e.target] || []).find((t) => t !== e.source);
    if (c) { qs.push(`What would happen to ${L(c)} if ${L(e.source)} disappeared?`); break; }
  }
  // the reverse of a real arrow: a good check that the answer follows arrow direction (expected: no)
  const r = E.find((e) => !E.some((f) => f.source === e.target && f.target === e.source));
  if (r) qs.push(`Is there an arrow from ${L(r.target)} to ${L(r.source)}?`);
  if (E.length >= 2) qs.push("Which entities have no incoming arrows?");
  return qs;
}

function refreshSuggestions() {
  const seen = new Set(), qs = [];
  for (const q of [...curatedQs, ...suggestFromGraph(graph)]) { const k = q.toLowerCase(); if (!seen.has(k)) { seen.add(k); qs.push(q); } }
  setChips(qs.slice(0, 6), "Try:");
}

// ------------------------------------------------------------------ graph panel + overlay
function renderGraph() {
  const nl = $("nodes"), el = $("edges");
  nl.innerHTML = ""; el.innerHTML = ""; $("warnings").innerHTML = "";
  $("gsource").textContent = graph ? `(source: ${graph.source})` : "";
  $("graphHint").hidden = !!graph;
  if (!graph) { drawOverlay(); return; }
  for (const w of graph.warnings || []) { const d = document.createElement("div"); d.textContent = "⚠ " + w; $("warnings").append(d); }
  if (!graph.edges.length) { const d = document.createElement("div"); d.textContent = "No arrows were extracted. E3 can only answer from the image for this diagram (labelled as image-only, without evidence). Add arrows below if the diagram has them."; $("warnings").append(d); }
  for (const n of graph.nodes) {
    const li = document.createElement("li");
    const RS = { shape: ["box", "region = enclosing shape (image analysis)"], grounded: ["AI", "region located by the model (approximate)"], label: ["text", "only the label position is known"] };
    const rs = RS[n.region_source];
    li.innerHTML = `<span>${esc(n.label)}</span>${rs ? `<span class="rsrc ${esc(n.region_source)}" title="${rs[1]}">${rs[0]}</span>` : ""}`;
    if (answerNodes.has(n.id)) li.className = "ev"; nl.append(li);
  }
  graph.edges.forEach((e, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span><span class="eid">e${i + 1}</span> ${esc(label(e.source))} → ${esc(label(e.target))}</span><button title="remove this arrow">✕</button>`;
    if (evidence.has(`${e.source}->${e.target}`)) li.className = "ev";
    li.querySelector("button").onclick = () => { graph.edges.splice(i, 1); markEdited(); };
    el.append(li);
  });
  for (const sel of [$("addSrc"), $("addTgt")]) {
    sel.innerHTML = graph.nodes.map((n) => `<option value="${esc(n.id)}">${esc(n.label)}</option>`).join("");
  }
  drawOverlay();
}

function markEdited() {
  if (!graph.source.includes("user-edited")) graph.source += " + user-edited";
  evidence.clear(); answerNodes.clear(); buildPipes(); renderGraph();
  refreshSuggestions();
}

$("addEdge").onclick = () => {
  if (!graph) return;
  const s = $("addSrc").value, t = $("addTgt").value;
  if (s === t || graph.edges.some((e) => e.source === s && e.target === t)) return;
  graph.edges.push({ source: s, target: t, relation: "points_to", confidence: null }); markEdited();
};

function center(b) { return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]; }
function clip(b, tx, ty) {
  const [cx, cy] = center(b), dx = tx - cx, dy = ty - cy;
  const hw = (b[2] - b[0]) / 2 + 3, hh = (b[3] - b[1]) / 2 + 3;
  const s = Math.min(Math.abs(dx) > 1e-6 ? hw / Math.abs(dx) : Infinity, Math.abs(dy) > 1e-6 ? hh / Math.abs(dy) : Infinity, 1);
  return [cx + dx * s, cy + dy * s];
}

function drawOverlay() {
  const svg = $("overlay");
  if (!graph || !graph.image_size) { svg.innerHTML = ""; return; }
  const [W, H] = graph.image_size;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const sw = Math.max(2, W / 400);
  let out = `<defs>
    <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0L10,5L0,10z" fill="#2f6fde"/></marker>
    <marker id="ahE" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0L10,5L0,10z" fill="#e5484d"/></marker></defs>`;
  const byId = Object.fromEntries(graph.nodes.map((n) => [n.id, n]));
  for (const n of graph.nodes) {
    if (!n.bbox) continue;
    const [x1, y1, x2, y2] = n.bbox, hit = answerNodes.has(n.id);
    out += `<rect x="${x1}" y="${y1}" width="${x2 - x1}" height="${y2 - y1}" fill="${hit ? "rgba(229,72,77,.12)" : "rgba(47,111,222,.06)"}" stroke="${hit ? "#e5484d" : "#2f6fde"}" stroke-width="${sw}" stroke-dasharray="${hit ? "" : sw * 3}"/>`;
  }
  for (const e of graph.edges) {
    const a = byId[e.source], b = byId[e.target];
    if (!a?.bbox || !b?.bbox) continue;
    const ev = evidence.has(`${e.source}->${e.target}`);
    const [bx, by] = center(b.bbox), [ax, ay] = center(a.bbox);
    const p = clip(a.bbox, bx, by), q = clip(b.bbox, ax, ay);
    out += `<line x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" stroke="${ev ? "#e5484d" : "#2f6fde"}" stroke-opacity="${ev ? 1 : 0.55}" stroke-width="${ev ? sw * 2.2 : sw}" marker-end="url(#${ev ? "ahE" : "ah"})"/>`;
  }
  if (focus) {
    const byId2 = Object.fromEntries(graph.nodes.map((n) => [n.id, n]));
    const boxes = [...focus.boxes, ...focus.nodes.map((id) => byId2[id]?.bbox).filter(Boolean)];
    const pad = Math.max(6, W / 120);
    out += `<defs><mask id="spot"><rect x="0" y="0" width="${W}" height="${H}" fill="white"/>${boxes.map((b) => `<rect x="${b[0] - pad}" y="${b[1] - pad}" width="${b[2] - b[0] + 2 * pad}" height="${b[3] - b[1] + 2 * pad}" rx="${pad}" fill="black"/>`).join("")}</mask>
      <marker id="ahF" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0L10,5L0,10z" fill="#f79009"/></marker></defs>`;
    out += `<rect x="0" y="0" width="${W}" height="${H}" fill="rgba(16,24,40,.55)" mask="url(#spot)"/>`;
    for (const b of boxes) out += `<rect class="spotbox ${focus.located ? "located" : ""}" x="${b[0] - pad}" y="${b[1] - pad}" width="${b[2] - b[0] + 2 * pad}" height="${b[3] - b[1] + 2 * pad}" rx="${pad}" fill="none" stroke="${focus.located ? "#f79009" : "#e5484d"}" stroke-width="${sw * 1.6}" ${focus.located ? `stroke-dasharray="${sw * 4}"` : ""}/>`;
    for (const a of focus.arrows) {
      const A = byId2[a.source], B = byId2[a.target];
      if (!A?.bbox || !B?.bbox) continue;
      const [bx, by] = center(B.bbox), [ax, ay] = center(A.bbox);
      const p = clip(A.bbox, bx, by), q = clip(B.bbox, ax, ay);
      out += `<line x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" stroke="#f79009" stroke-width="${sw * 2.6}" marker-end="url(#ahF)"/>`;
    }
    const lb = boxes[boxes.length - 1];
    if (lb && focus.label) out += `<text x="${lb[0]}" y="${Math.max(14, lb[1] - pad - 4)}" font-size="${Math.max(12, W / 55)}" fill="#fff" stroke="#101828" stroke-width="3" paint-order="stroke" font-weight="700">${esc(focus.label)}${focus.located ? " (approx.)" : ""}</text>`;
  }
  svg.innerHTML = out;
}

// ------------------------------------------------------------------ the three pipelines
// Architecture of each pipeline. Node ids are the stage names the server streams
// (studylens/pipelines.py STAGES); x/y place the boxes in a 400-wide SVG.
const PIPES = {
  EX: {
    title: "StudyLens answer", tag: "main",
    desc: "Structural facts come from the graph (NetworkX). Qwen writes the explanation; every part it names is linked to the scene graph by exact name matching (no AI), so each reference points at a real part.",
    nodes: [
      { id: "image", t: "Image", s: "uploaded diagram", x: 10, y: 8 },
      { id: "graph", t: "Scene graph", s: "OCR parts + arrows (step 1)", x: 210, y: 8 },
      { id: "facts", t: "Graph facts", s: "NetworkX, if structural", x: 110, y: 64 },
      { id: "explain", t: "Explanation", s: "Qwen writes it", x: 110, y: 120 },
      { id: "check", t: "Reference linking", s: "exact names, no AI", x: 10, y: 176 },
      { id: "locate", t: "Locate extras", s: "Qwen grounding", x: 210, y: 176 },
      { id: "answer", t: "Answer + references", s: "click to highlight", x: 110, y: 232 },
    ],
    edges: [["image", "facts"], ["graph", "facts"], ["facts", "explain"], ["explain", "check"], ["explain", "locate"], ["check", "answer"], ["locate", "answer"]], h: 272,
  },
  E1: {
    title: "E1 · OCR text only", tag: "baseline",
    desc: "PaddleOCR reads the text in the image; Qwen answers from that text alone. No image, no graph.",
    nodes: [
      { id: "image", t: "Image", s: "uploaded diagram", x: 100, y: 8 },
      { id: "ocr", t: "PaddleOCR", s: "text + positions", x: 100, y: 64 },
      { id: "llm_text", t: "Qwen2.5-VL-3B", s: "text-only prompt", x: 100, y: 120 },
      { id: "answer", t: "Answer", s: "no evidence", x: 100, y: 176 },
    ],
    edges: [["image", "ocr"], ["ocr", "llm_text"], ["llm_text", "answer"]], h: 216,
  },
  E2: {
    title: "E2 · Image → VLM", tag: "main baseline",
    desc: "The vision-language model answers from the image directly, like a general AI assistant. No graph, no evidence.",
    nodes: [
      { id: "image", t: "Image", s: "uploaded diagram", x: 100, y: 8 },
      { id: "vlm", t: "Qwen2.5-VL-3B", s: "image + question", x: 100, y: 92 },
      { id: "answer", t: "Answer", s: "no evidence", x: 100, y: 176 },
    ],
    edges: [["image", "vlm"], ["vlm", "answer"]], h: 216,
  },
  E3: {
    title: "E3 · Graph-grounded", tag: "proposed",
    desc: "The question becomes a graph query that NetworkX answers over the extracted graph, citing the arrows used. The model only parses and explains. If the query cannot apply, the model answers and its arrow citations are checked.",
    nodes: [
      { id: "image", t: "Image", s: "uploaded diagram", x: 10, y: 8 },
      { id: "extract", t: "Extracted graph", s: "Qwen + LoRA (step 1)", x: 10, y: 60 },
      { id: "parse", t: "Question → query", s: "rules, else Qwen parser", x: 10, y: 112 },
      { id: "query", t: "NetworkX query", s: "deterministic", x: 10, y: 164 },
      { id: "explain", t: "Explain", s: "Qwen, cited arrows only", x: 10, y: 216 },
      { id: "fallback", t: "Fallback", s: "Qwen + graph, citations checked", x: 210, y: 216 },
      { id: "answer", t: "Answer + evidence", s: "or abstain", x: 110, y: 268 },
    ],
    edges: [["image", "extract"], ["extract", "parse"], ["parse", "query"], ["query", "explain"], ["query", "fallback"], ["explain", "answer"], ["fallback", "answer"]], h: 308,
  },
};
const ORDER = ["E1", "E2", "E3"];  // comparison pipelines (the main answer is EX)
const NW = 180, NH = 40;
let ocrLines = null;          // OCR result for the current image (E1 runs OCR once per image)
let askCtl = null, pipeTimers = {};

function pipeSVG(k) {
  const P = PIPES[k], pos = Object.fromEntries(P.nodes.map((n) => [n.id, n]));
  let g = `<defs><marker id="ah-${k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" class="parrowhead"/></marker></defs>`;
  for (const [a, b] of P.edges) {
    const A = pos[a], B = pos[b];
    const x1 = A.x + NW / 2, y1 = A.y + NH, x2 = B.x + NW / 2, y2 = B.y - 2;
    g += `<line class="pedge" id="pe-${k}-${a}-${b}" x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" marker-end="url(#ah-${k})"/>`;
  }
  for (const n of P.nodes) {
    g += `<g class="pnode st-pending" id="pn-${k}-${n.id}"><rect x="${n.x}" y="${n.y}" width="${NW}" height="${NH}" rx="8"/>` +
      `<text x="${n.x + NW / 2}" y="${n.y + 17}" class="pt">${esc(n.t)}</text><text x="${n.x + NW / 2}" y="${n.y + 32}" class="ps">${esc(n.s)}</text></g>`;
  }
  return `<svg class="pipesvg" viewBox="0 0 400 ${P.h}" role="img" aria-label="${esc(P.title)} architecture">${g}</svg>`;
}

function pipeCard(k) {
  return `
    <div class="pipe" id="pipe-${k}">
      <div class="phead"><b>${esc(PIPES[k].title)}</b><span class="ptag ${k === "E3" || k === "EX" ? "prop" : ""}">${esc(PIPES[k].tag)}</span><span class="ptimer" id="pt-${k}"></span></div>
      <p class="hint">${esc(PIPES[k].desc)}</p>
      ${pipeSVG(k)}
      <ol class="plog" id="pl-${k}"></ol>
      <div class="pans" id="pa-${k}" hidden></div>
    </div>`;
}

function buildMain() { $("pipes-main").innerHTML = pipeCard("EX"); $("pa-EX").remove(); }

function buildPipes() {
  $("pipes").innerHTML = ORDER.map((k) => `
    <div class="pipe" id="pipe-${k}">
      <div class="phead"><b>${esc(PIPES[k].title)}</b><span class="ptag ${k === "E3" ? "prop" : ""}">${esc(PIPES[k].tag)}</span><span class="ptimer" id="pt-${k}"></span></div>
      <p class="hint">${esc(PIPES[k].desc)}</p>
      ${pipeSVG(k)}
      <ol class="plog" id="pl-${k}"></ol>
      <div class="pans" id="pa-${k}" hidden></div>
    </div>`).join("");
}

function setStage(k, stage, state) {
  const node = $(`pn-${k}-${stage}`);
  if (!node) return;
  node.setAttribute("class", `pnode st-${state}`);
  // light the arrow(s) that lead into this stage
  for (const [a, b] of PIPES[k].edges) {
    if (b !== stage) continue;
    const from = $(`pn-${k}-${a}`), e = $(`pe-${k}-${a}-${b}`);
    if (e && from && /st-(done)/.test(from.getAttribute("class")) && state !== "skipped") e.setAttribute("class", `pedge on ${state === "running" ? "flow" : ""}`);
  }
}

function plog(k, html, cls = "") { const li = document.createElement("li"); li.className = cls; li.innerHTML = html; $(`pl-${k}`).append(li); }

function startPipeTimer(k) {
  if (pipeTimers[k]) return;
  const t0 = performance.now();
  pipeTimers[k] = { t0, id: setInterval(() => { $(`pt-${k}`).textContent = ((performance.now() - t0) / 1000).toFixed(0) + " s"; }, 250) };
}
function stopPipeTimer(k) {
  const t = pipeTimers[k]; if (!t) return;
  clearInterval(t.id); $(`pt-${k}`).textContent = ((performance.now() - t.t0) / 1000).toFixed(1) + " s"; pipeTimers[k] = null;
}

function resetPipes(order) {
  for (const k of order) { stopPipeTimer(k); pipeTimers[k] = null; }
  if (order.includes("EX")) buildMain(); else buildPipes();
  order.forEach((k, i) => { $(`pt-${k}`).textContent = i === 0 ? "" : `queued (#${i + 1})`; });
}

const fmtS = (x) => (x == null ? "" : ` <span class="hint">${Number(x).toFixed(1)} s</span>`);
const opText = (q) => !q ? "" : q.op === "none" ? "none (not an arrow question)" : `${q.negate ? "NOT " : ""}${q.op}(${[q.x, q.y].filter(Boolean).map(esc).join(", ")})`;
const QSTAT = {
  answered: "found the answer in the graph",
  abstained: "the graph has no arrow that supports an answer → abstain",
  no_option_match: "the query result matches none (or several) of the options → fallback",
  not_structural: "not a question about arrows → fallback",
  unknown_entity: "the question names something that is not in the graph → fallback",
  unparsed: "could not turn the question into a query → fallback",
};

function onPipeEvent(ev) {
  const k = ev.pipeline;
  if (!k || !PIPES[k]) return;
  if ("result" in ev) { stopPipeTimer(k); renderResult(k, ev.result); return; }
  startPipeTimer(k);
  setStage(k, ev.stage, ev.state);
  if (ev.state === "error") { plog(k, `⚠ ${esc(ev.stage)}: ${esc(ev.detail || "error")}`, "err"); return; }
  if (ev.state !== "done" && ev.state !== "skipped") return;
  const s = ev.stage;
  if (k === "EX" && s === "graph") plog(k, `Scene graph from step 1: <b>${ev.entities}</b> parts, <b>${ev.arrows}</b> arrows`);
  if (k === "EX" && s === "facts") plog(k, ev.state === "done" ? `Structural question → <span class="mono">${opText(ev.query)}</span> answered by NetworkX: ${esc(ev.answer)}` : `Not a structural question (${esc(QSTAT[ev.status] ? ev.status : ev.status)}); explanation only`);
  if (k === "EX" && s === "explain") plog(k, `Qwen wrote the explanation${fmtS(ev.latency_s)}`);
  if (k === "EX" && s === "check") plog(k, `Linked <b>${ev.mentions}</b> mention(s) of graph parts → <b>${ev.n_refs}</b> reference(s), ${ev.flows} flow(s) along graph arrows (no AI)`);
  if (k === "EX" && s === "locate" && ev.state === "done") plog(k, `Located ${ev.n} part(s) that were not in the graph (approximate)`);
  if (k === "E1" && s === "ocr") plog(k, `PaddleOCR found <b>${ev.n_lines}</b> text lines${ev.cached ? " (reused from the previous question)" : fmtS(ev.latency_s)}: <span class="mono">${ev.preview.map(esc).join(" · ")}</span>`);
  if (k === "E1" && s === "llm_text") plog(k, `Qwen answered from the OCR text${fmtS(ev.latency_s)}`);
  if (k === "E2" && s === "vlm") plog(k, `Qwen answered from the image${fmtS(ev.latency_s)}`);
  if (k === "E3" && s === "extract") plog(k, `Graph from step 1: <b>${ev.entities}</b> entities, <b>${ev.arrows}</b> arrows <span class="hint">(${esc(ev.source)})</span>`);
  if (k === "E3" && s === "parse") plog(k, ev.query ? `Parsed by <b>${esc(ev.parser)}</b> → <span class="mono">${opText(ev.query)}</span>${fmtS(ev.latency_s)}` : `Parser: ${esc(ev.parser || "none")} → no query`);
  if (k === "E3" && s === "query") plog(k, `${ev.state === "skipped" ? "Query not run: " : "NetworkX "}${esc(QSTAT[ev.status] || ev.status)}`);
  if (k === "E3" && s === "explain" && ev.state === "done") plog(k, `Qwen wrote an explanation from the cited arrows${fmtS(ev.latency_s)}`);
  if (k === "E3" && s === "fallback" && ev.state === "done") plog(k, `Fallback answer from Qwen (image + graph); citation check: <b>${esc(ev.status)}</b>${fmtS(ev.latency_s)}`);
}

const E3STATUS = {
  graph_answer: ["ok", "Answer from the graph (NetworkX)", "Computed deterministically from the extracted arrows below."],
  abstained: ["warn", "Abstained", "No arrow in the extracted graph supports an answer, so nothing is guessed. If the diagram does show one, add it in step 1 and ask again."],
  grounded: ["ok", "Model answer, cites graph arrows", "The graph query could not apply; the model's cited arrows exist and involve its answer."],
  unsupported: ["warn", "Model answer, NOT supported by the graph", "The graph query could not apply and no extracted arrow supports this answer. It comes from the image alone; treat it with caution."],
  image_only: ["warn", "Image-only answer", "No arrows were extracted from this diagram, so the answer comes from the image alone and has no evidence."],
  error: ["err", "Error", ""],
};

function renderResult(k, r) {
  if (k === "EX") { renderExplanation(r); return; }
  const box = $(`pa-${k}`); box.hidden = false;
  if (!r || r.status === "error") { box.className = "pans err"; box.innerHTML = `<div class="pst">Error</div><p>${esc(r?.error || "failed")}</p>`; return; }
  if (k === "E1" && r.ocr_lines) ocrLines = r.ocr_lines;
  if (k !== "E3") {
    box.className = "pans neutral";
    box.innerHTML = `<div class="pst">${k === "E1" ? "Answer from OCR text (no evidence)" : "Answer from the image (no evidence)"}</div><p class="pansText">${esc(r.answer)}</p>`;
    return;
  }
  const [cls, title, note] = E3STATUS[r.status] || ["neutral", r.status, ""];
  box.className = `pans ${cls}`;
  let h = `<div class="pst">${esc(title)}</div><p class="pansText">${esc(r.answer)}</p>`;
  if (r.explanation) h += `<p class="pexpl">${esc(r.explanation)}</p>`;
  if (r.evidence?.length) h += `<div class="hint">Evidence: ${r.evidence.map((e) => `<span class="evi">${esc(e.text)}</span>`).join(", ")}</div>`;
  else h += `<div class="hint">${esc(note)}</div>`;
  const fb = r.fallback;
  if (fb?.invalid_citations?.length) h += `<div class="warnline">⚠ The model cited arrow id(s) that do not exist (${esc(fb.invalid_citations.join(", "))}); they were discarded.</div>`;
  if (fb?.grounding_note) h += `<div class="warnline">⚠ ${esc(fb.grounding_note)}</div>`;
  box.innerHTML = h;
  evidence = new Set((r.evidence || []).map((e) => `${e.source}->${e.target}`));
  answerNodes = new Set(r.answer_nodes || []);
  renderGraph();
}

function choicesList() { return $("choices").value.split("\n").map((s) => s.trim()).filter(Boolean); }

let lastAsk = null;    // {q, choices} of the latest question (for the comparison button)
let refs = [];         // references of the current explanation
let focus = null;      // what the overlay spotlights: {boxes, nodes, arrows, located, label}
let flowTimer = null;

async function runPipelines(order, q, ch) {
  if (askCtl) askCtl.abort();
  askCtl = new AbortController();
  resetPipes(order);
  const fd = new FormData();
  fd.append("question", q); fd.append("graph", JSON.stringify(graph)); fd.append("file", imageBlob, "upload");
  fd.append("pipelines", order.join(","));
  if (ch.length) fd.append("choices", JSON.stringify(ch));
  if (ocrLines) fd.append("ocr", JSON.stringify(ocrLines));
  try {
    const r = await fetch("/api/ask_stream", { method: "POST", body: fd, signal: askCtl.signal });
    if (!r.ok) { const e = await r.json(); for (const k of order) renderResult(k, { status: "error", error: e.detail || r.status }); }
    else {
      const reader = r.body.getReader(), dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, i).trim(); buf = buf.slice(i + 1);
          if (line) onPipeEvent(JSON.parse(line));
        }
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") for (const k of order) renderResult(k, { status: "error", error: String(e) });
  }
  for (const k of order) stopPipeTimer(k);
  askCtl = null;
}

async function ask() {
  const q = $("q").value.trim();
  if (!q) return;
  if (!graph) { setStatus(imageBlob ? "Click “Extract graph” first, then ask." : "Choose an image and extract its graph first."); return; }
  lastAsk = { q, ch: choicesList() };
  clearFocus();
  $("exAnswer").innerHTML = `<p class="hint">Working on it… (≈20–60 s on a laptop)</p>`; $("exRefs").innerHTML = "";
  $("askBtn").disabled = true; $("compareBtn").disabled = true;
  $("answerPanel").scrollIntoView({ behavior: "smooth", block: "start" });
  await runPipelines(["EX"], q, lastAsk.ch);
  $("askBtn").disabled = false; $("compareBtn").disabled = false;
}
$("askForm").onsubmit = (e) => { e.preventDefault(); ask(); };
$("compareBtn").onclick = async () => {
  if (!lastAsk || !graph) { setStatus("Ask a question first."); return; }
  $("compareBox").open = true; $("compareBtn").disabled = true; $("askBtn").disabled = true;
  await runPipelines(["E3", "E2", "E1"], lastAsk.q, lastAsk.ch);
  $("compareBtn").disabled = false; $("askBtn").disabled = false;
};

// ------------------------------------------------------------------ explanation + references
const PROV = { graph: "from the scene graph", located: "located by the model (approximate)" };

function renderExplanation(r) {
  const box = $("exAnswer");
  if (!r || r.status === "error") { box.innerHTML = `<div class="pans err"><div class="pst">Error</div><p>${esc(r?.error || "failed")}</p></div>`; return; }
  refs = r.references || [];
  const chip = (n) => {
    const ref = refs.find((x) => x.num === +n);
    if (!ref) return "";
    return `<button type="button" class="ref ${ref.provenance} ${ref.kind}" data-ref="${n}" title="${esc(ref.label)}">${n}</button>`;
  };
  const body = esc(r.text).replace(/\*\*([^*]+)\*\*/g, "<b>$1</b>").replace(/\[\[(\d+)\]\]/g, (m, n) => chip(n))
    .split(/\n{2,}/).map((para) => `<p>${para.replace(/\n/g, "<br>")}</p>`).join("");
  const q = r.query || {};
  let h = "";
  if (q.status === "answered") h += `<div class="facts"><b>Computed from the graph (NetworkX):</b> ${esc(q.answer)} <span class="mono">${opText(q.query)}</span></div>`;
  h += `<div class="extext">${body}</div>`;
  const c = r.checks || {};
  h += `<p class="hint">The explanation is written by a 3B model and may include statements the diagram does not show; the <b>references</b> are linked to the scene graph by exact name matching (computed, not generated), and the green box above, when present, is computed by NetworkX. ${refs.length} reference(s). ${r.total_s ? r.total_s.toFixed(1) + " s." : ""}</p>`;
  box.innerHTML = h;
  $("exRefs").innerHTML = refs.length ? `<h4>References</h4><ol class="reflist">${refs.map((x) => `<li><button type="button" class="ref ${x.provenance} ${x.kind}" data-ref="${x.num}">${x.num}</button> ${esc(x.label)} <span class="hint">${x.kind === "part" ? esc(PROV[x.provenance] || "") + (x.region_source === "label" ? " · label position only" : "") : x.kind === "flow" ? "flow of " + x.edges.length + " arrows" + (x.connected ? "" : " (not connected)") : "arrow"}</span></li>`).join("")}</ol>` : "";
  document.querySelectorAll(".ref").forEach((b) => {
    b.onclick = () => showRef(+b.dataset.ref);
    b.onmouseenter = () => { if (!flowTimer) showRef(+b.dataset.ref, true); };
  });
  evidence = new Set(); answerNodes = new Set(refs.filter((x) => x.node).map((x) => x.node));
  renderGraph();
}

function clearFocus() { clearInterval(flowTimer); flowTimer = null; focus = null; document.querySelectorAll(".ref.on").forEach((b) => b.classList.remove("on")); drawOverlay(); }

function showRef(n, quiet = false) {
  const ref = refs.find((x) => x.num === n);
  if (!ref) return;
  clearInterval(flowTimer); flowTimer = null;
  document.querySelectorAll(".ref").forEach((b) => b.classList.toggle("on", +b.dataset.ref === n));
  if (ref.kind === "part") {
    focus = { boxes: ref.bbox ? [ref.bbox] : [], nodes: ref.node ? [ref.node] : [], arrows: [], located: ref.provenance === "located", label: ref.label };
    drawOverlay();
  } else {
    // animate the flow arrow by arrow
    const arrows = ref.arrows;
    focus = { boxes: [], nodes: [arrows[0].source], arrows: [], label: ref.label };
    let i = 0;
    const step = () => {
      if (i >= arrows.length) { clearInterval(flowTimer); flowTimer = null; return; }
      focus.arrows.push(arrows[i]); focus.nodes.push(arrows[i].target); i++; drawOverlay();
    };
    drawOverlay(); step();
    flowTimer = setInterval(step, 750);
  }
  if (!quiet) $("stage").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

buildPipes();
buildMain();

// ------------------------------------------------------------------ evaluation tab
let runsLoaded = false;
async function loadRuns() {
  if (runsLoaded) return; runsLoaded = true;
  const r = await (await fetch("/api/eval")).json();
  $("runs").innerHTML = r.runs.map((x) => `<option value="${esc(x.run)}">${esc(x.dataset === "ai2d" ? "AI2D (real data): " : "Synthetic: ")}${esc(x.run)}${x.created ? " · " + esc(x.created) : ""}</option>`).join("");
  $("runs").onchange = () => showRun($("runs").value);
  if (r.runs.length) showRun(r.runs[0].run); else $("evalBody").innerHTML = "<p>No evaluation results found in results/.</p>";
}

function bar(x, ci) {
  const w = Math.round(100 * (x || 0));
  const ciTxt = ci ? ` <span class="hint">[${pct(ci[0])}, ${pct(ci[1])}]</span>` : "";
  const ciBar = ci ? `<span class="ci" style="left:${100 * ci[0]}%;width:${100 * (ci[1] - ci[0])}%"></span>` : "";
  return `<div class="barwrap"><div class="bar" style="width:${w}%"></div>${ciBar}</div><span>${pct(x)}${ciTxt}</span>`;
}

const COND_DESC = {
  "E1": "PaddleOCR text only (no image, no graph)",
  "E2": "image only (no graph)",
  "E3-vlm": "image + fine-tuned graph; the model answers and cites arrows (no graph query)",
  "E3-zs": "graph query over the zero-shot graph (model fallback when the query cannot apply)",
  "E3-ft": "graph query over the fine-tuned (LoRA) graph (model fallback when the query cannot apply)",
  "E3*": "graph query over the gold graph (upper bound: perfect extraction)",
  "E3-zs+fb": "E3-zs, but falls back to the model instead of abstaining",
  "E3-ft+fb": "E3-ft, but falls back to the model instead of abstaining",
  "E3*+fb": "E3*, but falls back to the model instead of abstaining",
  "E3-zs-vlm": "image + zero-shot graph; model answers and cites arrows",
  "E3*-vlm": "image + gold graph; model answers and cites arrows",
};

function accTable(s, conds) {
  let h = `<table class="t"><tr><th>Condition</th><th>What answers the question</th><th>Correct</th><th>Accuracy (95% bootstrap CI)</th></tr>`;
  for (const c of conds) { const a = s.accuracy[c]; if (!a) continue; h += `<tr><td><b>${esc(c)}</b></td><td>${esc(COND_DESC[c] || "")}</td><td>${a.correct}/${a.n}</td><td class="barcell">${bar(a.accuracy, a.ci95)}</td></tr>`; }
  return h + `</table>`;
}

function pairedTable(tests, firstLabel) {
  let h = `<table class="t"><tr><th>Comparison</th><th>Only ${esc(firstLabel || "first")} correct</th><th>Only second correct</th><th>p-value</th></tr>`;
  for (const [c, t] of Object.entries(tests || {})) h += `<tr><td>${esc(c)}</td><td>${t.only_first_correct}</td><td>${t.only_second_correct}</td><td>${t.p_value}${t.p_value < 0.05 ? " <b>(p&lt;0.05)</b>" : ""}</td></tr>`;
  return h + `</table>`;
}

function queryTable(gq) {
  if (!gq || !Object.keys(gq).length) return "";
  let h = `<h3>Graph query coverage (E3)</h3><p class="hint">How often the deterministic query answered, how it parsed the question, and accuracy on exactly those questions compared with E2.</p><table class="t"><tr><th>Graph</th><th>Answered by query</th><th>Query status</th><th>Parser</th><th>E3 accuracy there</th><th>E2 on same questions</th></tr>`;
  for (const [c, g] of Object.entries(gq)) h += `<tr><td>${esc(c)}</td><td>${g.answered_by_query}</td><td class="small">${Object.entries(g.query_status_counts).map(([k, v]) => `${esc(k)}: ${v}`).join("<br>")}</td><td class="small">${Object.entries(g.parser_counts).map(([k, v]) => `${esc(k)}: ${v}`).join("<br>")}</td><td>${pct(g.accuracy_when_query_answered)}</td><td>${pct(g.E2_accuracy_on_same_questions)}${g.paired_test_vs_E2_on_same_questions ? `<br><span class="hint">p=${g.paired_test_vs_E2_on_same_questions.p_value}</span>` : ""}</td></tr>`;
  return h + `</table>`;
}

async function showRun(run) {
  const d = await (await fetch(`/api/eval/${encodeURIComponent(run)}`)).json();
  if (d.dataset === "ai2d") renderAI2D(d); else renderPilot(d);
}

function lossChart(log) {
  const tr = log.filter((x) => x.loss != null), va = log.filter((x) => x.val_loss != null);
  if (!tr.length) return "";
  const W = 520, H = 160, P = 30, maxS = Math.max(...tr.map((x) => x.step)), all = tr.map((x) => x.loss).concat(va.map((x) => x.val_loss));
  const maxL = Math.max(...all), minL = Math.min(...all);
  const X = (s) => P + (W - 2 * P) * (s / maxS), Y = (l) => H - P + -(H - 2 * P) * ((l - minL) / (maxL - minL || 1));
  const sm = tr.map((x, i) => { const w = tr.slice(Math.max(0, i - 9), i + 1); return [x.step, w.reduce((a, b) => a + b.loss, 0) / w.length]; });
  return `<svg class="chart" viewBox="0 0 ${W} ${H}">
    <line x1="${P}" y1="${H - P}" x2="${W - P}" y2="${H - P}" class="axis"/><line x1="${P}" y1="${P}" x2="${P}" y2="${H - P}" class="axis"/>
    <polyline fill="none" class="s1" points="${sm.map(([s, l]) => `${X(s)},${Y(l)}`).join(" ")}"/>
    ${va.map((x) => `<circle cx="${X(x.step)}" cy="${Y(x.val_loss)}" r="3.5" class="s2"/>`).join("")}
    <text x="${P}" y="${P - 8}" class="lbl">loss ${maxL.toFixed(2)}</text><text x="${P}" y="${H - 8}" class="lbl">${minL.toFixed(2)}</text>
    <text x="${W - P}" y="${H - 8}" text-anchor="end" class="lbl">step ${maxS}</text>
    <text x="${W - P}" y="${P - 8}" text-anchor="end" class="lbl"><tspan class="s1t">— train (10-step mean)</tspan>  <tspan class="s2t">● validation</tspan></text></svg>`;
}

function renderAI2D(d) {
  const s = d.summary, conds = s.conditions;
  let h = `<p>${esc(d.title)}. <b>${s.n_questions}</b> human-written multiple-choice questions on <b>${s.n_diagrams}</b> diagrams from the official AI2D <b>test</b> split (never used for training); ${s.n_with_gold_edges} questions are on diagrams whose annotations contain arrows. Chance accuracy: ${pct(s.chance_accuracy)}. Model: ${esc(d.model)}${d.adapter ? " + LoRA adapter <code>" + esc(d.adapter) + "</code> (extraction only)" : ""}. Run date ${esc(d.created)}, seed ${d.seed}.</p>`;
  h += `<h3>Question-answering accuracy</h3>` + accTable(s, conds);
  const variants = Object.keys(s.accuracy).filter((c) => !conds.includes(c));
  if (variants.length) h += `<details><summary class="hint">Variants (abstention policy, model-only answering)</summary>${accTable(s, variants)}</details>`;
  h += `<h3>Paired comparison against E2, all questions (exact McNemar test)</h3>` + pairedTable(s.paired_tests_vs_E2, "E2");
  if (s.paired_tests_vs_E2_relation_questions) h += `<h3>Paired comparison against E2, relationship questions only</h3><p class="hint">The proposal's research question. A question counts as a relationship question if its diagram has annotated arrows and the question or its answer names an entity at an end of one (a deterministic proxy).</p>` + pairedTable(s.paired_tests_vs_E2_relation_questions, "E2");
  h += queryTable(s.graph_query);
  h += `<h3>By subset</h3><table class="t"><tr><th>Subset</th><th>n</th>${conds.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>`;
  for (const [k, v] of Object.entries(s.subsets)) { const n = v[conds[0]].n; h += `<tr><td>${esc(k.replaceAll("_", " "))}</td><td>${n}</td>${conds.map((c) => `<td>${pct(v[c].accuracy)}<br><span class="hint">[${pct(v[c].ci95[0])}, ${pct(v[c].ci95[1])}]</span></td>`).join("")}</tr>`; }
  h += `</table>`;
  h += `<h3>Graph extraction vs AI2D gold annotations</h3><table class="t"><tr><th>Extractor</th><th>Diagrams</th><th>Entity P / R / F1</th><th>Arrow P / R / F1</th><th>Reversed arrows</th><th>Mean latency</th></tr>`;
  for (const [k, v] of Object.entries(s.extraction)) h += `<tr><td>${k === "zs" ? "Zero-shot Qwen2.5-VL-3B" : "Fine-tuned (LoRA on AI2D train)"}</td><td>${v.n_diagrams}</td><td>${v.entities.precision} / ${v.entities.recall} / <b>${v.entities.f1}</b></td><td>${v.edges.precision} / ${v.edges.recall} / <b>${v.edges.f1}</b></td><td>${v.reversed_edges}</td><td>${v.mean_latency_s} s</td></tr>`;
  h += `</table>`;
  h += `<h3>Evidence and grounding (E3 conditions)</h3><table class="t"><tr><th>Condition</th><th>Grounded</th><th>Not supported</th><th>Abstained</th><th>Accuracy when grounded</th><th>Accuracy otherwise</th><th>Answers with invented arrow ids</th></tr>`;
  for (const [c, g] of Object.entries(s.grounding)) { const sc = g.status_counts; h += `<tr><td>${esc(c)}</td><td>${(sc.grounded || 0) + (sc.graph_answer || 0)}</td><td>${(sc.unsupported || 0) + (sc.image_only || 0)}</td><td>${sc.abstained || 0}</td><td>${pct(g.accuracy_when_grounded)}</td><td>${pct(g.accuracy_when_not_grounded)}</td><td>${g.answers_with_invalid_citations}</td></tr>`; }
  h += `</table>`;
  if (d.training_log && d.training_log.length) {
    const cfg = d.training_config || {};
    h += `<h3>Fine-tuning (LoRA)</h3><p class="hint">${cfg.n_train ?? "?"} AI2D train diagrams available, ${cfg.steps ?? "?"} optimizer steps × ${cfg.grad_accum ?? "?"} examples, LoRA rank ${cfg.rank ?? "?"} on ${cfg.targets === "attn" ? "attention projections" : "attention + MLP"} of the language model, lr ${cfg.lr ?? "?"}, images ≤ ${cfg.train_image_side ?? "?"} px.</p>${lossChart(d.training_log)}`;
  }
  h += `<h3>By AI2D category</h3><table class="t"><tr><th>Category</th><th>n</th>${conds.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>`;
  for (const [k, v] of Object.entries(s.by_category)) h += `<tr><td>${esc(k)}</td><td>${v.n}</td>${conds.map((c) => `<td>${pct(v[c])}</td>`).join("")}</tr>`;
  h += `</table>`;
  h += `<h3>Browse questions</h3><div class="row small"><select id="flt"><option value="all">all questions</option><option value="e3wins">E3-ft correct, E2 wrong</option><option value="e2wins">E2 correct, E3-ft wrong</option><option value="allwrong">all conditions wrong</option></select><span class="hint" id="fltn"></span></div><div class="browse"><div id="qlist"></div><div id="qdetail"><p class="hint">Click a question. Question text and images are read from your local AI2D copy (the dataset licence does not allow bundling them).</p></div></div>`;
  $("evalBody").innerHTML = h;
  const list = () => {
    const f = $("flt").value;
    const rs = d.records.filter((r) => f === "all" || (f === "e3wins" && r["E3-ft"]?.correct && !r.E2.correct) || (f === "e2wins" && r.E2.correct && !r["E3-ft"]?.correct) || (f === "allwrong" && conds.every((c) => !r[c].correct)));
    $("fltn").textContent = `${rs.length} questions`;
    $("qlist").innerHTML = rs.map((r) => `<div class="qrow" style="grid-template-columns:1fr 1.2fr repeat(${conds.length}, 22px)" data-q="${esc(r.qid)}"><span>${esc(r.qid)}</span><span class="hint">${esc(r.category)}</span>${conds.map((c) => `<span class="${r[c].correct ? "ok" : "bad"}" title="${esc(c)}">${r[c].correct ? "✓" : "✗"}</span>`).join("")}</div>`).join("");
    document.querySelectorAll(".qrow").forEach((el) => el.onclick = () => showItem(d, conds, el.dataset.q));
  };
  $("flt").onchange = list; list();
}

async function showItem(d, conds, qid) {
  const rec = d.records.find((r) => r.qid === qid);
  let h = `<b>${esc(qid)}</b> · ${esc(rec.category)} · ${rec.gold_edges} gold arrows`;
  const r = await fetch(`/api/eval/${encodeURIComponent(d.run || $("runs").value)}/item/${encodeURIComponent(qid)}`);
  let item = null;
  if (r.ok) item = await r.json(); else h += `<p class="hint">${esc((await r.json()).detail)}</p>`;
  if (item) {
    h += `<p class="qtext">${esc(item.question)}</p><ol type="a">${item.choices.map((c, i) => `<li class="${i === item.answer_index ? "gold" : ""}">${esc(c)}${i === item.answer_index ? " ✓" : ""}</li>`).join("")}</ol>`;
    h += `<img class="thumb" src="/api/ai2d/image/${encodeURIComponent(item.image_id)}" alt="">`;
  }
  h += `<table class="t"><tr><th>Condition</th><th>Chose</th><th></th><th>Status</th><th>Cited arrows</th></tr>`;
  for (const c of conds) { const x = rec[c]; const ch = x.choice == null ? "(unparsed)" : (item ? item.choices[x.choice] : "abcdefgh"[x.choice]); h += `<tr><td>${esc(c)}</td><td>${esc(ch)}</td><td class="${x.correct ? "ok" : "bad"}">${x.correct ? "✓" : "✗"}</td><td>${esc(x.status || "–")}</td><td>${x.n_evidence ?? "–"}${x.invalid_citations ? ` (+${x.invalid_citations} invalid)` : ""}</td></tr>`; }
  $("qdetail").innerHTML = h + "</table>";
}

function renderPilot(d) {
  const s = d.summary, conds = s.conditions;
  let h = `<p>${esc(d.title)}. <b>${s.n_questions}</b> questions on <b>${s.n_diagrams}</b> diagrams generated by this project (exact gold answers), including knowledge-free graphs with nonsense labels and counterfactual diagrams whose arrows contradict textbook knowledge. Questions use varied natural wording. Model: ${esc(d.model)}; OCR: ${esc(d.ocr || "")}. Run date ${esc(d.created)}.</p>`;
  h += `<p class="hint">Caveat: the keyword rules of the graph query were debugged on this pilot's questions, so E3 results here are optimistic; the AI2D run (human-written questions) is the unbiased test.</p>`;
  h += `<h3>Accuracy</h3>` + accTable(s, conds);
  const variants = Object.keys(s.accuracy).filter((c) => !conds.includes(c));
  if (variants.length) h += `<details><summary class="hint">Variants (fall back to the model instead of abstaining)</summary>${accTable(s, variants)}</details>`;
  h += `<h3>Paired comparisons (exact McNemar test)</h3>` + pairedTable(s.paired_tests);
  h += `<h3>By subset</h3><table class="t"><tr><th>Subset</th><th>n</th>${conds.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>`;
  for (const [k, v] of Object.entries(s.subsets)) { const n = v[conds[0]]?.n; h += `<tr><td>${esc(k.replaceAll("_", " "))}</td><td>${n}</td>${conds.map((c) => `<td>${pct(v[c].accuracy)}</td>`).join("")}</tr>`; }
  h += `</table><h3>By question type</h3><table class="t"><tr><th>Type</th><th>n</th>${conds.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>`;
  for (const [k, v] of Object.entries(s.by_kind)) h += `<tr><td>${esc(k.replaceAll("_", " "))}</td><td>${v.n}</td>${conds.map((c) => `<td>${pct(v[c])}</td>`).join("")}</tr>`;
  h += `</table>` + queryTable(s.graph_query);
  if (s.extraction) {
    h += `<h3>Graph extraction vs gold</h3><table class="t"><tr><th>Extractor</th><th>Diagrams</th><th>Entity F1</th><th>Arrow P / R / F1</th><th>Reversed arrows</th><th>Mean latency</th></tr>`;
    for (const [k, v] of Object.entries(s.extraction)) h += `<tr><td>${k === "zs" ? "Zero-shot" : "Fine-tuned (LoRA)"}</td><td>${v.n_diagrams}</td><td>${v.entities.f1}</td><td>${v.edges.precision} / ${v.edges.recall} / <b>${v.edges.f1}</b></td><td>${v.reversed_edges}</td><td>${v.mean_latency_s} s</td></tr>`;
    h += `</table>`;
  }
  h += `<h3>Browse questions</h3><div class="browse"><div id="qlist"></div><div id="qdetail"><p class="hint">Click a question to see every pipeline's answer.</p></div></div>`;
  $("evalBody").innerHTML = h;
  $("qlist").innerHTML = d.records.map((r, i) => `<div class="qrow" style="grid-template-columns:1fr repeat(${conds.length}, 22px)" data-i="${i}"><span>${esc(r.question)}</span>${conds.map((c) => `<span class="${r[c].correct ? "ok" : "bad"}" title="${esc(c)}">${r[c].correct ? "✓" : "✗"}</span>`).join("")}</div>`).join("");
  document.querySelectorAll(".qrow").forEach((el) => el.onclick = () => {
    const r = d.records[+el.dataset.i];
    let x = `<b>${esc(r.title)}</b> · ${esc(r.kind)}<p class="qtext">${esc(r.question)}</p><p class="hint">Gold: ${esc(JSON.stringify(r.gold.value ?? r.gold.type))}</p><img class="thumb" src="/api/samples/${encodeURIComponent(r.file)}" alt=""><table class="t"><tr><th>Condition</th><th>Answer</th><th></th><th>How</th></tr>`;
    for (const c of conds) { const a = r[c]; x += `<tr><td>${esc(c)}</td><td>${esc(a.answer)}</td><td class="${a.correct ? "ok" : "bad"}">${a.correct ? "✓" : "✗"}</td><td class="small">${esc([a.status, a.query ? opText(a.query) : "", a.parser || ""].filter(Boolean).join(" · "))}</td></tr>`; }
    $("qdetail").innerHTML = x + "</table>";
  });
}

init();
