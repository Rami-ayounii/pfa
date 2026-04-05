"""
web_interface/dashboard.py  v3
══════════════════════════════
GEO Pipeline — Real-time Streamlit Dashboard

CSS approach: all custom HTML rendered via components.html() with embedded
<style> blocks — guarantees styles apply regardless of Streamlit sanitisation.

Run:
    C:\\Users\\ayoun\\anaconda3\\python.exe -m streamlit run web_interface/dashboard.py
"""

import os
import sys
import json
import time
import queue
import threading
import subprocess
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "geo_output"
PYTHON     = os.environ.get("GEO_PYTHON", r"C:\Users\ayoun\anaconda3\envs\pfa\python.exe")
sys.path.insert(0, str(ROOT))

# ══════════════════════════════════════════════════════════════════════════════
# 1 — PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GEO Pipeline",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# 2 — MINIMAL GLOBAL CSS  (only for Streamlit's own chrome — reliably applied)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
#MainMenu            { visibility: hidden; }
footer               { visibility: hidden; }
.stDeployButton      { display: none !important; }
.main .block-container { padding-top: 1.4rem; padding-bottom: 2.5rem; max-width: 1480px; }
</style>
""", unsafe_allow_html=True)

# ── Reusable inline-style helper for section labels ───────────────────────────
_LABEL_STYLE = (
    'style="font-size:11px;font-weight:800;letter-spacing:.12em;'
    'text-transform:uppercase;color:#64748b;margin:0 0 10px 0;'
    'font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;"'
)


def section_label(text: str) -> None:
    st.markdown(f'<p {_LABEL_STYLE}>{text}</p>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3 — NODES + LOG PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
NODES = [
    {"id": "parse",        "label": "Parse\nQuery",      "icon": "🔍"},
    {"id": "agent0",       "label": "Agent 0\nPrompts",  "icon": "✍️"},
    {"id": "a1_load",      "label": "A1\nLoad",          "icon": "📂"},
    {"id": "a1_llm",       "label": "LLM\nFan-out",      "icon": "🤖"},
    {"id": "a1_aggregate", "label": "Aggregate",         "icon": "🔗"},
    {"id": "a1_extract",   "label": "Extract\nEntities", "icon": "🏷️"},
    {"id": "a1_enrich",    "label": "Enrich",            "icon": "➕"},
    {"id": "a1_clean",     "label": "Clean\n& Dedup",    "icon": "🧹"},
    {"id": "a1_metrics",   "label": "GEO\nMetrics",      "icon": "📊"},
    {"id": "a2_scrape",    "label": "Agent 2\nScrape",   "icon": "🌐"},
    {"id": "export",       "label": "Export",            "icon": "💾"},
]
NODE_IDS = [n["id"] for n in NODES]

NODE_TRIGGERS: list[tuple[str, list[str]]] = [
    ("parse",        ["Parsing query", "parse_query_node", "→ domain=", "query →", "[Parser]"]),
    ("agent0",       ["Agent 0", "GEO Prompt Generator", "agent0_node",
                      "Loop 1/", "intents discovered", "[Step 1]", "[Step 2]", "Self-Reflection"]),
    ("a1_load",      ["agent1_load_node", "Loading prompts", "prompt_df →"]),
    ("a1_llm",       ["Fan-out", "LLM tasks", "agent1_llm_query_node",
                      "async_query", "query_single", "[Router]"]),
    ("a1_aggregate", ["agent1_aggregate_node", "Aggregating", "raw_responses"]),
    ("a1_extract",   ["agent1_extract_node", "[Step 3]", "Entity extraction"]),
    ("a1_enrich",    ["agent1_enrich_node",   "[Step 4]", "enrich"]),
    ("a1_clean",     ["agent1_clean_node",    "[Step 5]", "dedup", "arbitration", "reflection"]),
    ("a1_metrics",   ["agent1_metrics_node",  "[Step 6]", "GEO metrics", "top-N", "top_n"]),
    ("a2_scrape",    ["ScrapeJob", "Agent 2", "Social Profile Scraper",
                      "► Step 1", "► Step 2", "► Step 4", "DuckDuckGo"]),
    ("export",       ["export_node", "pipeline_summary", "features.csv", "Export"]),
]

DONE_MARKERS   = ["✓", "Agent 0 complete", "Agent 2 complete", "Export complete",
                  "Pipeline complete", "complete —", "passed", "✅"]
ERROR_MARKERS  = ["✗", "Error:", "error:", "Exception", "Traceback", "FAILED",
                  "CRITICAL", "Unknown", "cannot find"]
SKIP_MARKERS   = ["[SKIP]", "skipped —", "skipping"]
HEADER_MARKERS = ["═══", "───", "====", "────", "[Pipeline]", "[Router]"]


# ══════════════════════════════════════════════════════════════════════════════
# 4 — SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_DEFAULTS: dict = {
    "is_running":   False,
    "output_lines": [],
    "node_status":  {n: "pending" for n in NODE_IDS},
    "run_start":    None,
    "run_end":      None,
    "run_ok":       None,
    "results":      {},
    "_queue":       None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════════════════════════════════════════════════════════════════════════
# 5 — HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _classify_line(line: str) -> str:
    if any(m in line for m in ERROR_MARKERS):  return "error"
    if any(m in line for m in SKIP_MARKERS):   return "warning"
    if any(m in line for m in DONE_MARKERS):   return "success"
    if any(m in line for m in HEADER_MARKERS): return "header"
    if "[DEBUG]" in line:                      return "debug"
    return "info"


def _update_node_status(line: str) -> None:
    ns       = st.session_state.node_status
    is_done  = any(m in line for m in DONE_MARKERS)
    is_error = any(m in line for m in ERROR_MARKERS)
    is_skip  = any(m in line for m in SKIP_MARKERS)
    for node_id, patterns in NODE_TRIGGERS:
        if any(p in line for p in patterns):
            cur = ns.get(node_id, "pending")
            if is_error and cur != "done":
                ns[node_id] = "error"
            elif is_skip and cur == "pending":
                ns[node_id] = "skip"
            elif is_done:
                ns[node_id] = "done"
            elif cur == "pending":
                ns[node_id] = "running"
            break
    st.session_state.node_status = ns


def _load_results() -> dict:
    r: dict = {}
    if not OUTPUT_DIR.exists():
        return r
    prompt_files = sorted(OUTPUT_DIR.glob("prompt_set_*.csv"),
                          key=lambda p: p.stat().st_mtime)
    if prompt_files:
        try:   r["prompts"] = pd.read_csv(prompt_files[-1])
        except Exception: pass
    for fname, key in [
        ("features.csv",        "features"),
        ("global_metrics.csv",  "global"),
        ("social_profiles.csv", "profiles"),
    ]:
        path = OUTPUT_DIR / fname
        if path.exists():
            try:   r[key] = pd.read_csv(path)
            except Exception: pass
    sp = OUTPUT_DIR / "pipeline_summary.json"
    if sp.exists():
        try:
            with open(sp) as f:
                r["summary"] = json.load(f)
        except Exception: pass
    return r


def _start_pipeline(cmd: list[str]) -> None:
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(ROOT),
            env={**os.environ},
        )
    except FileNotFoundError as exc:
        st.error(f"Could not start process: {exc}")
        return

    q: queue.Queue = queue.Queue()

    def _reader(proc: subprocess.Popen, q: queue.Queue) -> None:
        for line in iter(proc.stdout.readline, ""):
            q.put(("line", line))
        proc.wait()
        q.put(("done", proc.returncode))

    threading.Thread(target=_reader, args=(proc, q), daemon=True).start()
    st.session_state.is_running   = True
    st.session_state.output_lines = []
    st.session_state.node_status  = {n: "pending" for n in NODE_IDS}
    st.session_state.run_start    = time.time()
    st.session_state.run_end      = None
    st.session_state.run_ok       = None
    st.session_state.results      = {}
    st.session_state._queue       = q


def _drain_queue() -> bool:
    if not st.session_state.is_running:
        return False
    q = st.session_state._queue
    if q is None:
        return False
    changed = False
    while True:
        try:
            msg_type, data = q.get_nowait()
        except queue.Empty:
            break
        if msg_type == "line":
            line = data.rstrip()
            st.session_state.output_lines.append(line)
            _update_node_status(line)
            changed = True
        elif msg_type == "done":
            rc = data
            st.session_state.is_running = False
            st.session_state.run_end    = time.time()
            st.session_state.run_ok     = (rc == 0)
            for nid, status in st.session_state.node_status.items():
                if status == "running":
                    st.session_state.node_status[nid] = "done" if rc == 0 else "error"
            st.session_state.results = _load_results()
            changed = True
            break
    return changed


# ══════════════════════════════════════════════════════════════════════════════
# 6 — RENDER: AGENT WORKFLOW   (components.html → iframe, full CSS control)
# ══════════════════════════════════════════════════════════════════════════════
_WORKFLOW_CSS = """
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: transparent;
    padding: 4px 2px 8px;
}

/* ── outer wrapper ─────────────────────────────────────────────────────── */
.workflow-outer {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px 16px;
}

/* ── 3-col grid: card | arrow | card | arrow | card ────────────────────── */
.workflow-grid {
    display: grid;
    grid-template-columns: 1fr 56px 1fr 56px 1fr;
    align-items: start;
    gap: 0;
    width: 100%;
}

/* ── agent card base ───────────────────────────────────────────────────── */
.agent-card {
    border-radius: 14px;
    padding: 18px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,.07);
}
.agent-card-a0 { background: #fdf4ff; border: 2px solid #c084fc; }
.agent-card-a1 { background: #eff6ff; border: 2px solid #60a5fa; }
.agent-card-a2 { background: #f0fdf4; border: 2px solid #4ade80; }

/* ── agent badge ───────────────────────────────────────────────────────── */
.agent-badge {
    display: inline-block;
    font-size: 9px; font-weight: 900; letter-spacing: .18em; text-transform: uppercase;
    padding: 2px 10px; border-radius: 20px; margin-bottom: 10px;
}
.badge-a0 { background: #f3e8ff; color: #7c3aed; border: 1px solid #e9d5ff; }
.badge-a1 { background: #dbeafe; color: #1d4ed8; border: 1px solid #bfdbfe; }
.badge-a2 { background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }

/* ── title / subtitle ──────────────────────────────────────────────────── */
.agent-title {
    font-size: 15px; font-weight: 700; color: #1e293b;
    margin-bottom: 3px; line-height: 1.3;
}
.agent-subtitle {
    font-size: 11.5px; color: #64748b;
    margin-bottom: 14px; line-height: 1.5;
}

/* ── step list ─────────────────────────────────────────────────────────── */
.agent-steps { list-style: none; margin-bottom: 14px; }
.agent-steps li {
    font-size: 11.5px; color: #475569;
    padding: 3px 0;
    display: flex; align-items: flex-start; gap: 7px;
    line-height: 1.4;
}
.dot { font-size: 7px; margin-top: 4px; flex-shrink: 0; }
.dot-a0 { color: #9333ea; }
.dot-a1 { color: #2563eb; }
.dot-a2 { color: #16a34a; }

/* ── tool tags ─────────────────────────────────────────────────────────── */
.agent-tools { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 14px; }
.tool {
    font-size: 10px; font-weight: 600;
    padding: 2px 9px; border-radius: 20px; white-space: nowrap;
}
.tool-a0 { background: #f3e8ff; color: #6d28d9; border: 1px solid #ddd6fe; }
.tool-a1 { background: #dbeafe; color: #1e40af; border: 1px solid #bfdbfe; }
.tool-a2 { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }

/* ── I/O footer ────────────────────────────────────────────────────────── */
.agent-io {
    border-top: 1px solid rgba(0,0,0,.08);
    padding-top: 10px;
    font-size: 11px; color: #64748b;
    line-height: 1.7;
}
.agent-io strong { color: #374151; }
.agent-io code {
    font-size: 10.5px; background: rgba(0,0,0,.06);
    padding: 1px 5px; border-radius: 4px;
    font-family: "Courier New", monospace;
}

/* ── arrow connector ───────────────────────────────────────────────────── */
.connector {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 0 4px; padding-top: 100px;   /* vertically centred in card */
    gap: 4px;
}
.conn-arrow { font-size: 24px; color: #94a3b8; line-height: 1; }
.conn-label { font-size: 9px; color: #94a3b8; text-align: center; line-height: 1.4; }
</style>
"""

_WORKFLOW_HTML = """
<div class="workflow-outer">
<div class="workflow-grid">

  <!-- ── AGENT 0 ────────────────────────────────────────────────────── -->
  <div class="agent-card agent-card-a0">
    <div class="agent-badge badge-a0">AGENT 0</div>
    <div class="agent-title">✍️ Prompt Generator</div>
    <div class="agent-subtitle">
      Builds geographically-targeted LLM prompt sets via intent discovery
      and a self-reflection quality loop.
    </div>
    <ul class="agent-steps">
      <li><span class="dot dot-a0">●</span>Discovers intent categories
          <em>(discovery, comparison, recommendation…)</em></li>
      <li><span class="dot dot-a0">●</span>Generates N prompt variants per intent
          in each selected language</li>
      <li><span class="dot dot-a0">●</span>Self-reflection loop — drops off-topic
          or geographically misaligned prompts</li>
      <li><span class="dot dot-a0">●</span>All prompts anchored to Tunisia
          and its cities by default</li>
    </ul>
    <div class="agent-tools">
      <span class="tool tool-a0">Groq Compound</span>
      <span class="tool tool-a0">LLM reflection</span>
      <span class="tool tool-a0">fr / ar / en</span>
    </div>
    <div class="agent-io">
      <div><strong>IN&nbsp; :</strong> domain + location &mdash;
           e.g. <em>"Restaurants in Sfax"</em></div>
      <div><strong>OUT :</strong> <code>prompt_set_*.csv</code></div>
    </div>
  </div>

  <!-- ── ARROW 0 → 1 ───────────────────────────────────────────────── -->
  <div class="connector">
    <div class="conn-arrow">→</div>
    <div class="conn-label">prompts<br>CSV</div>
  </div>

  <!-- ── AGENT 1 ────────────────────────────────────────────────────── -->
  <div class="agent-card agent-card-a1">
    <div class="agent-badge badge-a1">AGENT 1</div>
    <div class="agent-title">📊 GEO Analyser</div>
    <div class="agent-subtitle">
      Fires prompts across multiple LLMs in parallel, extracts brand mentions,
      and scores GEO visibility features across 7 sequential steps.
    </div>
    <ul class="agent-steps">
      <li><span class="dot dot-a1">●</span>Loads prompt CSV — stratified sampling
          by intent (max N per intent)</li>
      <li><span class="dot dot-a1">●</span>Async fan-out — queries Fast + Strong
          + Qwen LLMs simultaneously</li>
      <li><span class="dot dot-a1">●</span>Aggregates all LLM responses
          into a unified table</li>
      <li><span class="dot dot-a1">●</span>Extracts brand/entity mentions
          with position rank and authority signals</li>
      <li><span class="dot dot-a1">●</span>Enriches mentions with citation context</li>
      <li><span class="dot dot-a1">●</span>Deduplication + LLM arbitration
          for conflicting entity mentions</li>
      <li><span class="dot dot-a1">●</span>Computes per-brand GEO metrics
          <em>(citation rate, stability, prompt coverage…)</em></li>
    </ul>
    <div class="agent-tools">
      <span class="tool tool-a1">async fan-out</span>
      <span class="tool tool-a1">Groq Fast</span>
      <span class="tool tool-a1">Groq Strong</span>
      <span class="tool tool-a1">Qwen3-32B</span>
    </div>
    <div class="agent-io">
      <div><strong>IN&nbsp; :</strong> <code>prompt_set_*.csv</code></div>
      <div><strong>OUT :</strong> <code>features.csv</code> &nbsp;·&nbsp;
           <code>global_metrics.csv</code></div>
    </div>
  </div>

  <!-- ── ARROW 1 → 2 ───────────────────────────────────────────────── -->
  <div class="connector">
    <div class="conn-arrow">→</div>
    <div class="conn-label">top-N<br>brands</div>
  </div>

  <!-- ── AGENT 2 ────────────────────────────────────────────────────── -->
  <div class="agent-card agent-card-a2">
    <div class="agent-badge badge-a2">AGENT 2</div>
    <div class="agent-title">🌐 Social Scraper</div>
    <div class="agent-subtitle">
      8-step parallel scraping pipeline per brand — web search, review sites,
      social media, hallucination detection, and authority scoring.
    </div>
    <ul class="agent-steps">
      <li><span class="dot dot-a2">●</span>DuckDuckGo web search
          → basic brand profile</li>
      <li><span class="dot dot-a2">●</span>Wikipedia presence
          and description check</li>
      <li><span class="dot dot-a2">●</span>TripAdvisor ratings
          and review count via Apify</li>
      <li><span class="dot dot-a2">●</span>Google Maps data
          via SerpApi + Selenium WebDriver</li>
      <li><span class="dot dot-a2">●</span>Instagram + Facebook
          follower and engagement metrics</li>
      <li><span class="dot dot-a2">●</span>Hallucination checker
          — flags invented or misidentified brands</li>
      <li><span class="dot dot-a2">●</span>LLM reflection computes
          social authority score <em>(0 – 100)</em></li>
    </ul>
    <div class="agent-tools">
      <span class="tool tool-a2">DuckDuckGo</span>
      <span class="tool tool-a2">SerpApi</span>
      <span class="tool tool-a2">Selenium</span>
      <span class="tool tool-a2">Apify</span>
      <span class="tool tool-a2">OpenRouter</span>
    </div>
    <div class="agent-io">
      <div><strong>IN&nbsp; :</strong> brand list <em>(top-N from Agent 1)</em></div>
      <div><strong>OUT :</strong> <code>social_profiles.csv</code></div>
    </div>
  </div>

</div>
</div>
"""

_PIPELINE_CSS = """
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: transparent;
    padding: 4px 2px;
}

/* ── wrapper ──────────────────────────────────────────────────────────── */
.pipeline-wrap {
    overflow-x: auto;
    white-space: nowrap;
    padding: 14px 16px 12px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}

/* ── single node card ─────────────────────────────────────────────────── */
.geo-node {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 78px;
    min-height: 68px;
    border-radius: 10px;
    border: 2px solid #e2e8f0;
    background: #ffffff;
    padding: 6px 4px;
    margin: 0 1px;
    font-size: 9.5px;
    font-weight: 500;
    text-align: center;
    vertical-align: middle;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
    transition: border-color .3s, background .3s, opacity .3s, box-shadow .3s;
}
.geo-node-icon  { font-size: 15px; margin-bottom: 2px; }
.geo-node-label { color: #94a3b8; line-height: 1.2; font-size: 9px; }
.geo-node-badge { font-size: 12px; margin-top: 3px; }

/* pending */
.geo-node.pending { opacity: .5; }

/* running */
.geo-node.running {
    border-color: #3b82f6;
    background: #eff6ff;
    opacity: 1;
    animation: pulse 1.4s ease-in-out infinite;
}
.geo-node.running .geo-node-label { color: #2563eb; font-weight: 700; }

/* done */
.geo-node.done { border-color: #22c55e; background: #f0fdf4; opacity: 1; }
.geo-node.done .geo-node-label  { color: #15803d; font-weight: 700; }

/* error */
.geo-node.error { border-color: #ef4444; background: #fef2f2; opacity: 1; }
.geo-node.error .geo-node-label { color: #dc2626; font-weight: 700; }

/* skip */
.geo-node.skip  { border-color: #f59e0b; background: #fffbeb; opacity: 1; }
.geo-node.skip  .geo-node-label { color: #b45309; font-weight: 700; }

@keyframes pulse {
    0%,  100% { box-shadow: 0 0 0  0   rgba(59,130,246,.5); }
    50%        { box-shadow: 0 0 0 10px rgba(59,130,246,0);  }
}

/* arrow between nodes */
.geo-arrow {
    display: inline-block;
    color: #cbd5e1;
    font-size: 13px;
    vertical-align: middle;
}

/* legend */
.pipeline-legend {
    margin-top: 10px;
    font-size: 11px;
    color: #94a3b8;
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    white-space: normal;
}
.legend-item { display: flex; align-items: center; gap: 4px; }
.leg-run  { color: #3b82f6; }
.leg-done { color: #22c55e; }
.leg-err  { color: #ef4444; }
.leg-skip { color: #f59e0b; }
</style>
"""


def render_agent_workflow() -> None:
    components.html(_WORKFLOW_CSS + _WORKFLOW_HTML, height=530, scrolling=False)


def render_pipeline_flow() -> None:
    ns = st.session_state.node_status

    _BADGE = {"pending": "○", "running": "⟳", "done": "✓", "error": "✗", "skip": "⤳"}

    parts: list[str] = []
    for i, node in enumerate(NODES):
        nid    = node["id"]
        status = ns.get(nid, "pending")
        label  = node["label"].replace("\n", "<br>")
        badge  = _BADGE.get(status, "○")
        parts.append(
            f'<div class="geo-node {status}">'
            f'<div class="geo-node-icon">{node["icon"]}</div>'
            f'<div class="geo-node-label">{label}</div>'
            f'<div class="geo-node-badge">{badge}</div>'
            f'</div>'
        )
        if i < len(NODES) - 1:
            parts.append('<span class="geo-arrow">&rsaquo;</span>')

    counts = {s: sum(1 for v in ns.values() if v == s)
              for s in ("pending", "running", "done", "error", "skip")}
    legend = (
        f'<span class="legend-item">○&nbsp;{counts["pending"]} pending</span>'
        f'<span class="legend-item leg-run">⟳&nbsp;{counts["running"]} running</span>'
        f'<span class="legend-item leg-done">✓&nbsp;{counts["done"]} done</span>'
        f'<span class="legend-item leg-err">✗&nbsp;{counts["error"]} error</span>'
        f'<span class="legend-item leg-skip">⤳&nbsp;{counts["skip"]} skip</span>'
    )
    html = (
        _PIPELINE_CSS
        + f'<div class="pipeline-wrap">{"".join(parts)}'
        + f'<div class="pipeline-legend">{legend}</div></div>'
    )
    components.html(html, height=145, scrolling=False)


# ══════════════════════════════════════════════════════════════════════════════
# 7 — RENDER: LOG CONSOLE
# ══════════════════════════════════════════════════════════════════════════════
_LOG_COLORS = {
    "error":   "#f87171",
    "warning": "#fbbf24",
    "success": "#4ade80",
    "header":  "#60a5fa",
    "debug":   "#475569",
    "info":    "#94a3b8",
}


def render_log_console(filter_level: str = "all", height: int = 400) -> None:
    lines = st.session_state.output_lines

    if not lines:
        components.html(
            f'<div style="height:{height}px;background:#0f172a;border-radius:10px;'
            f'display:flex;align-items:center;justify-content:center;'
            f'color:#334155;font-family:monospace;font-size:13px;'
            f'border:1px solid #1e293b;">Waiting for pipeline output…</div>',
            height=height + 4,
            scrolling=False,
        )
        return

    level_filter = {
        "all":          None,
        "errors only":  "error",
        "warnings":     "warning",
        "success only": "success",
        "debug":        "debug",
    }.get(filter_level)

    styled: list[str] = []
    for line in lines[-500:]:
        cls = _classify_line(line)
        if level_filter and cls != level_filter:
            continue
        color   = _LOG_COLORS.get(cls, "#94a3b8")
        escaped = (line
                   .replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;"))
        styled.append(f'<span style="color:{color}">{escaped}</span>')

    body = ("\n".join(styled)
            if styled
            else '<span style="color:#334155">No lines match this filter.</span>')

    html = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;background:#0f172a;">
<div id="geo-log" style="
    height:{height}px; overflow-y:auto;
    background:#0f172a; padding:12px 16px; border-radius:10px;
    font-family:'Courier New',Consolas,monospace; font-size:12px;
    line-height:1.7; white-space:pre-wrap; word-break:break-all;
    border:1px solid #1e293b; color:#94a3b8;">{body}
</div>
<script>
(function(){{
    var el = document.getElementById('geo-log');
    if (el) el.scrollTop = el.scrollHeight;
}})();
</script>
</body></html>"""
    components.html(html, height=height + 4, scrolling=False)

    n_err  = sum(1 for l in lines if any(m in l for m in ERROR_MARKERS))
    n_skip = sum(1 for l in lines if any(m in l for m in SKIP_MARKERS))
    st.caption(f"{len(lines)} lines · {n_err} errors · {n_skip} skips")


# ══════════════════════════════════════════════════════════════════════════════
# 8 — RENDER: RESULTS
# ══════════════════════════════════════════════════════════════════════════════
def render_kpi_bar() -> None:
    results   = st.session_state.results
    profiles  = results.get("profiles")
    features  = results.get("features")
    global_df = results.get("global")

    elapsed = ""
    if st.session_state.run_start and st.session_state.run_end:
        s = st.session_state.run_end - st.session_state.run_start
        m, sec = divmod(int(s), 60)
        elapsed = f"{m}m {sec}s" if m else f"{sec}s"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Brands Scraped", len(profiles) if profiles is not None else "—")
    with c2:
        if profiles is not None and "social_authority_score" in profiles.columns:
            avg = profiles["social_authority_score"].dropna().mean()
            st.metric("Avg Authority", f"{avg:.1f}/100")
        else:
            st.metric("Avg Authority", "—")
    with c3:
        if global_df is not None and "brand" in global_df.columns:
            sc = next((c for c in ["visibility_score", "citation_count"]
                       if c in global_df.columns), None)
            if sc and len(global_df):
                agg = global_df.groupby("brand")[sc].mean()
                st.metric("Top Brand", agg.idxmax())
            else:
                st.metric("Top Brand", "—")
        else:
            st.metric("Top Brand", "—")
    with c4:
        if features is not None and "prompt_id" in features.columns:
            st.metric("Prompts Queried", features["prompt_id"].nunique())
        else:
            st.metric("Prompts Queried", "—")
    with c5:
        icon = ("✅" if st.session_state.run_ok is True
                else "❌" if st.session_state.run_ok is False else "—")
        st.metric("Run", f"{icon}  {elapsed}")


def render_agent0_tab() -> None:
    df = st.session_state.results.get("prompts")
    if df is None:
        st.info("Agent 0 has not produced output yet. Run the pipeline first.")
        return
    st.caption(f"{len(df)} prompts generated")
    if "intent_id" in df.columns:
        all_i = sorted(df["intent_id"].unique().tolist())
        sel = st.multiselect("Filter by intent", all_i, default=all_i, key="a0_intent_filter")
        df = df[df["intent_id"].isin(sel)] if sel else df
    st.dataframe(df, width='stretch', height=420)


def render_agent1_tab() -> None:
    features_df = st.session_state.results.get("features")
    global_df   = st.session_state.results.get("global")

    if global_df is None and features_df is None:
        st.info("Agent 1 has not produced output yet. Run the pipeline first.")
        return

    if global_df is not None and len(global_df) > 0 and "brand" in global_df.columns:
        sc = next((c for c in ["visibility_score", "citation_count", "response_frequency",
                                "prompt_coverage", "stability_score"]
                   if c in global_df.columns), None)
        if sc:
            agg = global_df.groupby("brand")[sc].mean().sort_values(ascending=False).head(20)
            try:
                import plotly.express as px
                fig = px.bar(
                    agg.reset_index(),
                    x="brand", y=sc,
                    title=f"Brand Visibility — {sc.replace('_', ' ').title()}",
                    color=sc, color_continuous_scale="Blues",
                    labels={"brand": "Brand", sc: sc.replace("_", " ").title()},
                )
                fig.update_layout(xaxis_tickangle=-40, showlegend=False,
                                  height=380, margin=dict(b=90))
                st.plotly_chart(fig,width='stretch', height=420)
            except ImportError:
                st.bar_chart(agg)
        st.markdown("**Global metrics**")
        st.dataframe(global_df, width='stretch', height=260)

    if features_df is not None:
        with st.expander("Per-prompt features (raw)"):
            st.dataframe(features_df, width='stretch', height=260)


def render_agent2_tab() -> None:
    df = st.session_state.results.get("profiles")
    if df is None:
        st.info("Agent 2 has not produced output yet. Run the pipeline first.")
        return

    st.caption(f"{len(df)} brands processed")

    if "social_authority_score" in df.columns and df["social_authority_score"].notna().any():
        try:
            import plotly.express as px
            fig = px.bar(
                df.sort_values("social_authority_score", ascending=True),
                x="social_authority_score", y="brand", orientation="h",
                title="Social Authority Score (0 – 100)",
                color="social_authority_score",
                color_continuous_scale="Greens", range_x=[0, 100],
            )
            fig.update_layout(height=max(260, len(df) * 38 + 60),
                              showlegend=False, margin=dict(l=140))
            st.plotly_chart(fig, width='stretch')
        except ImportError:
            pass

    breakdown_cols = [c for c in ["score_review_quality", "score_social_reach",
                                   "score_web_completeness", "score_cross_platform"]
                      if c in df.columns]
    if breakdown_cols and len(df) >= 1:
        try:
            import plotly.graph_objects as go
            sel_brand = st.selectbox("Radar — select brand", df["brand"].tolist(), key="a2_radar")
            row  = df[df["brand"] == sel_brand].iloc[0]
            cats = [c.replace("score_", "").replace("_", " ").title() for c in breakdown_cols]
            vals = [float(row.get(c, 0) or 0) for c in breakdown_cols]
            fig  = go.Figure(go.Scatterpolar(
                r=vals + [vals[0]], theta=cats + [cats[0]],
                fill="toself", line_color="#22c55e",
            ))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 25])),
                              title=f"Score Breakdown — {sel_brand}", height=320)
            st.plotly_chart(fig, width='stretch')
        except ImportError:
            pass

    if "overall_confidence" in df.columns:
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Avg Data Confidence", f"{df['overall_confidence'].mean():.2f}")
        with c2:
            n_flags = (df["hallucination_count"].sum()
                       if "hallucination_count" in df.columns else "—")
            st.metric("Total Hallucination Flags", n_flags)

    priority  = ["brand", "social_authority_score", "gm_rating", "ta_rating",
                 "ig_followers", "fb_page_likes", "overall_confidence",
                 "geo_recommendations", "synthesis_notes"]
    show_cols = [c for c in priority if c in df.columns]
    rest_cols  = [c for c in df.columns if c not in show_cols]
    st.dataframe(df[show_cols + rest_cols], width='stretch', height=320)


def render_downloads_tab() -> None:
    if not OUTPUT_DIR.exists():
        st.info("Output directory `geo_output/` does not exist yet.")
        return
    files = sorted(OUTPUT_DIR.glob("*.csv")) + sorted(OUTPUT_DIR.glob("*.json"))
    if not files:
        st.info("No output files found.")
        return
    for path in files:
        size_kb = path.stat().st_size / 1024
        with open(path, "rb") as f:
            st.download_button(
                label=f"⬇  {path.name}  ({size_kb:.1f} KB)",
                data=f.read(),
                file_name=path.name,
                mime="text/csv" if path.suffix == ".csv" else "application/json",
                width='stretch',
                key=f"dl_{path.name}",
            )


# ══════════════════════════════════════════════════════════════════════════════
# 9 — SIDEBAR: Advanced parameters
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Parameters")

    ca, cb = st.columns(2)
    with ca:
        n_runs    = st.number_input("n_runs",     min_value=1, max_value=5,  value=1, step=1)
        top_n     = st.number_input("top_n",      min_value=1, max_value=20, value=5, step=1)
    with cb:
        n_intents = st.number_input("n_intents",  min_value=2, max_value=8,  value=3, step=1)
        max_ppi   = st.number_input("max/intent", min_value=1, max_value=5,  value=2, step=1,
                                    help="max_prompts_per_intent")

    languages = st.multiselect("Languages", ["fr", "ar", "en", "tn"], default=["fr"])

    with st.expander("🗂️ Legacy mode (domain / location)"):
        use_legacy     = st.checkbox("Use domain + location instead of query")
        domain_input   = st.text_input("Domain",   value="Tunisian restaurants",
                                       disabled=not use_legacy)
        location_input = st.text_input("Location", value="Tunisia",
                                       disabled=not use_legacy)

    st.markdown("---")
    python_path = st.text_input(
        "Python executable",
        value=PYTHON,
        help="Full path to the Anaconda/conda Python",
    )

    st.markdown("---")

    # Status indicator
    if st.session_state.is_running:
        elapsed = time.time() - (st.session_state.run_start or time.time())
        m, s = divmod(int(elapsed), 60)
        st.warning(f"⏳  Running… {m}m {s}s")
    elif st.session_state.run_ok is True:
        st.success("✅  Last run succeeded")
    elif st.session_state.run_ok is False:
        st.error("❌  Last run failed — see console")

    if st.session_state.is_running:
        if st.button("⏹  Request Stop", width='stretch'):
            st.session_state.is_running = False
            st.warning("Stop requested.")

    if st.button("🔄  Reload results from disk", width='stretch'):
        st.session_state.results = _load_results()
        st.success("Results reloaded.")


# ══════════════════════════════════════════════════════════════════════════════
# 10 — DRAIN QUEUE
# ══════════════════════════════════════════════════════════════════════════════
_drain_queue()


# ══════════════════════════════════════════════════════════════════════════════
# 11 — MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 📡 GEO Pipeline")
st.caption(
    "Generative Engine Optimization — Multi-Agent Brand Visibility Analysis  ·  "
    f"Output → `{OUTPUT_DIR.relative_to(ROOT)}/`"
)

st.divider()

# ── Query input ───────────────────────────────────────────────────────────────
section_label("🔍 Run the pipeline")

col_q, col_btn = st.columns([5, 1])
with col_q:
    query_input = st.text_input(
        "query",
        value="",
        placeholder=(
            "Type your query — e.g.  Restaurants in Sfax  ·  "
            "Best cafes in Tunis  ·  Patisseries in Monastir"
        ),
        label_visibility="collapsed",
        disabled=st.session_state.is_running,
    )
with col_btn:
    st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)
    run_clicked = st.button(
        "▶  Run",
        width='stretch',
        type="primary",
        disabled=st.session_state.is_running,
    )

# ── Trigger pipeline ──────────────────────────────────────────────────────────
if run_clicked:
    cmd = [python_path, str(ROOT / "pipeline.py")]
    if use_legacy:
        cmd += [
            "--domain",   domain_input   or "Tunisian restaurants",
            "--location", location_input or "Tunisia",
        ]
    else:
        cmd += ["--query", query_input or "Restaurants in Tunis"]
    cmd += [
        "--n_runs",                str(n_runs),
        "--top_n",                 str(top_n),
        "--n_intents",             str(n_intents),
        "--max_prompts_per_intent", str(max_ppi),
        "--languages",             ",".join(languages) if languages else "fr",
    ]
    _start_pipeline(cmd)
    st.rerun()

st.divider()

# ── Agent workflow ─────────────────────────────────────────────────────────────
section_label("🗺️ Pipeline workflow — what each agent does")
render_agent_workflow()

st.divider()

# ── Live node status ──────────────────────────────────────────────────────────
section_label("⚡ Live node status")
render_pipeline_flow()

# ── KPI bar ────────────────────────────────────────────────────────────────────
if st.session_state.run_start is not None:
    st.divider()
    render_kpi_bar()

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_log, tab_a0, tab_a1, tab_a2, tab_dl = st.tabs([
    "🖥️  Live Console",
    "✍️  Agent 0 — Prompts",
    "📊  Agent 1 — GEO Analysis",
    "🌐  Agent 2 — Social Profiles",
    "⬇️  Downloads",
])

with tab_log:
    log_filter = st.selectbox(
        "Show",
        ["all", "errors only", "warnings", "success only", "debug"],
        label_visibility="collapsed",
        key="log_filter",
    )
    render_log_console(filter_level=log_filter, height=420)

with tab_a0:
    render_agent0_tab()

with tab_a1:
    render_agent1_tab()

with tab_a2:
    render_agent2_tab()

with tab_dl:
    render_downloads_tab()


# ══════════════════════════════════════════════════════════════════════════════
# 12 — AUTO RERUN  (polls queue every 400 ms while pipeline is active)
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.is_running:
    time.sleep(0.4)
    st.rerun()
