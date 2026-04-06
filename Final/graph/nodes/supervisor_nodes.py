"""Supervisor nodes for the GEO LangGraph pipeline.

Each supervisor sits between two agents and decides:
  - "continue" : output is good enough, move to next agent
  - "retry"    : output is poor, send back to same agent for another attempt
  - "abort"    : max retries reached, skip to END

The LLM evaluates quality using concrete metrics from the state, then returns
a structured JSON decision. All decisions are appended to state["supervisor_notes"].
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langgraph.graph import END

from config import MODEL_ANALYST2
from core.llm_client import call_llm_json
from graph.state import GEOState

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_SUPERVISOR_SYSTEM = (
    "You are a strict quality supervisor for an AI research pipeline. "
    "Respond ONLY with valid JSON, no markdown, no explanation."
)


def _llm_decide(prompt: str) -> dict:
    """Call LLM with fallback and parse JSON decision."""
    try:
        result = call_llm_json(
            prompt=prompt,
            system=_SUPERVISOR_SYSTEM,
            preferred_model=MODEL_ANALYST2,
        )
        return result if isinstance(result, dict) else {}
    except Exception as exc:
        return {"decision": "continue", "reason": f"supervisor LLM failed ({exc}) - defaulting to continue"}


# ---------------------------------------------------------------------------
# Supervisor 0 - evaluates Agent 0 output (prompt set quality)
# ---------------------------------------------------------------------------

def supervisor0_node(state: dict) -> dict:
    notes   = list(state.get("supervisor_notes", []))
    retries = state.get("agent0_retries", 0)
    max_r   = state.get("max_retries", 2)

    prompt_set = state.get("prompts", [])  # Project 2 uses "prompts"
    n_prompts  = len(prompt_set)
    languages  = list({p.get("language") for p in prompt_set if isinstance(p, dict)})
    intents    = list({p.get("intent_id") for p in prompt_set if isinstance(p, dict)})

    if not prompt_set:
        note = "supervisor0: no prompts produced - aborting"
        notes.append(note)
        return {**state, "supervisor_notes": notes, "current_step": "supervisor0_abort"}

    n_intents_req = state.get('n_intents', 4)
    langs_req     = state.get('languages', ['fr'])
    # Minimum acceptable: 80% of requested intents, all languages present
    min_intents    = max(1, int(n_intents_req * 0.8))
    min_prompts    = min_intents * len(langs_req)

    prompt = f"""You are evaluating the output of an intent/prompt generation agent.

Domain        : {state.get('domain')}
Prompts count : {n_prompts} (minimum acceptable: {min_prompts})
Languages     : {languages} (required: {langs_req})
Intent types  : {intents} (minimum acceptable: {min_intents} of {n_intents_req} requested)
Sample prompts: {[p.get('prompt_text') if isinstance(p, dict) else str(p) for p in prompt_set[:5]]}

Decide "continue" if ALL of:
- At least {min_intents} distinct intents are present
- All required languages {langs_req} are present
- Prompts sound like real user queries (not template fillers)

Decide "retry" ONLY if the output clearly fails the above minimum thresholds.
Do NOT retry just because the set could theoretically be more diverse.

Respond ONLY with:
{{
  "decision": "continue" or "retry",
  "quality_score": 0-10,
  "reason": "one sentence"
}}"""

    result   = _llm_decide(prompt)
    decision = result.get("decision", "continue")
    score    = result.get("quality_score", 5)
    reason   = result.get("reason", "")

    note = f"supervisor0 [retry {retries}]: score={score}/10, decision={decision} - {reason}"
    notes.append(note)
    print(f"\n[Supervisor 0] {note}")

    if decision == "retry" and retries < max_r:
        return {
            **state,
            "prompts":          [],   # Project 2 field name
            "agent0_retries":   retries + 1,
            "supervisor_notes": notes,
            "current_step":     "supervisor0_retry",
        }

    return {
        **state,
        "agent0_quality_score": float(score),
        "supervisor_notes":     notes,
        "current_step":         "supervisor0_ok",
    }


def route_after_supervisor0(state: dict) -> str:
    step = state.get("current_step", "")
    if step == "supervisor0_retry":
        return "agent0_node"
    if step == "supervisor0_abort":
        return "__end__"
    return "agent1_load_node"


# ---------------------------------------------------------------------------
# Supervisor 1 - evaluates Agent 1 output (entity extraction quality)
# ---------------------------------------------------------------------------

def supervisor1_node(state: dict) -> dict:
    notes   = list(state.get("supervisor_notes", []))
    retries = state.get("agent1_retries", 0)
    max_r   = state.get("max_retries", 2)

    global_feats = state.get("entity_features_global", [])  # same field name
    n_entities   = len(global_feats)
    n_prompts    = len(state.get("prompts", []))  # Project 2 uses "prompts"

    if not global_feats:
        note = "supervisor1: no entity features produced - aborting"
        notes.append(note)
        return {**state, "supervisor_notes": notes, "current_step": "supervisor1_abort"}

    stable   = sum(1 for e in global_feats if e.get("consistency_label") == "stable")
    variable = sum(1 for e in global_feats if e.get("consistency_label") == "variable")
    top5     = [e.get("canonical_entity") for e in global_feats[:5]]

    # With small n_runs (e.g. 2), almost all entities will be "unstable" —
    # that is expected and NOT a reason to retry.
    min_entities = max(3, n_prompts // 4)

    prompt = f"""You are evaluating the output of an entity extraction and feature computation agent.

Domain           : {state.get('domain')}
Total prompts    : {n_prompts}
Entities found   : {n_entities} (minimum acceptable: {min_entities})
Stable entities  : {stable}
Variable entities: {variable}
Top 5 entities   : {top5}

Decide "continue" if ALL of:
- At least {min_entities} distinct entities were extracted
- Entity names look plausibly like real restaurants/brands (not just generic words)
- mention_rate and stability_score fields are present (even if scores are low)

IMPORTANT: "unstable" consistency labels are NORMAL and EXPECTED with small datasets.
Do NOT penalise low stability scores — they are a valid research finding, not a failure.
Decide "retry" ONLY if fewer than {min_entities} entities were found OR all entity names
look like generic non-restaurant terms.

Respond ONLY with:
{{
  "decision": "continue" or "retry",
  "quality_score": 0-10,
  "reason": "one sentence"
}}"""

    result   = _llm_decide(prompt)
    decision = result.get("decision", "continue")
    score    = result.get("quality_score", 5)
    reason   = result.get("reason", "")

    note = f"supervisor1 [retry {retries}]: score={score}/10, decision={decision} - {reason}"
    notes.append(note)
    print(f"\n[Supervisor 1] {note}")

    if decision == "retry" and retries < max_r:
        return {
            **state,
            "raw_responses":          [],
            "extracted_entities":     [],
            "clean_entities":         [],
            "entity_features":        [],
            "entity_features_global": [],
            "agent1_retries":         retries + 1,
            "supervisor_notes":       notes,
            "current_step":           "supervisor1_retry",
        }

    return {
        **state,
        "supervisor_notes": notes,
        "current_step":     "supervisor1_ok",
    }


def route_after_supervisor1(state: dict) -> str:
    step = state.get("current_step", "")
    if step == "supervisor1_retry":
        return "agent1_load_node"
    if step == "supervisor1_abort":
        return "__end__"
    return "monitor_node"


# ---------------------------------------------------------------------------
# Supervisor 2 - evaluates Agent 2 output (web feature quality)
# ---------------------------------------------------------------------------

def supervisor2_node(state: dict) -> dict:
    notes   = list(state.get("supervisor_notes", []))
    retries = state.get("agent2_retries", 0)
    max_r   = state.get("max_retries", 2)

    web_feats  = state.get("profiles", [])  # Project 2 uses "profiles"
    n_entities = len(state.get("entity_features_global", []))
    n_web      = len(web_feats)

    if not web_feats:
        note = "supervisor2: no web features produced - aborting"
        notes.append(note)
        return {**state, "supervisor_notes": notes, "current_step": "supervisor2_abort"}

    coverage = round(n_web / n_entities, 2) if n_entities else 0
    sample   = web_feats[:3]

    prompt = f"""You are evaluating the output of a web-feature extraction agent.

Domain            : {state.get('domain')}
Entities in scope : {n_entities}
Entities with web data: {n_web}
Coverage rate     : {coverage} (target >= 0.7)
Sample records    : {sample}

Quality criteria:
1. Coverage >= 0.7 (web data found for at least 70% of entities)
2. Records contain meaningful fields (not all None/empty)
3. No sign of scraping failures for the majority of entities

Respond ONLY with:
{{
  "decision": "continue" or "retry",
  "quality_score": 0-10,
  "reason": "one sentence"
}}"""

    result   = _llm_decide(prompt)
    decision = result.get("decision", "continue")
    score    = result.get("quality_score", 5)
    reason   = result.get("reason", "")

    note = f"supervisor2 [retry {retries}]: score={score}/10, decision={decision} - {reason}"
    notes.append(note)
    print(f"\n[Supervisor 2] {note}")

    if decision == "retry" and retries < max_r:
        return {
            **state,
            "profiles":         [],   # Project 2 field name
            "agent2_retries":   retries + 1,
            "supervisor_notes": notes,
            "current_step":     "supervisor2_retry",
        }

    return {
        **state,
        "supervisor_notes": notes,
        "current_step":     "supervisor2_ok",
    }


def route_after_supervisor2(state: dict) -> str:
    step = state.get("current_step", "")
    if step == "supervisor2_retry":
        return "monitor_node"
    if step == "supervisor2_ok":
        return "export_node"
    return "__end__"
