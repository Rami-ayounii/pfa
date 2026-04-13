"""
Final/graph/state.py
════════════════════
Merged GEOState TypedDict for the Final GEO pipeline.

Combines:
  - Project 2's fan-in merge operators (Annotated[list, operator.add])
  - Project 1's supervisor control fields (retries, quality scores, token accounting)

Key design choices:
  - Annotated[list, operator.add]  → LangGraph automatic fan-in for parallel branches
  - DataFrames stored in-memory    → no CSV hops between agents
  - supervisor_notes is a plain list (NOT Annotated — supervisors append sequentially)
  - total=False                    → all fields optional (allows partial state updates)
"""

import operator
from typing import Annotated, Any

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class GEOState(TypedDict, total=False):
    """Full shared state for the merged GEO LangGraph pipeline."""

    # ── Raw user input ────────────────────────────────────────────────────────
    query: str                      # "Restaurants in Sfax"

    # ── Parsed dimensions (filled by parse_query_node) ───────────────────────
    domain: str                     # "Restaurants"
    location: str                   # "Sfax"

    # ── Pipeline config ───────────────────────────────────────────────────────
    languages: list                 # e.g. ["fr", "ar"]
    n_intents: int
    n_variants: int
    n_runs: int
    max_prompts_per_intent: int     # max prompt variants selected per intent (default 6)
    top_n_brands: int
    output_dir: str

    # ── In-memory DataFrames (no CSV hops) ───────────────────────────────────
    prompt_df: Any                  # pd.DataFrame from agent0
    df_raw: Any                     # pd.DataFrame built once in agent1_aggregate; shared by enrich/clean/metrics
    entities_df: Any                # pd.DataFrame from agent1_extract
    enriched_df: Any                # pd.DataFrame from agent1_enrich
    clean_df: Any                   # pd.DataFrame from agent1_clean
    features_df: Any                # pd.DataFrame per-prompt features
    global_df: Any                  # pd.DataFrame global features (for monitor_node)

    # ── Fan-in fields (Annotated for LangGraph automatic merging) ────────────
    raw_responses: Annotated[list, operator.add]   # from parallel LLM nodes
    profiles: Annotated[list, operator.add]        # from parallel Agent 2 nodes
    errors: Annotated[list, operator.add]          # error strings from any node
    warnings: Annotated[list, operator.add]        # warning strings

    # ── Serialized outputs (list[dict] for JSON serialization) ───────────────
    prompts: list                   # list of prompt dicts for fan-out
    brands: list                    # top-N brands for Agent 2
    entity_features_global: list    # serialized global_df (for supervisor1 + agent2)

    # ── Metadata & run identity ───────────────────────────────────────────────
    run_id: str                     # unique run identifier e.g. "20260405_142301_abc3"
    knowledge_base: Any             # KnowledgeBase instance (not serialised)
    groq_client: Any                # Groq client instance (not serialised)
    quality_score: float            # 0.0–10.0, set by monitor_node after Agent 1 metrics
    quality_warnings: list          # non-fatal quality issues from monitor_node (plain list)

    # ── Supervisor control (from Project 1) ──────────────────────────────────
    agent0_retries: int             # how many times agent0 has been retried by supervisor
    agent1_retries: int
    agent2_retries: int
    max_retries: int                # ceiling applied by every supervisor node
    supervisor_notes: list          # supervisor decisions (plain list — sequential appends)
    current_step: str               # routing signal: "agent0_ok" | "agent1_ok" | "agent2_ok" | "retry" | "abort"

    # ── Quality scores from supervisors ──────────────────────────────────────
    agent0_quality_score: float
    agent1_quality_score: float
    agent2_coverage_score: float

    # ── Token accounting ──────────────────────────────────────────────────────
    token_usage: dict               # {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}


def initial_state(
    query: str = "",
    domain: str = "",
    location: str = "Tunisia",
    max_retries: int = 1,
    **kwargs,
) -> dict:
    """Factory for a minimal valid GEOState with sensible defaults."""
    return {
        "query": query,
        "domain": domain,
        "location": location,
        "languages": ["fr"],
        "n_intents": 4,
        "n_variants": 3,
        "n_runs": 2,
        "max_prompts_per_intent": 6,
        "top_n_brands": 10,
        "output_dir": "geo_output",
        # Fan-in accumulators — must start as empty lists
        "raw_responses": [],
        "profiles": [],
        "errors": [],
        "warnings": [],
        # Serialized outputs
        "prompts": [],
        "brands": [],
        "entity_features_global": [],
        # Supervisor control
        "supervisor_notes": [],
        "quality_warnings": [],
        "agent0_retries": 0,
        "agent1_retries": 0,
        "agent2_retries": 0,
        "max_retries": max_retries,
        # Quality & scoring
        "quality_score": 0.0,
        "agent0_quality_score": 0.0,
        "agent1_quality_score": 0.0,
        "agent2_coverage_score": 0.0,
        # Token accounting
        "token_usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        **kwargs,
    }
