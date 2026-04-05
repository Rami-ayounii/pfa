"""
graph/state.py
══════════════
Shared LangGraph pipeline state — TypedDict passed between all nodes.

Key design choices:
  - Annotated[list, operator.add]   → fan-in merge for parallel branches
  - DataFrames stored in-memory     → no CSV hops between agents
  - Config fields carried in state  → nodes are pure functions
"""

import operator
from typing import Any, Annotated
import pandas as pd

# TypedDict is available in Python 3.8+; use from typing_extensions if needed
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict


class GEOState(TypedDict, total=False):
    """Full shared state for the GEO LangGraph pipeline."""

    # ── Raw user input ────────────────────────────────────────────────────────
    query: str                     # "Restaurants in Sfax"

    # ── Parsed dimensions (filled by parse_query_node) ───────────────────────
    domain: str                    # "Restaurants"
    location: str                  # "Sfax"

    # ── Agent 0 output ────────────────────────────────────────────────────────
    prompt_df: Any                 # pd.DataFrame | None  (in-memory, not serialised)

    # ── Agent 1 intermediates ─────────────────────────────────────────────────
    prompts: list                  # list of prompt dicts loaded from prompt_df

    # Fan-in accumulator: each parallel LLM node appends one dict
    raw_responses: Annotated[list, operator.add]

    entities_df:  Any              # pd.DataFrame after step3
    enriched_df:  Any              # pd.DataFrame after step4
    clean_df:     Any              # pd.DataFrame after step5
    features_df:  Any              # pd.DataFrame after step6 (per-prompt)
    global_df:    Any              # pd.DataFrame after step6 (global)

    brands: list                   # top-N brand names → Agent 2 input

    # ── Agent 2 output ────────────────────────────────────────────────────────
    # Fan-in accumulator: each parallel scrape node appends one SocialProfile
    profiles: Annotated[list, operator.add]

    # ── Pipeline config (carried through all nodes) ───────────────────────────
    n_intents:              int
    n_variants:             int
    n_runs:                 int
    max_prompts_per_intent: int    # max prompt variants selected per intent (default 2)
    languages:              list   # e.g. ["fr", "ar"]
    query_models:           list   # Groq model IDs for Step 2
    analyst_model:          str    # model for cleaning/reflection
    output_dir:             str
    top_n_brands:           int

    # ── Shared API clients / keys ─────────────────────────────────────────────
    groq_client:    Any            # groq.Groq instance (not serialised)
    mistral_key:    str
    openrouter_key: str
    serpapi_key:    str
    apify_token:    str
    ig_user:        str
    ig_pass:        str
    is_colab:       bool

    # ── Diagnostics ───────────────────────────────────────────────────────────
    errors:   Annotated[list, operator.add]   # any error strings from nodes
    warnings: Annotated[list, operator.add]   # non-fatal warnings
