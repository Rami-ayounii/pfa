"""
graph/pipeline_graph.py
═══════════════════════
LangGraph StateGraph assembly for the GEO Multi-Agent Pipeline.

Graph structure:
  START
    → parse_query_node              (NL → domain + location)
    → agent0_node                   (generate prompt set)
    → agent1_load_node              (load prompts from DataFrame)
    → agent1_llm_query_all_node     (sequential prompts × parallel models per prompt)
    → agent1_aggregate_node         (merge all LLM responses → DataFrame)
    → agent1_extract_node           (entity extraction)
    → agent1_enrich_node            (entity enrichment)
    → agent1_clean_node             (fuzzy dedup + LLM arbitration)
    → agent1_metrics_node           (GEO feature computation + brand list)
    → [route_brand_scraping]    ──Send fan-out──▶ agent2_scrape_node (×brands parallel)
    → export_node               (write CSVs + pipeline_summary.json)
    → END

LLM querying strategy:
  agent1_llm_query_all_node processes prompts one-by-one (sequential).
  For each prompt, all (model × run) combos fire concurrently via asyncio.gather().
  This prevents Groq rate-limit exhaustion while still parallelising within each prompt.

Brand scraping parallelism is provided by LangGraph's Send() API.
Results are merged back via Annotated[list, operator.add] fields in GEOState.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from loguru import logger

from graph.state import GEOState
from graph.nodes.parse_node             import parse_query_node
from graph.nodes.agent0_node            import agent0_node
from graph.nodes.agent1_llm_query_node  import agent1_llm_query_all_node, agent1_llm_query_node
from graph.nodes.agent1_nodes           import (
    agent1_load_node,
    agent1_aggregate_node,
    agent1_extract_node,
    agent1_enrich_node,
    agent1_clean_node,
    agent1_metrics_node,
)
from graph.nodes.agent2_scrape_node     import agent2_scrape_node
from graph.nodes.export_node            import export_node


# ── Fan-out routers ────────────────────────────────────────────────────────────


def route_brand_scraping(state: GEOState) -> list[Send]:
    """
    Fan-out: generate one Send per brand.
    Each Send runs agent2_scrape_node in parallel.
    """
    brands         = state.get("brands", [])
    location       = state.get("location", "Tunisia")
    output_dir     = state.get("output_dir", "geo_output")
    serpapi_key    = state.get("serpapi_key", "")
    apify_token    = state.get("apify_token", "")
    openrouter_key = state.get("openrouter_key", "")
    groq_client    = state.get("groq_client")
    ig_user        = state.get("ig_user", "")
    ig_pass        = state.get("ig_pass", "")
    is_colab       = state.get("is_colab", False)

    if not brands:
        # No brands → skip Agent 2 entirely; go straight to export
        return [Send("export_node", {})]

    return [
        Send(
            "agent2_scrape_node",
            {
                "brand":          brand,
                "location":       location,
                "output_dir":     output_dir,
                "serpapi_key":    serpapi_key,
                "apify_token":    apify_token,
                "openrouter_key": openrouter_key,
                "groq_client":    groq_client,
                "ig_user":        ig_user,
                "ig_pass":        ig_pass,
                "is_colab":       is_colab,
            },
        )
        for brand in brands
    ]


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Compile and return the GEO pipeline StateGraph.

    Usage:
        graph = build_graph()
        result = graph.invoke(initial_state)
    """
    builder = StateGraph(GEOState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    builder.add_node("parse_query_node",        parse_query_node)
    builder.add_node("agent0_node",             agent0_node)
    builder.add_node("agent1_load_node",        agent1_load_node)
    builder.add_node("agent1_llm_query_all_node", agent1_llm_query_all_node)  # sequential-prompt, parallel-model
    builder.add_node("agent1_aggregate_node",   agent1_aggregate_node)
    builder.add_node("agent1_extract_node",     agent1_extract_node)
    builder.add_node("agent1_enrich_node",      agent1_enrich_node)
    builder.add_node("agent1_clean_node",       agent1_clean_node)
    builder.add_node("agent1_metrics_node",     agent1_metrics_node)
    builder.add_node("agent2_scrape_node",      agent2_scrape_node)      # fan-out target
    builder.add_node("export_node",             export_node)

    # ── Sequential edges ───────────────────────────────────────────────────────
    builder.add_edge(START,                    "parse_query_node")
    builder.add_edge("parse_query_node",       "agent0_node")
    builder.add_edge("agent0_node",            "agent1_load_node")

    # ── LLM queries: sequential prompts, parallel models per prompt ───────────
    builder.add_edge("agent1_load_node",            "agent1_llm_query_all_node")
    builder.add_edge("agent1_llm_query_all_node",   "agent1_aggregate_node")

    # ── Sequential Agent 1 processing ─────────────────────────────────────────
    builder.add_edge("agent1_aggregate_node", "agent1_extract_node")
    builder.add_edge("agent1_extract_node",   "agent1_enrich_node")
    builder.add_edge("agent1_enrich_node",    "agent1_clean_node")
    builder.add_edge("agent1_clean_node",     "agent1_metrics_node")

    # ── Fan-out: brand scraping (parallel) ────────────────────────────────────
    builder.add_conditional_edges(
        "agent1_metrics_node",
        route_brand_scraping,
        ["agent2_scrape_node", "export_node"],  # router may skip to export if no brands
    )
    builder.add_edge("agent2_scrape_node", "export_node")
    builder.add_edge("export_node",         END)

    return builder.compile()


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    from loguru import logger
    logger.info("Compiling GEO pipeline graph …")
    g = build_graph()
    logger.success("Graph compiled successfully.")
    try:
        print(g.get_graph().draw_ascii())
    except Exception:
        logger.info("(ASCII diagram not available)")
