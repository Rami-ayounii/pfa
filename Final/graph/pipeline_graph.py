"""
Final/graph/pipeline_graph.py
══════════════════════════════
Merged LangGraph StateGraph for the GEO Multi-Agent Pipeline.

Combines Project 2's Send fan-out architecture with Project 1's supervisor retry edges.

Graph structure:
  START
    → parse_query_node              (NL → domain + location)
    → agent0_node                   (generate prompt set)
    → supervisor0_node              (quality gate — retry agent0 or continue)
    → agent1_load_node              (load prompts from DataFrame)
    → agent1_llm_query_all_node     (sequential prompts × parallel models per prompt)
    → agent1_aggregate_node         (merge all LLM responses → DataFrame)
    → agent1_extract_node           (entity extraction)
    → agent1_enrich_node            (entity enrichment)
    → agent1_clean_node             (fuzzy dedup + LLM arbitration)
    → agent1_metrics_node           (GEO feature computation + brand list)
    → supervisor1_node              (quality gate — retry agent1 or continue)
    → monitor_node                  (zero-LLM quality gate — sets quality_score)
    → [route_brand_scraping]    ──Send fan-out──▶ agent2_scrape_node (×brands parallel)
    → export_node               (write CSVs + pipeline_summary.json)
    → supervisor2_node              (quality gate — retry agent2 or END)
    → END

Supervisor retry edges (Project 1 pattern):
  supervisor0: retry → agent0_node  |  continue → agent1_load_node  |  abort → END
  supervisor1: retry → agent1_load_node  |  continue → monitor_node  |  abort → END
  supervisor2: retry → monitor_node  |  END

Brand scraping parallelism is provided by LangGraph's Send() API (Project 2 pattern).
Results are merged back via Annotated[list, operator.add] fields in GEOState.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END
from langgraph.types import Send
from loguru import logger

from graph.state import GEOState
from graph.nodes.parse_node import parse_query_node
from graph.nodes.agent0_node import agent0_node
from graph.nodes.agent1_llm_query_node import agent1_llm_query_all_node
from graph.nodes.agent1_nodes import (
    agent1_load_node,
    agent1_aggregate_node,
    agent1_extract_node,
    agent1_enrich_node,
    agent1_clean_node,
    agent1_metrics_node,
)
from graph.nodes.supervisor_nodes import (
    supervisor0_node, supervisor1_node, supervisor2_node,
    route_after_supervisor0, route_after_supervisor1, route_after_supervisor2,
)
from graph.nodes.monitor_node import monitor_node
from graph.nodes.agent2_scrape_node import agent2_scrape_node
from graph.nodes.export_node import export_node


# ── Fan-out router ─────────────────────────────────────────────────────────────


def route_brand_scraping(state: GEOState) -> list[Send]:
    """
    Fan-out: generate one Send per brand.
    Each Send runs agent2_scrape_node in parallel.
    If no brands are present, skip Agent 2 and go straight to export.
    """
    brands = state.get("brands", [])
    if not brands:
        return [Send("export_node", state)]
    return [Send("agent2_scrape_node", {**state, "brand": b}) for b in brands]


# ── Graph builder ──────────────────────────────────────────────────────────────


def build_graph():
    """
    Compile and return the merged GEO pipeline StateGraph.

    Supervisor retry loops (Project 1) are injected between agents.
    Send fan-out for brand scraping (Project 2) is preserved intact.

    Usage:
        graph = build_graph()
        result = await graph.ainvoke(initial_state)
    """
    graph = StateGraph(GEOState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    graph.add_node("parse_query_node",          parse_query_node)
    graph.add_node("agent0_node",               agent0_node)
    graph.add_node("supervisor0_node",          supervisor0_node)
    graph.add_node("agent1_load_node",          agent1_load_node)
    graph.add_node("agent1_llm_query_all_node", agent1_llm_query_all_node)
    graph.add_node("agent1_aggregate_node",     agent1_aggregate_node)
    graph.add_node("agent1_extract_node",       agent1_extract_node)
    graph.add_node("agent1_enrich_node",        agent1_enrich_node)
    graph.add_node("agent1_clean_node",         agent1_clean_node)
    graph.add_node("agent1_metrics_node",       agent1_metrics_node)
    graph.add_node("supervisor1_node",          supervisor1_node)
    graph.add_node("monitor_node",              monitor_node)
    graph.add_node("agent2_scrape_node",        agent2_scrape_node)
    graph.add_node("export_node",               export_node)
    graph.add_node("supervisor2_node",          supervisor2_node)

    # ── Entry point and initial edges ──────────────────────────────────────────
    graph.set_entry_point("parse_query_node")
    graph.add_edge("parse_query_node", "agent0_node")
    graph.add_edge("agent0_node",      "supervisor0_node")

    # supervisor0: retry → agent0_node | continue → agent1_load_node | abort → END
    graph.add_conditional_edges(
        "supervisor0_node",
        route_after_supervisor0,
        {
            "agent0_node":      "agent0_node",
            "agent1_load_node": "agent1_load_node",
            END:                END,
        },
    )

    # ── Agent 1 sequential chain ───────────────────────────────────────────────
    graph.add_edge("agent1_load_node",          "agent1_llm_query_all_node")
    graph.add_edge("agent1_llm_query_all_node", "agent1_aggregate_node")
    graph.add_edge("agent1_aggregate_node",     "agent1_extract_node")
    graph.add_edge("agent1_extract_node",       "agent1_enrich_node")
    graph.add_edge("agent1_enrich_node",        "agent1_clean_node")
    graph.add_edge("agent1_clean_node",         "agent1_metrics_node")
    graph.add_edge("agent1_metrics_node",       "supervisor1_node")

    # supervisor1: retry → agent1_load_node | continue → monitor_node | abort → END
    graph.add_conditional_edges(
        "supervisor1_node",
        route_after_supervisor1,
        {
            "agent1_load_node": "agent1_load_node",
            "monitor_node":     "monitor_node",
            END:                END,
        },
    )

    # ── Fan-out: brand scraping (parallel) ────────────────────────────────────
    graph.add_conditional_edges(
        "monitor_node",
        route_brand_scraping,
        ["agent2_scrape_node", "export_node"],  # router may skip to export if no brands
    )
    graph.add_edge("agent2_scrape_node", "export_node")
    graph.add_edge("export_node",        "supervisor2_node")

    # supervisor2: retry → monitor_node (re-run agent2) | ok → END | abort → END
    # route_after_supervisor2 returns "export_node" for the ok case (already ran);
    # map it to END so the pipeline terminates cleanly.
    graph.add_conditional_edges(
        "supervisor2_node",
        route_after_supervisor2,
        {
            "monitor_node": "monitor_node",
            "export_node":  END,            # supervisor2_ok — export already done
            END:            END,            # supervisor2_abort
        },
    )

    return graph.compile()


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Compiling merged GEO pipeline graph …")
    g = build_graph()
    logger.success("Graph compiled successfully.")
    try:
        print(g.get_graph().draw_ascii())
    except Exception:
        logger.info("(ASCII diagram not available)")
