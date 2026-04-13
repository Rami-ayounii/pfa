"""
graph/nodes/parse_node.py
══════════════════════════
LangGraph node: parse the raw user query into domain + location.

Input state:  query
Output state: domain, location
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Final/ is root

from loguru import logger
from graph.state import GEOState
from core.query_parser import parse_query
from core.status_tracker import emit as _emit


def parse_query_node(state: GEOState) -> dict:
    """
    Extract domain and location from the raw natural-language query.

    Example:
        "Restaurants in Sfax"  →  domain="Restaurants", location="Sfax"
    """
    raw_query    = state.get("query", "")
    groq_client  = state.get("groq_client")
    output_dir   = state.get("output_dir", "geo_output")

    _emit("parse_query", "active", {"query": raw_query}, output_dir=output_dir)
    try:
        logger.info(f"[ParseNode] Parsing query: '{raw_query}'")
        parsed = parse_query(raw_query, groq_client=groq_client)

        logger.success(
            f"[ParseNode] domain='{parsed['domain']}', location='{parsed['location']}'"
        )
        _emit("parse_query", "done",
              {"domain": parsed["domain"], "location": parsed["location"]},
              output_dir=output_dir)
        return {
            "domain":   parsed["domain"],
            "location": parsed["location"],
        }
    except Exception as exc:
        logger.error(f"[ParseNode] {exc}")
        _emit("parse_query", "error", {"error": str(exc)}, output_dir=output_dir)
        raise
