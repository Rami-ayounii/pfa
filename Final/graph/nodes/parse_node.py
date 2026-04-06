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


def parse_query_node(state: GEOState) -> dict:
    """
    Extract domain and location from the raw natural-language query.

    Example:
        "Restaurants in Sfax"  →  domain="Restaurants", location="Sfax"
    """
    raw_query    = state.get("query", "")
    groq_client  = state.get("groq_client")

    logger.info(f"[ParseNode] Parsing query: '{raw_query}'")
    parsed = parse_query(raw_query, groq_client=groq_client)

    logger.success(
        f"[ParseNode] domain='{parsed['domain']}', location='{parsed['location']}'"
    )
    return {
        "domain":   parsed["domain"],
        "location": parsed["location"],
    }
