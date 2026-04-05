"""
core/query_parser.py
════════════════════
Natural-language query parser for the GEO pipeline.

Converts a single user prompt like "Restaurants in Sfax" into a
structured {domain, location} dict that the LangGraph pipeline can use.

Strategy:
  1. Try a fast LLM call (MODEL_FAST) with a structured extraction prompt
  2. Fallback: regex patterns  ("X in Y", "X near Y", "X around Y")
  3. Final fallback: domain = raw_query, location = "Tunisia"
"""

import re
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

# ── Lazy import to avoid circular deps ───────────────────────────────────────
def _get_llm_helpers():
    from core.llm_client import query_llm, parse_json, MODEL_EXTRACTOR
    return query_llm, parse_json, MODEL_EXTRACTOR


# ── Regex patterns (fallback) ─────────────────────────────────────────────────
_LOCATION_PATTERNS = [
    r"^(.+?)\s+(?:in|dans|à|a|en)\s+(.+)$",       # "Restaurants in Sfax" / "en Tunisie"
    r"^(.+?)\s+(?:near|près de|proche de)\s+(.+)$", # "Cafes near Tunis"
    r"^(.+?)\s+(?:around|autour de)\s+(.+)$",        # "Pizza around Hammamet"
    r"^(.+?)\s+(?:at|at the)\s+(.+)$",               # "Food at Djerba"
]

# ── System prompt ─────────────────────────────────────────────────────────────
_PARSE_SYSTEM = """\
You are a query parser for a restaurant GEO analysis pipeline.

TASK: Extract the business domain and location from a user query.

Rules:
- domain: the TYPE of business (e.g. "restaurants", "cafes", "street food")
- location: the CITY or REGION (e.g. "Sfax", "Tunis", "Djerba", "Tunisia")
- If no explicit location is found, default location to "Tunisia"
- If no explicit domain is found, default domain to "restaurants"
- Keep the domain in the same language as the query

Return ONLY valid JSON: {"domain": "...", "location": "..."}
No markdown, no prose.
"""


def parse_query(raw_query: str, groq_client=None) -> dict:
    """
    Parse a natural-language query into {domain, location}.

    Args:
        raw_query   : e.g. "Restaurants in Sfax", "Meilleurs cafés à Tunis"
        groq_client : optional pre-built Groq client

    Returns:
        dict with keys "domain" and "location"
    """
    raw_query = (raw_query or "").strip()
    if not raw_query:
        logger.warning("[QueryParser] Empty query — using defaults")
        return {"domain": "restaurants", "location": "Tunisia"}

    # ── 1. LLM-based extraction ───────────────────────────────────────────────
    try:
        query_llm, parse_json, MODEL_EXTRACTOR = _get_llm_helpers()
        result = query_llm(
            MODEL_EXTRACTOR,
            raw_query,
            system=_PARSE_SYSTEM,
            role="extractor",
            client=groq_client,
        )
        parsed = parse_json(result.get("raw_response", ""), label="query_parser")
        if isinstance(parsed, dict) and "domain" in parsed and "location" in parsed:
            domain   = str(parsed["domain"]).strip()   or "restaurants"
            location = str(parsed["location"]).strip() or "Tunisia"
            logger.info(f"[QueryParser] LLM parsed → domain='{domain}', location='{location}'")
            return {"domain": domain, "location": location}
    except Exception as exc:
        logger.warning(f"[QueryParser] LLM parse failed ({exc}), falling back to regex")

    # ── 2. Regex fallback ────────────────────────────────────────────────────
    for pattern in _LOCATION_PATTERNS:
        m = re.match(pattern, raw_query, re.IGNORECASE)
        if m:
            domain   = m.group(1).strip()
            location = m.group(2).strip()
            logger.info(f"[QueryParser] Regex parsed → domain='{domain}', location='{location}'")
            return {"domain": domain, "location": location}

    # ── 3. Final fallback ─────────────────────────────────────────────────────
    logger.warning(
        f"[QueryParser] Could not extract location from '{raw_query}' — "
        "defaulting location to 'Tunisia'"
    )
    return {"domain": raw_query, "location": "Tunisia"}
