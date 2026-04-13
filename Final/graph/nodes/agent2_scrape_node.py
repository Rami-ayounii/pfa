"""
graph/nodes/agent2_scrape_node.py
══════════════════════════════════
LangGraph fan-out node: scrape ONE brand's social profile.

This node is the target of the Send() fan-out in pipeline_graph.py.
Each brand is processed concurrently (LangGraph handles parallelism via threads).

Input  (sub-state from Send):
    brand, location, output_dir
Output (merged into parent state via Annotated[list, operator.add]):
    profiles += [one profile dict]
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Final/

from loguru import logger
from agents.agent2_react import run_agent2_react


def agent2_scrape_node(state: dict) -> dict:
    """
    Fan-out node: scrape a single brand using the social scraper pipeline.
    scrape_single_brand is sync; safe to call directly from a LangGraph node.
    """
    brand      = state.get("brand", "")
    domain     = state.get("domain", "")
    location   = state.get("location", "Tunisia")
    output_dir = state.get("output_dir", "geo_output")

    from core.status_tracker import emit as _emit, increment as _increment
    if not brand:
        logger.warning("[Agent2ScrapeNode] Empty brand name — skipping")
        return {"profiles": []}

    brands      = state.get("brands", [])
    total_brands = len(brands) if brands else 0
    _emit("agent2_scrape", "active",
          {"brand": brand, "total_brands": total_brands, "completed_brands": 0},
          output_dir=output_dir)

    logger.info(f"[Agent2ScrapeNode] Scraping '{brand}' | location='{location}'")

    # API keys from env (all optional — scraper degrades gracefully without them)
    serpapi_key    = os.environ.get("SERPAPI_KEY", "")
    apify_token    = os.environ.get("APIFY_API_TOKEN", "") or os.environ.get("APIFY_TOKEN", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    ig_user        = os.environ.get("IG_USERNAME", "")
    ig_pass        = os.environ.get("IG_PASSWORD", "")

    try:
        rows = run_agent2_react([brand])
        row  = rows[0] if rows else {}

        # Normalise to the profile dict shape the rest of the pipeline expects
        profile: dict = dict(row)
        profile["brand"]            = profile.pop("canonical_entity", None) or brand
        profile["confidence_score"] = profile.pop("overall_confidence", 0.0)
        profile["status"]           = "ok"

        logger.success(
            f"[Agent2ScrapeNode] '{brand}' done — "
            f"ig={profile.get('ig_handle')}, "
            f"gm_rating={profile.get('gm_rating')}"
        )
        completed = _increment("agent2_scrape", "completed_brands", output_dir)
        # Only emit "done" when ALL brands have finished; otherwise stay "active".
        final_status = "done" if (total_brands > 0 and completed >= total_brands) else "active"
        _emit("agent2_scrape", final_status,
              {"brand": brand, "confidence": profile.get("confidence_score", 0),
               "completed_brands": completed, "total_brands": total_brands},
              output_dir=output_dir)
        return {"profiles": [profile]}

    except TimeoutError:
        logger.warning(f"[Agent2ScrapeNode] Timeout scraping '{brand}' — skipping")
        completed = _increment("agent2_scrape", "completed_brands", output_dir)
        final_status = "done" if (total_brands > 0 and completed >= total_brands) else "active"
        _emit("agent2_scrape", final_status,
              {"brand": brand, "completed_brands": completed,
               "total_brands": total_brands, "last_error": f"timeout: {brand}"},
              output_dir=output_dir)
        return {"profiles": [], "errors": [f"Agent2 timeout for '{brand}'"]}

    except Exception as exc:
        logger.error(f"[Agent2ScrapeNode] Failed scraping '{brand}': {exc}")
        completed = _increment("agent2_scrape", "completed_brands", output_dir)
        final_status = "done" if (total_brands > 0 and completed >= total_brands) else "active"
        _emit("agent2_scrape", final_status,
              {"brand": brand, "completed_brands": completed,
               "total_brands": total_brands, "last_error": str(exc)},
              output_dir=output_dir)
        return {"profiles": [], "errors": [f"Agent2 failed for '{brand}': {exc}"]}


def agent2_all_node(state: dict) -> dict:
    """Single batched node: research ALL brands in one MCP client session.

    Replaces the per-brand fan-out so MCP subprocesses (search, scrape, wiki)
    are started once and reused across all brands instead of being spawned and
    killed once per brand.
    """
    brands     = state.get("brands", [])
    output_dir = state.get("output_dir", "geo_output")
    errors: list = []

    from core.status_tracker import emit as _emit
    if not brands:
        logger.warning("[Agent2AllNode] No brands — skipping")
        return {"profiles": [], "current_step": "agent2_skipped"}

    logger.info(f"[Agent2AllNode] Researching {len(brands)} brands in one session")
    _emit("agent2_scrape", "active", {"total_brands": len(brands), "completed_brands": 0},
          output_dir=output_dir)

    try:
        rows = run_agent2_react(brands)
    except Exception as exc:
        logger.error(f"[Agent2AllNode] run_agent2_react failed: {exc}")
        errors.append(f"Agent2 failed: {exc}")
        _emit("agent2_scrape", "done", {"total_brands": len(brands), "completed_brands": 0,
              "last_error": str(exc)}, output_dir=output_dir)
        return {"profiles": [], "errors": errors, "current_step": "agent2"}

    profiles = []
    for row in rows:
        profile = dict(row)
        profile["brand"]            = profile.pop("canonical_entity", None) or ""
        profile["confidence_score"] = profile.pop("overall_confidence", 0.0)
        profile["status"]           = "ok"
        profiles.append(profile)

    logger.success(f"[Agent2AllNode] Done — {len(profiles)} profiles collected")
    _emit("agent2_scrape", "done",
          {"total_brands": len(brands), "completed_brands": len(profiles)},
          output_dir=output_dir)
    return {"profiles": profiles, "errors": errors, "current_step": "agent2"}
