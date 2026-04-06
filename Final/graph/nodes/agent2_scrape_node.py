"""
graph/nodes/agent2_scrape_node.py
══════════════════════════════════
LangGraph fan-out node: scrape ONE brand's social profile using ReAct agent.

This node is the target of the Send() fan-out in pipeline_graph.py.
Each brand is processed concurrently (LangGraph handles parallelism).

Input  (sub-state from Send):
    brand, location, output_dir

Output (merged into parent state via Annotated[list, operator.add]):
    profiles += [one profile dict]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Final/ is root

import asyncio
import concurrent.futures
from loguru import logger


def agent2_scrape_node(state: dict) -> dict:
    """
    Fan-out node: scrape a single brand end-to-end using the ReAct agent.
    Returns one profile dict added to the fan-in accumulator.
    run_agent2_react is async; bridge to sync using a ThreadPoolExecutor.
    """
    brand    = state.get("brand", "")
    domain   = state.get("domain", "")
    location = state.get("location", "Tunisia")

    if not brand:
        logger.warning("[Agent2ScrapeNode] Empty brand name — skipping")
        return {"profiles": []}

    logger.info(f"[Agent2ScrapeNode] Scraping brand: '{brand}' in '{location}'")

    try:
        from agents.agent2_react import run_agent2_react  # lazy — avoids langchain_groq at import time
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, run_agent2_react([brand]))
            results = future.result(timeout=180)
        profile = results[0] if results else {"brand": brand, "status": "no_data"}
        logger.success(
            f"[Agent2ScrapeNode] '{brand}' done — status={profile.get('status', 'ok')}"
        )
        return {"profiles": [profile]}

    except concurrent.futures.TimeoutError:
        logger.warning(f"[Agent2ScrapeNode] Timeout scraping '{brand}' after 180 s — skipping")
        return {"profiles": [], "errors": [f"Agent2 scrape timed out for '{brand}'"]}

    except Exception as exc:
        logger.error(f"[Agent2ScrapeNode] Failed scraping '{brand}': {exc}")
        return {"profiles": [], "errors": [f"Agent2 scrape failed for '{brand}': {exc}"]}
