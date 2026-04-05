"""
graph/nodes/agent2_scrape_node.py
══════════════════════════════════
LangGraph fan-out node: scrape ONE brand's social profile.

This node is the target of the Send() fan-out in pipeline_graph.py.
Each brand is processed concurrently (LangGraph handles parallelism).

Input  (sub-state from Send):
    brand, location, output_dir, serpapi_key, apify_token,
    openrouter_key, groq_client, ig_user, ig_pass, is_colab

Output (merged into parent state via Annotated[list, operator.add]):
    profiles += [one SocialProfile]
"""

from loguru import logger
from agents.agent2_social_scraper import scrape_single_brand


def agent2_scrape_node(state: dict) -> dict:
    """
    Fan-out node: scrape a single brand end-to-end (steps 1–7 per brand).
    Returns one SocialProfile added to the fan-in accumulator.
    """
    brand          = state.get("brand", "")
    location       = state.get("location", "Tunisia")
    output_dir     = state.get("output_dir", "geo_output/agent2_output")
    serpapi_key    = state.get("serpapi_key", "")
    apify_token    = state.get("apify_token", "")
    openrouter_key = state.get("openrouter_key", "")
    groq_client    = state.get("groq_client")
    ig_user        = state.get("ig_user", "")
    ig_pass        = state.get("ig_pass", "")
    is_colab       = state.get("is_colab", False)

    if not brand:
        logger.warning("[Agent2ScrapeNode] Empty brand name — skipping")
        return {"profiles": []}

    logger.info(f"[Agent2ScrapeNode] Scraping brand: '{brand}' in '{location}'")

    try:
        profile = scrape_single_brand(
            brand          = brand,
            location       = location,
            output_dir     = output_dir,
            serpapi_key    = serpapi_key,
            apify_token    = apify_token,
            openrouter_key = openrouter_key,
            groq_client    = groq_client,
            ig_user        = ig_user,
            ig_pass        = ig_pass,
            is_colab       = is_colab,
        )
        logger.success(
            f"[Agent2ScrapeNode] '{brand}' done — "
            f"authority={getattr(profile, 'social_authority_score', 'N/A')}"
        )
        return {"profiles": [profile]}

    except Exception as exc:
        logger.error(f"[Agent2ScrapeNode] Failed scraping '{brand}': {exc}")
        return {"profiles": [], "errors": [f"Agent2 scrape failed for '{brand}': {exc}"]}
