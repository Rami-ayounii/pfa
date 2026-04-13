"""MCP Scrape Server — Apify (Instagram, Facebook, TripAdvisor), website social extraction.

Run as a standalone MCP server:
    python mcp_servers/scrape_server.py

Tools exposed:
    scrape_instagram(handle)             -> dict
    scrape_facebook(handle)              -> dict
    scrape_tripadvisor(entity, location) -> dict
    extract_website_socials(url)         -> dict

All Apify tools raise ValueError if APIFY_API_TOKEN is not configured.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import APIFY_API_TOKEN

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("scrape-server")


def _require_apify():
    """Raise a clear error if APIFY_API_TOKEN is missing or looks invalid."""
    if not APIFY_API_TOKEN or len(APIFY_API_TOKEN) < 10:
        raise ValueError(
            "APIFY_API_TOKEN is not configured or is too short. "
            "Set it in Final/.env to enable Instagram, Facebook, and TripAdvisor scraping."
        )


@mcp.tool()
def scrape_instagram(handle: str) -> dict:
    """Scrape an Instagram profile via Apify.

    Returns: ig_followers, ig_posts, ig_engagement_rate, ig_bio, ig_scraper.
    Raises ValueError if APIFY_API_TOKEN is not set.
    """
    _require_apify()

    result = {
        "ig_followers": 0, "ig_posts": 0, "ig_engagement_rate": 0.0,
        "ig_bio": "", "ig_scraper": "none",
    }
    if not handle:
        return result

    from apify_client import ApifyClient
    client = ApifyClient(APIFY_API_TOKEN)

    run = client.actor("apify/instagram-profile-scraper").call(
        run_input={"usernames": [handle], "resultsLimit": 1},
        timeout_secs=60,
    )
    if not run:
        return result

    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    if items:
        r = items[0]
        result["ig_followers"]       = r.get("followersCount", 0)
        result["ig_posts"]           = r.get("postsCount", 0)
        result["ig_engagement_rate"] = r.get("engagementRate", 0.0)
        result["ig_bio"]             = (r.get("biography") or "")[:200]
        result["ig_scraper"]         = "apify"

    return result


@mcp.tool()
def scrape_facebook(handle: str) -> dict:
    """Scrape a Facebook page via Apify.

    Returns: fb_page_likes, fb_post_engagement, fb_scraper.
    Raises ValueError if APIFY_API_TOKEN is not set.
    """
    _require_apify()

    result = {"fb_page_likes": 0, "fb_followers": 0, "fb_post_engagement": 0.0, "fb_scraper": "none"}
    if not handle:
        return result

    from apify_client import ApifyClient
    client = ApifyClient(APIFY_API_TOKEN)

    run = client.actor("apify/facebook-pages-scraper").call(
        run_input={
            "startUrls": [{"url": f"https://www.facebook.com/{handle}"}],
            "resultsLimit": 10,
            "maxPostComments": 0,
            "maxReviews": 0,
        },
        timeout_secs=60,
    )
    if not run:
        return result

    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    if items:
        page  = items[0]
        posts = page.get("posts", []) or []
        result["fb_page_likes"] = page.get("likes", 0) or 0
        result["fb_followers"]  = page.get("followers", 0) or 0
        if posts:
            total = sum(
                (p.get("likes", 0) or 0) + (p.get("comments", 0) or 0) +
                (p.get("shares", 0) or 0) for p in posts
            )
            result["fb_post_engagement"] = round(total / len(posts), 1)
        result["fb_scraper"] = "apify"

    return result


@mcp.tool()
def scrape_tripadvisor(entity: str, location: str = "Tunisia") -> dict:
    """Search TripAdvisor via Apify for a restaurant entity.

    Returns: ta_name, ta_rating, ta_review_count, ta_url, ta_price_range,
             ta_cuisine, ta_scraper.
    Raises ValueError if APIFY_API_TOKEN is not set.
    """
    _require_apify()

    result: dict = {
        "ta_name": None, "ta_rating": None, "ta_review_count": None,
        "ta_url": None, "ta_price_range": None, "ta_cuisine": None,
        "ta_scraper": "none",
    }
    if not entity:
        return result

    from apify_client import ApifyClient
    client = ApifyClient(APIFY_API_TOKEN)

    run = client.actor("maxcopell/tripadvisor").call(
        run_input={
            "query":    f"{entity} {location}",
            "language": "en",
            "currency": "USD",
            "maxItems": 3,
        },
        timeout_secs=60,
    )
    if not run:
        return result

    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    if not items:
        return result

    # Pick the first restaurant-type result
    item = None
    for candidate in items:
        category = (candidate.get("type") or candidate.get("category") or "").lower()
        if "restaurant" in category or "food" in category or not category:
            item = candidate
            break
    if item is None:
        item = items[0]

    result["ta_name"]         = item.get("name")
    result["ta_url"]          = item.get("url") or item.get("webUrl")
    result["ta_price_range"]  = item.get("priceRange") or item.get("priceLevelFormatted")
    result["ta_cuisine"]      = (
        ", ".join(item.get("cuisines", []))[:80] if item.get("cuisines")
        else item.get("cuisine")
    )

    # Rating — handle both float and string forms
    raw_rating = item.get("rating") or item.get("ratingImageUrl")
    if isinstance(raw_rating, (int, float)):
        result["ta_rating"] = float(raw_rating)
    elif isinstance(raw_rating, str):
        m = re.search(r"(\d+[.,]\d+|\d+)", raw_rating)
        if m:
            result["ta_rating"] = float(m.group(1).replace(",", "."))

    # Review count
    raw_reviews = item.get("numberOfReviews") or item.get("reviewsCount")
    if isinstance(raw_reviews, int):
        result["ta_review_count"] = raw_reviews
    elif isinstance(raw_reviews, str):
        m = re.search(r"([\d,]+)", raw_reviews)
        if m:
            result["ta_review_count"] = int(m.group(1).replace(",", ""))

    result["ta_scraper"] = "apify"

    return result


@mcp.tool()
def extract_website_socials(website_url: str) -> dict:
    """Fetch a website's HTML and extract Instagram and Facebook handles.

    Returns: website_ig, website_fb (handles as strings, or None).
    """
    socials: dict[str, str | None] = {"website_ig": None, "website_fb": None}
    if not website_url:
        return socials

    try:
        import httpx
        r = httpx.get(
            website_url, timeout=10, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        html = r.text

        ig = re.search(
            r'href=["\']?https?://(?:www\.)?instagram\.com/([A-Za-z0-9_.]{3,30})', html
        )
        fb = re.search(
            r'href=["\']?https?://(?:www\.)?facebook\.com/([A-Za-z0-9_.]{3,50})', html
        )

        if ig:
            socials["website_ig"] = ig.group(1).lower()
        if fb:
            socials["website_fb"] = fb.group(1).lower()

    except Exception as e:
        socials["status"] = "error"
        socials["error"]  = str(e)
        return socials

    return socials


if __name__ == "__main__":
    mcp.run(transport="stdio")
