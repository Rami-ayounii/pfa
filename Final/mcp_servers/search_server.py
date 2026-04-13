"""MCP Search Server — Google Maps (Selenium), DuckDuckGo.

Run as a standalone MCP server:
    python mcp_servers/search_server.py

Tools exposed:
    google_maps_search(entity, location) -> dict
    ddg_search(query, max_results)       -> list[dict]
"""

import re
import sys
import time
import random
from pathlib import Path
from urllib.parse import quote_plus

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("search-server")

# ---------------------------------------------------------------------------
# Reusable browser driver — created once per process, health-checked on reuse.
# Uses Edge (always present on Windows 11) with Chrome as fallback.
# Avoids the ~3s browser startup cost on every google_maps_search call.
# ---------------------------------------------------------------------------

_chrome_driver = None

_COMMON_ARGS = [
    "--headless=new",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-blink-features=AutomationControlled",
    (
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
]


def _get_driver():
    """Return a live browser driver (Edge preferred, Chrome fallback), recreating if crashed.

    Uses Selenium 4's built-in selenium-manager to auto-download the correct
    driver binary — no webdriver_manager required.
    """
    global _chrome_driver
    if _chrome_driver is not None:
        try:
            _ = _chrome_driver.current_url   # raises if driver is dead
            return _chrome_driver
        except Exception:
            try:
                _chrome_driver.quit()
            except Exception:
                pass
            _chrome_driver = None

    from selenium import webdriver

    # --- Try Edge first (always installed on Windows 11) ---
    try:
        from selenium.webdriver.edge.options import Options as EdgeOptions

        opts = EdgeOptions()
        for arg in _COMMON_ARGS:
            opts.add_argument(arg)
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        # No Service() — selenium-manager resolves the driver automatically
        _chrome_driver = webdriver.Edge(options=opts)
        return _chrome_driver
    except Exception:
        pass  # fall through to Chrome

    # --- Fallback: Chrome ---
    from selenium.webdriver.chrome.options import Options as ChromeOptions

    opts = ChromeOptions()
    for arg in _COMMON_ARGS:
        opts.add_argument(arg)
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    _chrome_driver = webdriver.Chrome(options=opts)
    return _chrome_driver


def _text(driver, css: str) -> str:
    """Return stripped text of first matching element, or ''."""
    try:
        from selenium.webdriver.common.by import By
        return driver.find_element(By.CSS_SELECTOR, css).text.strip()
    except Exception:
        return ""


def _attr(driver, css: str, attribute: str) -> str:
    """Return an attribute value of first matching element, or ''."""
    try:
        from selenium.webdriver.common.by import By
        return driver.find_element(By.CSS_SELECTOR, css).get_attribute(attribute) or ""
    except Exception:
        return ""


@mcp.tool()
def google_maps_search(entity: str, location: str = "Tunisia") -> dict:
    """Search Google Maps via Selenium (headless Chrome) for a restaurant entity.

    Returns structured venue data: name, rating, review_count, address,
    phone, website, category, status.
    """
    result = {
        "gm_name": None, "gm_rating": None, "gm_review_count": None,
        "gm_address": None, "gm_phone": None, "gm_website": None,
        "gm_category": None, "status": "no_results",
    }

    try:
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except ImportError:
        result["status"] = "selenium_unavailable"
        return result

    try:
        driver = _get_driver()
        wait = WebDriverWait(driver, 12)

        query = f"{entity} restaurant {location}"
        driver.get("https://www.google.com/maps/search/" + quote_plus(query))

        # Wait for the main panel to appear
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='main']")))
        time.sleep(1.5)

        # If we landed on a results list, click the first entry.
        # Use href-based selector instead of obfuscated class names (a.hfpxzc rotates).
        try:
            first = driver.find_element(By.CSS_SELECTOR, "a[href*='/maps/place/']")
            first.click()
            time.sleep(2.5)
        except Exception:
            pass  # already on a place page

        # ── Name ─────────────────────────────────────────────────────────────
        # Prefer attribute-stable selectors; obfuscated classes like DUwDvf rotate.
        for sel in ["h1[data-attrid]", "h1[class*='fontHeadline']", "h1"]:
            name = _text(driver, sel)
            if name:
                result["gm_name"] = name
                break

        # ── Rating ───────────────────────────────────────────────────────────
        # aria-label is locale-dependent: "stars" (en), "étoiles" (fr), "نجوم" (ar).
        # Use a broad numeric-text fallback so any locale works.
        # The obfuscated class F7nice rotates — avoid relying on it alone.
        for sel in [
            "span[aria-label*='stars']",
            "span[aria-label*='étoiles']",
            "span[aria-label*='\u0646\u062c\u0648\u0645']",   # نجوم
            "div[data-item-id='rating'] span[aria-hidden='true']",
            "span[aria-hidden='true'][role]",
        ]:
            try:
                el = driver.find_element(By.CSS_SELECTOR, sel)
                label = el.get_attribute("aria-label") or el.text
                m = re.search(r"(\d+[.,]\d+|\d+)", label)
                if m:
                    candidate = float(m.group(1).replace(",", "."))
                    if 1.0 <= candidate <= 5.0:
                        result["gm_rating"] = candidate
                        break
            except Exception:
                pass

        # ── Review count ─────────────────────────────────────────────────────
        for sel in ["span[aria-label*='reviews']", "button[jsaction*='review']"]:
            try:
                el = driver.find_element(By.CSS_SELECTOR, sel)
                label = el.get_attribute("aria-label") or el.text
                m = re.search(r"([\d\s,]+)", label)
                if m:
                    result["gm_review_count"] = int(
                        re.sub(r"[\s,]", "", m.group(1))
                    )
                    break
            except Exception:
                pass

        # ── Address ──────────────────────────────────────────────────────────
        for sel in [
            "button[data-item-id='address']",
            "[data-tooltip='Copy address']",
            "button[aria-label*='ddress']",
        ]:
            addr = _text(driver, sel)
            if addr:
                result["gm_address"] = addr
                break

        # ── Phone ────────────────────────────────────────────────────────────
        for sel in [
            "button[data-item-id*='phone']",
            "[data-tooltip='Copy phone number']",
            "button[aria-label*='hone']",
        ]:
            phone = _text(driver, sel)
            if phone:
                result["gm_phone"] = phone
                break

        # ── Website ──────────────────────────────────────────────────────────
        website = _attr(driver, "a[data-item-id='authority']", "href")
        if website:
            result["gm_website"] = website

        # ── Category ─────────────────────────────────────────────────────────
        # DkEaL is an obfuscated class that rotates; prefer jsaction-based selector.
        for sel in [
            "button[jsaction*='category']",
            "span[jsaction*='category']",
            "button[data-item-id*='category']",
        ]:
            cat = _text(driver, sel)
            if cat:
                result["gm_category"] = cat[:80]
                break

        if result["gm_name"]:
            result["status"] = "success"

    except Exception as e:
        # Driver is in unknown state — force recreation on next call.
        global _chrome_driver
        try:
            driver.get("about:blank")
        except Exception:
            try:
                driver.quit()
            except Exception:
                pass
            _chrome_driver = None
        result["status"] = "error"
        result["error"] = str(e)[:300]
    else:
        # Navigate away to reset page state for the next call.
        try:
            driver.get("about:blank")
        except Exception:
            pass

    return result


@mcp.tool()
def ddg_search(query: str, max_results: int = 5) -> list:
    """General DuckDuckGo text search.

    Returns list of {title, href, body} dicts.
    Useful for finding Instagram/Facebook handles and general entity info.
    """
    time.sleep(random.uniform(1.0, 2.0))   # DDG rate-limit avoidance
    try:
        from ddgs import DDGS
        return list(DDGS().text(query, max_results=max_results))
    except Exception as e:
        return [{"status": "error", "error": str(e)}]


if __name__ == "__main__":
    mcp.run(transport="stdio")
