"""
agents/agent2_social_scraper.py
════════════════════════════════
Agent 2 — Social Profile Scraper & Brand Intelligence Synthesiser

Pipeline (8 steps):
  Step 1  · DuckDuckGo URL resolution (Wikipedia + TripAdvisor)
  Step 2a · Wikipedia summary + categories
  Step 2b · TripAdvisor rating + review count (DDG snippet → Apify fallback)
  Step 3  · Venue data: SerpApi Google Maps → OSM Overpass fallback
  Step 4  · Google Maps details via Selenium (rating, phone, address, website, socials)
  Step 5a · Instagram: Instaloader (free, anonymous) → Apify fallback
  Step 5b · Facebook:  Apify → DDG snippet parse fallback
  Step 6  · Hallucination checker (rules + LLM semantic validation)
  Step 7  · LLM reflection & synthesis (social authority score, GEO recommendations)
  Step 8  · Export CSV + JSON

Input:  brand list (from Agent 1 output or manual)
Output: social_profiles.csv | hallucination_flags.csv | audit_log.json
"""

import os, re, sys, json, time, random, hashlib, requests
from typing  import Optional, Any
from dataclasses import dataclass, field
from datetime    import datetime
from pathlib     import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.llm_client import (
    query_llm, query_openrouter, cached_llm_call, parse_json,
    MODEL_FAST, MODEL_STRONG,
)

try:    from rapidfuzz import fuzz
except: fuzz = None
try:    from ddgs import DDGS
except: DDGS = None

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.common.by import By
    from webdriver_manager.chrome import ChromeDriverManager
    _SELENIUM_AVAILABLE = True
except Exception:
    _SELENIUM_AVAILABLE = False


# ── Tuning constants ──────────────────────────────────────────────────────────
DDGS_MAX_RESULTS     = 10
REQUEST_TIMEOUT      = 30
DELAY_MIN, DELAY_MAX = 0.5, 1.5
REFLECTION_THRESHOLD = 0.65
MAX_TOKENS           = 1024
SE_USER_AGENT        = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36")

def _delay(mn=DELAY_MIN, mx=DELAY_MAX): time.sleep(random.uniform(mn, mx))
def _is_key(k): return bool(k and not k.endswith(("HERE","TOKEN","_KEY_HERE")))


# ═══════════════════════════════════════════════════════════════════════════════
# Data models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HallucinationFlag:
    field_name:        str   = ""
    claimed_value:     Any   = None
    confidence_score:  float = 1.0
    validation_method: str   = ""
    anomaly_detected:  bool  = False
    anomaly_reason:    str   = ""
    timestamp:         str   = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SocialProfile:
    brand:              str   = ""

    # Wikipedia
    wiki_summary:       str   = ""
    wiki_categories:    list  = field(default_factory=list)

    # TripAdvisor
    ta_rating:          Optional[float] = None
    ta_review_count:    Optional[int]   = None
    ta_url:             str   = ""

    # Google Maps
    gm_rating:          Optional[float] = None
    gm_review_count:    Optional[int]   = None
    gm_address:         str   = ""
    gm_phone:           str   = ""
    gm_website:         str   = ""
    gm_facebook_handle: str   = ""
    gm_instagram_handle:str   = ""

    # Venue (SerpApi / OSM)
    fsq_venue_id:       str   = ""
    fsq_category:       str   = ""
    fsq_checkins:       Optional[int]   = None
    fsq_price_tier:     Optional[int]   = None
    fsq_rating:         Optional[float] = None
    venue_data_source:  str   = ""

    # Instagram
    ig_handle:          str   = ""
    ig_followers:       int   = 0
    ig_posts:           int   = 0
    ig_engagement_rate: float = 0.0
    ig_bio:             str   = ""
    ig_data:            dict  = field(default_factory=dict)
    ig_scraper_used:    str   = ""

    # Facebook
    fb_handle:          str   = ""
    fb_page_likes:      int   = 0
    fb_post_engagement: float = 0.0
    fb_data:            dict  = field(default_factory=dict)
    fb_scraper_used:    str   = ""

    # Synthesis
    social_authority_score: Optional[float] = None
    geo_recommendations:    list = field(default_factory=list)
    synthesis_notes:        str  = ""
    score_breakdown:        dict = field(default_factory=dict)

    # Audit
    hallucination_flags: list  = field(default_factory=list)
    overall_confidence:  float = 1.0
    correction_history:  list  = field(default_factory=list)
    reflection_rounds:   int   = 0
    final_llm_used:      str   = ""


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — DuckDuckGo URL resolution
# ═══════════════════════════════════════════════════════════════════════════════

def step1_resolve_urls(brand: str, location: str = "Tunisia") -> dict:
    result = {"wikipedia_url": "", "tripadvisor_url": ""}
    if DDGS is None:
        print("  [WARN] ddgs not installed - skipping URL resolution"); return result
    query = f"{brand} restaurant {location}"
    print(f"  [Step 1] DuckDuckGo: '{query}'")
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=DDGS_MAX_RESULTS))
        _delay()
        for hit in hits:
            url = hit.get("href", "")
            if not result["wikipedia_url"]   and "wikipedia.org"                       in url:
                result["wikipedia_url"]   = url
            if not result["tripadvisor_url"] and "tripadvisor.com/Restaurant_Review"   in url:
                result["tripadvisor_url"] = url
            if result["wikipedia_url"] and result["tripadvisor_url"]: break
    except Exception as e:
        print(f"  [WARN] DuckDuckGo error: {e}")
    print(f"    Wikipedia  : {result['wikipedia_url']  or '(not found)'}")
    print(f"    TripAdvisor: {result['tripadvisor_url'] or '(not found)'}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2a — Wikipedia
# ═══════════════════════════════════════════════════════════════════════════════

def step2a_wikipedia(profile: SocialProfile, wiki_url: str) -> None:
    if not wiki_url: return
    m = re.search(r"/wiki/([^#?]+)", wiki_url)
    if not m: return
    page_title = m.group(1).replace("_", " ")
    print(f"  [Step 2a] Wikipedia: '{page_title}'")
    try:
        import wikipediaapi
        for lang in ["en", "fr"]:
            wiki = wikipediaapi.Wikipedia(
                user_agent="GEOAgent/1.0 (research)", language=lang)
            page = wiki.page(page_title)
            if page.exists():
                profile.wiki_summary    = page.summary[:500]
                profile.wiki_categories = list(page.categories.keys())[:10]
                print(f"    ({lang}) {profile.wiki_summary[:80]}...")
                break
        else:
            print("    Not found in English or French")
    except Exception as e:
        print(f"  [WARN] Wikipedia error: {e}")
    _delay(0.3, 0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2b — TripAdvisor  (DDG snippet → Apify fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def step2b_tripadvisor(profile: SocialProfile, ta_url: str,
                       apify_token: str = "",
                       location: str = "Tunisia") -> None:
    print(f"  [Step 2b] TripAdvisor: '{profile.brand}'")
    if DDGS:
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(
                    f"{profile.brand} restaurant tripadvisor rating reviews {location}",
                    max_results=8))
            for hit in hits:
                if "tripadvisor.com" not in hit.get("href", ""): continue
                profile.ta_url     = hit.get("href", ta_url)
                snippet            = hit.get("body", "")
                m_r = re.search(r"(\d\.\d)\s*(?:of\s*5)?", snippet)
                m_n = re.search(r"([\d,]+)\s*reviews?", snippet, re.IGNORECASE)
                if m_r: profile.ta_rating       = float(m_r.group(1))
                if m_n: profile.ta_review_count = int(m_n.group(1).replace(",", ""))
                if profile.ta_rating:
                    print(f"    [OK] {profile.ta_rating} ({profile.ta_review_count} reviews) [DDG]")
                    _delay(0.3, 0.7); return
            print("    DDG: no rating in snippets")
        except Exception as e:
            print(f"    DDG error: {e}")

    if not ta_url or not _is_key(apify_token):
        print("    No TripAdvisor URL or Apify token - skipping"); return
    print("    Trying Apify fallback ...")
    try:
        from apify_client import ApifyClient
        run   = ApifyClient(apify_token).actor("maxcopell/tripadvisor").call(
            run_input={"startUrls": [{"url": ta_url}],
                       "maxItemsPerQuery": 1, "includeReviews": False},
            timeout_secs=120)
        items = list(ApifyClient(apify_token).dataset(
            run["defaultDatasetId"]).iterate_items())
        if items:
            item = items[0]
            profile.ta_rating       = (item.get("rating", {}).get("ratingValue")
                                       or item.get("ratingValue"))
            profile.ta_review_count = (item.get("numberOfReviews")
                                       or item.get("reviewCount"))
            print(f"    [OK] {profile.ta_rating} ({profile.ta_review_count} reviews) [Apify]")
    except Exception as e:
        print(f"    Apify error: {e}")
    _delay(0.3, 0.7)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Venue data  (SerpApi Google Maps → OSM fallback)
# ═══════════════════════════════════════════════════════════════════════════════

def _serpapi_places(profile: SocialProfile, brand: str,
                    serpapi_key: str, location: str = "Tunisia") -> bool:
    if not _is_key(serpapi_key): return False
    print("    Trying SerpApi Google Maps ...")
    try:
        from serpapi import GoogleSearch
        results = GoogleSearch({"engine": "google_maps", "q": f"{brand} {location}",
                                "type": "search", "api_key": serpapi_key}).get_dict()
        places  = results.get("local_results", [])
        if not places: return False
        place   = places[0]
        profile.fsq_venue_id   = place.get("place_id", "")
        profile.fsq_rating     = place.get("rating")
        profile.fsq_checkins   = place.get("reviews")
        profile.fsq_price_tier = len(place.get("price", ""))
        types_raw = place.get("types", [])
        if types_raw and isinstance(types_raw[0], dict):
            profile.fsq_category = ", ".join(t.get("name","") for t in types_raw)[:60]
        elif types_raw and isinstance(types_raw[0], str):
            profile.fsq_category = ", ".join(types_raw)[:60]
        else:
            profile.fsq_category = place.get("type", "")
        profile.venue_data_source = "serpapi"
        if not profile.gm_phone:   profile.gm_phone   = place.get("phone",   "")
        if not profile.gm_website: profile.gm_website = place.get("website", "")
        if not profile.gm_address: profile.gm_address = place.get("address", "")
        print(f"    [OK] {place.get('title','?')} | {profile.fsq_rating}* "
              f"({profile.fsq_checkins} reviews) [SerpApi]")
        return True
    except Exception as e:
        print(f"    SerpApi error: {e}"); return False


def _osm_fallback(profile: SocialProfile, brand: str,
                  location: str = "Tunisia") -> bool:
    print("    Trying OSM Overpass ...")
    try:
        import httpx
        query = (f'[out:json][timeout:25];area["name:en"="{location}"]->.a;'
                 f'(node["name"~"{brand}",i]["amenity"](area.a);'
                 f'way["name"~"{brand}",i]["amenity"](area.a););out body;')
        r = httpx.post("https://overpass-api.de/api/interpreter",
                       data={"data": query}, timeout=30)
        r.raise_for_status()
        elements = r.json().get("elements", [])
        if not elements: return False
        tags = elements[0].get("tags", {})
        profile.fsq_venue_id      = str(elements[0].get("id",""))
        profile.fsq_category      = tags.get("cuisine", tags.get("amenity",""))
        profile.venue_data_source = "osm"
        if not profile.gm_phone:   profile.gm_phone   = tags.get("phone","")
        if not profile.gm_website: profile.gm_website = tags.get("website","")
        print(f"    [OK] {tags.get('name','?')} [OSM]  [WARN] no ratings")
        return True
    except Exception as e:
        print(f"    OSM error: {e}"); return False


def step3_venue_data(profile: SocialProfile, brand: str,
                     serpapi_key: str = "", location: str = "Tunisia") -> None:
    print(f"  [Step 3] Venue data: '{brand}'")
    if not _serpapi_places(profile, brand, serpapi_key, location):
        _osm_fallback(profile, brand, location=location)
    if not profile.venue_data_source:
        profile.venue_data_source = "none"
        print("    [WARN] No venue data found")
    _delay(0.3, 0.7)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 — Google Maps via Selenium
# ═══════════════════════════════════════════════════════════════════════════════

def step4_google_maps(profile: SocialProfile, brand: str,
                      location: str = "Tunisia", is_colab: bool = False) -> None:
    print(f"  [Step 4] Google Maps: '{brand}'")

    # Skip Selenium if SerpApi already filled core data
    if profile.venue_data_source == "serpapi" and profile.fsq_rating is not None:
        print("    SerpApi already provided venue data - skipping Selenium")
        if profile.gm_rating is None:     profile.gm_rating       = profile.fsq_rating
        if profile.gm_review_count is None: profile.gm_review_count = profile.fsq_checkins
        return

    if not _SELENIUM_AVAILABLE:
        print("    [SKIP] Selenium not installed - step 4 skipped "
              "(pip install selenium webdriver-manager)")
        return

    # Detect Chrome binary on Windows if not in Colab
    import shutil, platform
    def _find_chrome_binary() -> str:
        if platform.system() == "Windows":
            candidates = [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
            ]
            for path in candidates:
                if os.path.exists(path):
                    return path
        return shutil.which("google-chrome") or shutil.which("chromium-browser") or ""

    try:
        opts = ChromeOptions()
        for arg in ["--headless","--no-sandbox","--disable-dev-shm-usage",
                    "--disable-gpu","--window-size=1920,1080"]:
            opts.add_argument(arg)
        if is_colab:
            opts.binary_location = "/usr/bin/google-chrome"
        else:
            chrome_bin = _find_chrome_binary()
            if not chrome_bin:
                print("    [SKIP] Chrome binary not found - step 4 skipped. "
                      "Install Chrome or set opts.binary_location manually.")
                return
            opts.binary_location = chrome_bin

        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()),
                                  options=opts)
        try:
            query = f"{brand} restaurant {location}".strip()
            driver.get(f"https://www.google.com/maps/search/{query.replace(' ','+')}")
            time.sleep(3)

            def _text(sel, default=""):
                try: return driver.find_element(By.CSS_SELECTOR, sel).text.strip()
                except: return default

            def _attr(sel, att, default=""):
                try: return driver.find_element(By.CSS_SELECTOR, sel).get_attribute(att)
                except: return default

            rating_text = _text("div.F7nice span[aria-hidden='true']")
            try:    profile.gm_rating = float(rating_text.replace(",","."))
            except: pass

            rev_aria = _attr("div.F7nice span[aria-label]", "aria-label")
            m = re.search(r"([\d,]+)", rev_aria)
            if m:
                try: profile.gm_review_count = int(m.group(1).replace(",",""))
                except: pass

            profile.gm_address = _attr("button[data-tooltip='Copy address']",
                                        "aria-label", "").replace("Address: ","")
            profile.gm_phone   = _attr("button[data-tooltip='Copy phone number']",
                                        "aria-label", "").replace("Phone: ","")
            profile.gm_website = _attr("a[data-tooltip='Open website']", "href", "")

            # Extract social links from the brand website
            if profile.gm_website:
                driver.get(profile.gm_website)
                time.sleep(2)
                for link in driver.find_elements(By.TAG_NAME, "a"):
                    href = (link.get_attribute("href") or "").lower()
                    if "facebook.com"  in href and not profile.gm_facebook_handle:
                        m = re.search(r"facebook\.com/([^/?#\s]+)", href)
                        if m: profile.gm_facebook_handle  = m.group(1)
                    if "instagram.com" in href and not profile.gm_instagram_handle:
                        m = re.search(r"instagram\.com/([^/?#\s]+)", href)
                        if m: profile.gm_instagram_handle = m.group(1)

            print(f"    [OK] {profile.gm_rating}* | {profile.gm_address or '(no address)'}")
        finally:
            driver.quit()
    except Exception as e:
        print(f"    Selenium error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5a — Instagram  (Instaloader → Apify fallback)
# ═══════════════════════════════════════════════════════════════════════════════

IG_BLACKLIST = {"popular","trending","explore","reel","reels","stories",
                "instagram","highlights","follow","top","best","p",
                "accounts","directory","about","tv","tags","locations"}

def _ig_handle_plausible(handle: str, brand: str) -> bool:
    if not handle or not brand: return False
    h = handle.lower().replace("_","").replace(".","")
    b = brand.lower().replace(" ","").replace("-","").replace("'","")
    if b in h or h in b: return True
    if fuzz: return fuzz.partial_ratio(h, b) >= 60
    return len(set(h) & set(b)) >= 3

def _resolve_ig_handle(brand: str, profile: SocialProfile) -> str:
    if profile.gm_instagram_handle: return profile.gm_instagram_handle
    if DDGS:
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(f'"{brand}" site:instagram.com', max_results=5))
            for hit in hits:
                m = re.search(r"instagram\.com/([a-zA-Z0-9_.]{3,30})/?",
                               hit.get("href",""))
                if m:
                    h = m.group(1).lower()
                    if h not in IG_BLACKLIST and not h.startswith("p/"):
                        return h
        except Exception: pass
    return ""

_instaloader_instance = None

def _get_instaloader(ig_user="", ig_pass=""):
    global _instaloader_instance
    if _instaloader_instance: return _instaloader_instance
    import instaloader
    L = instaloader.Instaloader(
        download_pictures=False, download_videos=False,
        download_video_thumbnails=False, download_geotags=False,
        download_comments=False, save_metadata=False,
        compress_json=False, quiet=True)
    if ig_user and ig_pass:
        try: L.login(ig_user, ig_pass)
        except Exception as e: print(f"  [WARN] Instaloader login failed: {e}")
    else: print("  [INFO] Instaloader anonymous mode")
    _instaloader_instance = L
    return L

def _ig_instaloader(handle: str, ig_user="", ig_pass="") -> dict:
    try:
        import instaloader
        L = _get_instaloader(ig_user, ig_pass)
        p = instaloader.Profile.from_username(L.context, handle)
        return {"followers": p.followers, "following": p.followees,
                "posts": p.mediacount, "engagement_rate": 0.0,
                "bio": (p.biography or "")[:150],
                "is_verified": p.is_verified,
                "is_business": p.is_business_account,
                "business_category": p.business_category_name or "",
                "source": "instaloader"}
    except Exception as e:
        print(f"    Instaloader error: {e}")
        if "rate limit" in str(e).lower():
            print("    Rate limited - waiting 30s ..."); time.sleep(30)
        return {}

def _ig_apify(handle: str, apify_token: str) -> dict:
    if not _is_key(apify_token): return {}
    try:
        from apify_client import ApifyClient
        run     = ApifyClient(apify_token).actor("apify/instagram-profile-scraper").call(
            run_input={"usernames": [handle], "resultsLimit": 1}, timeout_secs=300)
        dataset = list(ApifyClient(apify_token).dataset(
            run["defaultDatasetId"]).iterate_items())
        if not dataset: return {}
        r = dataset[0]
        return {"followers": r.get("followersCount",0),
                "following": r.get("followingCount",0),
                "posts": r.get("postsCount",0),
                "engagement_rate": r.get("engagementRate",0),
                "bio": r.get("biography","")[:150],
                "is_verified": r.get("verified",False),
                "is_business": r.get("isBusinessAccount",False),
                "business_category": r.get("businessCategoryName",""),
                "source": "apify"}
    except Exception as e:
        print(f"    Apify IG error: {e}"); return {}

def step5a_instagram(profile: SocialProfile, brand: str,
                     apify_token: str = "", ig_user: str = "",
                     ig_pass: str = "") -> None:
    print(f"  [Step 5a] Instagram: '{brand}'")
    handle = _resolve_ig_handle(brand, profile)
    if handle and not _ig_handle_plausible(handle, brand):
        print(f"    [WARN] Handle @{handle} doesn't match brand - skipping")
        handle = ""
    if not handle:
        print("    No handle found - skipping")
        profile.ig_scraper_used = "none"; return
    print(f"    Handle: @{handle}")

    data = _ig_instaloader(handle, ig_user, ig_pass)
    if not data:
        print("    Instaloader failed - trying Apify ...")
        data = _ig_apify(handle, apify_token)

    if data and (data.get("followers",0) > 0 or data.get("posts",0) > 0):
        profile.ig_handle          = handle
        profile.ig_followers       = data.get("followers", 0)
        profile.ig_posts           = data.get("posts", 0)
        profile.ig_engagement_rate = data.get("engagement_rate", 0)
        profile.ig_bio             = data.get("bio", "")
        profile.ig_data            = data
        profile.ig_scraper_used    = data.get("source", "unknown")
        print(f"    [OK] {profile.ig_followers:,} followers | "
              f"{profile.ig_posts} posts [{profile.ig_scraper_used}]")
    else:
        print(f"    [FAIL] Scraping failed for @{handle}")
        profile.ig_scraper_used = "none"


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5b — Facebook  (Apify → DDG snippet parse)
# ═══════════════════════════════════════════════════════════════════════════════

FB_BLACKLIST = {"pages","groups","events","watch","marketplace",
                "gaming","login","help","about","policies"}

def _resolve_fb_handle(brand: str, profile: SocialProfile) -> str:
    if profile.gm_facebook_handle: return profile.gm_facebook_handle
    if DDGS:
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(f'"{brand}" site:facebook.com', max_results=5))
            for hit in hits:
                m = re.search(r"facebook\.com/([A-Za-z0-9_.]{3,50})/?",
                               hit.get("href",""))
                if m:
                    h = m.group(1).lower()
                    if h not in FB_BLACKLIST and not h.startswith("profile.php"):
                        return h
        except Exception: pass
    return ""

def step5b_facebook(profile: SocialProfile, brand: str,
                    apify_token: str = "",
                    location: str = "Tunisia") -> None:
    print(f"  [Step 5b] Facebook: '{brand}'")
    handle = _resolve_fb_handle(brand, profile)
    if not handle:
        print("    No handle found - skipping")
        profile.fb_scraper_used = "none"; return
    print(f"    Handle: @{handle}")
    profile.fb_handle = handle
    data = {}

    # Apify
    if _is_key(apify_token):
        try:
            from apify_client import ApifyClient
            run     = ApifyClient(apify_token).actor("apify/facebook-pages-scraper").call(
                run_input={"startUrls": [{"url": f"https://www.facebook.com/{handle}"}],
                           "maxPosts": 10, "maxPostComments": 0, "maxReviews": 0},
                timeout_secs=180)
            dataset = list(ApifyClient(apify_token).dataset(
                run["defaultDatasetId"]).iterate_items())
            if dataset:
                page  = dataset[0]
                likes = page.get("likes", 0)
                posts = page.get("posts", []) or []
                eng   = 0.0
                if posts:
                    total = sum((p.get("likes",0) or 0) +
                                (p.get("comments",0) or 0) +
                                (p.get("shares",0) or 0) for p in posts)
                    eng   = round(total / len(posts), 1)
                data  = {"page_likes": likes, "post_engagement": eng, "source": "apify"}
        except Exception as e:
            print(f"    Apify FB error: {e}")

    # DDG fallback
    if not data and DDGS:
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(f"{brand} facebook page likes {location}", max_results=5))
            for hit in hits:
                if "facebook.com" not in hit.get("href",""): continue
                m = re.search(r"([\d,]+)\s*(?:people\s*)?(?:like|follow)",
                               hit.get("body",""), re.IGNORECASE)
                if m:
                    data = {"page_likes": int(m.group(1).replace(",","")),
                            "post_engagement": 0.0, "source": "ddg_parse"}
                    break
        except Exception as e:
            print(f"    DDG FB error: {e}")

    if data and data.get("page_likes",0) > 0:
        profile.fb_page_likes      = data["page_likes"]
        profile.fb_post_engagement = data["post_engagement"]
        profile.fb_data            = data
        profile.fb_scraper_used    = data.get("source","unknown")
        print(f"    [OK] {profile.fb_page_likes:,} likes [{profile.fb_scraper_used}]")
    else:
        print(f"    [FAIL] Facebook scraping failed for @{handle}")
        profile.fb_scraper_used = "none"


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6 — Hallucination Checker  (rules + LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class HallucinationChecker:
    """
    Two-layer hallucination detection:
      Layer 1 — Rule checks (fast, free)
      Layer 2 — LLM semantic validation
    Confidence = 70% field-level weighted average + 30% LLM holistic score.
    """

    SEVERITY   = {"missing": 0.0, "invalid": 0.4, "conflict": 0.6, "implausible": 0.7}
    SRC_WEIGHTS = {"gm_rating": 1.0, "ig_handle": 1.0, "ig_plausibility": 1.0,
                   "fb_handle": 0.9, "phone": 0.9, "website": 0.9,
                   "ta_rating": 0.7, "fsq_rating": 0.6,
                   "rating_consistency": 0.8, "address": 0.8}

    def __init__(self, profile: SocialProfile, groq_key: str = "",
                 groq_client=None, location: str = "Tunisia"):
        self.profile        = profile
        self.flags: list    = []
        self._checked: dict = {}
        self.llm_verdict    = {}
        self._groq_key      = groq_key
        self._client        = groq_client
        self.location       = location

    def _flag(self, field, value, severity, reason, method) -> None:
        conf = round(1.0 - self.SEVERITY.get(severity, 0.5), 2)
        self.flags.append(HallucinationFlag(
            field_name=field, claimed_value=value, confidence_score=conf,
            validation_method=f"rule:{method}", anomaly_detected=True,
            anomaly_reason=f"[{severity.upper()}] {reason}"))
        self._checked[field] = conf

    def _pass(self, field, conf=1.0):
        self._checked[field] = conf

    def _run_rules(self) -> None:
        p = self.profile
        # Handle plausibility
        for src, handle in [("ig_handle", p.ig_handle), ("fb_handle", p.fb_handle)]:
            if handle:
                if not _ig_handle_plausible(handle, p.brand):
                    self._flag(src, handle, "implausible",
                               f"@{handle} doesn't match '{p.brand}'", "name_match")
                else: self._pass(src)
            else: self._pass(src)

        # Phone format (Tunisia)
        if p.gm_phone and p.gm_phone != "N/A":
            if not re.match(r"^\+?[\d\s\-().]{7,20}$", p.gm_phone):
                self._flag("phone", p.gm_phone, "invalid", "Invalid phone format", "regex")
            elif p.gm_phone.startswith("+") and not p.gm_phone.startswith("+216"):
                self._flag("phone", p.gm_phone, "invalid", "Country code not +216", "regex")
            else: self._pass("phone")
        else: self._pass("phone")

        # Website URL
        if p.gm_website and p.gm_website != "N/A":
            if not re.match(r"^https?://[^\s]+$", p.gm_website):
                self._flag("website", p.gm_website, "invalid", "Malformed URL", "regex")
            else: self._pass("website")
        else: self._pass("website")

        # Ratings range
        for src in ["gm", "ta", "fsq"]:
            r = getattr(p, f"{src}_rating", None)
            if r is not None:
                if not (0.0 <= r <= 5.0):
                    self._flag(f"{src}_rating", r, "invalid", f"Rating {r} outside 0-5", "range")
                else: self._pass(f"{src}_rating")
            else: self._pass(f"{src}_rating")

        # Cross-source rating consistency
        if p.gm_rating and p.ta_rating:
            if abs(p.gm_rating - p.ta_rating) > 1.5:
                self._flag("rating_consistency",
                           f"GM={p.gm_rating} TA={p.ta_rating}",
                           "conflict", "Ratings diverge > 1.5", "cross_ref")
            else: self._pass("rating_consistency")

        # IG plausibility (handle but 0 followers + 0 posts)
        if p.ig_handle and p.ig_followers == 0 and p.ig_posts == 0:
            self._flag("ig_plausibility", p.ig_handle, "conflict",
                       "0 followers + 0 posts — scrape likely failed", "cross_ref")
        else: self._pass("ig_plausibility")

    def _llm_validate(self, groq_key: str = "") -> None:
        p = self.profile
        rule_findings = "\n".join(
            f"  - {f.field_name}: {f.anomaly_reason}" for f in self.flags
        ) or "  - None"

        prompt = f"""\
You are a data quality auditor for a {self.location} F&B brand intelligence platform.

BRAND: "{p.brand.upper()}"
Phone     : {p.gm_phone or "NOT SCRAPED"}
Website   : {p.gm_website or "NOT SCRAPED"}
Address   : {p.gm_address or "NOT SCRAPED"}
GM Rating : {p.gm_rating} ({p.gm_review_count} reviews)
TA Rating : {p.ta_rating} ({p.ta_review_count} reviews)
Instagram : @{p.ig_handle or "NOT FOUND"} | {p.ig_followers} followers | {p.ig_posts} posts
IG Bio    : {p.ig_bio[:100] if p.ig_bio else "N/A"}
Facebook  : @{p.fb_handle or "NOT FOUND"} | {p.fb_page_likes} likes

RULE FLAGS ALREADY DETECTED:
{rule_findings}

CHECK (semantic issues rules cannot catch):
1. Do phone, address, website belong to the SAME business?
2. Does @{p.ig_handle or "?"} plausibly match brand "{p.brand}"?
3. Is follower count plausible for this type/size of business?
4. Any internal contradictions?

RESPOND IN VALID JSON ONLY:
{{
  "issues": [{{"field":"<n>","severity":"invalid|conflict|implausible","reason":"<max 100 chars>"}}],
  "llm_confidence": <float 0.0-1.0>,
  "summary": "<max 120 chars>"
}}
If no issues: {{"issues":[],"llm_confidence":1.0,"summary":"All fields consistent."}}"""

        key = groq_key or os.environ.get("GROQ_API_KEY","")
        result = query_llm(MODEL_FAST, prompt, role="extractor",
                           client=self._client)
        verdict = parse_json(result["raw_response"], label="hallucination_llm")
        if not isinstance(verdict, dict): return
        self.llm_verdict = verdict

        for issue in verdict.get("issues", []):
            field    = issue.get("field","unknown")
            severity = issue.get("severity","invalid")
            reason   = issue.get("reason","")
            if field in self._checked and self._checked[field] < 1.0: continue
            conf = round(1.0 - self.SEVERITY.get(severity, 0.5), 2)
            self.flags.append(HallucinationFlag(
                field_name=field, claimed_value=getattr(p, field, "N/A"),
                confidence_score=conf, validation_method="llm:semantic",
                anomaly_detected=True, anomaly_reason=f"[LLM/{severity.upper()}] {reason}"))
            self._checked[field] = conf

        llm_conf = verdict.get("llm_confidence", 1.0)
        summary  = verdict.get("summary","")
        if summary: print(f"    LLM: {summary}")
        print(f"    LLM confidence: {llm_conf:.2f}")

    def compute_confidence(self) -> float:
        if not self._checked: return 1.0
        ws = sum(self.SRC_WEIGHTS.get(f,0.7) * c for f,c in self._checked.items())
        wt = sum(self.SRC_WEIGHTS.get(f,0.7)    for f   in self._checked)
        fc = ws / wt if wt else 1.0
        lc = self.llm_verdict.get("llm_confidence", fc)
        return max(0.0, min(1.0, round(fc*0.7 + lc*0.3, 3)))

    def run(self, use_llm: bool = True) -> None:
        print("    Running rule checks ...")
        self._run_rules()
        print(f"    Rule issues: {len(self.flags)}")
        if use_llm:
            print("    Running LLM semantic validation ...")
            self._llm_validate()
        self.profile.hallucination_flags = self.flags
        self.profile.overall_confidence  = self.compute_confidence()
        llm_n  = sum(1 for f in self.flags if "llm:" in f.validation_method)
        rule_n = len(self.flags) - llm_n
        print(f"    -- Rule: {rule_n} | LLM: {llm_n} | "
              f"Confidence: {self.profile.overall_confidence:.3f}")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 7 — ReflectionEngine  (LLM synthesis + self-reflection)
# ═══════════════════════════════════════════════════════════════════════════════

class ReflectionEngine:
    """
    Synthesises all scraped data into a brand intelligence report.
    2-round reflection loop: if confidence < threshold → re-evaluate.
    Uses LLM cache to skip redundant API calls.
    """

    def __init__(self, profile: SocialProfile,
                 openrouter_key: str = "", groq_client=None,
                 location: str = "Tunisia"):
        self.profile        = profile
        self.openrouter_key = openrouter_key or os.environ.get("OPENROUTER_API_KEY","")
        self.client         = groq_client
        self.location       = location

    def _build_prompt(self) -> str:
        p = self.profile
        def f(v, s=""): return f"{v}{s}" if v is not None else "NOT AVAILABLE"
        return f"""\
You are a brand intelligence analyst for F&B and hospitality in North Africa.
NOT AVAILABLE means the field was not scraped — do not penalise.

BRAND: {p.brand.upper()}

[ RATINGS & REVIEWS ]
- Google Maps  : {f(p.gm_rating)} stars | {f(p.gm_review_count," reviews")}
- TripAdvisor  : {f(p.ta_rating)} stars | {f(p.ta_review_count," reviews")}
- Venue ({p.venue_data_source or "none"}): {f(p.fsq_rating)} stars | {p.fsq_category or "N/A"}

[ SOCIAL MEDIA ]
- Instagram: @{p.ig_handle or "NOT FOUND"} | {f(p.ig_followers," followers")} | {f(p.ig_posts," posts")}
- Facebook : @{p.fb_handle or "NOT FOUND"} | {f(p.fb_page_likes," likes")} | {p.fb_post_engagement} avg engagement

[ LOCATION & WEB ]
- Address : {p.gm_address or "N/A"}
- Website : {p.gm_website or "N/A"}
- Phone   : {p.gm_phone   or "N/A"}

[ DATA QUALITY ]
- Hallucination flags : {len(p.hallucination_flags)}
- Known anomalies     : {"; ".join(f.anomaly_reason for f in p.hallucination_flags) or "none"}

[ SCORING RUBRIC — social_authority_score 0-100 ]
Weight equally (25 pts each):
  1. Review volume & rating  (GM + TA combined)
  2. Social reach            (IG followers + FB likes)
  3. Web & contact presence  (website, phone, address all present = full marks)
  4. Cross-platform match    (IG + FB handles match brand = full marks)
Deduct 5 pts per hallucination flag. Score 0 if no data.

[ GEO EXPANSION ]
Brand is based in {self.location}. Recommend 2-3 cities to expand next.
Use: cuisine type, price tier, social reach, regional tourism.
Prefer nearby cities and regional tourism hubs.

Return ONLY valid JSON (no markdown):
{{
  "social_authority_score": <float 0-100>,
  "geo_recommendations":    ["<city, country>", ...],
  "synthesis_notes":        "<max 120 chars>",
  "confidence_assessment":  <float 0-1>,
  "score_breakdown": {{
    "review_quality":  <float 0-25>,
    "social_reach":    <float 0-25>,
    "web_completeness":<float 0-25>,
    "cross_platform":  <float 0-25>
  }}
}}"""

    def _call_llm(self, prompt: str) -> str:
        # Try OpenRouter first (free models), fall back to Groq
        result = query_openrouter(prompt, self.openrouter_key)
        if not result:
            print("    OpenRouter failed - falling back to Groq ...")
            r = query_llm(MODEL_FAST, prompt, role="analyst", client=self.client)
            result = r["raw_response"]
        return result

    def run(self, max_rounds: int = 2) -> None:
        for rnd in range(1, max_rounds + 1):
            print(f"    Round {rnd}/{max_rounds} ...")
            self.profile.reflection_rounds = rnd

            prompt = self._build_prompt()
            raw    = cached_llm_call(prompt, self._call_llm)

            if not raw:
                print("    [WARN] No LLM response - skipping synthesis")
                self.profile.synthesis_notes = "LLM unavailable."
                self.profile.final_llm_used  = "none"
                return

            parsed = parse_json(raw, label=f"reflection round {rnd}")
            if not isinstance(parsed, dict):
                print(f"    [WARN] Parse error: {raw[:100]}")
                return

            self.profile.social_authority_score = parsed.get("social_authority_score")
            self.profile.geo_recommendations    = parsed.get("geo_recommendations", [])
            self.profile.synthesis_notes        = parsed.get("synthesis_notes", "")
            self.profile.score_breakdown        = parsed.get("score_breakdown", {})
            assessed_conf                        = parsed.get("confidence_assessment",
                                                              self.profile.overall_confidence)
            self.profile.final_llm_used = f"openrouter_r{rnd}"

            print(f"    Authority : {self.profile.social_authority_score}")
            print(f"    Geo recs  : {self.profile.geo_recommendations}")
            print(f"    Confidence: {assessed_conf:.2f}")
            print(f"    Notes     : {self.profile.synthesis_notes[:80]}")

            if assessed_conf >= REFLECTION_THRESHOLD:
                print(f"    [OK] Confidence >= {REFLECTION_THRESHOLD} - done")
                break
            print("    [WARN] Low confidence - re-evaluating ...")
            self.profile.overall_confidence = assessed_conf


# ═══════════════════════════════════════════════════════════════════════════════
# Step 8 — Export
# ═══════════════════════════════════════════════════════════════════════════════

def step8_export(profiles: list[SocialProfile], audit_log: list,
                 output_dir: str = ".") -> None:
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for p in profiles:
        rows.append({
            "brand": p.brand,
            "wiki_summary": p.wiki_summary[:100],
            "ta_rating": p.ta_rating, "ta_reviews": p.ta_review_count,
            "gm_rating": p.gm_rating, "gm_reviews": p.gm_review_count,
            "gm_address": p.gm_address, "gm_phone": p.gm_phone,
            "gm_website": p.gm_website,
            "venue_source": p.venue_data_source, "venue_category": p.fsq_category,
            "venue_rating": p.fsq_rating, "venue_reviews": p.fsq_checkins,
            "venue_price_tier": p.fsq_price_tier,
            "ig_handle": p.ig_handle, "ig_followers": p.ig_followers,
            "ig_posts": p.ig_posts, "ig_bio": p.ig_bio[:100],
            "ig_scraper": p.ig_scraper_used,
            "fb_handle": p.fb_handle, "fb_page_likes": p.fb_page_likes,
            "fb_post_engagement": p.fb_post_engagement, "fb_scraper": p.fb_scraper_used,
            "social_authority_score": p.social_authority_score,
            "score_review_quality":   p.score_breakdown.get("review_quality"),
            "score_social_reach":     p.score_breakdown.get("social_reach"),
            "score_web_completeness": p.score_breakdown.get("web_completeness"),
            "score_cross_platform":   p.score_breakdown.get("cross_platform"),
            "overall_confidence": p.overall_confidence,
            "hallucination_count": len(p.hallucination_flags),
            "geo_recommendations": ", ".join(p.geo_recommendations),
            "synthesis_notes": p.synthesis_notes,
            "reflection_rounds": p.reflection_rounds,
            "llm_used": p.final_llm_used,
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "social_profiles.csv"), index=False)
    print(f"  [OK] social_profiles.csv")

    flag_rows = []
    for p in profiles:
        for flag in p.hallucination_flags:
            flag_rows.append({
                "brand": p.brand, "field_name": flag.field_name,
                "claimed_value": str(flag.claimed_value),
                "confidence_score": flag.confidence_score,
                "validation_method": flag.validation_method,
                "anomaly_reason": flag.anomaly_reason,
                "timestamp": flag.timestamp,
            })
    if flag_rows:
        pd.DataFrame(flag_rows).to_csv(
            os.path.join(output_dir, "hallucination_flags.csv"), index=False)
        print(f"  [OK] hallucination_flags.csv")

    with open(os.path.join(output_dir, "audit_log.json"), "w") as f:
        json.dump(audit_log, f, indent=2)
    print(f"  [OK] audit_log.json")


# ═══════════════════════════════════════════════════════════════════════════════
# Agent 2 — Master orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class Agent2SocialScraper:
    """
    Agent 2 — Social Profile Scraper & Brand Intelligence Synthesiser.

    Full pipeline: URL resolution → Wikipedia → TripAdvisor → Venue →
                   Google Maps → Instagram → Facebook →
                   Hallucination check → LLM reflection → Export.

    Args:
        brands          : list of brand names to process
        output_dir      : folder for all output files
        serpapi_key     : SerpApi key (100 free searches/month)
        apify_token     : Apify token (for TripAdvisor / FB / IG fallbacks)
        openrouter_key  : OpenRouter key (free tier, for synthesis)
        groq_client     : pre-built Groq client
        ig_user/ig_pass : optional Instagram credentials
        is_colab        : set True when running in Google Colab
    """

    def __init__(
        self,
        brands:         list[str],
        output_dir:     str  = "agent2_output",
        serpapi_key:    str  = "",
        apify_token:    str  = "",
        openrouter_key: str  = "",
        groq_client          = None,
        ig_user:        str  = "",
        ig_pass:        str  = "",
        is_colab:       bool = False,
        location:       str  = "Tunisia",
    ):
        self.brands         = brands
        self.output_dir     = output_dir
        self.serpapi_key    = serpapi_key
        self.apify_token    = apify_token
        self.openrouter_key = openrouter_key
        self.client         = groq_client
        self.ig_user        = ig_user
        self.ig_pass        = ig_pass
        self.is_colab       = is_colab
        self.location       = location
        self.profiles: list[SocialProfile] = []
        self.audit:    list[dict]           = []

    def run(self) -> list[SocialProfile]:
        print(f"\n{'='*60}")
        print(f" Agent 2 - Social Profile Scraper v3.0")
        print(f" Brands  : {self.brands}")
        print(f" Output  : {self.output_dir}/")
        print(f"{'='*60}")

        for i, brand in enumerate(self.brands):
            if i > 0:
                print(f"\n[WAIT] Waiting 15s before next brand ...")
                time.sleep(15)

            print(f"\n{'-'*60}")
            print(f" {brand.upper()}")
            print(f"{'-'*60}")

            profile = SocialProfile(brand=brand)

            print("[Step 1] URL Resolution")
            links = step1_resolve_urls(brand, location=self.location)

            print("[Step 2a] Wikipedia")
            step2a_wikipedia(profile, links["wikipedia_url"])

            print("[Step 2b] TripAdvisor")
            step2b_tripadvisor(profile, links["tripadvisor_url"], self.apify_token,
                               location=self.location)

            print("[Step 3] Venue Data")
            step3_venue_data(profile, brand, self.serpapi_key, location=self.location)

            print("[Step 4] Google Maps")
            step4_google_maps(profile, brand, location=self.location,
                              is_colab=self.is_colab)

            print("[Step 5a] Instagram")
            step5a_instagram(profile, brand, self.apify_token,
                             self.ig_user, self.ig_pass)

            print("[Step 5b] Facebook")
            step5b_facebook(profile, brand, self.apify_token,
                            location=self.location)

            print("[Step 6] Hallucination Detection")
            checker = HallucinationChecker(profile, groq_client=self.client,
                                           location=self.location)
            checker.run(use_llm=True)

            print("[Step 7] LLM Reflection & Synthesis")
            reflector = ReflectionEngine(profile, self.openrouter_key, self.client,
                                         location=self.location)
            reflector.run()

            self.audit.append({
                "brand":             brand,
                "confidence":        profile.overall_confidence,
                "reflection_rounds": profile.reflection_rounds,
                "hallucinations":    len(profile.hallucination_flags),
                "llm_used":          profile.final_llm_used,
                "ig_scraper":        profile.ig_scraper_used,
                "fb_scraper":        profile.fb_scraper_used,
                "venue_source":      profile.venue_data_source,
            })
            self.profiles.append(profile)
            print(f"\n  [OK] Done: {brand}")

        # Step 8: Export
        print(f"\n[Step 8] Export")
        step8_export(self.profiles, self.audit, self.output_dir)

        # Summary
        avg_conf  = sum(p.overall_confidence for p in self.profiles) / len(self.profiles)
        total_flg = sum(len(p.hallucination_flags) for p in self.profiles)
        print(f"\n{'='*60}")
        print(f" Agent 2 complete")
        print(f"   Brands processed   : {len(self.profiles)}")
        print(f"   Avg confidence     : {avg_conf:.2f}")
        print(f"   Total flags        : {total_flg}")
        print(f"   IG scrapers used   : {set(p.ig_scraper_used for p in self.profiles)}")
        print(f"   FB scrapers used   : {set(p.fb_scraper_used for p in self.profiles)}")
        print(f"   Venue sources      : {set(p.venue_data_source for p in self.profiles)}")
        print(f"{'='*60}")

        return self.profiles


# ══════════════════════════════════════════════════════════════════════════════
# Standalone helper for LangGraph fan-out node
# ══════════════════════════════════════════════════════════════════════════════

def scrape_single_brand(
    brand:          str,
    location:       str  = "Tunisia",
    output_dir:     str  = "geo_output/agent2_output",
    serpapi_key:    str  = "",
    apify_token:    str  = "",
    openrouter_key: str  = "",
    groq_client          = None,
    ig_user:        str  = "",
    ig_pass:        str  = "",
    is_colab:       bool = False,
) -> "SocialProfile":
    """
    Execute the full 7-step scraping pipeline for a single brand.

    This function is used by agent2_scrape_node in the LangGraph fan-out,
    where each brand is processed concurrently.

    Returns:
        SocialProfile — populated with all scraped + validated data
    """
    from loguru import logger

    logger.info(f"[ScrapeJob] Starting: '{brand}' | location='{location}'")
    os.makedirs(output_dir, exist_ok=True)

    profile = SocialProfile(brand=brand)

    # Step 1 — URL resolution
    logger.debug(f"[ScrapeJob] {brand}: Step 1 — URL resolution")
    links = step1_resolve_urls(brand, location=location)

    # Step 2a — Wikipedia
    logger.debug(f"[ScrapeJob] {brand}: Step 2a — Wikipedia")
    step2a_wikipedia(profile, links["wikipedia_url"])

    # Step 2b — TripAdvisor
    logger.debug(f"[ScrapeJob] {brand}: Step 2b — TripAdvisor")
    step2b_tripadvisor(profile, links["tripadvisor_url"], apify_token, location=location)

    # Step 3 — Venue data
    logger.debug(f"[ScrapeJob] {brand}: Step 3 — Venue data")
    step3_venue_data(profile, brand, serpapi_key, location=location)

    # Step 4 — Google Maps
    logger.debug(f"[ScrapeJob] {brand}: Step 4 — Google Maps")
    step4_google_maps(profile, brand, location=location, is_colab=is_colab)

    # Step 5a — Instagram
    logger.debug(f"[ScrapeJob] {brand}: Step 5a — Instagram")
    step5a_instagram(profile, brand, apify_token, ig_user, ig_pass)

    # Step 5b — Facebook
    logger.debug(f"[ScrapeJob] {brand}: Step 5b — Facebook")
    step5b_facebook(profile, brand, apify_token, location=location)

    # Step 6 — Hallucination checker
    logger.debug(f"[ScrapeJob] {brand}: Step 6 — Hallucination check")
    checker = HallucinationChecker(profile, groq_client=groq_client, location=location)
    checker.run(use_llm=True)

    # Step 7 — LLM reflection & synthesis
    logger.debug(f"[ScrapeJob] {brand}: Step 7 — Reflection & synthesis")
    reflector = ReflectionEngine(profile, openrouter_key, groq_client, location=location)
    reflector.run()

    logger.success(
        f"[ScrapeJob] '{brand}' complete — "
        f"confidence={profile.overall_confidence:.2f}, "
        f"authority={profile.social_authority_score}"
    )
    return profile
