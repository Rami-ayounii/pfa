"""
core/llm_client.py
══════════════════
Shared LLM utilities used by all GEO agents.

Provides:
  - query_llm()            Groq call with retry + rate-limit handling  (sync)
  - async_query_llm()      Async Groq call with semaphore + retry       (async)
  - query_mistral()        Mistral call (used by Agent 1 extractor)     (sync)
  - query_openrouter()     OpenRouter call with model fallback (Agent 2) (sync)
  - parse_json()           Robust multi-layer JSON parser
  - cached_llm_call()      Cache wrapper for identical prompts
  - drain_retry_queue()    Flush failed async calls that were queued
  - TOKEN_USAGE            Global token counter
"""

import os, re, json, time, hashlib, asyncio, requests
from groq import Groq
from dotenv import load_dotenv

from loguru import logger

# Load .env as early as possible so all os.environ.get() calls below pick it up
load_dotenv()

# ── Model constants (overridable via env vars) ────────────────────────────────
# Production models (stable)
MODEL_FAST      = os.environ.get("MODEL_FAST",      "llama-3.1-8b-instant")            # fastest, cheapest
MODEL_STRONG    = os.environ.get("MODEL_STRONG",    "llama-3.3-70b-versatile")         # highest quality
MODEL_COMPOUND  = os.environ.get("MODEL_COMPOUND",  "groq/compound")                   # Groq compound agentic system (was compound-beta)
MODEL_ANALYST   = os.environ.get("MODEL_ANALYST",   "openai/gpt-oss-120b")             # OpenAI OSS 120B reasoning (via Groq)
MODEL_EXTRACTOR = os.environ.get("MODEL_EXTRACTOR", "llama-3.1-8b-instant")            # fast structured extraction
# Additional query models — diversify GEO measurement across model families
MODEL_QWEN      = os.environ.get("MODEL_QWEN",      "qwen/qwen3-32b")                  # Qwen 3 32B — strong multilingual (fr/ar)
MODEL_LLAMA4    = os.environ.get("MODEL_LLAMA4",    "meta-llama/llama-4-scout-17b-16e-instruct")  # Llama 4 Scout MoE (preview)
MODEL_KIMI      = os.environ.get("MODEL_KIMI",      "moonshotai/kimi-k2-instruct-0905") # Moonshot Kimi K2 (⚠ verify availability on your Groq plan)
MODEL_GPTOSS    = os.environ.get("MODEL_GPTOSS",    "openai/gpt-oss-20b")              # GPT-OSS 20B — lighter reasoning model
# Comma-separated env var overrides the default free-model list
_or_env = os.environ.get("OPENROUTER_MODELS", "")
OPENROUTER_MODELS = (
    [m.strip() for m in _or_env.split(",") if m.strip()]
    if _or_env else [
        "meta-llama/llama-3.1-8b-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
    ]
)

# ── OpenRouter fallback map ───────────────────────────────────────────────────
# When a Groq model exhausts all retries (non-fatal), query_llm / async_query_llm
# will automatically try the mapped OpenRouter model before returning an empty result.
#
# Free-tier limits (Apr 2026): 20 req/min · ~200 req/day shared across ALL free models
# Failed calls count toward the daily quota.
# Use $10+ credits to raise the daily cap to 1 000 req/day.
#
# "openrouter/free" is a special router (launched Feb 2026) that auto-selects
# from the current free model pool — most resilient option when a specific ID
# goes stale, but gives up control over which model is used.
#
# Model IDs verified against OpenRouter free pool (Apr 2026):
#   meta-llama/llama-3.2-3b-instruct:free   — 131K ctx
#   meta-llama/llama-3.3-70b-instruct:free  — 66K  ctx
#   nousresearch/hermes-3-llama-3.1-405b:free — 131K ctx  (strong open model)
#   openai/gpt-oss-120b:free                — 131K ctx  (same family as Groq GPT-OSS)
#   openai/gpt-oss-20b:free                 — 131K ctx  (same family as Groq GPT-OSS)
#   qwen/qwen3-next-80b-a3b-instruct:free   — 262K ctx  (Qwen3 MoE)
#   qwen/qwen3.6-plus:free                  — 1M   ctx  (latest Qwen free)
#   google/gemma-3-27b-it:free              — 131K ctx  (vision-capable)
#   openrouter/free                         — 200K ctx  (auto-router, always available)
GROQ_TO_OPENROUTER_FALLBACK: dict[str, str] = {
    # Fast 8B → Llama 3.2 3B free (Llama 3.1-8b:free is retired)
    "llama-3.1-8b-instant":                         "meta-llama/llama-3.2-3b-instruct:free",
    # Strong 70B → Llama 3.3 70B free (same family, verified available)
    "llama-3.3-70b-versatile":                       "meta-llama/llama-3.3-70b-instruct:free",
    # Compound/agentic → Hermes 3 405B free (strong instruction-following)
    "groq/compound":                                  "nousresearch/hermes-3-llama-3.1-405b:free",
    # GPT-OSS 120B → exact same model available free on OpenRouter
    "openai/gpt-oss-120b":                            "openai/gpt-oss-120b:free",
    # GPT-OSS 20B → exact same model available free on OpenRouter
    "openai/gpt-oss-20b":                             "openai/gpt-oss-20b:free",
    # Qwen3 32B → Qwen3 Next 80B MoE free (262K ctx, same family)
    "qwen/qwen3-32b":                                 "qwen/qwen3-next-80b-a3b-instruct:free",
    # Llama 4 Scout → Llama 3.3 70B free (best available Llama free tier)
    "meta-llama/llama-4-scout-17b-16e-instruct":      "meta-llama/llama-3.3-70b-instruct:free",
    # Kimi K2 → Qwen3.6 Plus free (1M context, strong multilingual)
    "moonshotai/kimi-k2-instruct-0905":               "qwen/qwen3.6-plus:free",
}

# ── Concurrency controls (overridable via env vars) ───────────────────────────
MAX_CONCURRENT_LLM    = int(os.environ.get("MAX_CONCURRENT_LLM",    "5"))
MAX_CONCURRENT_BRANDS = int(os.environ.get("MAX_CONCURRENT_BRANDS", "3"))

# Lazily initialised so it is always created inside the running event loop
_llm_semaphore: asyncio.Semaphore | None = None

def _get_semaphore() -> asyncio.Semaphore:
    """Return the module-level semaphore, creating it in the current loop."""
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)
    return _llm_semaphore

# ── Async retry queue ─────────────────────────────────────────────────────────
_retry_queue: asyncio.Queue | None = None

def _get_retry_queue() -> asyncio.Queue:
    global _retry_queue
    if _retry_queue is None:
        _retry_queue = asyncio.Queue()
    return _retry_queue

# ── Global token tracker ──────────────────────────────────────────────────────
TOKEN_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def _track(usage: dict):
    for k in TOKEN_USAGE:
        TOKEN_USAGE[k] += usage.get(k, 0)

# ── Per-role call parameters ──────────────────────────────────────────────────
def _model_params(model: str, role: str) -> dict:
    _m = model.lower()
    if role == "query":
        # Small/instant models hit Groq free-tier request size limits at 8192.
        # Large models: capped at 2048 — a GEO restaurant response listing
        # 10 restaurants with descriptions fits comfortably in ~500-1500 tokens.
        _small = any(x in _m for x in ("8b", "3b", "1b", "instant"))
        base = {"temperature": 0.7, "max_tokens": 1024 if _small else 2048, "top_p": 0.9}
    elif role == "extractor":
        base = {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0}
    elif role == "analyst":
        base = {"temperature": 0.1, "max_tokens": 8192, "top_p": 0.95}
        if "gpt-oss" in _m:
            base["reasoning_effort"] = "low"
    else:
        base = {"temperature": 0.2, "max_tokens": 4096, "top_p": 0.9}
    # Model-family overrides
    if "qwen3" in _m:
        base["reasoning_effort"] = "none"   # disable thinking mode → faster + fewer tokens
    if "kimi" in _m and role == "query":
        base["max_tokens"] = min(base["max_tokens"], 2048)  # Kimi K2 is expensive per token
    return base

def _parse_wait(err: str) -> float:
    m = re.search(r"try again in (\d+)m([\d.]+)s", err)
    if m: return int(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r"try again in ([\d.]+)s", err)
    if m: return float(m.group(1))
    m = re.search(r"try again in (\d+) minute", err)
    if m: return int(m.group(1)) * 60
    return 0.0

FATAL_ERRORS = ["model not found", "invalid api key", "authentication",
                "permission denied", "does not exist"]

# ── Groq (sync) ───────────────────────────────────────────────────────────────
def query_llm(model: str, prompt: str, system: str = "",
              retries: int = 3, role: str = "analyst",
              client: Groq | None = None) -> dict:
    """
    Groq LLM call with retry, rate-limit back-off, and token tracking.
    role: query | extractor | analyst
    """
    _base_url = os.environ.get("GROQ_BASE_URL")  # set to swap Groq for any OpenAI-compat endpoint
    _client = client or Groq(
        api_key=os.environ.get("GROQ_API_KEY", ""),
        **({"base_url": _base_url} if _base_url else {}),
    )
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]

    _fatal = False
    for attempt in range(1, retries + 1):
        try:
            params = _model_params(model, role)
            resp   = _client.chat.completions.create(
                model=model, messages=messages, **params)
            result = {
                "raw_response":      resp.choices[0].message.content.strip(),
                "completion_tokens": resp.usage.completion_tokens,
                "prompt_tokens":     resp.usage.prompt_tokens,
                "total_tokens":      resp.usage.total_tokens,
            }
            _track(result)
            return result
        except Exception as e:
            err = str(e)
            logger.warning(f"Attempt {attempt}/{retries} ({model}): {err[:80]}")
            if any(f in err.lower() for f in FATAL_ERRORS):
                _fatal = True
                break
            wait = _parse_wait(err) or (2 ** attempt)
            logger.debug(f"Waiting {wait:.0f}s before retry …")
            time.sleep(wait + 1)

    # OpenRouter fallback — only for non-fatal failures (rate limits, timeouts, etc.)
    # Tries: (1) mapped specific model, then (2) openrouter/free auto-router
    if not _fatal and os.environ.get("OPENROUTER_API_KEY"):
        _or_model = GROQ_TO_OPENROUTER_FALLBACK.get(model)
        _or_models = ([_or_model] if _or_model else []) + ["openrouter/free"]
        logger.info(f"[Fallback] Groq '{model}' exhausted → OpenRouter {_or_models}")
        raw = query_openrouter(prompt, models=_or_models)
        if raw:
            return {"raw_response": raw, "completion_tokens": 0,
                    "prompt_tokens": 0, "total_tokens": 0}

    return {"raw_response": "", "completion_tokens": 0,
            "prompt_tokens": 0, "total_tokens": 0}

# ── Groq (async) ──────────────────────────────────────────────────────────────
async def async_query_llm(model: str, prompt: str, system: str = "",
                           retries: int = 3, role: str = "analyst") -> dict:
    """
    Async Groq LLM call — uses AsyncGroq + semaphore for controlled concurrency.

    The semaphore limit is MAX_CONCURRENT_LLM (default 5, override via env var).
    Failed calls are pushed to the retry queue instead of silently returning "".
    """
    from groq import AsyncGroq  # only import async client when needed

    _base_url = os.environ.get("GROQ_BASE_URL")
    api_key   = os.environ.get("GROQ_API_KEY", "")
    messages  = ([{"role": "system", "content": system}] if system else []) + \
                [{"role": "user", "content": prompt}]

    _fatal = False
    async with _get_semaphore():
        async with AsyncGroq(
            api_key=api_key,
            **({"base_url": _base_url} if _base_url else {}),
        ) as aclient:
            for attempt in range(1, retries + 1):
                try:
                    params = _model_params(model, role)
                    resp   = await aclient.chat.completions.create(
                        model=model, messages=messages, **params)
                    result = {
                        "raw_response":      resp.choices[0].message.content.strip(),
                        "completion_tokens": resp.usage.completion_tokens,
                        "prompt_tokens":     resp.usage.prompt_tokens,
                        "total_tokens":      resp.usage.total_tokens,
                    }
                    _track(result)
                    return result
                except Exception as e:
                    err = str(e)
                    logger.warning(f"[async] Attempt {attempt}/{retries} ({model}): {err[:80]}")
                    if any(f in err.lower() for f in FATAL_ERRORS):
                        logger.error(f"[async] Fatal error for model={model} — not retrying.")
                        _fatal = True
                        return {"raw_response": "", "completion_tokens": 0,
                                "prompt_tokens": 0, "total_tokens": 0}
                    wait = _parse_wait(err) or (2 ** attempt)
                    logger.debug(f"[async] Waiting {wait:.0f}s …")
                    await asyncio.sleep(wait + 1)

    # OpenRouter fallback — run sync call in thread pool to stay async-safe
    # Tries: (1) mapped specific model, then (2) openrouter/free auto-router
    if not _fatal and os.environ.get("OPENROUTER_API_KEY"):
        _or_model  = GROQ_TO_OPENROUTER_FALLBACK.get(model)
        _or_models = ([_or_model] if _or_model else []) + ["openrouter/free"]
        logger.info(f"[Async Fallback] Groq '{model}' exhausted → OpenRouter {_or_models}")
        raw = await asyncio.to_thread(query_openrouter, prompt, None, 2048, 0.7, 3, _or_models)
        if raw:
            return {"raw_response": raw, "completion_tokens": 0,
                    "prompt_tokens": 0, "total_tokens": 0}

    # All retries and fallback exhausted — push to retry queue
    logger.error(f"[async] All retries exhausted for model={model}. Queueing for retry.")
    await _get_retry_queue().put({"model": model, "prompt": prompt,
                                   "system": system, "role": role})
    return {"raw_response": "", "completion_tokens": 0,
            "prompt_tokens": 0, "total_tokens": 0}


async def drain_retry_queue() -> list[dict]:
    """
    Flush all items from the retry queue, attempting each call once more.
    Returns a list of results (may include empty dicts for still-failing calls).
    """
    q = _get_retry_queue()
    results = []
    while not q.empty():
        task = await q.get()
        logger.info(f"[RetryQueue] Retrying: model={task['model']}")
        result = await async_query_llm(
            task["model"], task["prompt"],
            system=task.get("system", ""),
            role=task.get("role", "analyst"),
        )
        results.append(result)
    return results


# ── Mistral ───────────────────────────────────────────────────────────────────
def query_mistral(prompt: str, system: str = "",
                  api_key: str | None = None) -> dict:
    """
    Mistral API call — drop-in replacement for query_llm for extraction tasks.
    """
    key = api_key or os.environ.get("MISTRAL_API_KEY", "")
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    try:
        resp = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": MODEL_EXTRACTOR, "messages": messages,
                  "temperature": 0.0, "max_tokens": 1024},
            timeout=30,
        )
        data = resp.json()
        if "choices" not in data:
            logger.warning(f"Mistral API error: {data}")
            return {"raw_response": "", "completion_tokens": 0,
                    "prompt_tokens": 0, "total_tokens": 0}
        result = {
            "raw_response":      data["choices"][0]["message"]["content"].strip(),
            "completion_tokens": data["usage"]["completion_tokens"],
            "prompt_tokens":     data["usage"]["prompt_tokens"],
            "total_tokens":      data["usage"]["total_tokens"],
        }
        _track(result)
        return result
    except Exception as e:
        logger.warning(f"Mistral error: {e}")
        return {"raw_response": "", "completion_tokens": 0,
                "prompt_tokens": 0, "total_tokens": 0}

# ── OpenRouter ────────────────────────────────────────────────────────────────
def query_openrouter(prompt: str, api_key: str | None = None,
                     max_tokens: int = 1024, temperature: float = 0.2,
                     max_retries: int = 3,
                     models: list | None = None) -> str:
    """
    OpenRouter call with model fallback and exponential back-off on 429.

    models : optional list of model IDs to try in order.
             Defaults to OPENROUTER_MODELS when not provided.
    Returns raw text or "" on failure.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    _models = models if models is not None else OPENROUTER_MODELS

    for model in _models:
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    endpoint,
                    headers={"Authorization": f"Bearer {key}",
                             "HTTP-Referer": "https://geo-agent.local",
                             "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens, "temperature": temperature},
                    timeout=45,
                )
                if resp.status_code == 429:
                    wait = (2 ** attempt) * 5
                    logger.warning(f"[429] {model} — waiting {wait}s …")
                    time.sleep(wait); continue
                if resp.status_code == 404:
                    logger.debug(f"[404] {model} — trying next model"); break
                if resp.status_code != 200:
                    break
                data = resp.json()
                if "error" in data:
                    if data["error"].get("code") == 429:
                        time.sleep((2 ** attempt) * 5); continue
                    break
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    logger.debug(f"[OK] OpenRouter ({model})")
                    return content
                break
            except requests.exceptions.Timeout:
                time.sleep(3)
            except Exception as exc:
                logger.error(f"OpenRouter error: {exc}"); break

    logger.warning("All OpenRouter models exhausted.")
    return ""

# ── LLM response cache ────────────────────────────────────────────────────────
LLM_CACHE_FILE = "llm_cache.json"

def _load_cache() -> dict:
    if os.path.exists(LLM_CACHE_FILE):
        try:
            with open(LLM_CACHE_FILE) as f: return json.load(f)
        except Exception: pass
    return {}

def _save_cache(cache: dict):
    try:
        with open(LLM_CACHE_FILE, "w") as f: json.dump(cache, f, indent=2)
    except Exception: pass

def cached_llm_call(prompt: str, call_fn) -> str:
    """Call LLM only if this exact prompt has not been called before."""
    cache = _load_cache()
    key   = hashlib.md5(prompt.encode()).hexdigest()
    if key in cache:
        logger.debug("Using cached LLM response")
        return cache[key]
    result = call_fn(prompt)
    if result:
        cache[key] = result
        _save_cache(cache)
    return result

# ── Robust JSON parser ────────────────────────────────────────────────────────
def parse_json(raw: str, label: str = "") -> list | dict:
    """
    Multi-layer parser: strips markdown fences, fixes trailing commas,
    smart quotes, unescaped newlines, qwen3 thinking blocks.
    """
    if not raw or not raw.strip():
        logger.warning(f"{label}: empty response.")
        return []

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$",          "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _unwrap(parsed):
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], list):
            return parsed[0]
        return parsed

    try: return _unwrap(json.loads(text))
    except json.JSONDecodeError: pass

    for pattern in (r"(\[.*\])", r"(\{.*\})"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try: return _unwrap(json.loads(m.group(1)))
            except json.JSONDecodeError: pass

    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    fixed = fixed.replace("\u201c", '"').replace("\u201d", '"')
    try: return _unwrap(json.loads(fixed))
    except json.JSONDecodeError: pass

    try:
        repaired = re.sub(r'(?<!\\)\n', r'\\n', text)
        return _unwrap(json.loads(repaired))
    except json.JSONDecodeError: pass

    logger.warning(f"{label}: could not parse JSON.\n  RAW: {raw[:300]}")
    return []

# ── Provider-agnostic unified query ──────────────────────────────────────────
def query_provider(provider: str, model: str, prompt: str,
                   system: str = "", **kwargs) -> dict:
    """
    Unified entry point — route to the right backend by provider name.

    provider : "groq" | "mistral" | "openrouter"
    model    : model id (overrides defaults for groq/mistral; ignored for openrouter)
    Returns  : same dict as query_llm — {"raw_response", "total_tokens", ...}
               For openrouter the dict has {"raw_response": <str>, "total_tokens": 0}
    """
    provider = provider.lower()
    if provider == "groq":
        return query_llm(model, prompt, system=system, **kwargs)
    if provider == "mistral":
        return query_mistral(prompt, system=system, **kwargs)
    if provider == "openrouter":
        raw = query_openrouter(prompt, **kwargs)
        return {"raw_response": raw, "completion_tokens": 0,
                "prompt_tokens": 0, "total_tokens": 0}
    raise ValueError(f"Unknown provider '{provider}'. Use 'groq', 'mistral', or 'openrouter'.")
