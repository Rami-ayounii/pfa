"""
core/llm_client.py
══════════════════
Hybrid LLM client for the merged GEO pipeline.

Combines Project 2's full async/retry infrastructure with Project 1's
ModelRegistry-driven dynamic model selection.

Exports:
  async_query_llm()   Async Groq call with semaphore + registry selection
  query_llm()         Sync Groq call with registry selection + retry
  parse_json()        Robust multi-layer JSON parser
  clean_and_parse_json  Alias for parse_json (Project 1 compat)
  call_llm()          Thin alias over query_llm (Project 1 compat)
  call_llm_json()     call_llm + parse_json (Project 1 compat)
  query_provider()    Provider-agnostic unified call
  query_mistral()     Mistral API direct call
  query_openrouter()  OpenRouter fallback call
  cached_llm_call()   MD5-keyed cache wrapper
  drain_retry_queue() Async retry queue flush
  TOKEN_USAGE         Global token counter dict
  MODEL_FAST, MODEL_STRONG, MODEL_EXTRACTOR, MODEL_COMPOUND  Model constants
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # Final/ is root

import os, re, json, time, hashlib, asyncio, requests, random, threading
from collections import deque as _deque
from groq import Groq
from loguru import logger

from config import (
    GROQ_API_KEY, MISTRAL_API_KEY, OPENROUTER_API_KEY,
    MAX_CONCURRENT_LLM, GROQ_RPM, GROQ_TPM_LIMIT,
    MODEL_FAST, MODEL_STRONG, MODEL_EXTRACTOR,
    MODEL_COMPOUND, MODEL_GPTOSS, MODEL_QWEN, MODEL_LLAMA4,
    MODEL_ANALYST, MODEL_KIMI,
    OPENROUTER_MODELS, MAX_CONCURRENT_OR, OR_RPM,
)
from core.model_registry import registry  # type: ignore[import]

# ── OpenRouter fallback map ───────────────────────────────────────────────────
GROQ_TO_OPENROUTER_FALLBACK: dict[str, str] = {
    "llama-3.1-8b-instant":                         "meta-llama/llama-3.2-3b-instruct:free",
    "llama-3.3-70b-versatile":                       "meta-llama/llama-3.3-70b-instruct:free",
    "openai/gpt-oss-20b":                            "openai/gpt-oss-20b:free",
    "qwen/qwen3-32b":                                "qwen/qwen3-next-80b-a3b-instruct:free",
    "meta-llama/llama-4-scout-17b-16e-instruct":     "meta-llama/llama-3.3-70b-instruct:free",
    "groq/compound":                                  "nousresearch/hermes-3-llama-3.1-405b:free",
    "openai/gpt-oss-120b":                            "openai/gpt-oss-120b:free",
    "moonshotai/kimi-k2-instruct-0905":               "qwen/qwen3.6-plus:free",
}

# ── Concurrency / rate-limit controls ────────────────────────────────────────
_COMPOUND_EXTRA_DELAY    = float(os.environ.get("COMPOUND_EXTRA_DELAY", "0.3"))
_SWITCH_MODEL_THRESHOLD  = 4.0   # switch Groq model if wait > 4s AND another model available
MAX_BACKOFF_SLEEP        = 5.0   # never sleep longer than 5s; switch instead
_OR_MAX_WAIT             = 10.0  # OpenRouter: skip to next model if 429 wait exceeds this

_llm_semaphore: asyncio.Semaphore | None = None

# ── OpenRouter rate controls ──────────────────────────────────────────────────
# Free tier: ~20 req/min. Cap concurrency and track the sliding window.
_or_semaphore   = threading.Semaphore(MAX_CONCURRENT_OR)
_or_window: _deque = _deque()        # monotonic timestamps of recent OR requests
_or_window_lock = threading.Lock()

def _or_wait_for_slot() -> None:
    """Block until there is capacity within the OR_RPM sliding-window budget."""
    while True:
        with _or_window_lock:
            now = time.monotonic()
            while _or_window and now - _or_window[0] > 60.0:
                _or_window.popleft()
            if len(_or_window) < OR_RPM:
                _or_window.append(now)
                return
            sleep_for = 61.0 - (now - _or_window[0])
        jitter = random.uniform(0.5, 2.0)
        wait   = max(sleep_for, 0.0) + jitter
        logger.info(f"[OR gate] {len(_or_window)}/{OR_RPM} req/min — pausing {wait:.1f}s")
        time.sleep(wait)

def _parse_retry_after(headers: dict) -> float:
    """Return seconds to wait from a Retry-After header (integer or HTTP-date)."""
    ra = headers.get("Retry-After") or headers.get("retry-after", "")
    if not ra:
        return 0.0
    try:
        return float(ra)
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime
        retry_dt = parsedate_to_datetime(ra)
        delta = (retry_dt - __import__("datetime").datetime.now(retry_dt.tzinfo)).total_seconds()
        return max(delta, 0.0)
    except Exception:
        return 0.0

def _get_semaphore() -> asyncio.Semaphore:
    global _llm_semaphore
    if _llm_semaphore is None:
        _llm_semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)
    return _llm_semaphore

# ── Per-minute TPM (tokens-per-minute) gate ───────────────────────────────────
_tpm_window: _deque = _deque()  # (monotonic_timestamp, tokens) pairs
_tpm_lock: asyncio.Lock | None = None

async def _wait_for_tpm_headroom(tokens_needed: int = 800) -> None:
    """Block if adding tokens_needed would exceed GROQ_TPM_LIMIT in current minute."""
    global _tpm_lock
    if _tpm_lock is None:
        _tpm_lock = asyncio.Lock()
    async with _tpm_lock:
        now = time.monotonic()
        while _tpm_window and now - _tpm_window[0][0] > 60.0:
            _tpm_window.popleft()
        used = sum(t for _, t in _tpm_window)
        wait = 0.0
        if used + tokens_needed > GROQ_TPM_LIMIT and _tpm_window:
            wait = (_tpm_window[0][0] + 61.0) - now
    # Sleep outside the lock so other callers can proceed concurrently.
    if wait > 0:
        logger.info(f"[TPM gate] {used:,}/{GROQ_TPM_LIMIT:,} tokens/min — pausing {wait:.1f}s")
        await asyncio.sleep(wait)

def _record_tpm(tokens: int) -> None:
    _tpm_window.append((time.monotonic(), tokens))

# ── Token-bucket rate limiter ─────────────────────────────────────────────────
class _TokenBucket:
    def __init__(self, rpm: int):
        self._capacity    = float(rpm)
        self._tokens      = float(rpm)
        self._rate        = rpm / 60.0
        self._last_refill = time.monotonic()
        self._lock: asyncio.Lock | None = None

    def _refill(self) -> None:
        now   = time.monotonic()
        delta = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + delta * self._rate)
        self._last_refill = now

    async def consume_async(self) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            wait = (1.0 - self._tokens) / self._rate
            self._tokens = 0.0
        await asyncio.sleep(wait)

_rate_limiter = _TokenBucket(GROQ_RPM)

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
    if "qwen3" in _m:
        base["reasoning_effort"] = "none"
    if "kimi" in _m and role == "query":
        base["max_tokens"] = min(base["max_tokens"], 2048)
    return base

def _parse_wait(err: str) -> float:
    m = re.search(r"try again in (\d+)m([\d.]+)s", err)
    if m: return int(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r"try again in ([\d.]+)s", err)
    if m: return float(m.group(1))
    m = re.search(r"try again in (\d+) minute", err)
    if m: return int(m.group(1)) * 60
    return 0.0

def _is_daily_limit(err: str) -> bool:
    return "tokens per day" in err or "per day" in err

FATAL_ERRORS = ["model not found", "invalid api key", "authentication",
                "permission denied", "does not exist"]

# Errors that indicate the request is too large for this model — skip it
# immediately (mark tpd_exhausted so registry picks another model) but do
# NOT treat as globally fatal (another model may handle the same prompt).
_CTX_OVERFLOW_MARKERS = ["request too large", "context_length_exceeded",
                         "maximum context length"]

def _empty_result() -> dict:
    return {"raw_response": "", "completion_tokens": 0,
            "prompt_tokens": 0, "total_tokens": 0}

# ── Groq (sync) ───────────────────────────────────────────────────────────────
def query_llm(model: str, prompt: str, system: str = "",
              retries: int = 3, role: str = "analyst",
              client: Groq | None = None) -> dict:
    """
    Sync Groq LLM call with registry-driven model selection, retry, and
    rate-limit back-off. On each attempt the registry selects the best
    available model, so a single call may fan across multiple models.

    role: query | extractor | analyst
    """
    _base_url = os.environ.get("GROQ_BASE_URL", "")
    if _base_url:
        _client = client or Groq(api_key=GROQ_API_KEY, base_url=_base_url)
    else:
        _client = client or Groq(api_key=GROQ_API_KEY)
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]

    _fatal = False
    for attempt in range(1, retries + 1):
        model = registry.select(preferred=model)
        try:
            params = _model_params(model, role)
            resp   = _client.chat.completions.create(
                model=model, messages=messages, **params)  # type: ignore[arg-type]
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
            if any(m in err.lower() for m in _CTX_OVERFLOW_MARKERS):
                logger.warning(f"Request too large for {model} — skipping this model")
                registry.mark_tpd_exhausted(model)
                continue  # pick a different model next attempt, no sleep
            if _is_daily_limit(err):
                registry.mark_tpd_exhausted(model)
            else:
                wait_base = _parse_wait(err) or (2 ** attempt)
                if wait_base > _SWITCH_MODEL_THRESHOLD and registry.available_count() > 1:
                    registry.mark_tpm(model, min(wait_base, 60.0))
                    model = registry.select(preferred=model)
                    logger.info(f"429 wait {wait_base:.0f}s → switching to {model}")
                    # no sleep — immediately retry with new model
                else:
                    registry.mark_tpm(model, wait_base)
                    wait = min(wait_base, MAX_BACKOFF_SLEEP) * random.uniform(0.8, 1.2)
                    logger.debug(f"Waiting {wait:.1f}s (base={wait_base:.0f}s, jittered) …")
                    time.sleep(wait)

    if not _fatal and OPENROUTER_API_KEY:
        _or_model  = GROQ_TO_OPENROUTER_FALLBACK.get(model)
        _or_models = ([_or_model] if _or_model else []) + ["openrouter/free"]
        logger.info(f"[Fallback] Groq '{model}' exhausted → OpenRouter {_or_models}")
        raw = query_openrouter(prompt, models=_or_models)
        if raw:
            return {**_empty_result(), "raw_response": raw}

    return _empty_result()

# ── Groq (async) ──────────────────────────────────────────────────────────────
async def async_query_llm(model: str, prompt: str, system: str = "",
                           retries: int = 3, role: str = "analyst") -> dict:
    """
    Async Groq LLM call with three-layer rate protection and registry
    model selection:

    1. Token-bucket pre-throttle  — enforces GROQ_RPM before entering loop.
    2. Semaphore per-attempt      — MAX_CONCURRENT_LLM slots, released before sleep.
    3. Jittered backoff — wait *= random.uniform(0.5, 1.8); unknown errors get 8–20 s base.

    Registry integration: model is re-selected after acquiring the semaphore,
    so a temporarily rate-limited model is swapped out automatically.
    """
    await _wait_for_tpm_headroom()

    from groq import AsyncGroq

    _base_url    = os.environ.get("GROQ_BASE_URL", "")
    messages     = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]

    await _rate_limiter.consume_async()

    _fatal     = False
    _succeeded = False
    _result    = None
    _wait      = 0.0

    for attempt in range(1, retries + 1):
        async with _get_semaphore():
            _requested = model
            model = registry.select(preferred=_requested)
            if _requested and model != _requested:
                await asyncio.sleep(random.uniform(0.0, 0.35))
            _is_compound = "compound" in model.lower()
            _aclient = AsyncGroq(api_key=GROQ_API_KEY, base_url=_base_url) if _base_url \
                else AsyncGroq(api_key=GROQ_API_KEY)
            async with _aclient as aclient:
                try:
                    params = _model_params(model, role)
                    resp   = await aclient.chat.completions.create(
                        model=model, messages=messages, **params)  # type: ignore[arg-type]
                    _result = {
                        "raw_response":      resp.choices[0].message.content.strip(),
                        "completion_tokens": resp.usage.completion_tokens,
                        "prompt_tokens":     resp.usage.prompt_tokens,
                        "total_tokens":      resp.usage.total_tokens,
                    }
                    _track(_result)
                    _record_tpm(_result.get("total_tokens", 0))
                    _succeeded = True
                except Exception as e:
                    err = str(e)
                    logger.warning(f"[async] Attempt {attempt}/{retries} ({model}): {err[:80]}")
                    if any(f in err.lower() for f in FATAL_ERRORS):
                        logger.error(f"[async] Fatal error for model={model} — not retrying.")
                        _fatal = True
                    elif any(m in err.lower() for m in _CTX_OVERFLOW_MARKERS):
                        logger.warning(f"[async] Request too large for {model} — skipping")
                        registry.mark_tpd_exhausted(model)
                        # _wait stays 0 so we immediately retry with a different model
                    else:
                        wait_base = _parse_wait(err) or random.uniform(1.0, 3.0)
                        if _is_daily_limit(err):
                            registry.mark_tpd_exhausted(model)
                        elif wait_base > _SWITCH_MODEL_THRESHOLD and registry.available_count() > 1:
                            registry.mark_tpm(model, min(wait_base, 60.0))
                            model = registry.select(preferred=_requested)
                            logger.info(f"[async] 429 wait {wait_base:.0f}s → switching to {model}")
                            _wait = 0.0  # immediate retry with new model
                        else:
                            registry.mark_tpm(model, wait_base)
                            _wait = min(wait_base, MAX_BACKOFF_SLEEP) * random.uniform(0.8, 1.2)
                            logger.debug(f"[async] Waiting {_wait:.1f}s "
                                         f"(base={wait_base:.0f}s, jittered) …")

        if _succeeded:
            if _is_compound:
                await asyncio.sleep(_COMPOUND_EXTRA_DELAY)
            return _result or _empty_result()

        if _fatal:
            return _empty_result()

        await asyncio.sleep(_wait)

    if not _fatal and OPENROUTER_API_KEY:
        _or_model  = GROQ_TO_OPENROUTER_FALLBACK.get(model)
        _or_models = ([_or_model] if _or_model else []) + ["openrouter/free"]
        logger.info(f"[Async Fallback] Groq '{model}' exhausted → OpenRouter {_or_models}")
        raw = await asyncio.to_thread(query_openrouter, prompt, None, 2048, 0.7, 3, _or_models)
        if raw:
            return {**_empty_result(), "raw_response": raw}

    logger.error(f"[async] All retries exhausted for model={model}. Queueing for retry.")
    await _get_retry_queue().put({"model": model, "prompt": prompt,
                                   "system": system, "role": role})
    asyncio.ensure_future(drain_retry_queue())
    return _empty_result()


async def drain_retry_queue() -> list[dict]:
    """Flush all items from the retry queue, attempting each call once more."""
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
                  model: str = "mistral-small-latest",
                  api_key: str | None = None) -> dict:
    """Mistral API call — drop-in for query_llm for extraction tasks."""
    key = api_key or MISTRAL_API_KEY
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
    try:
        resp = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages,
                  "temperature": 0.0, "max_tokens": 1024},
            timeout=30,
        )
        if resp.status_code in (401, 403):
            logger.error("Mistral API key invalid — disabling Mistral for this session")
            return {**_empty_result(), "error_code": resp.status_code}
        data = resp.json()
        if "choices" not in data:
            logger.warning(f"Mistral API error: {data}")
            return _empty_result()
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
        return _empty_result()

# ── OpenRouter ────────────────────────────────────────────────────────────────
def query_openrouter(prompt: str, api_key: str | None = None,
                     max_tokens: int = 1024, temperature: float = 0.2,
                     max_retries: int = 3,
                     models: list | None = None) -> str:
    """
    OpenRouter call with:
      - threading.Semaphore to cap concurrent calls (MAX_CONCURRENT_OR)
      - Sliding-window RPM guard (OR_RPM)
      - Retry-After header parsing for accurate 429 back-off
      - Exponential back-off with jitter as fallback
      - Model rotation across the fallback list
    """
    key      = api_key or OPENROUTER_API_KEY
    endpoint = "https://openrouter.ai/api/v1/chat/completions"
    _models  = models if models is not None else OPENROUTER_MODELS

    with _or_semaphore:                     # cap concurrent OR calls
        for model in _models:
            for attempt in range(max_retries):
                _or_wait_for_slot()         # honour RPM budget before every request
                try:
                    resp = requests.post(
                        endpoint,
                        headers={"Authorization": f"Bearer {key}",
                                 "HTTP-Referer": "https://geo-agent.local",
                                 "Content-Type": "application/json"},
                        json={"model": model,
                              "messages": [{"role": "user", "content": prompt}],
                              "max_tokens": max_tokens, "temperature": temperature},
                        timeout=60,
                    )

                    if resp.status_code == 429:
                        ra   = _parse_retry_after(dict(resp.headers))
                        wait = ra if ra > 0 else (2 ** attempt) * 3 + random.uniform(0.5, 2)
                        if wait > _OR_MAX_WAIT:
                            logger.warning(f"[OR 429] {model} wait {wait:.0f}s > {_OR_MAX_WAIT:.0f}s — rotating to next model")
                            break
                        logger.warning(f"[OR 429] {model} attempt {attempt+1}/{max_retries} — waiting {wait:.1f}s")
                        time.sleep(wait)
                        continue

                    if resp.status_code == 404:
                        logger.debug(f"[OR 404] {model} — trying next model")
                        break

                    if resp.status_code != 200:
                        logger.warning(f"[OR {resp.status_code}] {model} — trying next model")
                        break

                    data = resp.json()
                    if "error" in data:
                        code = data["error"].get("code") or data["error"].get("status")
                        if code == 429 or str(code) == "429":
                            wait = (2 ** attempt) * 3 + random.uniform(0.5, 2)
                            if wait > _OR_MAX_WAIT:
                                logger.warning(f"[OR 429 body] {model} wait {wait:.0f}s — rotating to next model")
                                break
                            logger.warning(f"[OR 429 body] {model} — waiting {wait:.1f}s")
                            time.sleep(wait)
                            continue
                        logger.warning(f"[OR error] {model}: {data['error']}")
                        break

                    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                    if content:
                        logger.debug(f"[OR OK] {model}")
                        return content
                    break

                except requests.exceptions.Timeout:
                    wait = (2 ** attempt) * 3 + random.uniform(0, 2)
                    logger.warning(f"[OR timeout] {model} attempt {attempt+1} — retrying in {wait:.1f}s")
                    time.sleep(wait)
                except Exception as exc:
                    logger.error(f"[OR error] {model}: {exc}")
                    break

    logger.warning("[OR] All models exhausted — no response from OpenRouter.")
    return ""

# ── LLM response cache ────────────────────────────────────────────────────────
LLM_CACHE_FILE = str(Path(__file__).parent.parent / "llm_cache.json")

# Load once at import time; keep in memory for the process lifetime.
def _load_cache() -> dict:
    try:
        with open(LLM_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

_llm_cache: dict = _load_cache()
_cache_lock = threading.Lock()

def _save_cache() -> None:
    try:
        with open(LLM_CACHE_FILE, "w") as f:
            json.dump(_llm_cache, f, indent=2)
    except Exception:
        pass

def cached_llm_call(prompt: str, call_fn) -> str:
    """Call LLM only if this exact prompt has not been called before."""
    key = hashlib.md5(prompt.encode()).hexdigest()
    with _cache_lock:
        if key in _llm_cache:
            logger.debug("Using cached LLM response")
            return _llm_cache[key]
    result = call_fn(prompt)
    if result:
        with _cache_lock:
            _llm_cache[key] = result
        _save_cache()
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

    first_bracket = next((i for i, c in enumerate(text) if c in "[{"), -1)
    last_bracket  = next((i for i, c in enumerate(reversed(text)) if c in "]}"), -1)
    if first_bracket != -1 and last_bracket != -1:
        sliced = text[first_bracket : len(text) - last_bracket]
        try: return _unwrap(json.loads(sliced))
        except json.JSONDecodeError: pass

    try: return _unwrap(json.loads(text))
    except json.JSONDecodeError: pass

    for pattern in (r"(\[[\s\S]*?\])", r"(\{[\s\S]*?\})"):
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

    entities = re.findall(r'"([^"]{2,80})"', raw)
    if entities:
        logger.warning(f"{label}: JSON broken — salvaged {len(entities)} quoted strings.")
        return entities

    logger.warning(f"{label}: could not parse JSON.\n  RAW: {raw[:300]}")
    return []

clean_and_parse_json = parse_json  # alias for Project 1 agent compatibility

# ── Project 1 compatibility wrappers ─────────────────────────────────────────
def call_llm(prompt: str, system: str = "", preferred_model: str | None = None,
             max_completion_tokens: int = 4096, temperature: float = 0.2,
             tpm_max_wait: float = 90.0, max_attempts: int = 6) -> dict:
    """Thin alias over query_llm for Project 1 agent compatibility."""
    model = preferred_model or MODEL_STRONG
    return query_llm(model, prompt, system=system, retries=max_attempts)


def call_llm_json(prompt: str, system: str = "",
                  preferred_model: str | None = None) -> list | dict:
    """Thin alias: call_llm + parse_json for Project 1 agent compatibility."""
    result = call_llm(prompt, system=system, preferred_model=preferred_model)
    raw = result.get("raw_response", "") if isinstance(result, dict) else str(result)
    return parse_json(raw, label="call_llm_json")

# ── Provider-agnostic unified query ──────────────────────────────────────────
def query_provider(provider: str, model: str, prompt: str,
                   system: str = "", **kwargs) -> dict:
    """
    Unified entry point — route to the right backend by provider name.

    provider : "groq" | "mistral" | "openrouter"
    Returns  : same dict as query_llm — {"raw_response", "total_tokens", ...}
    """
    provider = provider.lower()
    if provider == "groq":
        return query_llm(model, prompt, system=system, **kwargs)
    if provider == "mistral":
        return query_mistral(prompt, system=system, **kwargs)
    if provider == "openrouter":
        raw = query_openrouter(prompt, **kwargs)
        return {**_empty_result(), "raw_response": raw}
    raise ValueError(f"Unknown provider '{provider}'. Use 'groq', 'mistral', or 'openrouter'.")
