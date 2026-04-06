"""Dynamic model registry for the GEO pipeline.

Instead of a hardcoded fallback chain, this module:
1. Discovers available models from Groq's API at startup.
2. Tracks per-model rate-limit state (TPM / TPD) at runtime.
3. Selects the best available model for each call based on current state.

This makes model selection a runtime decision, not a hardcoded sequence.
"""

import re
import time
import threading
import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GROQ_API_KEY

# ---------------------------------------------------------------------------
# Model capability tiers — used to prefer higher-quality models when available.
# The registry discovers which models actually exist; these tiers determine
# preference order among the ones that are available and not rate-limited.
# ---------------------------------------------------------------------------

# Keywords that identify model quality tier (higher index = lower preference)
_TIER_KEYWORDS = [
    # Tier 0 — largest / best
    ["70b", "72b", "qwen3-32b", "qwen-32b"],
    # Tier 1 — medium
    ["32b", "13b", "14b", "8b-instant", "small"],
    # Tier 2 — fast / small
    ["8b", "7b", "instant", "mini"],
]


def _tier(model_id: str) -> int:
    mid = model_id.lower()
    for i, keywords in enumerate(_TIER_KEYWORDS):
        if any(kw in mid for kw in keywords):
            return i
    return 99  # unknown — lowest preference


# ---------------------------------------------------------------------------
# Per-model state
# ---------------------------------------------------------------------------

class _ModelState:
    def __init__(self, model_id: str):
        self.id            = model_id
        self.tpd_exhausted = False     # True = skip for rest of session
        self.tpm_retry_at  = 0.0       # epoch time when TPM resets
        self.recent_calls  = 0         # increments on mark_tpm, resets when tpm clears

    @property
    def available(self) -> bool:
        if self.tpd_exhausted:
            return False
        if time.time() >= self.tpm_retry_at:
            self.recent_calls = 0   # TPM window expired, reset counter
            return True
        return False

    def mark_tpd_exhausted(self):
        self.tpd_exhausted = True
        print(f"[registry] {self.id}: daily limit exhausted — removed from pool")

    def mark_tpm(self, retry_after_seconds: float):
        self.tpm_retry_at = time.time() + retry_after_seconds
        self.recent_calls += 1
        print(f"[registry] {self.id}: TPM limit — available again in {retry_after_seconds:.0f}s")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """Discovers and tracks Groq models. Thread-safe."""

    # Models that are irrelevant for text generation (whisper, tts, vision-only, etc.)
    _SKIP_PATTERNS = re.compile(
        r"whisper|tts|vision|guard|distil|embed|rerank|reward|tool", re.I
    )

    def __init__(self):
        self._lock   = threading.Lock()
        self._states: dict[str, _ModelState] = {}
        self._loaded = False

    def _load(self):
        """Discover available models from Groq API. Called once lazily."""
        try:
            resp = requests.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                timeout=10,
            )
            resp.raise_for_status()
            models = resp.json().get("data", [])
            discovered = [
                m["id"] for m in models
                if not self._SKIP_PATTERNS.search(m.get("id", ""))
            ]
            for mid in discovered:
                self._states[mid] = _ModelState(mid)
            print(f"[registry] Discovered {len(discovered)} models: {discovered}")
        except Exception as e:
            print(f"[registry] Could not discover models from API ({e}) — using defaults")
            # Minimal fallback so the pipeline can still run
            defaults = [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "qwen/qwen3-32b",
                "llama3-8b-8192",
            ]
            for mid in defaults:
                self._states[mid] = _ModelState(mid)
        self._loaded = True

    def _ensure_loaded(self):
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._load()

    # ── Public API ─────────────────────────────────────────────────────────────

    def select(self, preferred: str | None = None) -> str:
        """Return the best currently-available model.

        Preference order:
        1. `preferred` if available.
        2. Available models sorted by capability tier (best first).

        Raises RuntimeError if no models are available.
        """
        self._ensure_loaded()
        with self._lock:
            if preferred and preferred in self._states:
                if self._states[preferred].available:
                    return preferred

            available = [s for s in self._states.values() if s.available]
            if not available:
                exhausted = [s.id for s in self._states.values() if s.tpd_exhausted]
                tpm_blocked = [
                    s.id for s in self._states.values()
                    if not s.tpd_exhausted and not s.available
                ]
                raise RuntimeError(
                    f"No models available.\n"
                    f"  Daily exhausted : {exhausted}\n"
                    f"  TPM blocked     : {tpm_blocked}"
                )

            # Sort by tier (lower = better), then by recent_calls (fewer = preferred),
            # then alphabetically for stability
            available.sort(key=lambda s: (_tier(s.id), s.recent_calls, s.id))
            chosen = available[0].id
            avail = sum(1 for s in self._states.values() if s.available)
            if avail < 2:
                from loguru import logger
                logger.warning(f"[Registry] Only {avail} model(s) available — rate limits imminent!")
            if preferred and chosen != preferred:
                print(f"[registry] Selected model: {chosen} (preferred '{preferred}' unavailable)")
            return chosen

    def mark_tpd_exhausted(self, model_id: str):
        self._ensure_loaded()
        with self._lock:
            if model_id in self._states:
                self._states[model_id].mark_tpd_exhausted()

    def mark_tpm(self, model_id: str, retry_after_seconds: float):
        self._ensure_loaded()
        with self._lock:
            if model_id in self._states:
                self._states[model_id].mark_tpm(retry_after_seconds)

    def available_count(self) -> int:
        """Returns number of models currently available (not TPD-exhausted, not TPM-blocked)."""
        self._ensure_loaded()
        with self._lock:
            return sum(1 for s in self._states.values() if s.available)

    def status(self) -> dict:
        """Return a summary of model availability (useful for debugging)."""
        self._ensure_loaded()
        with self._lock:
            return {
                mid: {
                    "available":      s.available,
                    "tpd_exhausted":  s.tpd_exhausted,
                    "tpm_retry_in":   max(0.0, s.tpm_retry_at - time.time()),
                }
                for mid, s in self._states.items()
            }


# Singleton — shared across all agents in the process.
registry = ModelRegistry()
