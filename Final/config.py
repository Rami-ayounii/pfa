"""Shared configuration for the merged GEO pipeline.

All secrets are read from environment variables; no hard-coded keys.
Copy .env.example -> .env and fill in your keys.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Fix UTF-8 stdout/stderr on Windows (avoids cp1252 UnicodeEncodeError
# when print statements contain em-dashes, arrows, or other non-ASCII chars).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

load_dotenv(Path(__file__).parent / ".env")

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY       = os.environ.get("GROQ_API_KEY", "")
MISTRAL_API_KEY    = os.environ.get("MISTRAL_API_KEY", "")
SERPAPI_KEY        = os.environ.get("SERPAPI_KEY", "")
APIFY_API_TOKEN    = os.environ.get("APIFY_API_TOKEN", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Optional: override Groq base URL (e.g. for any OpenAI-compatible endpoint)
GROQ_BASE_URL = os.environ.get("GROQ_BASE_URL", "")

# ── Model constants (overridable via env vars) ────────────────────────────────
#
# GROQ MODEL TIERS (verified Apr 2026 — https://console.groq.com/docs/models):
#
#   TIER 1 — FREE  (safe defaults, 1 000 RPM)
#   llama-3.1-8b-instant          250K TPM  1K RPM  fastest
#   llama-3.3-70b-versatile       300K TPM  1K RPM  best free quality
#   openai/gpt-oss-20b            250K TPM  1K RPM  free reasoning model
#
#   TIER 2 — PREVIEW  (may return 429 on free tier)
#   qwen/qwen3-32b                300K TPM  1K RPM  multilingual fr/ar
#   meta-llama/llama-4-scout-17b-16e-instruct  300K TPM  1K RPM
#
#   TIER 3 — PAID ONLY
#   groq/compound                 200K TPM  200 RPM  lowest RPM — avoid as default
#   openai/gpt-oss-120b           250K TPM  1K RPM
#   moonshotai/kimi-k2-instruct-0905  250K TPM  1K RPM  $1.00/$3.00 per 1M

# Tier 1 — Free, high RPM (safe defaults)
MODEL_FAST      = os.environ.get("MODEL_FAST",      "llama-3.1-8b-instant")
MODEL_STRONG    = os.environ.get("MODEL_STRONG",    "llama-3.3-70b-versatile")
MODEL_GPTOSS    = os.environ.get("MODEL_GPTOSS",    "openai/gpt-oss-20b")
MODEL_EXTRACTOR = os.environ.get("MODEL_EXTRACTOR", "gemma2-9b-it")

# Tier 2 — Preview
MODEL_QWEN      = os.environ.get("MODEL_QWEN",   "qwen/qwen3-32b")
MODEL_LLAMA4    = os.environ.get("MODEL_LLAMA4", "meta-llama/llama-4-scout-17b-16e-instruct")

# Tier 3 — Paid only (defaults fall back to free equivalents)
MODEL_COMPOUND  = os.environ.get("MODEL_COMPOUND", "llama-3.3-70b-versatile")
MODEL_ANALYST   = os.environ.get("MODEL_ANALYST",  "openai/gpt-oss-120b")
MODEL_KIMI      = os.environ.get("MODEL_KIMI",     "moonshotai/kimi-k2-instruct-0905")

# Project 1 aliases (kept for backward-compat with Geo/ nodes)
MODEL_INTENT         = os.environ.get("MODEL_INTENT",         "llama-3.3-70b-versatile")
MODEL_ANALYST2       = os.environ.get("MODEL_ANALYST2",       "llama-3.3-70b-versatile")
MODEL_FALLBACK_HEAVY = os.environ.get("MODEL_FALLBACK_HEAVY", "qwen/qwen3-32b")
MODEL_EXTRACTOR2     = os.environ.get("MODEL_EXTRACTOR2",     "llama-3.3-70b-versatile")
MODEL_EXTRACTOR3     = os.environ.get("MODEL_EXTRACTOR3",     "llama-3.1-8b-instant")
GROQ_MODEL           = os.environ.get("GROQ_MODEL",           "llama-3.3-70b-versatile")

# 4 diverse query models: 2 Groq + 2 OpenRouter (different TPM pools)
DEFAULT_QUERY_MODELS = [
    MODEL_STRONG,   # llama-3.3-70b-versatile  (Groq, tier 0, 300K TPM)
    MODEL_LLAMA4,   # llama-4-scout             (Groq, tier 0, 300K TPM — separate quota)
    MODEL_QWEN,     # qwen/qwen3-32b            (OpenRouter, tier 0)
    MODEL_GPTOSS,   # openai/gpt-oss-20b        (OpenRouter, tier 1)
]

# Groq fallback chain (tried in order on 429 rate-limit errors)
GROQ_FALLBACK_CHAIN = [
    MODEL_STRONG,           # 70B — best quality
    MODEL_LLAMA4,           # llama-4-scout — new strong
    MODEL_QWEN,             # qwen3-32b — multilingual
    MODEL_EXTRACTOR,        # gemma2-9B — fast structured
    MODEL_FAST,             # 8B-instant — fastest
    "llama3-8b-8192",       # legacy last resort
]

# OpenRouter model list (comma-separated env var overrides)
# Ordered by quality; rotate through all on 429 — having 7+ models means
# even if the first 2-3 are rate-limited we still get a fast response.
_or_env = os.environ.get("OPENROUTER_MODELS", "")
OPENROUTER_MODELS = (
    [m.strip() for m in _or_env.split(",") if m.strip()]
    if _or_env else [
        # Tier 0 — largest / strongest
        "openai/gpt-oss-120b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "nousresearch/hermes-3-llama-3.1-405b:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "qwen/qwen3.6-plus:free",
        # Tier 1 — solid mid-size
        "openai/gpt-oss-20b:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemma-3-27b-it:free",
        "minimax/minimax-m2.5:free",
        "z-ai/glm-4.5-air:free",
        # Tier 2 — fast / small fallbacks
        "google/gemma-3-12b-it:free",
        "stepfun/step-3.5-flash:free",
        "nvidia/nemotron-nano-12b-v2-vl:free",
        "meta-llama/llama-3.2-3b-instruct:free",
    ]
)

# ── Concurrency controls ──────────────────────────────────────────────────────
MAX_CONCURRENT_LLM    = int(os.environ.get("MAX_CONCURRENT_LLM",    "5"))
MAX_CONCURRENT_BRANDS = int(os.environ.get("MAX_CONCURRENT_BRANDS", "3"))
MAX_CONCURRENT_OR     = int(os.environ.get("MAX_CONCURRENT_OR",     "2"))   # OpenRouter free tier
GROQ_RPM              = int(os.environ.get("GROQ_RPM",              "1000"))
OR_RPM                = int(os.environ.get("OR_RPM",                "18"))  # OpenRouter free ~20/min, keep 10% headroom

# ── Agent defaults ────────────────────────────────────────────────────────────
N_RUNS               = int(os.environ.get("N_RUNS",               "1"))
N_INTENTS            = int(os.environ.get("N_INTENTS",            "4"))
N_VARIANTS           = int(os.environ.get("N_VARIANTS",           "2"))
LANGUAGES            = os.environ.get("LANGUAGES", "en,fr,ar").split(",")
MAX_REFLECTION_LOOPS = int(os.environ.get("MAX_REFLECTION_LOOPS", "1"))
MAX_RETRIES          = int(os.environ.get("MAX_RETRIES",          "3"))

# Inter-prompt delay: breathing room between prompt batches in agent1_llm_query_node.py
INTER_PROMPT_DELAY_S = float(os.environ.get("INTER_PROMPT_DELAY_S", "0.6"))
# Per-minute token cap: 93% of 300K Groq TPM limit (conservative headroom)
GROQ_TPM_LIMIT = int(os.environ.get("GROQ_TPM_LIMIT", "280000"))

# ── Output paths ──────────────────────────────────────────────────────────────
# Use an absolute path anchored to Final/ so the pipeline and status_server.py
# always agree on the output location regardless of the working directory.
_THIS_DIR        = Path(__file__).parent
OUTPUT_DIR       = os.environ.get("OUTPUT_DIR", str(_THIS_DIR / "geo_output"))
AGENT2_OUTPUT_DIR = os.environ.get("AGENT2_OUTPUT_DIR", "agent2_output")

RAW_OUTPUT_PATH         = f"{OUTPUT_DIR}/raw_responses.csv"
EXTRACTED_ENTITIES_PATH = f"{OUTPUT_DIR}/extracted_entities.csv"
CLEAN_ENTITIES_PATH     = f"{OUTPUT_DIR}/clean_entities.csv"
FEATURES_PATH           = f"{OUTPUT_DIR}/entity_features.csv"
FEATURES_GLOBAL_PATH    = f"{OUTPUT_DIR}/entity_features_global.csv"

CHECKPOINT_FILE = f"{AGENT2_OUTPUT_DIR}/web_features.csv"
VALIDATION_LOG  = f"{AGENT2_OUTPUT_DIR}/validation_log.csv"
AUDIT_LOG       = f"{AGENT2_OUTPUT_DIR}/audit_log.json"
