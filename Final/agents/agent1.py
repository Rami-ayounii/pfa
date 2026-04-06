# -*- coding: utf-8 -*-
"""Agent 1 — Entity extraction, enrichment, cleaning, and GEO feature computation."""

import os
import json
import re
import time
import requests
import pandas as pd
import csv
import math
from groq import Groq, RateLimitError

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import GROQ_API_KEY, MISTRAL_API_KEY
from core.model_registry import registry as _model_registry

# Lazy Groq client — never instantiated at import time.
_groq_client: Groq | None = None

def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ── config ────────────────────────────────────────────────────────────────────

QUERY_MODELS = [
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
]

MODEL_EXTRACTOR       = "mistral-small-latest"
MODEL_EXTRACTOR2      = "llama-3.3-70b-versatile"
MODEL_EXTRACTOR3      = "llama-3.1-8b-instant"
MODEL_ANALYST         = "openai/gpt-oss-120b"
MODEL_ANALYST2        = "llama-3.3-70b-versatile"
MODEL_FALLBACK_HEAVY  = "qwen/qwen3-32b"

N_RUNS          = 2
OUTPUT_DIR      = "geo_output"
RAW_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "raw_responses.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── token tracking ─────────────────────────────────────────────────────────────

TOKEN_USAGE: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def reset_token_usage() -> None:
    """Reset TOKEN_USAGE before a new node execution."""
    TOKEN_USAGE.update({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

def _track_tokens(usage: dict) -> None:
    for key in TOKEN_USAGE:
        TOKEN_USAGE[key] += usage.get(key, 0)


# ── model params ───────────────────────────────────────────────────────────────

def _model_params(model: str, role: str = "analyst") -> dict:
    if role == "query":
        base = {"temperature": 0.7, "max_tokens": 8192, "top_p": 0.9}
    elif role == "extractor":
        base = {"temperature": 0.0, "max_tokens": 1024, "top_p": 1.0}
    elif role == "analyst":
        base = {"temperature": 0.1, "max_tokens": 8192, "top_p": 0.95}
        if "gpt-oss" in model.lower():
            base["reasoning_effort"] = "low"
    else:
        base = {"temperature": 0.2, "max_tokens": 8192, "top_p": 0.9}

    if "qwen3" in model.lower():
        base["reasoning_effort"] = "none"

    return base


def _parse_wait_time(err_msg: str) -> float:
    m = re.search(r"try again in (\d+)m([\d.]+)s", err_msg)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r"try again in ([\d.]+)s", err_msg)
    if m:
        return float(m.group(1))
    m = re.search(r"try again in (\d+) minute", err_msg)
    if m:
        return int(m.group(1)) * 60
    return 0.0


FATAL_ERRORS = [
    "model not found",
    "invalid api key",
    "authentication",
    "permission denied",
    "does not exist"
]


# ── query_llm ─────────────────────────────────────────────────────────────────

def query_llm(model: str, prompt: str, system: str = "",
              retries: int = 3, role: str = "analyst") -> dict:
    """
    Single LLM call with retry, rate limit handling, token tracking,
    and automatic model fallback via the dynamic ModelRegistry.
    role: query | extractor | analyst
    """
    empty = {"raw_response": "", "completion_tokens": 0,
             "prompt_tokens": 0, "total_tokens": 0}

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(retries * 4):  # enough room to cycle through models
        try:
            candidate = _model_registry.select(preferred=model)
        except RuntimeError as e:
            print(f"[ERROR] {e}")
            return empty

        try:
            params   = _model_params(candidate, role=role)
            response = _get_client().chat.completions.create(
                model    = candidate,
                messages = messages,
                stop     = None,
                **params,
            )
            result = {
                "raw_response":      response.choices[0].message.content.strip(),
                "completion_tokens": response.usage.completion_tokens,
                "prompt_tokens":     response.usage.prompt_tokens,
                "total_tokens":      response.usage.total_tokens,
            }
            _track_tokens(result)
            if candidate != model:
                print(f"[llm] Used fallback model: {candidate}")
            return result

        except RateLimitError as e:
            err_msg = str(e)
            if "tokens per day" in err_msg or "per day" in err_msg:
                _model_registry.mark_tpd_exhausted(candidate)
            else:
                wait = _parse_wait_time(err_msg)
                if wait > 0 and wait <= 90:
                    _model_registry.mark_tpm(candidate, wait)
                    print(f"[RATE LIMIT] {candidate}: waiting {wait:.0f}s...")
                    time.sleep(wait + 2)
                else:
                    _model_registry.mark_tpm(candidate, wait)
                    print(f"[RATE LIMIT] {candidate}: wait {wait:.0f}s — trying another model")

        except Exception as e:
            err_msg = str(e)
            print(f"[WARN] {candidate}: {e}")
            if any(msg in err_msg.lower() for msg in FATAL_ERRORS):
                print(f"[FATAL] Non-recoverable error: {err_msg}")
                return empty
            time.sleep(2 ** min(attempt, 4))

    print(f"[ERROR] All models exhausted.")
    return empty


# ── clean_and_parse_json ───────────────────────────────────────────────────────

def clean_and_parse_json(raw: str, label: str = "") -> list:
    """
    Multi-layer JSON parser.
    Handles markdown fences, nested arrays, trailing commas,
    smart quotes, unescaped newlines, qwen3 thinking blocks.
    """
    if not raw or not raw.strip():
        print(f"[WARN] {label}: empty response.")
        return []

    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    def _unwrap(parsed):
        if (isinstance(parsed, list)
                and len(parsed) == 1
                and isinstance(parsed[0], list)):
            return parsed[0]
        return parsed

    try:
        return _unwrap(json.loads(text))
    except json.JSONDecodeError:
        pass

    for pattern in (r"(\[.*\])", r"(\{.*\})"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return _unwrap(json.loads(match.group(1)))
            except json.JSONDecodeError:
                pass

    fixed = re.sub(r",\s*([}\]])", r"\1", text)
    fixed = fixed.replace("\u201c", '"').replace("\u201d", '"')
    try:
        return _unwrap(json.loads(fixed))
    except json.JSONDecodeError:
        pass

    try:
        repaired = re.sub(r'(?<!\\)\n', r'\\n', text)
        return _unwrap(json.loads(repaired))
    except json.JSONDecodeError:
        pass

    print(f"[WARN] {label}: could not parse JSON.")
    print(f"[RAW] {raw[:400]}")
    return []


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Batch loader
# Input  : prompt_set.csv
# Output : list of prompt dicts
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_SET_PATH = "prompt_set_Tunisian_restaurants.csv"  # default path used by node


def agent1_load_prompts(prompt_set_path: str) -> list:

    if not os.path.exists(prompt_set_path):
        print(f"[ERROR] Prompt set not found: {prompt_set_path}")
        return []

    df = pd.read_csv(prompt_set_path, encoding='utf-8-sig')

    # verify required columns
    required_cols = ['prompt_id', 'intent_id', 'language', 'prompt_text']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Missing columns in prompt set: {missing}")
        return []

    prompts = df.to_dict(orient='records')

    print(f"   Step 1 : Batch loader complete")
    print(f"   Prompts loaded : {len(prompts)}")
    print(f"   Intent types   : {df['intent_id'].unique().tolist()}")
    print(f"   Languages      : {df['language'].unique().tolist()}")

    return prompts


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — LLM querying + progressive saving
# Input  : list of prompt dicts from Step 1
# Output : raw_responses.csv
# Schema : response_id · prompt_id · run_id · model_id · intent_id ·
#          language · prompt_text · response_text · completion_tokens ·
#          prompt_tokens · total_tokens · timestamp
# ══════════════════════════════════════════════════════════════════════════════

def step2_query_single(prompt_id: str, prompt_text: str,
                       model_id: str, run_id: str, output_dir: str) -> dict:
    """
    Fan-out target for LangGraph: execute one LLM call per
    (prompt × model × run) combination.

    Compatible with LangGraph Send fan-out — stateless, returns a dict
    that the graph merges into raw_responses (Annotated[list, operator.add]).
    """
    response_id = f"{prompt_id}__{model_id.replace('/', '-')}__{run_id}"

    result = query_llm(
        model  = model_id,
        prompt = prompt_text,
        role   = "query",
    )

    if not result["raw_response"]:
        return {}

    record = {
        "response_id"      : response_id,
        "prompt_id"        : prompt_id,
        "model_id"         : model_id,
        "run_id"           : run_id,
        "response_text"    : result["raw_response"],
        "completion_tokens": result["completion_tokens"],
        "prompt_tokens"    : result["prompt_tokens"],
        "total_tokens"     : result["total_tokens"],
        "timestamp"        : time.time(),
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "raw_responses.csv")
    header_needed = not os.path.exists(out_path)
    pd.DataFrame([record]).to_csv(
        out_path, mode="a", header=header_needed,
        index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL,
    )

    return record


def agent1_query_prompts(
    prompts:     list,
    models:      list = QUERY_MODELS,
    n_runs:      int  = N_RUNS,
    output_path: str  = RAW_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Step 2: LLM querying with progressive saving and resume support.

    For each prompt x each model x n_runs:
        -> query_llm(role="query")
        -> save record to CSV immediately
    """

    # resume support
    done_keys = set()
    if os.path.exists(output_path):
        df_existing = pd.read_csv(
            output_path, encoding='utf-8-sig', quoting=csv.QUOTE_ALL
        )
        for _, row in df_existing.iterrows():
            done_keys.add((row['prompt_id'], row['model_id'], row['run_id']))
        print(f"Resuming - {len(done_keys)} records already saved")

    total_prompts  = len(prompts)
    total_calls    = total_prompts * len(models) * n_runs
    done_calls     = len(done_keys)
    header_written = os.path.exists(output_path)

    print(f"\nStep 2: LLM querying")
    print(f"   Prompts        : {total_prompts}")
    print(f"   Models         : {models}")
    print(f"   Runs/prompt    : {n_runs}")
    print(f"   Total calls    : {total_calls}")
    print(f"   Already done   : {done_calls}")
    print(f"   Remaining      : {total_calls - done_calls}\n")

    for i, prompt_dict in enumerate(prompts):
        prompt_id   = prompt_dict['prompt_id']
        prompt_text = prompt_dict['prompt_text']
        intent_id   = prompt_dict.get('intent_id', '')
        language    = prompt_dict.get('language', '')

        print(f"\n[{i+1}/{total_prompts}] {prompt_id} | {intent_id} | {language}")
        print(f"  Prompt: {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")

        for model_id in models:
            print(f"\n  Model: {model_id}")

            for run_idx in range(1, n_runs + 1):
                run_id      = f"run_{run_idx}"
                response_id = f"{prompt_id}__{model_id.replace('/', '-')}__{run_id}"

                # skip if already done
                if (prompt_id, model_id, run_id) in done_keys:
                    print(f"     [{run_id}] skipped ({response_id})")
                    continue

                result = query_llm(
                    model  = model_id,
                    prompt = prompt_text,
                    role   = "query"
                )

                if not result['raw_response']:
                    print(f"     [{run_id}] empty response, skipping")
                    continue

                record = {
                    "response_id"      : response_id,
                    "prompt_id"        : prompt_id,
                    "model_id"         : model_id,
                    "run_id"           : run_id,
                    "response_text"    : result['raw_response'],
                    "completion_tokens": result['completion_tokens'],
                    "prompt_tokens"    : result['prompt_tokens'],
                    "total_tokens"     : result['total_tokens'],
                    "timestamp"        : time.time(),
                }
                # progressive save - one record at a time
                pd.DataFrame([record]).to_csv(
                    output_path,
                    mode     = 'a',
                    header   = not header_written,
                    index    = False,
                    encoding = 'utf-8-sig',
                    quoting  = csv.QUOTE_ALL
                )
                header_written = True
                done_keys.add((prompt_id, model_id, run_id))

                print(f"     [{run_id}] {result['total_tokens']} tokens | {response_id}")

    # final summary
    df_raw = pd.read_csv(output_path, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)

    print(f"\n{'='*55}")
    print(f"Step 2 complete")
    print(f"   Total records  : {len(df_raw)}")
    print(f"   Prompts covered: {df_raw['prompt_id'].nunique()}")
    print(f"   Models used    : {df_raw['model_id'].unique().tolist()}")
    print(f"   Total tokens   : {TOKEN_USAGE['total_tokens']}")
    print(f"   Saved to       : {output_path}")
    print(f"{'='*55}")

    return df_raw


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Entity extraction
# Input  : raw_responses.csv
# Output : extracted_entities.csv
# ══════════════════════════════════════════════════════════════════════════════

def extract_entity_description(raw_text: str, entity: str) -> str:
    """
    Extract the text block dedicated to this entity from a full LLM response.
    Handles 4 formats: numbered list, bold header, markdown table, free-form prose.
    """
    if not raw_text or not entity:
        return ""

    entity_lower      = entity.lower()
    lines             = raw_text.split("\n")
    description_lines = []
    inside_entity     = False
    consecutive_empty = 0

    # Format 3: markdown table
    for line in lines:
        if "|" in line and entity_lower in line.lower():
            cells = [c.strip() for c in line.split("|") if c.strip()]
            return " — ".join(cells)

    # Formats 1, 2, 4: list / bold header / prose
    for line in lines:
        if not inside_entity:
            if entity_lower in line.lower():
                inside_entity = True
                description_lines.append(line)
            continue

        stripped = line.strip()

        # stop: new numbered item → next entity
        if re.match(r"^\s*\d+\.\s+", line) and description_lines:
            break

        # stop: bold header that is NOT this entity → next entity
        if re.match(r"^\s*\*\*[^*]+\*\*", line) and description_lines:
            if entity_lower not in line.lower():
                break

        # stop: markdown table row for a different entity
        if "|" in line and entity_lower not in line.lower() and description_lines:
            break

        # stop: two consecutive blank lines
        if stripped == "":
            consecutive_empty += 1
            if consecutive_empty >= 2:
                break
            description_lines.append(line)
            continue

        consecutive_empty = 0
        description_lines.append(line)

    return "\n".join(description_lines).strip()


MISTRAL_MODEL = "mistral-small-latest"


def query_mistral(prompt: str, system: str = "") -> dict:
    """
    Mistral API call — returns same dict format as query_llm
    so it's a drop-in replacement for extraction tasks.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {MISTRAL_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model"      : MISTRAL_MODEL,
            "messages"   : messages,
            "temperature": 0.0,
            "max_tokens" : 1024,
        }
    )

    data = response.json()

    if "choices" not in data:
        print(f"[WARN] Mistral API error: {data}")
        return {"raw_response": "", "completion_tokens": 0,
                "prompt_tokens": 0, "total_tokens": 0}

    return {
        "raw_response"     : data["choices"][0]["message"]["content"].strip(),
        "completion_tokens": data["usage"]["completion_tokens"],
        "prompt_tokens"    : data["usage"]["prompt_tokens"],
        "total_tokens"     : data["usage"]["total_tokens"],
    }


ENTITIES_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "extracted_entities.csv")


BRAND_EXTRACTION_SYSTEM = """\
You are a restaurant name extractor working in a GEO analysis pipeline.

Your ONLY task is to find every named restaurant, cafe, or food business
in the text. The text may be a numbered list, markdown table, bold headers,
or free-form prose - read ALL formats.

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
WHAT TO EXTRACT
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
Extract ONLY named establishments - places with a proper name:
  + Named restaurants     "Dar El Jeld", "Le Corsaire", "Chez Ahmed"
  + Named cafes           "Cafe des Nattes", "Le Petit Cafe"
  + Named sandwich shops  "Sandwich Chez Ali", "Le Snack du Port"
  + Named food businesses "Patisserie Mabrouk", "Boulangerie Paris"

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
WHAT TO EXCLUDE
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
Exclude items that are NOT a proper establishment name:
  - Standalone dish names with no business name
    ("couscous", "brik", "mlewi", "harissa", "shakshuka", "falafel")
  - Purely generic descriptions with no proper name
    ("a local restaurant", "the best cafe", "street food stalls")
  - City or region names  ("Tunis", "Sfax", "Djerba", "Tunisia")
  - People names          ("Chef Ahmed", "Mohamed")

NOTE: "sandwich" or "pizza" as PART of a business name is allowed.
  KEEP  -> "sandwich chez ahmed"
  KEEP  -> "pizza napoli"
  KEEP  -> "Mlewi Lazher"
  SKIP  -> "sandwich"
  SKIP  -> "mlewi"

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
NORMALIZATION - CRITICAL
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
Every name must be byte-for-byte identical across all runs.

  1. Lowercase only
       "Dar El Jeld"       -> "dar el jeld"
       "Le Corsaire"       -> "le corsaire"

  2. Keep geographic/cultural articles
       "Dar", "El", "Le", "La", "Les", "Al", "Chez" -> always keep

  3. Strip generic prefix "Restaurant" if a proper name follows
       "Restaurant Rami"      -> "rami"
       "Restaurant La Medina" -> "la medina"
       "Restaurant El Ksar"   -> "el ksar"
       BUT: if "restaurant" IS the name -> keep as-is
       "Le Restaurant"        -> "le restaurant"

  3b. Strip ONLY the leading word "Restaurant/Cafe/Snack"
      when immediately followed by a proper name or article
       "Restaurant du Port"   -> "du port"
       "Cafe Chez Ali"        -> "chez ali"
       "Snack Le Palmier"     -> "le palmier"

  4. Remove duplicates - return each name ONCE only

  5. Arabic names - keep as-is in Arabic script
       "\u062f\u0627\u0631 \u0627\u0644\u062c\u0644\u062f"  -> "\u062f\u0627\u0631 \u0627\u0644\u062c\u0644\u062f"
       "\u0645\u0637\u0639\u0645 \u0631\u0627\u0645\u064a"  -> "\u0631\u0627\u0645\u064a"

  6. Mixed script - normalize Latin part only
       "Dar El Jeld \u062f\u0627\u0631 \u0627\u0644\u062c\u0644\u062f" -> "dar el jeld"

\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
OUTPUT FORMAT - STRICT
\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550
Return ONLY a flat JSON array of strings.
First character MUST be [   Last character MUST be ]
No objects. No nesting. No prose. No markdown. No explanation.
Return at most 20 restaurant names per response.

Correct:  ["dar el jeld", "le corsaire", "sandwich chez ahmed"]
Wrong:    [{"brand": "dar el jeld"}]
Wrong:    [["dar el jeld"]]

If NO named establishments found: []
Never return null, never return {}, never return an empty string.
The response must ALWAYS be a valid JSON array, even when empty.
"""


def build_brand_extraction_prompt(raw_text: str, language: str = "fr") -> str:
    return (
        f"Language of text: {language}\n"
        "Extract all restaurant names from the text below.\n"
        "Return ONLY a flat JSON array of lowercase strings.\n"
        "Example: [\"dar el jeld\", \"le corsaire\"]\n\n"
        f"---\n{raw_text}\n---"
    )


def agent1_extract_entities(
    df_raw:      pd.DataFrame,
    model:       str = MODEL_EXTRACTOR,
    output_path: str = ENTITIES_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Step 3: Entity extraction.
    For each raw response extract entity names using extractor LLM.

    Input  : raw_responses.csv (df_raw)
    Output : extracted_entities.csv
    Schema : entity_id · response_id · prompt_id · run_id · model_id ·
             intent_id · language · entity · brand_raw_text
    """

    # resume support
    done_response_ids = set()
    if os.path.exists(output_path):
        df_existing = pd.read_csv(
            output_path, encoding='utf-8-sig', quoting=csv.QUOTE_ALL
        )
        done_response_ids = set(df_existing['response_id'].unique())
        print(f"Resuming - {len(done_response_ids)} responses already extracted")

    header_written = os.path.exists(output_path)
    total = len(df_raw)

    print(f"\nStep 3: Entity extraction")
    print(f"   Responses to process : {total}")
    print(f"   Already done         : {len(done_response_ids)}")
    print(f"   Remaining            : {total - len(done_response_ids)}")
    print(f"   Extractor model      : {model}\n")

    for i, row in df_raw.iterrows():
        response_id = row['response_id']

        if response_id in done_response_ids:
            print(f"  [{i+1}/{total}] {response_id} skipped")
            continue

        print(f"  [{i+1}/{total}] {response_id}")

        extraction_prompt = build_brand_extraction_prompt(
            raw_text=row['response_text'],
            language=row.get('language', 'fr')
        )

        result = query_mistral(
            prompt = extraction_prompt,
            system = BRAND_EXTRACTION_SYSTEM
        )

        raw_entities = clean_and_parse_json(
            result['raw_response'],
            label=f"Entity extraction [{response_id}]"
        )

        if not isinstance(raw_entities, list):
            raw_entities = []

        records = []
        for item in raw_entities:
            if isinstance(item, str):
                entity_name = item.lower().strip()
            elif isinstance(item, dict):
                entity_name = item.get("brand", "").lower().strip()
            else:
                continue

            if not entity_name:
                continue

            records.append({
                "entity_id"     : f"{response_id}__{entity_name.replace(' ', '_')}",
                "response_id"   : response_id,
                "entity"        : entity_name,
                "brand_raw_text": extract_entity_description(
                                    row['response_text'], entity_name
                                  ),
            })

        print(f"           {len(records)} entities: {[r['entity'] for r in records]}")

        if records:
            pd.DataFrame(records).to_csv(
                output_path,
                mode='a',
                header=not header_written,
                index=False,
                encoding='utf-8-sig',
                quoting=csv.QUOTE_ALL
            )
            header_written = True

        done_response_ids.add(response_id)

    # final summary
    df_entities = pd.read_csv(
        output_path, encoding='utf-8-sig', quoting=csv.QUOTE_ALL
    )

    print(f"\n{'='*55}")
    print(f"Step 3 complete")
    print(f"   Total entity records : {len(df_entities)}")
    print(f"   Unique entities      : {df_entities['entity'].nunique()}")
    print(f"   Responses processed  : {df_entities['response_id'].nunique()}")
    print(f"   Saved to             : {output_path}")
    print(f"{'='*55}")

    return df_entities


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Entity enrichment
# Input  : raw_responses.csv + extracted_entities.csv
# Output : enriched_entities.csv
# ══════════════════════════════════════════════════════════════════════════════

ENRICHED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "enriched_entities.csv")

RANKING_EXTRACTION_SYSTEM = """\
You are a ranking extraction specialist for a GEO analysis pipeline.

Your ONLY task is to read a restaurant recommendation response and return
the ordered list of ALL restaurant entities mentioned, in the exact order
they first appear.

HOW TO RANK
- Numbered list response: use the item numbers directly
- Prose response: use the sentence order of first mention
- Position 1 = first mentioned entity

OUTPUT FORMAT \u2014 STRICT
Return ONLY a flat JSON array of strings in ranked order.
First character MUST be [   Last character MUST be ]

Example:
["dar el jeld", "le corsaire", "la kasbah", "dar zarrouk"]

If no entities found: []
"""


def compute_description_length(entity: str, run_records: list) -> int:
    """
    Estimate average tokens dedicated to this entity across all runs.
    Uses char ratio x completion_tokens as proxy. No API call.
    """
    estimates = []

    for rec in run_records:
        raw_text = rec.get("response_text", "")
        completion_tokens = rec.get("completion_tokens", 0)

        description_block = rec.get("brand_raw_text") or extract_entity_description(raw_text, entity)
        description_block = description_block if isinstance(description_block, str) else ""

        resp_chars = len(raw_text)
        desc_chars = len(description_block)

        if resp_chars > 0 and completion_tokens > 0:
            char_ratio = desc_chars / resp_chars
            estimated_tokens = round(char_ratio * completion_tokens)
        else:
            estimated_tokens = 0

        estimates.append(estimated_tokens)

    return round(sum(estimates) / len(estimates)) if estimates else 0


def extract_rankings_for_response(response_id: str, response_text: str, entities: list, model: str) -> dict:
    """
    Extract ranking positions for ALL entities in one response in ONE LLM call.
    Returns: {entity_name: ranking_position}
    """
    prompt = (
        "Read this restaurant recommendation response and return "
        "ALL restaurant names in the exact order they first appear.\n\n"
        "Known entities to look for (but include any others too):\n"
        f"{json.dumps(entities, ensure_ascii=False)}\n\n"
        f"Response:\n---\n{response_text}\n---\n\n"
        "Return ONLY a flat JSON array of entity names in ranked order."
    )

    result = query_llm(
        model=model,
        prompt=prompt,
        system=RANKING_EXTRACTION_SYSTEM,
        role="extractor"
    )

    ranked_list = clean_and_parse_json(
        result["raw_response"],
        label=f"Ranking [{response_id}]"
    )

    if not isinstance(ranked_list, list):
        return {}

    return {
        entity.lower().strip(): idx + 1
        for idx, entity in enumerate(ranked_list)
        if isinstance(entity, str)
    }


def agent1_enrich_entities(
    df_entities: pd.DataFrame,
    df_raw: pd.DataFrame,
    model: str = MODEL_EXTRACTOR,
    output_path: str = ENRICHED_OUTPUT_PATH,
) -> pd.DataFrame:
    """
    Step 4 — Entity enrichment.

    Phase A : ranking extraction — 1 LLM call per response
    Phase B : description length — pure Python, no LLM

    Input  : extracted_entities.csv + raw_responses.csv
    Output : enriched_entities.csv
    Schema : response_id · prompt_id · entity · run_id ·
             ranking_position · description_length_tokens
    """

    # join entities with raw responses
    df_joined = df_entities.merge(
        df_raw[["response_id", "prompt_id", "response_text",
                "run_id", "model_id", "completion_tokens"]],
        on="response_id",
        how="left"
    )

    # resume support
    done_response_ids = set()
    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
        done_response_ids = set(df_existing["response_id"].unique())
        print(f"Resuming \u2014 {len(done_response_ids)} responses already done")

    header_written = os.path.exists(output_path)

    # PHASE A \u2014 ranking extraction
    print("\nPhase A \u2014 Ranking extraction per response")

    ranking_map = {}
    responses = df_joined[["response_id", "response_text"]].drop_duplicates()
    total_resp = len(responses)

    for idx, (_, row) in enumerate(responses.iterrows()):
        response_id = row["response_id"]
        response_text = row["response_text"]

        entities_in_response = df_joined[
            df_joined["response_id"] == response_id
        ]["entity"].tolist()

        print(f"[{idx+1}/{total_resp}] {response_id} \u2014 {len(entities_in_response)} entities")

        ranking_map[response_id] = extract_rankings_for_response(
            response_id=response_id,
            response_text=response_text,
            entities=entities_in_response,
            model=model,
        )

        print(f" -> {ranking_map[response_id]}")

    # PHASE B \u2014 description length
    print("\nPhase B \u2014 Description length computation")

    groups = list(df_joined.groupby(["prompt_id", "entity"]))
    desc_map = {}

    for (prompt_id, entity), group in groups:
        run_records = group.to_dict(orient="records")
        desc_map[(prompt_id, entity)] = compute_description_length(
            entity=entity,
            run_records=run_records,
        )

    print(f"Computed {len(groups)} (prompt x entity) pairs")

    # BUILD FINAL RECORDS
    print("\nBuilding final records")

    records = []

    for _, row in df_joined.iterrows():
        response_id = row["response_id"]

        if response_id in done_response_ids:
            continue

        prompt_id = row["prompt_id"]
        entity = row["entity"]
        run_id = row["run_id"]

        records.append({
            "response_id": response_id,
            "prompt_id": prompt_id,
            "entity": entity,
            "run_id": run_id,
            "ranking_position": ranking_map.get(response_id, {}).get(entity),
            "description_length_tokens": desc_map.get((prompt_id, entity), 0),
        })

    if records:
        pd.DataFrame(records).to_csv(
            output_path,
            mode="a",
            header=not header_written,
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_ALL
        )

    # final summary
    df_enriched = pd.read_csv(output_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    print("\nStep 4 complete")
    print(f"Total records: {len(df_enriched)}")
    print(f"Unique entities: {df_enriched['entity'].nunique()}")
    print(f"Prompts covered: {df_enriched['prompt_id'].nunique()}")
    print(f"Saved to: {output_path}")

    return df_enriched


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Data cleaning
# Input  : enriched_entities.csv
# Output : clean_entities.csv
# ══════════════════════════════════════════════════════════════════════════════

from rapidfuzz import fuzz, process

CLEAN_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "clean_entities.csv")
CLEAN_LOG_PATH    = os.path.join(OUTPUT_DIR, "cleaning_log.csv")

FUZZY_MERGE_THRESHOLD = 88
FUZZY_AMBIG_LOWER     = 75
FUZZY_AMBIG_UPPER     = 87
MIN_ENTITY_LENGTH     = 3
NULL_RANK_PENALTY     = 999


# PHASE A — Hard rules only

def phase_a_prefilter(df: pd.DataFrame) -> tuple:
    """
    Hard drop only — empty, numeric, too short.
    No domain lists — LLM handles semantic validation in Phase C.
    """
    log = []
    df  = df.copy()
    df['quality_flag'] = 'ok'

    def hard_invalid(entity: str) -> str:
        e = str(entity).strip().lower()
        if not e:
            return "empty"
        if len(e) < MIN_ENTITY_LENGTH:
            return "too_short"
        if e.isdigit():
            return "numeric"
        return None

    drop_mask = pd.Series([False] * len(df), index=df.index)

    for idx, row in df.iterrows():
        reason = hard_invalid(str(row.get('entity', '')))
        if reason:
            drop_mask[idx] = True
            log.append({
                "entity": row['entity'],
                "action": "dropped",
                "reason": reason,
                "phase" : "A",
            })

    dropped = df[drop_mask]
    df      = df[~drop_mask].reset_index(drop=True)

    print(f"   Dropped (hard) : {len(dropped)}")
    print(f"   Kept           : {len(df)}")

    return df, log


# PHASE B — Fuzzy clustering with frequency-based canonical

def phase_b_fuzzy_cluster(
    entities:      list,
    entity_counts: dict,
) -> tuple:
    """
    Fuzzy deduplication using token_sort_ratio.
    Canonical = most frequent entity in cluster.
    """
    entities    = sorted(set(e.strip().lower() for e in entities if e))
    assigned    = {}
    clusters    = {}
    pair_scores = {}

    for entity in entities:
        if entity in assigned:
            continue

        candidates = process.extract(
            entity,
            [e for e in entities if e not in assigned],
            scorer       = fuzz.token_sort_ratio,
            limit        = None,
            score_cutoff = FUZZY_AMBIG_LOWER,
        )

        group = {entity}
        for match, score, _ in candidates:
            if match == entity:
                continue
            pair_scores[(entity, match)] = score
            if score >= FUZZY_MERGE_THRESHOLD:
                group.add(match)

        canonical = max(group, key=lambda e: entity_counts.get(e, 0))
        clusters[canonical] = list(group)
        for member in group:
            assigned[member] = canonical

    for entity in entities:
        if entity not in assigned:
            assigned[entity] = entity
            clusters[entity] = [entity]

    return clusters, pair_scores, assigned


# PHASE C — LLM: pair resolution + entity validation using brand context

ARBITRATION_SYSTEM = """\
You are an entity resolution and validation specialist for a restaurant
GEO analysis pipeline analyzing Tunisian restaurants
(in Tunisia and abroad including Paris and other cities).

You have TWO tasks:

TASK 1 \u2014 PAIR RESOLUTION
Decide if each pair refers to the SAME physical restaurant or DIFFERENT.
Be CONSERVATIVE \u2014 when in doubt return DIFFERENT.

SAME only when:
  - Pure spelling or article variant of identical place
      kasbah / casbah                  -> SAME
      el kasbah / la kasbah            -> SAME
      le corsaire / corsaire           -> SAME
      restaurant rami / rami           -> SAME
DIFFERENT when:
  - Any meaningful word differs
  - Different proper nouns even if similar theme
  - Any genuine doubt                  -> DIFFERENT

TASK 2 \u2014 ENTITY VALIDATION
For each entity and its context, decide if it is a real
restaurant, cafe, or food business.

IMPORTANT \u2014 be very conservative:
  - If context mentions food, dining, service, cuisine -> valid = true
  - Names like "le sfax", "le djerba", "le marrakech", "le bardo"
    are Paris restaurants named after cities -> valid = true
  - Only invalidate if context CLEARLY describes something
    that is NOT a restaurant:
      a mosque, museum, monument, mathematician,
      a standalone dish with no restaurant context,
      a city or region with no restaurant context
  - When uncertain -> valid = true
  - Prefer false negatives over false positives

OUTPUT FORMAT \u2014 strict JSON object, no prose, no markdown:
{
  "pair_resolutions": [
    {
      "pair"      : ["name_a", "name_b"],
      "decision"  : "SAME" | "DIFFERENT",
      "canonical" : "chosen_name" | null,
      "confidence": "high" | "low"
    }
  ],
  "entity_validations": [
    {
      "entity": "entity_name",
      "valid" : true | false,
      "reason": "brief explanation"
    }
  ]
}

If no pairs to resolve: "pair_resolutions": []
If no entities to validate: "entity_validations": []
"""


def phase_c_llm_arbitration(
    ambiguous_pairs: list,
    all_entities:    list,
    entity_context:  dict,
    entity_counts:   dict,
    model:           str = MODEL_ANALYST,
    batch_size:      int = 20,
) -> tuple:
    """
    Phase C — LLM arbitration and entity validation.
    Sends brand_raw_text context so LLM judges semantic validity accurately.
    """
    merge_decisions = {}
    invalid_set     = set()
    flag_set        = set()
    log             = []

    entities_with_context = [
        {
            "entity" : e,
            "context": entity_context.get(e, "")[:400],
        }
        for e in all_entities
    ]

    pair_batches   = [ambiguous_pairs[i:i+batch_size]
                      for i in range(0, max(len(ambiguous_pairs), 1), batch_size)]
    entity_batches = [entities_with_context[i:i+batch_size]
                      for i in range(0, len(entities_with_context), batch_size)]

    while len(pair_batches)   < len(entity_batches):
        pair_batches.append([])
    while len(entity_batches) < len(pair_batches):
        entity_batches.append([])

    print(f"   Ambiguous pairs : {len(ambiguous_pairs)}")
    print(f"   Entities        : {len(all_entities)}")
    print(f"   Batches         : {len(pair_batches)}")

    for b_idx, (pair_batch, entity_batch) in enumerate(
        zip(pair_batches, entity_batches)
    ):
        print(f"   Batch {b_idx+1}/{len(pair_batches)} "
              f"\u2014 {len(pair_batch)} pairs, {len(entity_batch)} entities")

        prompt = (
            f"TASK 1 \u2014 Resolve {len(pair_batch)} restaurant name pairs.\n"
            f"TASK 2 \u2014 Validate {len(entity_batch)} entities using context.\n\n"
            + (
                "Pairs:\n"
                + json.dumps(
                    [{"name_a": a, "name_b": b, "fuzzy_score": round(s, 1)}
                     for a, b, s in pair_batch],
                    ensure_ascii=False, indent=2
                ) + "\n\n"
                if pair_batch else
                "Pairs: []\n\n"
            )
            + "Entities:\n"
            + json.dumps(entity_batch, ensure_ascii=False, indent=2)
            + "\n\nReturn ONLY the JSON object."
        )

        result = query_llm(
            model  = model,
            prompt = prompt,
            system = ARBITRATION_SYSTEM,
            role   = "analyst"
        )

        parsed = clean_and_parse_json(
            result['raw_response'],
            label=f"Phase C batch {b_idx+1}"
        )

        if not isinstance(parsed, dict):
            print(f"   [WARN] Batch {b_idx+1} parse failed \u2014 skipping")
            for a, b, _ in pair_batch:
                flag_set.add(a)
                flag_set.add(b)
            continue

        for item in parsed.get('pair_resolutions', []):
            pair       = item.get('pair', [])
            decision   = item.get('decision', 'DIFFERENT')
            chosen     = item.get('canonical')
            confidence = item.get('confidence', 'low')

            if len(pair) != 2:
                continue
            a, b = pair[0], pair[1]

            if decision == 'SAME' and chosen and confidence == 'high':
                other = b if chosen == a else a
                merge_decisions[other] = chosen
                log.append({
                    "entity": other,
                    "action": "merged_by_llm",
                    "reason": f"LLM -> {chosen} (conf=high)",
                    "phase" : "C",
                })
            elif confidence == 'low':
                flag_set.add(a)
                flag_set.add(b)
                log.append({
                    "entity": f"{a} | {b}",
                    "action": "flagged",
                    "reason": "LLM low confidence",
                    "phase" : "C",
                })

        for item in parsed.get('entity_validations', []):
            entity = item.get('entity', '').strip().lower()
            valid  = item.get('valid', True)
            reason = item.get('reason', '')

            if not valid and entity:
                invalid_set.add(entity)
                log.append({
                    "entity": entity,
                    "action": "invalidated",
                    "reason": reason,
                    "phase" : "C",
                })

        time.sleep(0.3)

    print(f"   LLM merged    : {len(merge_decisions)}")
    print(f"   Invalidated   : {len(invalid_set)}")
    print(f"   Flagged       : {len(flag_set)}")

    return merge_decisions, invalid_set, flag_set, log


# PHASE D — Self-reflection on all cleaning decisions

REFLECTION_SYSTEM = """\
You are a quality assurance specialist reviewing data cleaning decisions
for a Tunisian restaurant GEO analysis pipeline.

Review the cleaning log and current canonical mappings.
Identify any suspicious or incorrect decisions.

Be critical \u2014 a wrong merge is worse than a missed merge.
A wrongly invalidated restaurant loses real data.

IMPORTANT \u2014 restaurants named after cities like "le sfax", "le djerba",
"le bardo", "le marrakech" are valid Paris Tunisian restaurants.
Flag any invalidation of these as suspicious.

Return ONLY valid JSON:
{
  "quality_score": <0-10>,
  "proceed"      : <true | false>,
  "suspicious"   : [
    {
      "entity" : <string>,
      "issue"  : <what looks wrong>,
      "action" : "undo_merge" | "undo_flag" | "undo_invalidation" | "review"
    }
  ]
}
"""


def phase_d_self_reflect(
    cleaning_log:  list,
    canonical_map: dict,
    invalid_set:   set,
    model:         str = MODEL_ANALYST,
) -> dict:
    """
    Agent reflects on its own cleaning decisions.
    Returns correction instructions.
    """
    if not cleaning_log:
        print("   No decisions to reflect on")
        return {"quality_score": 10, "proceed": True, "suspicious": []}

    prompt = (
        f"Review these {len(cleaning_log)} data cleaning decisions "
        f"for Tunisian restaurant entities.\n\n"
        f"Cleaning log:\n"
        + json.dumps(cleaning_log, ensure_ascii=False, indent=2)
        + f"\n\nCurrent merges:\n"
        + json.dumps(
            {k: v for k, v in canonical_map.items() if k != v},
            ensure_ascii=False, indent=2
          )
        + f"\n\nCurrently invalidated:\n"
        + json.dumps(list(invalid_set), ensure_ascii=False, indent=2)
        + "\n\nIdentify suspicious or incorrect decisions."
    )

    result = query_llm(
        model  = model,
        prompt = prompt,
        system = REFLECTION_SYSTEM,
        role   = "analyst"
    )

    reflection = clean_and_parse_json(
        result['raw_response'],
        label="Phase D reflection"
    )

    if not isinstance(reflection, dict):
        print("   Parse failed \u2014 proceeding without correction")
        return {"quality_score": 5, "proceed": True, "suspicious": []}

    print(f"   Quality score : {reflection.get('quality_score', '?')}/10")
    print(f"   Proceed       : {reflection.get('proceed', True)}")

    suspicious = reflection.get('suspicious', [])
    if suspicious:
        print(f"   Suspicious    : {len(suspicious)}")
        for s in suspicious:
            print(f"   -> '{s.get('entity')}': {s.get('issue')}")

    return reflection


# PHASE E — Apply canonical mapping + null rank penalty

def phase_e_apply_mapping(
    df:            pd.DataFrame,
    canonical_map: dict,
    invalid_set:   set,
    flag_set:      set,
) -> pd.DataFrame:
    """
    Apply all decisions from Phases A-D.
    """
    df = df.copy()

    df['canonical_entity'] = df['entity'].map(
        lambda e: canonical_map.get(e, e)
    )

    df['ranking_position_filled'] = df['ranking_position'].fillna(
        NULL_RANK_PENALTY
    ).astype(int)

    df['quality_flag'] = df['entity'].apply(
        lambda e: 'invalid' if e in invalid_set else 'ok'
    )

    df['clean_flag'] = df['entity'].apply(
        lambda e: 'invalid' if e in invalid_set
             else 'flagged' if e in flag_set
             else 'clean'
    )

    df['merge_source'] = df.apply(
        lambda row: row['entity']
        if row['entity'] != row['canonical_entity'] else "",
        axis=1
    )

    return df


# MAIN — Step 5

def agent1_clean_entities(
    df_enriched: pd.DataFrame,
    df_raw:      pd.DataFrame,
    df_entities: pd.DataFrame,
    model:       str = MODEL_ANALYST,
    output_path: str = CLEAN_OUTPUT_PATH,
    log_path:    str = CLEAN_LOG_PATH,
) -> pd.DataFrame:
    """
    Step 5 — Agentic data cleaning pipeline.

    Phase A : Hard rules only — empty, numeric, too short    (Python)
    Phase B : Fuzzy clustering, frequency-based canonical    (rapidfuzz)
    Phase C : LLM pair resolution + entity validation        (LLM)
    Phase D : Self-reflection + correction                   (LLM)
    Phase E : Apply all decisions                            (Python)

    Input  : enriched_entities.csv + raw_responses.csv +
             extracted_entities.csv (for brand context)
    Output : clean_entities.csv + cleaning_log.csv
    Schema : all enriched columns + canonical_entity +
             quality_flag + clean_flag +
             ranking_position_filled + merge_source
    """
    print(f"\n{'='*55}")
    print(f"Step 5 \u2014 Agentic data cleaning")
    print(f"   Input rows      : {len(df_enriched)}")
    print(f"   Unique entities : {df_enriched['entity'].nunique()}")
    print(f"   Model           : {model}")
    print(f"{'='*55}\n")

    cleaning_log  = []
    entity_counts = df_enriched['entity'].value_counts().to_dict()

    # build entity context from brand_raw_text (extracted entities)
    print("Building entity context from brand descriptions...")
    entity_context = {}

    for _, row in df_enriched.iterrows():
        entity = row['entity']
        if entity not in entity_context:
            # try brand_raw_text from extracted entities first
            match = df_entities[
                (df_entities['response_id'] == row['response_id']) &
                (df_entities['entity'] == entity)
            ]
            if not match.empty:
                brand_text = match.iloc[0].get('brand_raw_text', '')
                if pd.notna(brand_text) and str(brand_text).strip():
                    entity_context[entity] = str(brand_text)[:400]
                    continue

            # fallback — use response_text skipping first 100 chars (intro)
            raw_match = df_raw[df_raw['response_id'] == row['response_id']]
            if not raw_match.empty:
                entity_context[entity] = str(
                    raw_match.iloc[0]['response_text']
                )[100:400]

    print(f"   Context built for {len(entity_context)} entities\n")

    # Phase A
    print("Phase A \u2014 Hard pre-filter")
    df_filtered, log_a = phase_a_prefilter(df_enriched)
    cleaning_log.extend(log_a)

    # Phase B
    print("\nPhase B \u2014 Fuzzy clustering")
    all_entities = (
        df_filtered['entity']
        .dropna().str.strip().str.lower()
        .unique().tolist()
    )

    clusters, pair_scores, canonical_map = phase_b_fuzzy_cluster(
        entities      = all_entities,
        entity_counts = entity_counts,
    )

    auto_merged = 0
    for canonical, members in clusters.items():
        for m in members:
            if m != canonical:
                auto_merged += 1
                cleaning_log.append({
                    "entity": m,
                    "action": "merged",
                    "reason": f"fuzzy >= {FUZZY_MERGE_THRESHOLD} -> {canonical}",
                    "phase" : "B",
                })

    ambiguous_pairs = [
        (a, b, score)
        for (a, b), score in pair_scores.items()
        if FUZZY_AMBIG_LOWER <= score <= FUZZY_AMBIG_UPPER
    ]

    print(f"   Unique entities : {len(all_entities)}")
    print(f"   Auto-merged     : {auto_merged}")
    print(f"   Ambiguous pairs : {len(ambiguous_pairs)}")

    # Phase C
    print("\nPhase C \u2014 LLM arbitration + entity validation")
    merge_decisions, invalid_set, flag_set, log_c = phase_c_llm_arbitration(
        ambiguous_pairs = ambiguous_pairs,
        all_entities    = all_entities,
        entity_context  = entity_context,
        entity_counts   = entity_counts,
        model           = model,
    )
    cleaning_log.extend(log_c)
    canonical_map.update(merge_decisions)

    # Phase D
    print("\nPhase D \u2014 Self-reflection")
    reflection = phase_d_self_reflect(
        cleaning_log  = cleaning_log,
        canonical_map = canonical_map,
        invalid_set   = invalid_set,
        model         = model,
    )

    for correction in reflection.get('suspicious', []):
        entity = correction.get('entity', '').strip().lower()
        action = correction.get('action', '')

        if action == 'undo_merge' and entity in canonical_map:
            old_canonical = canonical_map[entity]
            canonical_map[entity] = entity
            cleaning_log.append({
                "entity": entity,
                "action": "merge_undone",
                "reason": correction.get('issue', ''),
                "phase" : "D",
            })
            print(f"   Merge undone : '{entity}' <- '{old_canonical}'")

        elif action == 'undo_flag' and entity in flag_set:
            flag_set.discard(entity)
            cleaning_log.append({
                "entity": entity,
                "action": "flag_removed",
                "reason": correction.get('issue', ''),
                "phase" : "D",
            })
            print(f"   Flag removed : '{entity}'")

        elif action == 'undo_invalidation' and entity in invalid_set:
            invalid_set.discard(entity)
            cleaning_log.append({
                "entity": entity,
                "action": "invalidation_undone",
                "reason": correction.get('issue', ''),
                "phase" : "D",
            })
            print(f"   Invalidation undone : '{entity}'")

    # Phase E
    print("\nPhase E \u2014 Apply canonical mapping")
    df_clean = phase_e_apply_mapping(
        df            = df_filtered,
        canonical_map = canonical_map,
        invalid_set   = invalid_set,
        flag_set      = flag_set,
    )

    # Save
    df_clean.to_csv(
        output_path, index=False,
        encoding='utf-8-sig', quoting=csv.QUOTE_ALL
    )
    pd.DataFrame(cleaning_log).to_csv(
        log_path, index=False,
        encoding='utf-8-sig', quoting=csv.QUOTE_ALL
    )

    # Summary
    merged_rows  = (df_clean['merge_source'] != "").sum()
    clean_rows   = df_clean[df_clean['clean_flag'] == 'clean']
    flagged_rows = df_clean[df_clean['clean_flag'] == 'flagged']
    invalid_rows = df_clean[df_clean['clean_flag'] == 'invalid']
    null_penalty = (df_clean['ranking_position_filled'] == NULL_RANK_PENALTY).sum()

    print(f"\n{'='*55}")
    print(f"Step 5 complete")
    print(f"   Input rows          : {len(df_enriched)}")
    print(f"   After Phase A       : {len(df_filtered)}")
    print(f"   Output rows         : {len(df_clean)}")
    print(f"   Canonical entities  : {df_clean['canonical_entity'].nunique()}")
    print(f"   Clean entities      : {clean_rows['canonical_entity'].nunique()}")
    print(f"   Flagged entities    : {flagged_rows['canonical_entity'].nunique()}")
    print(f"   Invalid entities    : {invalid_rows['canonical_entity'].nunique()}")
    print(f"   Rows with merge     : {merged_rows}")
    print(f"   Null rank->penalty  : {null_penalty}")
    print(f"   Saved to            : {output_path}")
    print(f"   Log saved to        : {log_path}")
    print(f"{'='*55}")

    return df_clean


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Compute metrics
# Input  : clean_entities.csv
# Output : entity_features.csv + entity_features_global.csv
# ══════════════════════════════════════════════════════════════════════════════

FEATURES_PATH        = os.path.join(OUTPUT_DIR, "entity_features.csv")
FEATURES_GLOBAL_PATH = os.path.join(OUTPUT_DIR, "entity_features_global.csv")


def compute_co_mention_rate(df: pd.DataFrame) -> dict:
    """
    For each entity, compute how often it appears
    alongside each other entity in the same response.
    Returns: {entity: {co_entity: rate}}
    """
    co_mention_map = {}

    responses = df.groupby('response_id')['canonical_entity'].apply(list).to_dict()

    for response_id, entities in responses.items():
        entities = list(set(entities))
        for entity in entities:
            if entity not in co_mention_map:
                co_mention_map[entity] = {}
            for other in entities:
                if other == entity:
                    continue
                co_mention_map[entity][other] = \
                    co_mention_map[entity].get(other, 0) + 1

    # normalize by entity mention count
    entity_counts = df.groupby('canonical_entity').size().to_dict()
    co_rates = {}
    for entity, co_counts in co_mention_map.items():
        total = entity_counts.get(entity, 1)
        co_rates[entity] = {
            other: round(count / total, 4)
            for other, count in co_counts.items()
        }

    return co_rates


def compute_prompt_type_response(
    df:      pd.DataFrame,
    df_raw:  pd.DataFrame,
    prompts: list,
) -> dict:
    """
    For each entity, compute mention_rate per intent_id.
    intent_id comes from the original prompt set, joined via prompt_id.
    """
    # build prompt_id -> intent_id lookup from Step 1 output
    intent_lookup = {
        str(p['prompt_id']): str(p['intent_id'])
        for p in prompts
        if 'intent_id' in p and 'prompt_id' in p
    }

    df_with_intent = df.copy()
    df_with_intent['intent_id'] = (
        df_with_intent['prompt_id'].astype(str).map(intent_lookup)
    )

    result = {}
    intents = df_with_intent['intent_id'].dropna().unique().tolist()
    total_per_intent = (
        df_with_intent.groupby('intent_id')['response_id']
        .nunique().to_dict()
    )

    for entity, group in df_with_intent.groupby('canonical_entity'):
        intent_rates = {}
        for intent in intents:
            intent_group = group[group['intent_id'] == intent]
            total = total_per_intent.get(intent, 1)
            intent_rates[intent] = round(len(intent_group) / total, 4)
        result[entity] = intent_rates

    return result


def compute_entity_features_per_prompt(
    df:           pd.DataFrame,
    total_models: int,
) -> pd.DataFrame:
    """
    Compute features per (prompt_id x canonical_entity).
    Denominators use actual valid response counts per prompt.
    """
    total_responses_per_prompt = (
        df.groupby('prompt_id')['response_id'].nunique().to_dict()
    )

    rows = []

    for (prompt_id, entity), group in df.groupby(
        ['prompt_id', 'canonical_entity']
    ):
        total_resp  = total_responses_per_prompt.get(prompt_id, 1)
        valid_ranks = group[
            group['ranking_position_filled'] != NULL_RANK_PENALTY
        ]['ranking_position_filled'].tolist()

        mention_count = len(group)
        mention_rate  = round(mention_count / total_resp, 4)
        top1_rate     = round(
            sum(1 for r in valid_ranks if r == 1) / mention_count, 4
        ) if mention_count > 0 else 0.0

        avg_rank = round(
            sum(valid_ranks) / len(valid_ranks), 4
        ) if valid_ranks else None

        rank_variance = round(
            math.sqrt(
                sum((r - (sum(valid_ranks) / len(valid_ranks))) ** 2
                    for r in valid_ranks) / len(valid_ranks)
            ), 4
        ) if len(valid_ranks) >= 2 else 0.0

        stability_score = round(
            mention_rate * (1 / (1 + rank_variance)), 4
        )

        if stability_score >= 0.75:
            consistency_label = "stable"
        elif stability_score >= 0.40:
            consistency_label = "variable"
        else:
            consistency_label = "unstable"

        distinct_models = group['response_id'].apply(
            lambda r: r.split('__')[1] if '__' in str(r) else ''
        ).nunique()
        model_presence_count = distinct_models
        cross_model_rate     = round(model_presence_count / total_models, 4)

        rows.append({
            "prompt_id":                prompt_id,
            "canonical_entity":         entity,
            "mention_count":            mention_count,
            "mention_rate":             mention_rate,
            "average_ranking_position": avg_rank,
            "rank_variance":            rank_variance,
            "top1_rate":                top1_rate,
            "frequency_across_runs":    group['run_id'].nunique(),
            "stability_score":          stability_score,
            "consistency_label":        consistency_label,
            "avg_description_length":   round(
                group['description_length_tokens'].mean(), 1
            ),
            "model_presence_count":     model_presence_count,
            "cross_model_rate":         cross_model_rate,
        })

    return pd.DataFrame(rows)


def compute_entity_features_global(
    df:           pd.DataFrame,
    df_raw:       pd.DataFrame,
    total_models: int,
    prompts:      list,
) -> pd.DataFrame:
    """
    Compute global features per canonical_entity across all prompts and runs.
    This is the main input to the Phase 3 recommendation model.
    """
    total_responses = df['response_id'].nunique()
    co_mention_map  = compute_co_mention_rate(df)
    ptr_map         = compute_prompt_type_response(df, df_raw, prompts)

    rows = []

    for entity, group in df.groupby('canonical_entity'):
        valid_ranks = group[
            group['ranking_position_filled'] != NULL_RANK_PENALTY
        ]['ranking_position_filled'].tolist()

        mention_count = len(group)
        mention_rate  = round(mention_count / total_responses, 4)
        top1_rate     = round(
            sum(1 for r in valid_ranks if r == 1) / mention_count, 4
        ) if mention_count > 0 else 0.0

        avg_rank = round(
            sum(valid_ranks) / len(valid_ranks), 4
        ) if valid_ranks else None

        rank_variance = round(
            math.sqrt(
                sum((r - (sum(valid_ranks) / len(valid_ranks))) ** 2
                    for r in valid_ranks) / len(valid_ranks)
            ), 4
        ) if len(valid_ranks) >= 2 else 0.0

        stability_score = round(
            mention_rate * (1 / (1 + rank_variance)), 4
        )

        if stability_score >= 0.75:
            consistency_label = "stable"
        elif stability_score >= 0.40:
            consistency_label = "variable"
        else:
            consistency_label = "unstable"

        distinct_models = group['response_id'].apply(
            lambda r: r.split('__')[1] if '__' in str(r) else ''
        ).nunique()
        model_presence_count = distinct_models
        cross_model_rate     = round(model_presence_count / total_models, 4)

        rows.append({
            "canonical_entity":         entity,
            "mention_count":            mention_count,
            "mention_rate":             mention_rate,
            "average_ranking_position": avg_rank,
            "rank_variance":            rank_variance,
            "top1_rate":                top1_rate,
            "frequency_across_runs":    group['run_id'].nunique(),
            "stability_score":          stability_score,
            "consistency_label":        consistency_label,
            "avg_description_length":   round(
                group['description_length_tokens'].mean(), 1
            ),
            "model_presence_count":     model_presence_count,
            "cross_model_rate":         cross_model_rate,
            "prompt_type_response":     json.dumps(
                ptr_map.get(entity, {}), ensure_ascii=False
            ),
            "co_mention_rate":          json.dumps(
                co_mention_map.get(entity, {}), ensure_ascii=False
            ),
        })

    df_global = pd.DataFrame(rows).sort_values(
        'stability_score', ascending=False
    ).reset_index(drop=True)

    return df_global


def agent1_compute_metrics(
    df_clean:     pd.DataFrame,
    df_raw:       pd.DataFrame,
    prompts:      list,
    output_path:  str = FEATURES_PATH,
    global_path:  str = FEATURES_GLOBAL_PATH,
) -> tuple:
    """
    Step 6 — Compute all GEO features.
    Pure Python — no LLM calls.

    Input  : clean_entities.csv + raw_responses.csv
    Output : entity_features.csv (per prompt x entity)
             entity_features_global.csv (global — input to Phase 3)
    """
    print(f"\n{'='*55}")
    print(f"Step 6 \u2014 Compute metrics")
    print(f"   Input rows      : {len(df_clean)}")
    print(f"   Unique entities : {df_clean['canonical_entity'].nunique()}")
    print(f"{'='*55}\n")

    # Filter: keep clean + flagged, drop invalid only
    df = df_clean[df_clean['clean_flag'] != 'invalid'].copy()
    print(f"   After filtering invalid : {len(df)} rows "
          f"({df['canonical_entity'].nunique()} entities)")

    if len(df) == 0:
        print("   [ERROR] No valid entities after filtering \u2014 aborting")
        return pd.DataFrame(), pd.DataFrame()

    # Derive total_models from actual data
    total_models = df['response_id'].apply(
        lambda r: r.split('__')[1] if '__' in str(r) else ''
    ).nunique()

    total_responses = df['response_id'].nunique()

    print(f"   Total models (actual)   : {total_models}")
    print(f"   Total responses (valid) : {total_responses}\n")

    # Per-prompt features
    print("Computing per-prompt features...")
    df_features = compute_entity_features_per_prompt(
        df           = df,
        total_models = total_models,
    )
    df_features.to_csv(
        output_path, index=False,
        encoding='utf-8-sig', quoting=csv.QUOTE_ALL,
    )
    print(f"   Saved : {output_path}")

    # Global features
    print("\nComputing global features...")
    df_global = compute_entity_features_global(
        df           = df,
        df_raw       = df_raw,
        total_models = total_models,
        prompts      = prompts,
    )
    df_global.to_csv(
        global_path, index=False,
        encoding='utf-8-sig', quoting=csv.QUOTE_ALL,
    )
    print(f"   Saved : {global_path}")

    # Summary
    print(f"\n{'='*55}")
    print(f"Step 6 complete")
    print(f"   Per-prompt rows    : {len(df_features)}")
    print(f"   Global entities    : {len(df_global)}")
    print(f"   Stable entities    : "
          f"{len(df_global[df_global['consistency_label']=='stable'])}")
    print(f"   Variable entities  : "
          f"{len(df_global[df_global['consistency_label']=='variable'])}")
    print(f"   Unstable entities  : "
          f"{len(df_global[df_global['consistency_label']=='unstable'])}")
    print(f"   Top 5 by stability :")
    print(df_global[['canonical_entity', 'stability_score',
                     'mention_rate', 'average_ranking_position',
                     'cross_model_rate']].head(5).to_string(index=False))
    print(f"\n   Running on partial data: {total_responses} valid responses")
    print(f"   Rates normalized to actual valid responses.")
    print(f"   Re-run after backfilling truncated responses.")
    print(f"{'='*55}")

    return df_features, df_global


# ── LangGraph node ────────────────────────────────────────────────────────────

from graph.state import GEOState


def run_agent1_node(state: GEOState) -> dict:
    """LangGraph node: runs Agent 1 steps 1-6 and merges results into state."""
    errors = list(state.get("errors", []))
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    reset_token_usage()

    # Prefer prompt_set already in state; fall back to CSV on disk.
    prompt_records = state.get("prompt_set") or []
    if prompt_records:
        prompts = prompt_records
    else:
        prompts = agent1_load_prompts(PROMPT_SET_PATH)

    if not prompts:
        errors.append("agent1: no prompts available \u2014 skipping")
        return {**state, "errors": errors, "current_step": "agent1_aborted"}

    try:
        # Step 2
        df_raw = agent1_query_prompts(
            prompts=prompts, models=QUERY_MODELS,
            n_runs=N_RUNS, output_path=RAW_OUTPUT_PATH,
        )
        # Step 3
        df_entities = agent1_extract_entities(
            df_raw=df_raw, model=MODEL_EXTRACTOR,
            output_path=ENTITIES_OUTPUT_PATH,
        )
        # Step 4
        df_enriched = agent1_enrich_entities(
            df_entities=df_entities, df_raw=df_raw,
            model=MODEL_EXTRACTOR3, output_path=ENRICHED_OUTPUT_PATH,
        )
        # Step 5
        df_clean = agent1_clean_entities(
            df_enriched=df_enriched, df_raw=df_raw, df_entities=df_entities,
            model=MODEL_ANALYST, output_path=CLEAN_OUTPUT_PATH,
            log_path=CLEAN_LOG_PATH,
        )
        # Step 6
        df_features, df_global = agent1_compute_metrics(
            df_clean=df_clean, df_raw=df_raw, prompts=prompts,
            output_path=FEATURES_PATH, global_path=FEATURES_GLOBAL_PATH,
        )
    except Exception as exc:
        errors.append(f"agent1: {exc}")
        return {**state, "errors": errors, "current_step": "agent1_error"}

    # Merge per-run token counts into the shared state accumulator.
    prior = state.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    merged_tokens = {k: prior.get(k, 0) + TOKEN_USAGE.get(k, 0) for k in prior}

    return {
        **state,
        "raw_responses":          df_raw.to_dict(orient="records"),
        "extracted_entities":     df_entities.to_dict(orient="records"),
        "clean_entities":         df_clean.to_dict(orient="records"),
        "entity_features":        df_features.to_dict(orient="records"),
        "entity_features_global": df_global.to_dict(orient="records"),
        "token_usage":            merged_tokens,
        "errors":                 errors,
        "current_step":           "agent1",
    }
