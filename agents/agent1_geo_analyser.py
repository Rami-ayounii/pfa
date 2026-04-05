"""
agents/agent1_geo_analyser.py
═════════════════════════════
Agent 1 — GEO Brand Analyser

Pipeline (7 steps):
  Step 1 · Load prompt set CSV
  Step 2 · Query LLMs (multi-model × multi-run) with progressive saving
  Step 3 · Extract brand entities from each LLM response       (Mistral/Groq)
  Step 4 · Enrich entities: ranking position + description length
  Step 5 · Clean entities: fuzzy dedup + LLM arbitration + self-reflection
  Step 6 · Compute GEO features (mention rate, stability score, …)
  Step 7 · Orchestrate steps 1-6 with resume support

Output files (in geo_output/):
  raw_responses.csv        · prompt_id × model × run → LLM response
  extracted_entities.csv   · entity per response
  enriched_entities.csv    · entity + ranking_position + description_length
  clean_entities.csv       · canonical entity + quality flags
  cleaning_log.csv         · every cleaning decision with phase tag
  entity_features.csv      · per (prompt × entity) GEO features
  entity_features_global.csv · global features (input to Agent 2)
"""

import os, sys, re, csv, json, math, time
import pandas as pd
from pathlib import Path
from rapidfuzz import fuzz, process
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.llm_client import (
    query_llm, query_mistral, parse_json,
    MODEL_FAST, MODEL_STRONG, MODEL_ANALYST, MODEL_EXTRACTOR,
    TOKEN_USAGE,
)

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR            = "geo_output"
QUERY_MODELS          = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
N_RUNS                = 2
FUZZY_MERGE_THRESHOLD = 88
FUZZY_AMBIG_LOWER     = 75
FUZZY_AMBIG_UPPER     = 87
MIN_ENTITY_LENGTH     = 3
NULL_RANK_PENALTY     = 999

# ── System prompts ────────────────────────────────────────────────────────────

BRAND_EXTRACTION_SYSTEM = """\
You are a restaurant name extractor working in a GEO analysis pipeline.

Your ONLY task: find every named restaurant, cafe, or food business in the text.

EXTRACT (named establishments with a proper name):
  + "Dar El Jeld", "Le Corsaire", "Chez Ahmed", "Sandwich Chez Ali"

EXCLUDE:
  - Standalone dish names ("couscous", "brik", "harissa")
  - Generic descriptions ("a local restaurant", "street food stalls")
  - City / region names ("Tunis", "Sfax", "Djerba")
  - People names ("Chef Ahmed")

NORMALISATION (byte-for-byte identical across runs):
  1. Lowercase only           "Dar El Jeld" → "dar el jeld"
  2. Keep cultural articles   Dar / El / Le / La / Chez — always keep
  3. Strip leading generic word when followed by a proper name
       "Restaurant Rami" → "rami"  |  "Cafe Chez Ali" → "chez ali"
  4. No duplicates — each name once only
  5. Arabic names → keep as-is in Arabic script
  6. Return at most 20 names per response

OUTPUT — STRICT:
  [ "name1", "name2", ... ]
  First char = [   Last char = ]
  No objects, no nesting, no prose, no markdown.
  If nothing found: []
"""

RANKING_SYSTEM = """\
You are a ranking extraction specialist for a GEO analysis pipeline.
Return ALL restaurant names mentioned in the response in the exact order
they first appear. Position 1 = first mentioned.
Return ONLY a flat JSON array of strings in ranked order.
If nothing found: []
"""

def _arbitration_system(domain: str = "Tunisian restaurant") -> str:
    return f"""\
You are an entity resolution and validation specialist for a {domain} GEO analysis pipeline.

TASK 1 — PAIR RESOLUTION: decide SAME or DIFFERENT.
  SAME only for pure spelling/article variants of the identical place.
  DIFFERENT when ANY meaningful word differs or genuine doubt exists.

TASK 2 — ENTITY VALIDATION: is this a real restaurant / food business?
  - If context mentions food, dining, service → valid = true
  - Names like "le sfax", "le djerba" are valid Paris Tunisian restaurants
  - Only invalidate if context CLEARLY describes a non-restaurant
  - When uncertain → valid = true

Return ONLY:
{{
  "pair_resolutions":  [{{"pair": ["a","b"], "decision": "SAME"|"DIFFERENT",
                         "canonical": "...", "confidence": "high"|"low"}}],
  "entity_validations":[{{"entity": "...", "valid": true|false, "reason": "..."}}]
}}
"""


def _reflection_system(domain: str = "Tunisian restaurant") -> str:
    return f"""\
You are a quality auditor reviewing data cleaning decisions for a {domain} GEO analysis pipeline.

Review the cleaning log and canonical mappings.
Flag suspicious or incorrect decisions. A wrong merge is worse than a missed merge.
Names like "le sfax", "le bardo", "le marrakech" are valid Paris restaurants.
Flag any invalidation of these as suspicious.

Return ONLY:
{{
  "quality_score": <0-10>,
  "proceed":       <true|false>,
  "suspicious": [
    {{"entity":"...", "issue":"...",
     "action": "undo_merge"|"undo_flag"|"undo_invalidation"|"review"}}
  ]
}}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load prompts
# ═══════════════════════════════════════════════════════════════════════════════

def step1_load_prompts(prompt_set_path: str) -> list:
    if not os.path.exists(prompt_set_path):
        raise FileNotFoundError(f"Prompt set not found: {prompt_set_path}")
    df = pd.read_csv(prompt_set_path, encoding="utf-8-sig")
    required = ["prompt_id", "intent_id", "language", "prompt_text"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    prompts = df.to_dict(orient="records")
    print(f"  [Step 1] {len(prompts)} prompts loaded | "
          f"intents: {df['intent_id'].unique().tolist()} | "
          f"languages: {df['language'].unique().tolist()}")
    return prompts


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — LLM querying with progressive saving + resume
# ═══════════════════════════════════════════════════════════════════════════════

def step2_query_prompts(
    prompts:     list,
    models:      list = None,
    n_runs:      int  = N_RUNS,
    output_path: str  = os.path.join(OUTPUT_DIR, "raw_responses.csv"),
    groq_client       = None,
) -> pd.DataFrame:

    models = models or QUERY_MODELS

    # Resume support
    done_keys      = set()
    header_written = os.path.exists(output_path)
    if header_written:
        df_ex = pd.read_csv(output_path, encoding="utf-8-sig", on_bad_lines="skip")
        done_keys = {(r.prompt_id, r.model_id, r.run_id)
                     for r in df_ex.itertuples()}
        print(f"  [Step 2] Resuming - {len(done_keys)} records already saved")

    total = len(prompts) * len(models) * n_runs
    print(f"  [Step 2] {total} total calls | remaining: {total - len(done_keys)}")

    for i, p in enumerate(prompts):
        print(f"\n  [{i+1}/{len(prompts)}] {p['prompt_id']} | {p['intent_id']} | {p['language']}")
        for model_id in models:
            for run_idx in range(1, n_runs + 1):
                run_id      = f"run_{run_idx}"
                response_id = f"{p['prompt_id']}__{model_id.replace('/','--')}__{run_id}"

                if (p["prompt_id"], model_id, run_id) in done_keys:
                    print(f"    [{run_id}] {model_id} skipped"); continue

                result = query_llm(model_id, p["prompt_text"],
                                   role="query", client=groq_client)
                if not result["raw_response"]:
                    print(f"    [{run_id}] empty response - skipping"); continue

                record = {
                    "response_id":       response_id,
                    "prompt_id":         p["prompt_id"],
                    "model_id":          model_id,
                    "run_id":            run_id,
                    "intent_id":         p.get("intent_id", ""),
                    "language":          p.get("language", ""),
                    "prompt_text":       p["prompt_text"],
                    "response_text":     result["raw_response"],
                    "completion_tokens": result["completion_tokens"],
                    "prompt_tokens":     result["prompt_tokens"],
                    "total_tokens":      result["total_tokens"],
                    "timestamp":         time.time(),
                }

                os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                pd.DataFrame([record]).to_csv(
                    output_path, mode="a", header=not header_written,
                    index=False, encoding="utf-8-sig")
                header_written = True
                done_keys.add((p["prompt_id"], model_id, run_id))
                print(f"    [{run_id}] {model_id} - {result['total_tokens']} tokens")

    df_raw = pd.read_csv(output_path, encoding="utf-8-sig", on_bad_lines="skip")
    logger.info(f"[Step 2] OK {len(df_raw)} records | total tokens: {TOKEN_USAGE['total_tokens']}")
    return df_raw


# ─────────────────────────────────────────────────────────────────────────────
# step2_query_single — used by LangGraph agent1_llm_query_node (fan-out)
# ─────────────────────────────────────────────────────────────────────────────

def step2_query_single(
    prompt_id:   str,
    prompt_text: str,
    model_id:    str,
    run_idx:     int  = 1,
    groq_client        = None,
    done_keys:   set  = None,
) -> dict | None:
    """
    Query a single (prompt, model, run) combination synchronously.

    Used by the LangGraph fan-out node agent1_llm_query_node.
    Returns a result dict or None if the call should be skipped (resume support).

    Args:
        prompt_id   : unique prompt identifier
        prompt_text : the actual prompt text to send to the LLM
        model_id    : Groq model ID
        run_idx     : run number (1-based)
        groq_client : optional pre-built Groq client
        done_keys   : set of (prompt_id, model_id, run_id) already processed

    Returns:
        dict with keys: prompt_id, prompt_text, model_id, run_idx,
                        response_text, completion_tokens, prompt_tokens,
                        total_tokens
        or None if skipped
    """
    run_id = f"run_{run_idx}"
    if done_keys and (prompt_id, model_id, run_id) in done_keys:
        logger.debug(f"[Step2Single] Skipping {prompt_id}/{model_id}/{run_id} (already done)")
        return None

    result = query_llm(model_id, prompt_text, role="query", client=groq_client)
    if not result["raw_response"]:
        logger.warning(f"[Step2Single] Empty response: {prompt_id}/{model_id}/{run_id}")
        return None

    return {
        "prompt_id":         prompt_id,
        "prompt_text":       prompt_text,
        "model_id":          model_id,
        "run_idx":           run_idx,
        "response_text":     result["raw_response"],
        "completion_tokens": result["completion_tokens"],
        "prompt_tokens":     result["prompt_tokens"],
        "total_tokens":      result["total_tokens"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Entity extraction
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_entity_description(raw_text: str, entity: str) -> str:
    """Extract the text block dedicated to this entity (numbered list / bold / table / prose)."""
    if not raw_text or not entity: return ""
    entity_lower = entity.lower()
    lines        = raw_text.split("\n")
    desc_lines   = []
    inside        = False
    empty_streak  = 0

    for line in lines:
        if "|" in line and entity_lower in line.lower():
            cells = [c.strip() for c in line.split("|") if c.strip()]
            return " — ".join(cells)

    for line in lines:
        if not inside:
            if entity_lower in line.lower():
                inside = True; desc_lines.append(line)
            continue
        stripped = line.strip()
        if re.match(r"^\s*\d+\.\s+", line) and desc_lines:  break
        if re.match(r"^\s*\*\*[^*]+\*\*", line) and desc_lines:
            if entity_lower not in line.lower(): break
        if "|" in line and entity_lower not in line.lower() and desc_lines: break
        if stripped == "":
            empty_streak += 1
            if empty_streak >= 2: break
            desc_lines.append(line)
            continue
        empty_streak = 0
        desc_lines.append(line)

    return "\n".join(desc_lines).strip()


def step3_extract_entities(
    df_raw:      pd.DataFrame,
    output_path: str = os.path.join(OUTPUT_DIR, "extracted_entities.csv"),
    mistral_key: str = None,
    groq_client       = None,
) -> pd.DataFrame:

    done_ids       = set()
    header_written = os.path.exists(output_path)
    if header_written:
        df_ex    = pd.read_csv(output_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
        done_ids = set(df_ex["response_id"].unique())
        print(f"  [Step 3] Resuming - {len(done_ids)} responses done")

    total = len(df_raw)
    print(f"  [Step 3] {total - len(done_ids)} responses remaining")

    for i, row in df_raw.iterrows():
        rid = row["response_id"]
        if rid in done_ids:
            continue
        print(f"  [{i+1}/{total}] {rid}")

        prompt = (
            f"Language of text: {row.get('language','fr')}\n"
            "Extract all restaurant names from the text below.\n"
            "Return ONLY a flat JSON array of lowercase strings.\n"
            "Example: [\"dar el jeld\", \"le corsaire\"]\n\n"
            f"---\n{row['response_text']}\n---"
        )

        # Use Mistral if key is available, then fall back to Groq if it fails
        result = {"raw_response": ""}
        if mistral_key or os.environ.get("MISTRAL_API_KEY"):
            result = query_mistral(prompt, system=BRAND_EXTRACTION_SYSTEM,
                                   api_key=mistral_key)
        if not result["raw_response"]:
            result = query_llm(MODEL_FAST, prompt,
                               system=BRAND_EXTRACTION_SYSTEM,
                               role="extractor", client=groq_client)

        raw_entities = parse_json(result["raw_response"],
                                  label=f"extraction[{rid}]")
        raw_entities = raw_entities if isinstance(raw_entities, list) else []

        records = []
        for item in raw_entities:
            name = (item if isinstance(item, str)
                    else item.get("brand", "") if isinstance(item, dict) else "")
            name = name.lower().strip()
            if not name: continue
            records.append({
                "entity_id":      f"{rid}__{name.replace(' ','_')}",
                "response_id":    rid,
                "entity":         name,
                "brand_raw_text": _extract_entity_description(
                                      row["response_text"], name),
            })

        print(f"    {len(records)} entities: {[r['entity'] for r in records]}")
        if records:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            pd.DataFrame(records).to_csv(
                output_path, mode="a", header=not header_written,
                index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
            header_written = True
        done_ids.add(rid)

    if not os.path.exists(output_path):
        logger.warning("[Step 3] No entities extracted from any response — returning empty DataFrame")
        return pd.DataFrame(columns=["entity_id", "response_id", "entity", "brand_raw_text"])
    df_ent = pd.read_csv(output_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    print(f"\n  [Step 3] [OK] {df_ent['entity'].nunique()} unique entities "
          f"from {df_ent['response_id'].nunique()} responses")
    return df_ent


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 — Entity enrichment  (ranking position + description length)
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_rankings(response_id: str, response_text: str,
                      entities: list, groq_client=None) -> dict:
    """1 LLM call → {entity: rank_position} for the whole response."""
    prompt = (
        "Read this restaurant recommendation response and return "
        "ALL restaurant names in the exact order they first appear.\n\n"
        f"Known entities (include others too):\n{json.dumps(entities, ensure_ascii=False)}\n\n"
        f"Response:\n---\n{response_text}\n---\n\n"
        "Return ONLY a flat JSON array of entity names in ranked order."
    )
    result  = query_llm(MODEL_FAST, prompt, system=RANKING_SYSTEM,
                        role="extractor", client=groq_client)
    ranked  = parse_json(result["raw_response"], label=f"ranking[{response_id}]")
    ranked  = ranked if isinstance(ranked, list) else []
    return {e.lower().strip(): idx + 1
            for idx, e in enumerate(ranked) if isinstance(e, str)}


def _description_length(entity: str, run_records: list) -> int:
    estimates = []
    for rec in run_records:
        raw       = rec.get("response_text", "")
        tokens    = rec.get("completion_tokens", 0)
        desc      = rec.get("brand_raw_text") or _extract_entity_description(raw, entity)
        desc      = desc if isinstance(desc, str) else ""
        if len(raw) > 0 and tokens > 0:
            estimates.append(round(len(desc) / len(raw) * tokens))
        else:
            estimates.append(0)
    return round(sum(estimates) / len(estimates)) if estimates else 0


def step4_enrich_entities(
    df_entities: pd.DataFrame,
    df_raw:      pd.DataFrame,
    output_path: str = os.path.join(OUTPUT_DIR, "enriched_entities.csv"),
    groq_client       = None,
) -> pd.DataFrame:

    df_joined = df_entities.merge(
        df_raw[["response_id","prompt_id","response_text",
                "run_id","model_id","completion_tokens"]],
        on="response_id", how="left")

    done_ids       = set()
    header_written = os.path.exists(output_path)
    if header_written:
        df_ex    = pd.read_csv(output_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
        done_ids = set(df_ex["response_id"].unique())
        print(f"  [Step 4] Resuming - {len(done_ids)} responses done")

    # Phase A — rankings (1 LLM call per response)
    print("  [Step 4] Phase A - ranking extraction")
    ranking_map = {}
    responses   = df_joined[["response_id","response_text"]].drop_duplicates()
    for idx, (_, row) in enumerate(responses.iterrows()):
        rid     = row["response_id"]
        ents    = df_joined[df_joined["response_id"] == rid]["entity"].tolist()
        print(f"    [{idx+1}/{len(responses)}] {rid} - {len(ents)} entities")
        ranking_map[rid] = _extract_rankings(rid, row["response_text"],
                                             ents, groq_client)

    # Phase B — description lengths (pure Python)
    print("  [Step 4] Phase B - description lengths")
    desc_map = {}
    for (pid, entity), grp in df_joined.groupby(["prompt_id","entity"]):
        desc_map[(pid, entity)] = _description_length(entity, grp.to_dict("records"))

    # Build records
    records = []
    for _, row in df_joined.iterrows():
        rid = row["response_id"]
        if rid in done_ids: continue
        pid, entity, run_id = row["prompt_id"], row["entity"], row["run_id"]
        records.append({
            "response_id":              rid,
            "prompt_id":                pid,
            "entity":                   entity,
            "run_id":                   run_id,
            "ranking_position":         ranking_map.get(rid, {}).get(entity),
            "description_length_tokens":desc_map.get((pid, entity), 0),
        })

    if records:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        pd.DataFrame(records).to_csv(
            output_path, mode="a", header=not header_written,
            index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    if not os.path.exists(output_path):
        logger.warning("[Step 4] No enriched entities — returning empty DataFrame")
        return pd.DataFrame(columns=["response_id","prompt_id","entity","run_id",
                                     "ranking_position","description_length_tokens"])
    df_enriched = pd.read_csv(output_path, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    print(f"\n  [Step 4] [OK] {df_enriched['entity'].nunique()} entities enriched")
    return df_enriched


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5 — Agentic data cleaning  (5 phases)
# ═══════════════════════════════════════════════════════════════════════════════

def _phase_a_prefilter(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Hard drop: empty / numeric / too short."""
    log, drop = [], pd.Series(False, index=df.index)
    for idx, row in df.iterrows():
        e = str(row.get("entity", "")).strip().lower()
        reason = ("empty"    if not e          else
                  "too_short" if len(e) < MIN_ENTITY_LENGTH else
                  "numeric"  if e.isdigit()   else None)
        if reason:
            drop[idx] = True
            log.append({"entity": row["entity"], "action": "dropped",
                        "reason": reason, "phase": "A"})
    df_out = df[~drop].reset_index(drop=True)
    print(f"    Dropped (hard): {drop.sum()} | Kept: {len(df_out)}")
    return df_out, log


def _phase_b_fuzzy(entities: list, counts: dict) -> tuple[dict, dict, dict]:
    """Fuzzy clustering with frequency-based canonical selection."""
    entities    = sorted(set(e.strip().lower() for e in entities if e))
    assigned, clusters, pair_scores = {}, {}, {}

    for entity in entities:
        if entity in assigned: continue
        cands = process.extract(
            entity, [e for e in entities if e not in assigned],
            scorer=fuzz.token_sort_ratio, limit=None,
            score_cutoff=FUZZY_AMBIG_LOWER)
        group = {entity}
        for match, score, _ in cands:
            if match == entity: continue
            pair_scores[(entity, match)] = score
            if score >= FUZZY_MERGE_THRESHOLD: group.add(match)
        canonical = max(group, key=lambda e: counts.get(e, 0))
        clusters[canonical] = list(group)
        for m in group: assigned[m] = canonical

    for e in entities:
        if e not in assigned:
            assigned[e] = e; clusters[e] = [e]
    return clusters, pair_scores, assigned


def _phase_c_llm(ambiguous_pairs: list, all_entities: list,
                 entity_context: dict, entity_counts: dict,
                 model: str, groq_client=None, batch_size: int = 20,
                 domain: str = "Tunisian restaurant") -> tuple:
    """LLM arbitration: pair resolution + entity validation."""
    merge_decisions, invalid_set, flag_set, log = {}, set(), set(), []

    ents_with_ctx = [{"entity": e, "context": entity_context.get(e, "")[:400]}
                     for e in all_entities]

    pair_batches   = [ambiguous_pairs[i:i+batch_size]
                      for i in range(0, max(len(ambiguous_pairs), 1), batch_size)]
    entity_batches = [ents_with_ctx[i:i+batch_size]
                      for i in range(0, len(ents_with_ctx), batch_size)]

    while len(pair_batches)   < len(entity_batches): pair_batches.append([])
    while len(entity_batches) < len(pair_batches):   entity_batches.append([])

    print(f"    {len(ambiguous_pairs)} ambiguous pairs | "
          f"{len(all_entities)} entities | {len(pair_batches)} batches")

    for b_idx, (pb, eb) in enumerate(zip(pair_batches, entity_batches)):
        print(f"    Batch {b_idx+1}/{len(pair_batches)} "
              f"— {len(pb)} pairs, {len(eb)} entities")
        prompt = (
            f"TASK 1 — Resolve {len(pb)} restaurant name pairs.\n"
            f"TASK 2 — Validate {len(eb)} entities using context.\n\n"
            + ("Pairs:\n" + json.dumps(
                [{"name_a": a, "name_b": b, "fuzzy_score": round(s, 1)}
                 for a, b, s in pb], ensure_ascii=False, indent=2) + "\n\n"
               if pb else "Pairs: []\n\n")
            + "Entities:\n" + json.dumps(eb, ensure_ascii=False, indent=2)
            + "\n\nReturn ONLY the JSON object."
        )
        result  = query_llm(model, prompt, system=_arbitration_system(domain),
                            role="analyst", client=groq_client)
        parsed  = parse_json(result["raw_response"], label=f"arb batch {b_idx+1}")
        if not isinstance(parsed, dict):
            print(f"    [WARN] Batch {b_idx+1} parse failed")
            for a, b, _ in pb: flag_set.update([a, b])
            continue

        for item in parsed.get("pair_resolutions", []):
            pair, decision, chosen, conf = (
                item.get("pair", []), item.get("decision", "DIFFERENT"),
                item.get("canonical"), item.get("confidence", "low"))
            if len(pair) != 2: continue
            a, b = pair
            if decision == "SAME" and chosen and conf == "high":
                other = b if chosen == a else a
                merge_decisions[other] = chosen
                log.append({"entity": other, "action": "merged_by_llm",
                             "reason": f"→ {chosen} (conf=high)", "phase": "C"})
            elif conf == "low":
                flag_set.update([a, b])
                log.append({"entity": f"{a}|{b}", "action": "flagged",
                             "reason": "LLM low confidence", "phase": "C"})

        for item in parsed.get("entity_validations", []):
            entity = item.get("entity", "").strip().lower()
            if not item.get("valid", True) and entity:
                invalid_set.add(entity)
                log.append({"entity": entity, "action": "invalidated",
                             "reason": item.get("reason", ""), "phase": "C"})
        time.sleep(0.3)

    print(f"    Merged: {len(merge_decisions)} | "
          f"Invalidated: {len(invalid_set)} | Flagged: {len(flag_set)}")
    return merge_decisions, invalid_set, flag_set, log


def _phase_d_reflect(cleaning_log: list, canonical_map: dict,
                     invalid_set: set, model: str,
                     groq_client=None,
                     domain: str = "Tunisian restaurant") -> dict:
    """Agent reflects on its own cleaning decisions and proposes corrections."""
    if not cleaning_log:
        return {"quality_score": 10, "proceed": True, "suspicious": []}

    prompt = (
        f"Review these {len(cleaning_log)} data cleaning decisions "
        f"for {domain} entities.\n\n"
        f"Cleaning log:\n{json.dumps(cleaning_log, ensure_ascii=False, indent=2)}\n\n"
        f"Current merges:\n{json.dumps({k:v for k,v in canonical_map.items() if k!=v}, ensure_ascii=False, indent=2)}\n\n"
        f"Invalidated:\n{json.dumps(list(invalid_set), ensure_ascii=False, indent=2)}\n\n"
        "Identify suspicious or incorrect decisions."
    )
    result     = query_llm(model, prompt, system=_reflection_system(domain),
                           role="analyst", client=groq_client)
    reflection = parse_json(result["raw_response"], label="Phase D reflection")
    reflection = reflection if isinstance(reflection, dict) else {}

    print(f"    Quality: {reflection.get('quality_score','?')}/10 | "
          f"Proceed: {reflection.get('proceed', True)}")
    for s in reflection.get("suspicious", []):
        print(f"    [WARN] '{s.get('entity')}': {s.get('issue')}")
    return reflection


def step5_clean_entities(
    df_enriched: pd.DataFrame,
    df_raw:      pd.DataFrame,
    df_entities: pd.DataFrame,
    model:       str = MODEL_STRONG,
    output_path: str = os.path.join(OUTPUT_DIR, "clean_entities.csv"),
    log_path:    str = os.path.join(OUTPUT_DIR, "cleaning_log.csv"),
    groq_client       = None,
    domain:      str = "Tunisian restaurant",
) -> pd.DataFrame:

    print(f"\n  [Step 5] {df_enriched['entity'].nunique()} unique entities in")
    cleaning_log  = []
    entity_counts = df_enriched["entity"].value_counts().to_dict()

    # Build entity context from brand_raw_text
    entity_context = {}
    for _, row in df_enriched.iterrows():
        entity = row["entity"]
        if entity in entity_context: continue
        match = df_entities[
            (df_entities["response_id"] == row["response_id"]) &
            (df_entities["entity"] == entity)]
        if not match.empty:
            bt = match.iloc[0].get("brand_raw_text", "")
            if pd.notna(bt) and str(bt).strip():
                entity_context[entity] = str(bt)[:400]; continue
        raw_match = df_raw[df_raw["response_id"] == row["response_id"]]
        if not raw_match.empty:
            entity_context[entity] = str(raw_match.iloc[0]["response_text"])[100:400]

    # Phase A
    print("  Phase A - hard pre-filter")
    df_filtered, log_a = _phase_a_prefilter(df_enriched)
    cleaning_log.extend(log_a)

    # Phase B
    print("  Phase B - fuzzy clustering")
    all_entities = df_filtered["entity"].dropna().str.strip().str.lower().unique().tolist()
    clusters, pair_scores, canonical_map = _phase_b_fuzzy(all_entities, entity_counts)

    for canonical, members in clusters.items():
        for m in members:
            if m != canonical:
                cleaning_log.append({"entity": m, "action": "merged",
                                     "reason": f"fuzzy ≥ {FUZZY_MERGE_THRESHOLD} → {canonical}",
                                     "phase": "B"})

    ambiguous = [(a, b, s) for (a, b), s in pair_scores.items()
                 if FUZZY_AMBIG_LOWER <= s <= FUZZY_AMBIG_UPPER]
    print(f"    {len(all_entities)} entities | {len(ambiguous)} ambiguous pairs")

    # Phase C
    print("  Phase C - LLM arbitration + validation")
    merge_decisions, invalid_set, flag_set, log_c = _phase_c_llm(
        ambiguous, all_entities, entity_context, entity_counts, model, groq_client,
        domain=domain)
    cleaning_log.extend(log_c)
    canonical_map.update(merge_decisions)

    # Phase D
    print("  Phase D - self-reflection")
    reflection = _phase_d_reflect(cleaning_log, canonical_map,
                                  invalid_set, model, groq_client, domain=domain)
    for corr in reflection.get("suspicious", []):
        entity = corr.get("entity", "").strip().lower()
        action = corr.get("action", "")
        if action == "undo_merge" and entity in canonical_map:
            cleaning_log.append({"entity": entity, "action": "merge_undone",
                                  "reason": corr.get("issue", ""), "phase": "D"})
            canonical_map[entity] = entity
        elif action == "undo_flag" and entity in flag_set:
            flag_set.discard(entity)
            cleaning_log.append({"entity": entity, "action": "flag_removed",
                                  "reason": corr.get("issue", ""), "phase": "D"})
        elif action == "undo_invalidation" and entity in invalid_set:
            invalid_set.discard(entity)
            cleaning_log.append({"entity": entity, "action": "invalidation_undone",
                                  "reason": corr.get("issue", ""), "phase": "D"})

    # Phase E — apply
    print("  Phase E - apply canonical mapping")
    df_clean = df_filtered.copy()
    df_clean["canonical_entity"]          = df_clean["entity"].map(
        lambda e: canonical_map.get(e, e))
    df_clean["ranking_position_filled"]   = df_clean["ranking_position"].fillna(
        NULL_RANK_PENALTY).astype(int)
    df_clean["quality_flag"]              = df_clean["entity"].apply(
        lambda e: "invalid" if e in invalid_set else "ok")
    df_clean["clean_flag"]                = df_clean["entity"].apply(
        lambda e: "invalid" if e in invalid_set
             else "flagged" if e in flag_set else "clean")
    df_clean["merge_source"]              = df_clean.apply(
        lambda r: r["entity"] if r["entity"] != r["canonical_entity"] else "", axis=1)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_clean.to_csv(output_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    pd.DataFrame(cleaning_log).to_csv(
        log_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    print(f"\n  [Step 5] [OK] {df_clean['canonical_entity'].nunique()} canonical entities | "
          f"clean: {df_clean[df_clean['clean_flag']=='clean']['canonical_entity'].nunique()} | "
          f"flagged: {df_clean[df_clean['clean_flag']=='flagged']['canonical_entity'].nunique()} | "
          f"invalid: {df_clean[df_clean['clean_flag']=='invalid']['canonical_entity'].nunique()}")
    return df_clean


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6 — Compute GEO features
# ═══════════════════════════════════════════════════════════════════════════════

def _co_mention_rate(df: pd.DataFrame) -> dict:
    co_map  = {}
    resp_ents = df.groupby("response_id")["canonical_entity"].apply(list).to_dict()
    for _, ents in resp_ents.items():
        ents = list(set(ents))
        for e in ents:
            if e not in co_map: co_map[e] = {}
            for o in ents:
                if o != e: co_map[e][o] = co_map[e].get(o, 0) + 1
    counts = df.groupby("canonical_entity").size().to_dict()
    return {e: {o: round(c/counts.get(e,1), 4) for o, c in co.items()}
            for e, co in co_map.items()}


def _prompt_type_response(df: pd.DataFrame, prompts: list) -> dict:
    intent_lkp = {str(p["prompt_id"]): str(p["intent_id"]) for p in prompts
                  if "intent_id" in p}
    df2 = df.copy()
    df2["intent_id"] = df2["prompt_id"].astype(str).map(intent_lkp)
    total_per_intent = df2.groupby("intent_id")["response_id"].nunique().to_dict()
    result = {}
    for entity, grp in df2.groupby("canonical_entity"):
        result[entity] = {
            intent: round(len(grp[grp["intent_id"]==intent]) / total_per_intent.get(intent,1), 4)
            for intent in df2["intent_id"].dropna().unique()
        }
    return result


def _entity_stats(group: pd.DataFrame, total_resp: int, total_models: int) -> dict:
    valid_ranks = group[group["ranking_position_filled"] != NULL_RANK_PENALTY
                        ]["ranking_position_filled"].tolist()
    mc  = len(group)
    mr  = round(mc / total_resp, 4)
    t1  = round(sum(1 for r in valid_ranks if r == 1) / mc, 4) if mc else 0.0
    avg = round(sum(valid_ranks)/len(valid_ranks), 4) if valid_ranks else None
    var = round(math.sqrt(sum((r-avg)**2 for r in valid_ranks)/len(valid_ranks)), 4) \
          if avg and len(valid_ranks) >= 2 else 0.0
    ss  = round(mr * (1 / (1 + var)), 4)
    lbl = "stable" if ss >= 0.75 else "variable" if ss >= 0.40 else "unstable"
    dm  = group["response_id"].apply(
        lambda r: r.split("__")[1] if "__" in str(r) else "").nunique()
    return {
        "mention_count":            mc,
        "mention_rate":             mr,
        "average_ranking_position": avg,
        "rank_variance":            var,
        "top1_rate":                t1,
        "frequency_across_runs":    group["run_id"].nunique(),
        "stability_score":          ss,
        "consistency_label":        lbl,
        "avg_description_length":   round(group["description_length_tokens"].mean(), 1),
        "model_presence_count":     dm,
        "cross_model_rate":         round(dm / total_models, 4),
    }


def step6_compute_metrics(
    df_clean:    pd.DataFrame,
    df_raw:      pd.DataFrame,
    prompts:     list,
    output_path: str = os.path.join(OUTPUT_DIR, "entity_features.csv"),
    global_path: str = os.path.join(OUTPUT_DIR, "entity_features_global.csv"),
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = df_clean[df_clean["clean_flag"] != "invalid"].copy()
    total_resp   = df["response_id"].nunique()
    total_models = df["response_id"].apply(
        lambda r: r.split("__")[1] if "__" in str(r) else "").nunique()

    print(f"  [Step 6] {total_resp} valid responses | "
          f"{total_models} models | {df['canonical_entity'].nunique()} entities")

    # Per-prompt features
    per_prompt_rows = []
    tr_per_prompt   = df.groupby("prompt_id")["response_id"].nunique().to_dict()
    for (pid, entity), grp in df.groupby(["prompt_id","canonical_entity"]):
        row = {"prompt_id": pid, "canonical_entity": entity}
        row.update(_entity_stats(grp, tr_per_prompt.get(pid, 1), total_models))
        per_prompt_rows.append(row)
    df_features = pd.DataFrame(per_prompt_rows)

    # Global features
    co_map  = _co_mention_rate(df)
    ptr_map = _prompt_type_response(df, prompts)
    global_rows = []
    for entity, grp in df.groupby("canonical_entity"):
        row = {"canonical_entity": entity}
        row.update(_entity_stats(grp, total_resp, total_models))
        row["prompt_type_response"] = json.dumps(ptr_map.get(entity, {}), ensure_ascii=False)
        row["co_mention_rate"]      = json.dumps(co_map.get(entity, {}), ensure_ascii=False)
        global_rows.append(row)
    df_global = pd.DataFrame(global_rows).sort_values(
        "stability_score", ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_features.to_csv(output_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    df_global.to_csv(global_path,   index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    print(f"\n  [Step 6] [OK] Per-prompt: {len(df_features)} rows | "
          f"Global: {len(df_global)} entities")
    print(f"  Top 5 by stability score:")
    print(df_global[["canonical_entity","stability_score","mention_rate",
                      "average_ranking_position"]].head(5).to_string(index=False))
    return df_features, df_global


# ═══════════════════════════════════════════════════════════════════════════════
# Agent 1 — Master orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class Agent1GeoAnalyser:
    """
    Agent 1 — Full GEO analysis pipeline (Steps 1-6).

    Args:
        prompt_set_path : path to prompt_set CSV from Agent 0
        output_dir      : folder for all output files
        query_models    : list of Groq model IDs to use for querying
        n_runs          : number of runs per (prompt × model)
        analyst_model   : model for cleaning / reflection steps
        groq_client     : optional pre-built Groq client
        mistral_key     : optional Mistral API key (for extraction)
    """

    def __init__(
        self,
        prompt_set_path: str,
        output_dir:      str       = OUTPUT_DIR,
        query_models:    list[str] = None,
        n_runs:          int       = N_RUNS,
        analyst_model:   str       = MODEL_STRONG,
        groq_client                = None,
        mistral_key:     str       = None,
        domain:          str       = "Tunisian restaurant",
    ):
        self.prompt_set_path = prompt_set_path
        self.output_dir      = output_dir
        self.query_models    = query_models or QUERY_MODELS
        self.n_runs          = n_runs
        self.analyst_model   = analyst_model
        self.client          = groq_client
        self.mistral_key     = mistral_key
        self.domain          = domain
        self.audit: list     = []

        os.makedirs(output_dir, exist_ok=True)

    def _p(self, fname: str) -> str:
        return os.path.join(self.output_dir, fname)

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        print(f"\n{'='*60}")
        print(f" Agent 1 - GEO Brand Analyser")
        print(f" Prompt set  : {self.prompt_set_path}")
        print(f" Models      : {self.query_models}")
        print(f" Runs/prompt : {self.n_runs}")
        print(f" Output dir  : {self.output_dir}")
        print(f"{'='*60}")

        # Step 1
        prompts = step1_load_prompts(self.prompt_set_path)
        self.audit.append({"step": 1, "prompts_loaded": len(prompts)})

        # Step 2
        df_raw = step2_query_prompts(
            prompts, self.query_models, self.n_runs,
            self._p("raw_responses.csv"), self.client)
        self.audit.append({"step": 2, "responses": len(df_raw)})

        # Step 3
        df_entities = step3_extract_entities(
            df_raw, self._p("extracted_entities.csv"),
            self.mistral_key, self.client)
        self.audit.append({"step": 3, "entities": df_entities["entity"].nunique()})

        # Step 4
        df_enriched = step4_enrich_entities(
            df_entities, df_raw,
            self._p("enriched_entities.csv"), self.client)
        self.audit.append({"step": 4, "enriched_rows": len(df_enriched)})

        # Step 5
        df_clean = step5_clean_entities(
            df_enriched, df_raw, df_entities,
            self.analyst_model,
            self._p("clean_entities.csv"),
            self._p("cleaning_log.csv"),
            self.client,
            domain=self.domain)
        self.audit.append({"step": 5,
                           "canonical_entities": df_clean["canonical_entity"].nunique()})

        # Step 6
        df_features, df_global = step6_compute_metrics(
            df_clean, df_raw, prompts,
            self._p("entity_features.csv"),
            self._p("entity_features_global.csv"))
        self.audit.append({"step": 6, "global_entities": len(df_global)})

        print(f"\n{'='*60}")
        print(f" Agent 1 complete - outputs in '{self.output_dir}/'") 
        print(f"   raw_responses.csv          {len(df_raw)} rows")
        print(f"   extracted_entities.csv     {len(df_entities)} rows")
        print(f"   enriched_entities.csv      {len(df_enriched)} rows")
        print(f"   clean_entities.csv         {len(df_clean)} rows")
        print(f"   entity_features.csv        {len(df_features)} rows")
        print(f"   entity_features_global.csv {len(df_global)} rows")
        print(f"   Total tokens used          {TOKEN_USAGE['total_tokens']}")
        print(f"{'='*60}")

        return df_features, df_global
