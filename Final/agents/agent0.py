# -*- coding: utf-8 -*-
"""Agent 0 - Intent discovery and prompt generation."""

import time
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from config import MODEL_INTENT, INTER_PROMPT_DELAY_S
from core.llm_client import call_llm, parse_json


def agent0_generate_intents(domain: str, n_intents: int, model: str) -> list:
    """Step 1: LLM discovers relevant intent types for the given domain."""
    system ="""You are an expert in user behavior and search intent analysis, specialising in GEO (Generative Engine Optimization) research.
Your job is to identify the most realistic and diverse intent types that users express when querying an AI assistant about a specific domain.

STRICT RULES for GEO-valid intents:
1. Every intent must produce prompts where a user asks an AI ABOUT establishments — never TO an establishment.
   INVALID: "Quels sont les plats de votre restaurant ?" (addresses the AI as if it IS the restaurant)
   VALID:   "Quels sont les restaurants tunisiens les plus réputés à Tunis ?"
2. Every intent must be geographically anchored to the country/city in the domain — never drift to other cities or countries.
   INVALID for domain "Tunisian restaurants": prompts about Paris, Lyon, France
   VALID: prompts about Tunis, Sousse, Sfax, Djerba, Monastir, Hammamet
3. Every intent must naturally lead to naming SPECIFIC establishments — not generic cuisine descriptions.
   INVALID: "Comment prépare-t-on le couscous ?" (about food, not brands)
   VALID:   "Quel restaurant tunisien est réputé pour son couscous à Sousse ?"

Always respond in valid JSON only, no explanation, no markdown."""

    prompt = f"""Domain: {domain}

Generate exactly {n_intents} distinct intent categories that users would have
when asking an AI assistant about this domain.

For each intent provide:
- intent_id: short snake_case identifier
- intent_name: clear label
- description: one sentence explaining this intent type (must follow the 3 GEO rules above)

Respond ONLY with a JSON array like:
[
  {{
    "intent_id": "top_recommendation",
    "intent_name": "Top Restaurant Recommendation",
    "description": "User wants the AI to name the best-known or most reputed restaurants in a specific Tunisian city"
  }}
]"""


    result  = call_llm(prompt=prompt, system=system, preferred_model=model)
    raw     = result.get("raw_response", "") if isinstance(result, dict) else str(result)
    intents = parse_json(raw, label="agent0_generate_intents")

    # LLM sometimes wraps the array in an object: {"intents": [...]}
    if isinstance(intents, dict):
        for key in ("intents", "intent_types", "categories", "results", "data"):
            if key in intents and isinstance(intents[key], list):
                intents = intents[key]
                break
        else:
            intents = list(intents.values())[0] if intents else []

    intents = [
        i for i in intents
        if isinstance(i, dict) and "intent_id" in i and "intent_name" in i
    ]

    logger.info(f"[Agent0] Step 1: {len(intents)} intents discovered")
    return intents


def agent0_generate_prompts(domain: str, intents: list, languages: list,
                             n_variants: int, model: str) -> pd.DataFrame:
    """Step 2: For each intent, generate n_variants prompts per language."""
    system = """You are an expert in prompt engineering and multilingual NLP, specialising in GEO (Generative Engine Optimization) research.
Your job is to generate realistic, diverse user queries that will reveal which specific brands or establishments an AI assistant knows.

CRITICAL GEO RULE: At least 60% of prompts must be phrased to force the AI to recall and name SPECIFIC, WELL-KNOWN establishments.
- GOOD: "Quel est le restaurant tunisien le plus réputé à Tunis ?" → forces naming a known brand
- GOOD: "Cite-moi les 3 meilleures adresses pour manger tunisien à Tunis"  → forces a ranked list of real names
- BAD:  "Tu connais un bon resto tunisien ?" → too vague, invites invention
- BAD:  "Quels sont les plats de votre restaurant ?" → asks the AI as if it IS the restaurant

Prompts that ask for hours, menus of unnamed restaurants, or generic cuisine descriptions do NOT help GEO — avoid them.
Always respond in valid JSON only, no explanation, no markdown."""

    all_prompts = []

    for intent in intents:
        if not isinstance(intent, dict):
            logger.warning(f"[Agent0] Skipping non-dict intent: {intent!r}")
            continue
        prompt = f"""Domain: {domain}
Intent: {intent['intent_name']} — {intent['description']}
Languages: {languages}
Number of variants per language: {n_variants}

Generate {n_variants} natural user queries for EACH of these languages: {languages}

Queries must:
- Sound like real user input, not formal questions
- Be phrased to elicit SPECIFIC establishment names (use superlatives, rankings, "cite", "liste", "le meilleur", "les plus connus")
- Be geographically precise (name a city or neighbourhood, not just "Tunisie" generically)
- Be diverse in phrasing and structure
- Stay within the domain and intent

Respond ONLY with a JSON array like:
[
  {{
    "intent_id": "{intent['intent_id']}",
    "language": "fr",
    "variant_id": 1,
    "prompt_text": "..."
  }}
]"""

        result   = call_llm(prompt=prompt, system=system, preferred_model=model)
        raw      = result.get("raw_response", "") if isinstance(result, dict) else str(result)
        variants = parse_json(raw, label=f"agent0_prompts/{intent['intent_id']}")

        all_prompts.extend(variants)
        logger.info(f"[Agent0] Intent '{intent['intent_id']}' — {len(variants)} prompts")
        time.sleep(INTER_PROMPT_DELAY_S)

    df = pd.DataFrame(all_prompts).reset_index(drop=True)
    df['prompt_id'] = ['P' + str(i + 1).zfill(3) for i in range(len(df))]
    df = df[['prompt_id', 'intent_id', 'language', 'variant_id', 'prompt_text']]
    return df


def agent0_reflect(domain: str, intents: list,
                   prompts_df: pd.DataFrame, model: str) -> dict:
    """Agent reflects on its own output quality before passing to Agent 1."""
    system = """You are a critical evaluator of prompt sets for NLP research.
    Respond in valid JSON only."""

    prompt = f"""You generated this prompt set for domain: '{domain}'

Intents discovered: {[i['intent_id'] for i in intents]}
Total prompts generated: {len(prompts_df)}
Languages: {prompts_df['language'].unique().tolist()}
All prompts:
{prompts_df['prompt_text'].tolist()}

Evaluate the quality of this prompt set for GEO (Generative Engine Optimization) research:

1. Are the intents diverse enough for GEO analysis?
2. Are there important intents missing?
3. Are prompts natural and realistic?
4. Is language quality acceptable?
5. GEO CRITICAL — Do at least 60% of prompts force the AI to name SPECIFIC known establishments?
   Count prompts that use superlatives, rankings, "cite", "liste", "le meilleur", "les plus connus", or name a specific place.
   Prompts asking for hours, menus, or generic cuisine info do NOT count.
   If fewer than 60% force specific naming → set proceed: false and list them in issues.

Respond ONLY with:
{{
  "quality_score": 0-10,
  "proceed": true/false,
  "brand_eliciting_pct": <percentage of prompts that force specific brand naming>,
  "missing_intents": [],
  "issues": [],
  "recommendation": "..."
}}"""

    result = call_llm(prompt=prompt, system=system, preferred_model=model)
    raw    = result.get("raw_response", "") if isinstance(result, dict) else str(result)
    parsed = parse_json(raw, label="agent0_reflect")

    # LLM may wrap the object in an array — unwrap if needed
    if isinstance(parsed, list):
        report: dict = parsed[0] if parsed else {}
    elif isinstance(parsed, dict):
        report = parsed
    else:
        report = {}

    logger.info(
        f"[Agent0] Self-Reflection — score={report.get('quality_score', '?')}/10 "
        f"proceed={report.get('proceed', True)} issues={report.get('issues', [])}"
    )
    return report


def agent0_run(domain: str, model: str = None, languages: list = None,
               n_intents: int = 4, n_variants: int = 3,
               max_reflection_loops: int = 2,
               output_dir: str = None,
               groq_client=None) -> pd.DataFrame:
    """Run Agent 0: intent discovery + prompt generation with self-reflection.

    Args:
        domain:               Domain string (e.g. "Tunisian restaurants").
        model:                Groq model ID to use. Defaults to MODEL_INTENT.
        languages:            List of language codes. Defaults to ["fr", "ar"].
        n_intents:            Number of intent categories to discover.
        n_variants:           Number of prompt variants per intent per language.
        max_reflection_loops: How many self-reflection loops to allow.
        output_dir:           Directory for the prompt CSV output.
        groq_client:          Ignored (kept for API compatibility).

    Returns:
        DataFrame with columns: prompt_id, intent_id, language, variant_id, prompt_text.
    """
    if model is None:
        model = MODEL_INTENT
    if languages is None:
        languages = ["fr", "ar"]

    logger.info(f"[Agent0] Starting — domain='{domain}'")

    prompt_df = pd.DataFrame()

    for loop in range(max_reflection_loops):
        logger.info(f"[Agent0] Loop {loop + 1}/{max_reflection_loops}")

        intents   = agent0_generate_intents(domain=domain, n_intents=n_intents, model=model)
        prompt_df = agent0_generate_prompts(
            domain=domain, intents=intents,
            languages=languages, n_variants=n_variants, model=model,
        )
        report = agent0_reflect(domain=domain, intents=intents, prompts_df=prompt_df, model=model)

        if report.get('proceed', True):
            logger.info(f"[Agent0] Satisfied at loop {loop + 1}")
            break
        n_intents = n_intents + len(report.get('missing_intents', []))
    else:
        logger.warning("[Agent0] Reached max loops — saving best available output")

    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        out_path = f"{output_dir}/prompt_set_{domain.replace(' ', '_')}.csv"
    else:
        out_path = f"prompt_set_{domain.replace(' ', '_')}.csv"
    prompt_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    logger.success(f"[Agent0] Complete — {len(prompt_df)} prompts → '{out_path}'")
    return prompt_df


# ── LangGraph node ────────────────────────────────────────────────────────────

from graph.state import GEOState


def run_agent0_node(state: GEOState) -> dict:
    """LangGraph node: runs Agent 0 and merges results into state."""
    errors = list(state.get("errors", []))
    try:
        prompt_df = agent0_run(
            domain               = state["domain"],
            model                = MODEL_INTENT,
            languages            = state.get("languages", ["fr"]),
            n_intents            = state.get("n_intents", 4),
            n_variants           = state.get("n_variants", 3),
            max_reflection_loops = state.get("max_reflection_loops", 2),
            output_dir           = state.get("output_dir"),
        )
        prompt_set = prompt_df.to_dict(orient="records")
    except Exception as exc:
        errors.append(f"agent0: {exc}")
        prompt_set = []

    return {
        "prompt_set":   prompt_set,
        "errors":       errors,
        "current_step": "agent0",
    }
