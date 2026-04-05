"""
agents/agent0_prompt_generator.py
══════════════════════════════════
Agent 0 — GEO Prompt Set Generator

Pipeline:
  Step 1 · Discover intent types for the domain           (LLM)
  Step 2 · Generate n_variants prompts per intent/language (LLM)
  Step 3 · Self-reflect on quality → regenerate if needed  (LLM)
  Step 4 · Save prompt_set_{domain}.csv

Output schema:
  prompt_id | intent_id | language | variant_id | prompt_text
"""

import os, sys, json, time, re
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.llm_client import query_llm, parse_json, MODEL_STRONG


# ── System prompts ────────────────────────────────────────────────────────────

INTENT_SYSTEM = """\
You are an expert in user behavior and search intent analysis.
Your job is to identify the most realistic and diverse intent types
that users express when querying an AI assistant about a specific domain.
Always respond in valid JSON only — no explanation, no markdown.
"""

PROMPT_GEN_SYSTEM = """\
You are an expert in prompt engineering and multilingual NLP.
Your job is to generate realistic, diverse user queries for a given intent and domain.
Queries must sound natural, like something a real user would type.
Always respond in valid JSON only — no explanation, no markdown.
"""

REFLECTION_SYSTEM = """\
You are a critical evaluator of prompt sets for GEO (Generative Engine Optimization) research.
Respond in valid JSON only.
"""


# ── Step 1: Intent discovery ──────────────────────────────────────────────────

def _generate_intents(domain: str, n_intents: int, model: str,
                      client=None, location: str = "Tunisia") -> list:
    """Discover n_intents distinct user intent types for the domain."""
    prompt = f"""\
Domain: {domain}
Location: {location}

Generate exactly {n_intents} distinct intent categories that users would have
when asking an AI assistant about this domain in {location}.

For each intent provide:
- intent_id:   short snake_case identifier
- intent_name: clear label
- description: one sentence explaining this intent type

Respond ONLY with a JSON array like:
[
  {{
    "intent_id":   "recommendation",
    "intent_name": "General Recommendation",
    "description": "User wants the AI to suggest the best options in the domain"
  }}
]"""

    result   = query_llm(model, prompt, system=INTENT_SYSTEM,
                         role="analyst", client=client)
    intents  = parse_json(result["raw_response"], label="intent_discovery")
    intents  = intents if isinstance(intents, list) else []
    print(f"  [Step 1] {len(intents)} intents discovered")
    return intents


# ── Step 2: Prompt generation ─────────────────────────────────────────────────

def _generate_prompts(domain: str, intents: list, languages: list,
                      n_variants: int, model: str, client=None,
                      location: str = "Tunisia") -> pd.DataFrame:
    """
    For each intent × language, generate n_variants natural user queries.
    Saves progressively — safe to restart.
    """
    all_prompts = []

    for intent in intents:
        prompt = f"""\
Domain: {domain}
Location: {location}
Intent: {intent['intent_name']} — {intent['description']}
Languages: {languages}
Number of variants per language: {n_variants}

Generate {n_variants} natural user queries for EACH of these languages: {languages}
Queries must:
- Sound like real user input, not formal questions
- Be diverse in phrasing (avoid repetition)
- Be relevant to the domain and intent in {location}
- Vary in length and structure
- Reference {location} or cities/regions within {location} where geographically relevant

Respond ONLY with a JSON array like:
[
  {{
    "intent_id":   "{intent['intent_id']}",
    "language":    "fr",
    "variant_id":  1,
    "prompt_text": "..."
  }}
]"""

        result   = query_llm(model, prompt, system=PROMPT_GEN_SYSTEM,
                             role="query", client=client)
        variants = parse_json(result["raw_response"],
                              label=f"prompts[{intent['intent_id']}]")
        variants = variants if isinstance(variants, list) else []
        all_prompts.extend(variants)
        print(f"  [Step 2] Intent '{intent['intent_id']}' > "
              f"{len(variants)} prompts generated")
        time.sleep(0.5)

    df = pd.DataFrame(all_prompts).reset_index(drop=True)
    df["prompt_id"] = ["P" + str(i + 1).zfill(3) for i in range(len(df))]
    # Ensure all expected columns exist
    for col in ["intent_id", "language", "variant_id", "prompt_text"]:
        if col not in df.columns:
            df[col] = ""
    return df[["prompt_id", "intent_id", "language", "variant_id", "prompt_text"]]


# ── Step 3: Self-reflection ───────────────────────────────────────────────────

def _reflect(domain: str, intents: list, prompts_df: pd.DataFrame,
             model: str, client=None, location: str = "Tunisia") -> dict:
    """
    Agent reflects on its own output quality before passing to Agent 1.
    Returns { quality_score, proceed, missing_intents, issues, recommendation }.
    """
    prompt = f"""\
You generated this prompt set for domain: '{domain}' in {location}

Intents discovered: {[i['intent_id'] for i in intents]}
Total prompts generated: {len(prompts_df)}
Languages: {prompts_df['language'].unique().tolist() if 'language' in prompts_df.columns else []}
Sample prompts: {prompts_df['prompt_text'].head(6).tolist() if 'prompt_text' in prompts_df.columns else []}

Evaluate the quality of this prompt set:
1. Are the intents diverse enough for GEO analysis?
2. Are there important intents missing?
3. Are prompts natural and realistic?
4. Is language quality acceptable?

Respond ONLY with:
{{
  "quality_score":    <int 0-10>,
  "proceed":          <true|false>,
  "missing_intents":  [],
  "issues":           [],
  "recommendation":   "..."
}}"""

    result = query_llm(model, prompt, system=REFLECTION_SYSTEM,
                       role="analyst", client=client)
    report = parse_json(result["raw_response"], label="reflection")
    report = report if isinstance(report, dict) else {}

    print(f"\n  [Self-Reflection]")
    print(f"    Quality score : {report.get('quality_score', '?')}/10")
    print(f"    Proceed       : {report.get('proceed', True)}")
    print(f"    Issues        : {report.get('issues', [])}")
    print(f"    Missing       : {report.get('missing_intents', [])}")
    return report


# ── Agent 0 main ──────────────────────────────────────────────────────────────

class Agent0PromptGenerator:
    """
    Agent 0 — Generates a diverse, high-quality prompt set for GEO analysis.

    Args:
        domain               : e.g. "Tunisian restaurants"
        model                : LLM model identifier
        languages            : list of language codes, e.g. ["fr", "ar"]
        n_intents            : number of intent types to discover
        n_variants           : prompts per intent per language
        max_reflection_loops : max regeneration cycles
        output_dir           : where to save prompt_set CSV
    """

    def __init__(
        self,
        domain:               str,
        model:                str       = MODEL_STRONG,
        languages:            list[str] = None,
        n_intents:            int       = 4,
        n_variants:           int       = 3,
        max_reflection_loops: int       = 2,
        output_dir:           str       = ".",
        groq_client                     = None,
        location:             str       = "Tunisia",
    ):
        self.domain               = domain
        self.model                = model
        self.languages            = languages or ["fr"]
        self.n_intents            = n_intents
        self.n_variants           = n_variants
        self.max_reflection_loops = max_reflection_loops
        self.output_dir           = output_dir
        self.client               = groq_client
        self.location             = location
        self.audit: list[dict]    = []

    def run(self) -> pd.DataFrame:
        print(f"\n{'='*60}")
        print(f" Agent 0 - GEO Prompt Generator")
        print(f" Domain   : {self.domain}")
        print(f" Location : {self.location}")
        print(f" Languages: {self.languages}")
        print(f" Intents  : {self.n_intents}  |  Variants: {self.n_variants}")
        print(f"{'='*60}")

        prompt_df = pd.DataFrame()
        intents   = []
        n_intents = self.n_intents

        for loop in range(1, self.max_reflection_loops + 1):
            print(f"\n  Loop {loop}/{self.max_reflection_loops}")

            # Step 1 — discover intents
            intents = _generate_intents(
                self.domain, n_intents, self.model, self.client, self.location)

            # Step 2 — generate prompts
            prompt_df = _generate_prompts(
                self.domain, intents, self.languages,
                self.n_variants, self.model, self.client, self.location)

            # Step 3 — self-reflect
            report = _reflect(
                self.domain, intents, prompt_df, self.model, self.client, self.location)

            self.audit.append({
                "loop":           loop,
                "quality_score":  report.get("quality_score"),
                "proceed":        report.get("proceed", True),
                "issues":         report.get("issues", []),
                "n_prompts":      len(prompt_df),
            })

            if report.get("proceed", True):
                print(f"  [OK] Agent 0 satisfied with output at loop {loop}")
                break
            else:
                print(f"  [FAIL] Agent 0 not satisfied - regenerating ...")
                # Grow intent count to fill gaps
                n_intents = n_intents + len(report.get("missing_intents", []))
        else:
            print(f"  [WARNING] Agent 0 reached max loops - using best available output")

        # Save
        os.makedirs(self.output_dir, exist_ok=True)
        slug        = self.domain.replace(" ", "_")
        output_path = os.path.join(self.output_dir, f"prompt_set_{slug}.csv")
        prompt_df.to_csv(output_path, index=False, encoding="utf-8")

        print(f"\n  Agent 0 complete - {len(prompt_df)} prompts > '{output_path}'")
        return prompt_df
