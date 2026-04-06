# -*- coding: utf-8 -*-
"""Agent 0 - Intent discovery and prompt generation."""

import json
import time
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MODEL_INTENT
from core.llm_client import call_llm

MODEL_1 = MODEL_INTENT


def query_llm_structured(model: str, prompt: str, system: str = "") -> str:
    """Call LLM with automatic fallback on rate-limit errors."""
    return call_llm(prompt=prompt, system=system, preferred_model=model,
                    max_completion_tokens=4096, temperature=0.2)


def agent0_generate_intents(domain: str, n_intents: int, model: str) -> list:
    """
    Step 1: LLM discovers relevant intent types for the given domain.
    """
    system = """You are an expert in user behavior and search intent analysis.
Your job is to identify the most realistic and diverse intent types
that users express when querying an AI assistant about a specific domain.
Always respond in valid JSON only, no explanation, no markdown."""

    prompt = f"""Domain: {domain}

Generate exactly {n_intents} distinct intent categories that users would have
when asking an AI assistant about this domain.

For each intent provide:
- intent_id: short snake_case identifier
- intent_name: clear label
- description: one sentence explaining this intent type

Respond ONLY with a JSON array like:
[
  {{
    "intent_id": "recommendation",
    "intent_name": "General Recommendation",
    "description": "User wants the AI to suggest the best options in the domain"
  }}
]"""

    response = query_llm_structured(model=model, prompt=prompt, system=system)

    # clean response in case LLM adds markdown
    intents = parse_json_response(response, model, prompt, system)

    print(f"Agent 0, Step 1: {len(intents)} intents discovered")
    return intents


def agent0_generate_prompts(domain: str, intents: list, languages: list,
                             n_variants: int, model: str) -> pd.DataFrame:
    """
    Step 2: For each intent, generate n_variants prompts per language.
    """
    system = """You are an expert in prompt engineering and multilingual NLP.
Your job is to generate realistic, diverse user queries for a given intent and domain.
Queries must sound natural, like something a real user would type.
Always respond in valid JSON only, no explanation, no markdown."""

    all_prompts = []

    for intent in intents:
        prompt = f"""Domain: {domain}
Intent: {intent['intent_name']} {intent['description']}
Languages: {languages}
Number of variants per language: {n_variants}

Generate {n_variants} natural user queries for EACH of these languages: {languages}
Queries must:
- Sound like real user input, not formal questions
- Be diverse in phrasing (avoid repetition)
- Be relevant to the domain and intent
- Vary in length and structure

Respond ONLY with a JSON array like:
[
  {{
    "intent_id": "{intent['intent_id']}",
    "language": "fr",
    "variant_id": 1,
    "prompt_text": "..."
  }}
]"""

        response = query_llm_structured(model=model, prompt=prompt, system=system)
        variants = parse_json_response(response, model, prompt, system)

        all_prompts.extend(variants)
        print(f"Intent '{intent['intent_id']}'  {len(variants)} prompts generated")
        time.sleep(0.5)  # avoid rate limiting

    # build DataFrame
    df = pd.DataFrame(all_prompts)

    # add prompt_id
    df = df.reset_index(drop=True)
    df['prompt_id'] = ['P' + str(i+1).zfill(3) for i in range(len(df))]

    # reorder columns
    df = df[['prompt_id', 'intent_id', 'language', 'variant_id', 'prompt_text']]

    return df


def parse_json_response(response: str, model: str,
                        original_prompt: str, system: str,
                        max_retries: int = 3) -> list:
    for attempt in range(max_retries):
        try:
            clean = (response.strip()
                     .removeprefix("```json")
                     .removeprefix("```")
                     .removesuffix("```")
                     .strip())
            return json.loads(clean)

        except json.JSONDecodeError as e:
            print(f"JSON parse failed (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                fix_prompt = f"""Your previous response was not valid JSON.
Error: {e}
Your response was:
{response}
Fix it and return ONLY valid JSON, nothing else."""
                response = query_llm_structured(
                    model=model,
                    prompt=fix_prompt,
                    system=system
                )
            else:
                print(f"[ERROR] Failed to parse JSON after {max_retries} attempts - returning empty list")
                return []


def agent0_reflect(domain: str, intents: list,
                   prompts_df: pd.DataFrame, model: str) -> dict:
    """
    Agent reflects on its own output quality before passing to Agent 1.
    Returns quality report + decision to proceed or regenerate.
    """
    system = """You are a critical evaluator of prompt sets for NLP research.
    Respond in valid JSON only."""

    prompt = f"""You generated this prompt set for domain: '{domain}'

Intents discovered: {[i['intent_id'] for i in intents]}
Total prompts generated: {len(prompts_df)}
Languages: {prompts_df['language'].unique().tolist()}
Sample prompts: {prompts_df['prompt_text'].head(6).tolist()}

Evaluate the quality of this prompt set:
1. Are the intents diverse enough for GEO analysis?
2. Are there important intents missing?
3. Are prompts natural and realistic?
4. Is language quality acceptable?

Respond ONLY with:
{{
  "quality_score": 0-10,
  "proceed": true/false,
  "missing_intents": [],
  "issues": [],
  "recommendation": "..."
}}"""

    response = query_llm_structured(model=model, prompt=prompt, system=system)
    report = parse_json_response(response, model, prompt, system)

    print(f"\n Agent 0 Self-Reflection:")
    print(f"   Quality score : {report['quality_score']}/10")
    print(f"   Proceed       : {report['proceed']}")
    print(f"   Issues        : {report.get('issues', [])}")
    print(f"   Missing       : {report.get('missing_intents', [])}")

    return report


def agent0_run(domain: str, model: str = None, languages: list = None,
               n_intents: int = 4, n_variants: int = 3,
               max_reflection_loops: int = 2, groq_client=None) -> pd.DataFrame:
    """Run Agent 0: intent discovery + prompt generation with self-reflection.

    Args:
        domain:               Domain string (e.g. "Tunisian restaurants").
        model:                Groq model ID to use. Defaults to MODEL_1.
        languages:            List of language codes. Defaults to ["fr", "ar"].
        n_intents:            Number of intent categories to discover.
        n_variants:           Number of prompt variants per intent per language.
        max_reflection_loops: How many self-reflection loops to allow.
        groq_client:          Ignored (kept for API compatibility).

    Returns:
        DataFrame with columns: prompt_id, intent_id, language, variant_id,
        prompt_text.
    """
    if model is None:
        model = MODEL_1
    if languages is None:
        languages = ["fr", "ar"]

    print(f"\n Agent 0 starting, Domain: '{domain}'")

    prompt_df = pd.DataFrame()

    for loop in range(max_reflection_loops):
        print(f"\nLoop {loop + 1}/{max_reflection_loops}")

        # 1: discover intents
        intents = agent0_generate_intents(
            domain=domain, n_intents=n_intents, model=model
        )

        # 2: generate prompts
        prompt_df = agent0_generate_prompts(
            domain=domain, intents=intents,
            languages=languages, n_variants=n_variants, model=model
        )

        # 3: self reflection
        report = agent0_reflect(
            domain=domain, intents=intents,
            prompts_df=prompt_df, model=model
        )

        if report['proceed']:
            print(f"\n Agent 0 satisfied with output at loop {loop+1}")
            break
        else:
            print(f"\n Agent 0 not satisfied - regenerating...")
            # inject missing intents into next loop
            n_intents = n_intents + len(report.get('missing_intents', []))

    else:
        print(f"\n Agent 0 reached max loops - saving best available output")

    output_path = f"prompt_set_{domain.replace(' ', '_')}.csv"
    prompt_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n Agent 0 complete - {len(prompt_df)} prompts -> '{output_path}'")
    return prompt_df


# ── LangGraph node ────────────────────────────────────────────────────────────

from pipeline_state import PipelineState


def run_agent0_node(state: PipelineState) -> PipelineState:
    """LangGraph node: runs Agent 0 and merges results into state."""
    errors = list(state.get("errors", []))
    try:
        prompt_df = agent0_run(
            domain               = state["domain"],
            model                = MODEL_1,
            languages            = state.get("languages", ["fr"]),
            n_intents            = state.get("n_intents", 4),
            n_variants           = state.get("n_variants", 3),
            max_reflection_loops = state.get("max_reflection_loops", 2),
        )
        prompt_set = prompt_df.to_dict(orient="records")
    except Exception as exc:
        errors.append(f"agent0: {exc}")
        prompt_set = []

    return {
        **state,
        "prompt_set":  prompt_set,
        "errors":      errors,
        "current_step": "agent0",
    }
