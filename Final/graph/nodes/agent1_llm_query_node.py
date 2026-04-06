"""
graph/nodes/agent1_llm_query_node.py
══════════════════════════════════════
LangGraph nodes for Agent 1 LLM querying.

agent1_llm_query_all_node  (PRIMARY)
    Sequential-per-prompt, parallel-per-model node.
    For each prompt: fires asyncio.gather() over all (model × run) combos,
    then moves to the next prompt. This respects Groq rate limits while
    still parallelising models within each prompt.

    Input  (from state):
        prompts, query_models, n_runs, domain
    Output (merged into state via Annotated[list, operator.add]):
        raw_responses += [all rows for all prompts]

agent1_llm_query_node  (LEGACY — kept for backward compatibility)
    Original single-task fan-out target; one (prompt, model, run) at a time.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Final/ is root

import asyncio
from loguru import logger

from core.llm_client import async_query_llm, query_llm, MODEL_FAST

# ── throttle constants (overridable via env / config.py) ────────────────────
try:
    from config import INTER_PROMPT_DELAY_S as _INTER_PROMPT_DELAY_S  # type: ignore
except ImportError:
    _INTER_PROMPT_DELAY_S = 0.6  # seconds; override via INTER_PROMPT_DELAY_S env var

_STAGGER_S = 0.05  # 50 ms between coroutine launches within one prompt

# System prompt for restaurant GEO queries
_QUERY_SYSTEM_TEMPLATE = """\
You are a helpful assistant knowledgeable about {domain}s.
Answer naturally and helpfully. Mention specific establishment names when relevant.
"""


# ── stagger helper ──────────────────────────────────────────────────────────

async def _delayed_query(delay: float, model_id: str, prompt_text: str,
                         system: str, role: str) -> dict:
    """Wrapper that delays the start of an LLM call to stagger concurrent launches."""
    if delay > 0.0:
        await asyncio.sleep(delay)
    return await async_query_llm(model_id, prompt_text, system=system, role=role)


# ═══════════════════════════════════════════════════════════════════════════════
# PRIMARY node — sequential prompts, parallel models
# ═══════════════════════════════════════════════════════════════════════════════

async def agent1_llm_query_all_node(state: dict) -> dict:
    """
    For each prompt (sequentially): fire all (model × run) LLM calls in
    parallel using asyncio.gather(), then proceed to the next prompt.

    This prevents thundering-herd rate-limit issues while still giving full
    concurrency within each prompt's model set.
    """
    prompts      = state.get("prompts", [])
    query_models = state.get("query_models", [MODEL_FAST])
    n_runs       = state.get("n_runs", 1)
    domain       = state.get("domain", "Tunisian restaurant")

    system   = _QUERY_SYSTEM_TEMPLATE.format(domain=domain)
    all_rows = []

    logger.info(
        f"[LLMQueryAll] {len(prompts)} prompts × {len(query_models)} models "
        f"× {n_runs} run(s) — processing prompt-by-prompt"
    )

    for i, prompt in enumerate(prompts):
        prompt_id   = prompt.get("prompt_id",   "")
        prompt_text = prompt.get("prompt_text", "")

        # All (model × run) combos for this single prompt
        combos = [
            (model_id, run_idx)
            for model_id in query_models
            for run_idx  in range(1, n_runs + 1)
        ]

        logger.info(
            f"[LLMQueryAll] {i+1}/{len(prompts)} · {prompt_id} "
            f"— firing {len(combos)} parallel calls"
        )

        coros = [
            _delayed_query(
                delay=idx * _STAGGER_S,
                model_id=model_id,
                prompt_text=prompt_text,
                system=system,
                role="query",
            )
            for idx, (model_id, _) in enumerate(combos)
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)

        # ── inter-prompt breathing room ──────────────────────────────────────
        if i < len(prompts) - 1:   # no sleep after the last prompt
            await asyncio.sleep(_INTER_PROMPT_DELAY_S)

        for (model_id, run_idx), result in zip(combos, results):
            if isinstance(result, BaseException):
                logger.error(
                    f"[LLMQueryAll] {prompt_id}/{model_id}/run_{run_idx}: {result}"
                )
                continue
            raw_text = result.get("raw_response", "")
            if not raw_text:
                logger.warning(
                    f"[LLMQueryAll] Empty response: {prompt_id}/{model_id}/run_{run_idx}"
                )
                continue
            all_rows.append({
                "prompt_id":         prompt_id,
                "prompt_text":       prompt_text,
                "model_id":          model_id,
                "run_idx":           run_idx,
                "response_text":     raw_text,
                "completion_tokens": result.get("completion_tokens", 0),
                "prompt_tokens":     result.get("prompt_tokens", 0),
                "total_tokens":      result.get("total_tokens", 0),
            })

    logger.info(f"[LLMQueryAll] Complete — {len(all_rows)} responses collected")
    return {"raw_responses": all_rows}


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY node — single (prompt, model, run) fan-out target
# ═══════════════════════════════════════════════════════════════════════════════

async def agent1_llm_query_node(state: dict) -> dict:
    """
    Legacy fan-out node: run a single LLM query for one (prompt, model, run).
    Kept for backward compatibility. Prefer agent1_llm_query_all_node.
    """
    prompt_id   = state.get("prompt_id",   "")
    prompt_text = state.get("prompt_text", "")
    model_id    = state.get("model_id",    MODEL_FAST)
    run_idx     = state.get("run_idx",     1)
    domain      = state.get("domain",      "Tunisian restaurant")

    system = _QUERY_SYSTEM_TEMPLATE.format(domain=domain)

    logger.debug(
        f"[LLMQueryNode] prompt={prompt_id} | model={model_id} | run={run_idx}"
    )

    result   = await async_query_llm(model_id, prompt_text, system=system, role="query")
    raw_text = result.get("raw_response", "")

    if not raw_text:
        logger.warning(
            f"[LLMQueryNode] Empty response — "
            f"prompt={prompt_id}, model={model_id}, run={run_idx}"
        )

    return {"raw_responses": [{
        "prompt_id":         prompt_id,
        "prompt_text":       prompt_text,
        "model_id":          model_id,
        "run_idx":           run_idx,
        "response_text":     raw_text,
        "completion_tokens": result.get("completion_tokens", 0),
        "prompt_tokens":     result.get("prompt_tokens", 0),
        "total_tokens":      result.get("total_tokens", 0),
    }]}
