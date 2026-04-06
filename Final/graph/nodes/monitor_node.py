"""
graph/nodes/monitor_node.py
═══════════════════════════
Zero-LLM quality monitor node.

Runs after agent1_metrics_node, before brand-scraping fan-out.
Computes a quality_score (0.0 – 10.0) from three heuristics:

  1. Entity count     ≥ MIN_ENTITIES_THRESHOLD  (default 10, env: MIN_ENTITIES)
  2. Diversity ratio  ≥ 0.4  (unique entities / total mentions)
  3. Stability std    ≤ 0.3  (stdev of stability_score column)

No LLM calls — pure pandas + statistics computation.
Sets quality_score and quality_warnings in GEOState.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Final/ is root

import os
import statistics
from loguru import logger

from graph.state import GEOState

MIN_ENTITIES_THRESHOLD = int(os.environ.get("MIN_ENTITIES", "10"))


def monitor_node(state: GEOState) -> dict:
    """
    Quality gate: compute quality_score (0–10) and emit quality_warnings.

    Deductions:
      - Entity count below threshold : up to  4 pts
      - Low diversity ratio          : up to  3 pts
      - High stability variance      : up to  2 pts

    Returns {"quality_score": float, "quality_warnings": list}.
    """
    global_df = state.get("global_df")

    if global_df is None or len(global_df) == 0:
        logger.warning("[MonitorNode] global_df is empty — quality_score = 0.0")
        return {
            "quality_score":    0.0,
            "quality_warnings": ["Empty global_df — no entities extracted"],
        }

    score    = 10.0
    warnings = []
    n_entities = len(global_df)

    # ── Check 1: Entity count ──────────────────────────────────────────────────
    if n_entities < MIN_ENTITIES_THRESHOLD:
        deduction = min(4.0, (MIN_ENTITIES_THRESHOLD - n_entities) * 0.4)
        score    -= deduction
        warnings.append(
            f"Low entity count: {n_entities} (threshold: {MIN_ENTITIES_THRESHOLD})"
        )
        logger.debug(
            f"[MonitorNode] Check 1 FAIL — {n_entities} entities < "
            f"{MIN_ENTITIES_THRESHOLD} → -{deduction:.1f} pts"
        )

    # ── Check 2: Diversity ratio (unique entities ÷ total mentions) ────────────
    if "mention_count" in global_df.columns:
        total_mentions = global_df["mention_count"].sum()
        if total_mentions > 0:
            diversity = n_entities / total_mentions
            if diversity < 0.4:
                deduction = min(3.0, (0.4 - diversity) * 10.0)
                score    -= deduction
                warnings.append(
                    f"Low diversity ratio: {diversity:.2f} (threshold: 0.40)"
                )
                logger.debug(
                    f"[MonitorNode] Check 2 FAIL — diversity {diversity:.2f} < 0.40 "
                    f"→ -{deduction:.1f} pts"
                )

    # ── Check 3: Stability variance ────────────────────────────────────────────
    if "stability_score" in global_df.columns and n_entities >= 2:
        scores_list = global_df["stability_score"].dropna().tolist()
        if len(scores_list) >= 2:
            std = statistics.stdev(scores_list)
            if std > 0.3:
                deduction = min(2.0, (std - 0.3) * 5.0)
                score    -= deduction
                warnings.append(
                    f"High stability variance: std={std:.2f} (threshold: 0.30)"
                )
                logger.debug(
                    f"[MonitorNode] Check 3 FAIL — stability std {std:.2f} > 0.30 "
                    f"→ -{deduction:.1f} pts"
                )

    quality_score = round(max(0.0, score), 2)

    if warnings:
        logger.warning(
            f"[MonitorNode] quality_score={quality_score} | {len(warnings)} warning(s)"
        )
        for w in warnings:
            logger.warning(f"  [!]  {w}")
    else:
        logger.success(
            f"[MonitorNode] quality_score={quality_score} — all checks passed"
        )

    return {"quality_score": quality_score, "quality_warnings": warnings}
