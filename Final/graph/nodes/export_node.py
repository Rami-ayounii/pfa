"""
graph/nodes/export_node.py
══════════════════════════
LangGraph final node: export all outputs to disk.

Writes:
  - agent2_output/social_profiles.csv
  - agent2_output/hallucination_flags.csv
  - agent2_output/audit_log.json
  - pipeline_summary.json

Input state:  profiles, global_df, features_df, domain, location,
              output_dir, query, brands, errors, warnings
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Final/ is root

import os
import json
import pandas as pd
from datetime import datetime
from loguru import logger

from graph.state import GEOState
from core.llm_client import TOKEN_USAGE
from core.status_tracker import emit as _emit


def _serialize(obj):
    """
    Recursively serialize objects to JSON-safe types.
    Handles dataclass instances, plain dicts (new ReAct output),
    lists/tuples, pandas Series, and numpy scalars.
    """
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _serialize(v) for k, v in vars(obj).items()}
    elif isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize(i) for i in obj]
    elif hasattr(obj, 'to_dict'):  # pandas Series
        return obj.to_dict()
    elif hasattr(obj, 'item'):     # numpy scalar
        return obj.item()
    else:
        return obj


def export_node(state: GEOState) -> dict:
    """
    Final pipeline node — export all collected data to disk and print summary.
    """
    profiles         = state.get("profiles", [])
    global_df        = state.get("global_df")
    features_df      = state.get("features_df")
    output_dir       = state.get("output_dir", "geo_output")
    domain           = state.get("domain", "")
    location         = state.get("location", "")
    query            = state.get("query", "")
    brands           = state.get("brands", [])
    errors           = state.get("errors", [])
    warnings         = state.get("warnings", [])
    quality_score    = state.get("quality_score", 0.0)
    quality_warnings = state.get("quality_warnings", [])
    run_id           = state.get("run_id", "")

    _emit("export", "active", output_dir=output_dir)
    try:
        agent2_dir = os.path.join(output_dir, "agent2_output")
        os.makedirs(agent2_dir, exist_ok=True)

        # ── Export social profiles ─────────────────────────────────────────────
        if profiles:
            _export_profiles_manual(profiles, agent2_dir)
            logger.success(f"[ExportNode] {len(profiles)} social profiles exported → {agent2_dir}")

        # ── Merge Agent 1 + Agent 2 → Final Output ────────────────────────────
        final_output_path = os.path.join(output_dir, "final_output.csv")
        _export_final_output(global_df, profiles, final_output_path)

        # ── Compute avg authority from profiles (dict or dataclass) ───────────
        avg_authority = 0.0
        if profiles:
            scores = []
            for p in profiles:
                if isinstance(p, dict):
                    score = p.get("social_authority_score") or p.get("authority_score") or 0
                else:
                    score = getattr(p, "social_authority_score", None) or 0
                scores.append(score)
            avg_authority = round(sum(scores) / len(scores), 1) if scores else 0.0

        # ── Derive brand names from profiles ──────────────────────────────────
        profiled_brands = []
        for p in profiles:
            if isinstance(p, dict):
                profiled_brands.append(p.get("brand", str(p)))
            else:
                profiled_brands.append(getattr(p, "brand", str(p)))

        # ── Export pipeline summary ────────────────────────────────────────────
        summary = {
            "run_id":            run_id,
            "query":             query,
            "domain":            domain,
            "location":          location,
            "brands_targeted":   brands,
            "brands_profiled":   profiled_brands,
            "entities_found":    len(global_df) if global_df is not None else 0,
            "token_usage":       TOKEN_USAGE.copy(),
            "avg_authority":     avg_authority,
            "quality_score":     quality_score,
            "quality_warnings":  quality_warnings,
            "errors":            errors,
            "warnings":          warnings,
        }
        summary_path = os.path.join(output_dir, "pipeline_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(_serialize(summary), f, indent=2, ensure_ascii=False)

        # ── Print final banner ─────────────────────────────────────────────────
        logger.success("=" * 60)
        logger.success(" PIPELINE COMPLETE")
        logger.success("=" * 60)
        logger.info(f"  Query           : {query}")
        logger.info(f"  Domain          : {domain}")
        logger.info(f"  Location        : {location}")
        logger.info(f"  Output dir      : {output_dir}/")
        logger.info(f"  Total tokens    : {TOKEN_USAGE['total_tokens']}")
        if global_df is not None:
            logger.info(f"  Entities found  : {len(global_df)}")
        if profiles:
            logger.info(f"  Brands profiled : {len(profiles)}")
            logger.info(f"  Avg authority   : {avg_authority}/100")
        logger.info(f"  Quality score   : {quality_score}/10.0")
        if quality_warnings:
            for qw in quality_warnings:
                logger.warning(f"  [!]  {qw}")
        if errors:
            logger.warning(f"  Errors          : {len(errors)}")
        logger.info(f"  Summary saved   : {summary_path}")
        logger.info(f"  Final output    : {final_output_path}")
        logger.success("=" * 60)

        _emit("export", "done",
              {"entities": summary["entities_found"],
               "brands": len(profiled_brands),
               "tokens": TOKEN_USAGE.get("total_tokens", 0)},
              output_dir=output_dir)
        return {}
    except Exception as exc:
        logger.error(f"[ExportNode] {exc}")
        _emit("export", "error", {"error": str(exc)}, output_dir=output_dir)
        raise


def _safe_to_csv(df: pd.DataFrame, path: str) -> None:
    """Write DataFrame to CSV; fall back to a timestamped file if the target is locked."""
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(path)
        fallback = f"{base}_{ts}{ext}"
        logger.warning(
            f"[ExportNode] '{path}' is locked (open in another program?). "
            f"Writing to '{fallback}' instead."
        )
        df.to_csv(fallback, index=False, encoding="utf-8-sig")


def _export_final_output(global_df, profiles, path: str) -> None:
    """
    Merge Agent 1 (global_df) and Agent 2 (profiles) into a single Final Output CSV,
    then deduplicate:
      1. Drop exact duplicate rows.
      2. Collapse rows sharing the same canonical_entity (aggregate GEO scores).
      3. Fuzzy-merge near-duplicate entity names that slipped through Agent 1 cleaning.

    Join key: global_df['canonical_entity']  ←→  profile['brand']  (case-insensitive)
    Rows from Agent 1 with no matching profile are kept (left join).
    """
    # ── Build Agent 1 DataFrame ───────────────────────────────────────────────
    if global_df is not None and not global_df.empty:
        df1 = global_df.copy()
    else:
        df1 = pd.DataFrame()

    # ── Build Agent 2 DataFrame ───────────────────────────────────────────────
    rows2 = []
    for p in (profiles or []):
        if hasattr(p, '__dataclass_fields__'):
            rows2.append({k: _serialize(v) for k, v in vars(p).items()})
        elif isinstance(p, dict):
            rows2.append({k: _serialize(v) for k, v in p.items()})
        elif hasattr(p, "__dict__"):
            rows2.append(p.__dict__)
    df2 = pd.DataFrame(rows2) if rows2 else pd.DataFrame()

    # ── Merge ─────────────────────────────────────────────────────────────────
    if df1.empty and df2.empty:
        logger.warning("[ExportNode] Both Agent 1 and Agent 2 outputs are empty — skipping Final Output")
        return

    if df1.empty:
        merged = df2.rename(columns={"brand": "canonical_entity"}) if "brand" in df2.columns else df2
    elif df2.empty:
        merged = df1
    else:
        df1["_join_key"] = df1["canonical_entity"].str.strip().str.lower()
        df2["_join_key"] = df2["brand"].str.strip().str.lower()
        df2 = df2.drop(columns=["brand"], errors="ignore")
        merged = df1.merge(df2, on="_join_key", how="left")
        merged = merged.drop(columns=["_join_key"], errors="ignore")

    before = len(merged)
    merged = _dedup_final(merged)
    after  = len(merged)
    if before != after:
        logger.info(f"[ExportNode] Dedup: {before} → {after} rows ({before - after} removed/merged)")

    _safe_to_csv(merged, path)
    logger.success(f"[ExportNode] Final Output ({len(merged)} rows, {len(merged.columns)} cols) → {path}")


# GEO numeric columns: how to aggregate duplicates
_GEO_AGG = {
    "mention_count":            "sum",
    "mention_rate":             "max",
    "average_ranking_position": "min",   # lower rank = better
    "rank_variance":            "mean",
    "top1_rate":                "max",
    "frequency_across_runs":    "max",
    "stability_score":          "max",
    "model_presence_count":     "max",
    "cross_model_rate":         "max",
    "avg_description_length":   "mean",
}

def _dedup_final(df: pd.DataFrame) -> pd.DataFrame:
    """
    Three-pass deduplication for the final merged DataFrame.
    Pass 1 — Drop exact duplicate rows.
    Pass 2 — Collapse rows with the same canonical_entity (case-insensitive).
    Pass 3 — Fuzzy-merge near-duplicate entity names (rapidfuzz, threshold 88).
    """
    if df.empty or "canonical_entity" not in df.columns:
        return df

    # Pass 1: exact row duplicates
    # Some cells may contain lists/dicts (from JSON parsing) — stringify them
    # before drop_duplicates() which requires all values to be hashable.
    for _col in df.columns:
        if df[_col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[_col] = df[_col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
    df = df.drop_duplicates()

    # Normalise the entity key in-place for passes 2 & 3
    df = df.copy()
    df["canonical_entity"] = df["canonical_entity"].astype(str).str.strip()

    # Pass 2: same name (case-insensitive) → aggregate
    df["_norm"] = df["canonical_entity"].str.lower()
    df = _aggregate_by_key(df, "_norm")
    df = df.drop(columns=["_norm"], errors="ignore")

    # Pass 3: fuzzy near-duplicates
    try:
        from rapidfuzz import fuzz
        df = _fuzzy_merge(df, fuzz, threshold=88)
    except ImportError:
        logger.warning("[ExportNode] rapidfuzz not installed — skipping fuzzy dedup (pip install rapidfuzz)")

    return df.reset_index(drop=True)


def _aggregate_by_key(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Collapse rows that share the same key_col value."""
    if df[key_col].nunique() == len(df):
        return df  # nothing to collapse

    groups = []
    for _, group in df.groupby(key_col, sort=False):
        if len(group) == 1:
            groups.append(group.iloc[0].to_dict())
            continue
        # Pick the anchor row (highest stability_score, or first if not present)
        if "stability_score" in group.columns:
            anchor = group.loc[group["stability_score"].idxmax()].to_dict()
        else:
            anchor = group.iloc[0].to_dict()
        # Aggregate numeric GEO columns
        for col, agg in _GEO_AGG.items():
            if col in group.columns:
                vals = pd.Series(pd.to_numeric(group[col], errors="coerce")).dropna()
                if not vals.empty:
                    anchor[col] = getattr(vals, agg)() if agg in ("sum","max","min","mean") else vals.iloc[0]
        # For Agent 2 columns: take first non-null value across the group
        _A2_PREFIXES = ("gm_", "ig_", "fb_", "ta_", "has_wikipedia", "scrape_")
        for col in group.columns:
            if col.startswith(_A2_PREFIXES) and (anchor.get(col) is None or anchor.get(col) != anchor.get(col)):
                non_null = group[col].dropna()
                if not non_null.empty:
                    anchor[col] = non_null.iloc[0]
        groups.append(anchor)

    return pd.DataFrame(groups)


def _fuzzy_merge(df: pd.DataFrame, fuzz, threshold: int = 88) -> pd.DataFrame:
    """Cluster canonical_entity names by fuzzy similarity and collapse each cluster."""
    entities = df["canonical_entity"].tolist()
    n = len(entities)
    if n < 2:
        return df

    # Union-find
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(n):
        for j in range(i + 1, n):
            score = fuzz.token_sort_ratio(entities[i], entities[j])
            if score >= threshold:
                union(i, j)

    # Build cluster map: cluster_root → list of df-indices
    clusters: dict = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    # Check if any merges happened
    if all(len(v) == 1 for v in clusters.values()):
        return df

    merged_rows = []
    for root, idxs in clusters.items():
        group = df.iloc[idxs]
        merged_rows.append(_aggregate_by_key(
            group.assign(_norm=group["canonical_entity"].str.lower()), "_norm"
        ).drop(columns=["_norm"], errors="ignore").iloc[0].to_dict())

    n_merged = sum(len(v) - 1 for v in clusters.values() if len(v) > 1)
    logger.info(f"[ExportNode] Fuzzy dedup merged {n_merged} near-duplicate entities")
    return pd.DataFrame(merged_rows)


def _export_profiles_manual(profiles, output_dir: str) -> None:
    """Export profiles to CSV — handles both dataclass and plain dict profiles."""
    rows = []
    for p in profiles:
        if hasattr(p, '__dataclass_fields__'):
            rows.append({k: _serialize(v) for k, v in vars(p).items()})
        elif isinstance(p, dict):
            rows.append({k: _serialize(v) for k, v in p.items()})
        elif hasattr(p, "__dict__"):
            rows.append(p.__dict__)
    if rows:
        _safe_to_csv(pd.DataFrame(rows),
                     os.path.join(output_dir, "social_profiles.csv"))
