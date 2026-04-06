"""
Final/tests/test_agent1.py
──────────────────────────
Unit tests for agents/agent1_geo_analyser.py (shared with Final/).
Adapted from Project 2's tests/test_agent1.py — same import paths since
agent1_geo_analyser.py is not duplicated in Final/agents/.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from agents.agent1_geo_analyser import (
    step1_load_prompts,
    step6_compute_metrics,
    _phase_a_prefilter,
    _phase_b_fuzzy,
    MIN_ENTITY_LENGTH,
    FUZZY_MERGE_THRESHOLD,
)


# ── step1_load_prompts ────────────────────────────────────────────────────────

def test_step1_load_prompts_valid(sample_prompt_csv):
    prompts = step1_load_prompts(sample_prompt_csv)
    assert isinstance(prompts, list)
    assert len(prompts) > 0
    assert all("prompt_text" in p for p in prompts)


def test_step1_load_prompts_missing_file():
    with pytest.raises(FileNotFoundError):
        step1_load_prompts("/nonexistent/path/prompt_set.csv")


def test_step1_load_prompts_missing_column(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame([{"prompt_id": "p1", "prompt_text": "hello"}]).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="Missing"):
        step1_load_prompts(str(bad_csv))


# ── _phase_a_prefilter ────────────────────────────────────────────────────────

def test_phase_a_filters_short_entities(sample_entities):
    # Add a short entity that should be filtered
    short_row = sample_entities.iloc[0].copy()
    short_row["entity"] = "ab"  # len < MIN_ENTITY_LENGTH
    df_with_short = pd.concat([sample_entities, pd.DataFrame([short_row])],
                               ignore_index=True)
    df_filtered, log = _phase_a_prefilter(df_with_short)
    assert "ab" not in df_filtered["entity"].values
    assert any(e["entity"] == "ab" for e in log)


def test_phase_a_keeps_valid_entities(sample_entities):
    df_filtered, log = _phase_a_prefilter(sample_entities)
    assert "dar el jeld" in df_filtered["entity"].values
    assert "le corsaire"  in df_filtered["entity"].values


# ── _phase_b_fuzzy ────────────────────────────────────────────────────────────

def test_fuzzy_dedup_merges_variants():
    entities = ["dar el jeld", "dar el-jeld", "le corsaire"]
    counts   = {"dar el jeld": 5, "dar el-jeld": 2, "le corsaire": 3}
    clusters, pair_scores, canonical_map = _phase_b_fuzzy(entities, counts)

    # Both variants should map to the same canonical
    c1 = canonical_map.get("dar el jeld")
    c2 = canonical_map.get("dar el-jeld")
    assert c1 == c2, "Spelling variants should be merged to same canonical"


def test_fuzzy_keeps_distinct_entities():
    entities = ["dar el jeld", "le corsaire", "dar zarrouk"]
    counts   = {e: 2 for e in entities}
    _, _, canonical_map = _phase_b_fuzzy(entities, counts)

    # All three should have distinct canonicals
    canonicals = set(canonical_map.values())
    assert len(canonicals) == 3


# ── step6_compute_metrics — GEO feature calculations ─────────────────────────

@pytest.fixture
def clean_entities_df(sample_entities):
    """Add canonical_entity column to simulate post-cleaning data."""
    df = sample_entities.copy()
    df["canonical_entity"]        = df["entity"]
    df["ranking_position_filled"] = df["ranking_position"].fillna(99).astype(int)
    df["clean_flag"]              = "clean"
    df["quality_flag"]            = "ok"
    df["merge_source"]            = ""
    return df


def test_geo_features_mention_rate(clean_entities_df, sample_raw_responses, sample_prompt_df, tmp_path):
    prompts = sample_prompt_df.to_dict("records")
    df_features, df_global = step6_compute_metrics(
        clean_entities_df, sample_raw_responses, prompts,
        str(tmp_path / "entity_features.csv"),
        str(tmp_path / "entity_features_global.csv"),
    )

    # "dar el jeld" appears in r1, r2, r3 out of 4 responses -> mention_rate = 0.75
    row = df_global[df_global["canonical_entity"] == "dar el jeld"]
    assert len(row) == 1
    rate = float(row.iloc[0]["mention_rate"])
    assert 0.6 <= rate <= 1.0, f"Expected mention_rate ~0.75, got {rate}"


def test_geo_features_top1_rate(clean_entities_df, sample_raw_responses, sample_prompt_df, tmp_path):
    prompts = sample_prompt_df.to_dict("records")
    _, df_global = step6_compute_metrics(
        clean_entities_df, sample_raw_responses, prompts,
        str(tmp_path / "features.csv"),
        str(tmp_path / "global.csv"),
    )
    # "dar el jeld" is rank 1 in r1, r2, r3 — top1 in all 3 of its mentions
    row = df_global[df_global["canonical_entity"] == "dar el jeld"]
    top1 = float(row.iloc[0]["top1_rate"])
    assert top1 > 0.5


def test_geo_features_stability_score(clean_entities_df, sample_raw_responses, sample_prompt_df, tmp_path):
    prompts = sample_prompt_df.to_dict("records")
    _, df_global = step6_compute_metrics(
        clean_entities_df, sample_raw_responses, prompts,
        str(tmp_path / "features.csv"),
        str(tmp_path / "global.csv"),
    )
    # stability_score = mention_rate x 1/(1 + rank_variance) — must be <= mention_rate
    for _, row in df_global.iterrows():
        assert float(row["stability_score"]) <= float(row["mention_rate"]) + 1e-6


def test_geo_features_consistency_label(clean_entities_df, sample_raw_responses, sample_prompt_df, tmp_path):
    prompts = sample_prompt_df.to_dict("records")
    _, df_global = step6_compute_metrics(
        clean_entities_df, sample_raw_responses, prompts,
        str(tmp_path / "features.csv"),
        str(tmp_path / "global.csv"),
    )
    valid_labels = {"stable", "variable", "unstable"}
    for label in df_global["consistency_label"]:
        assert label in valid_labels, f"Unexpected label: {label}"


def test_output_files_created(clean_entities_df, sample_raw_responses, sample_prompt_df, tmp_path):
    prompts = sample_prompt_df.to_dict("records")
    feat_path   = str(tmp_path / "entity_features.csv")
    global_path = str(tmp_path / "entity_features_global.csv")
    step6_compute_metrics(clean_entities_df, sample_raw_responses, prompts,
                          feat_path, global_path)
    assert Path(feat_path).exists()
    assert Path(global_path).exists()
