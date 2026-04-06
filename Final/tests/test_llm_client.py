"""
Final/tests/test_llm_client.py
────────────────────────────────
Unit tests for core/llm_client.py (Final/ merged version).
Adapted from Project 2's tests/test_llm_client.py — same import paths.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from unittest.mock import MagicMock, patch

import pytest

from core.llm_client import (
    parse_json, _track, TOKEN_USAGE, cached_llm_call,
    _parse_wait, query_llm, query_provider,
)


# ── parse_json ────────────────────────────────────────────────────────────────

def test_parse_json_plain_array():
    assert parse_json('["A", "B", "C"]') == ["A", "B", "C"]


def test_parse_json_plain_object():
    result = parse_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_parse_json_markdown_fences():
    raw = '```json\n["X", "Y"]\n```'
    assert parse_json(raw) == ["X", "Y"]


def test_parse_json_trailing_comma():
    raw = '["a", "b",]'
    result = parse_json(raw)
    assert result == ["a", "b"]


def test_parse_json_smart_quotes():
    raw = '[\u201cDar El Jeld\u201d, \u201cLe Corsaire\u201d]'
    result = parse_json(raw)
    assert result == ["Dar El Jeld", "Le Corsaire"]


def test_parse_json_qwen3_thinking():
    raw = '<think>some internal reasoning</think>\n["Brand A"]'
    result = parse_json(raw)
    assert result == ["Brand A"]


def test_parse_json_empty_returns_list():
    assert parse_json("") == []
    assert parse_json("   ") == []


def test_parse_json_nested_extraction():
    """Even if wrapped in text, should extract the JSON."""
    raw = 'Here are the brands: ["Dar El Jeld", "Le Corsaire"] end'
    result = parse_json(raw)
    assert "Dar El Jeld" in result


# ── _track / TOKEN_USAGE ──────────────────────────────────────────────────────

def test_token_tracking():
    import core.llm_client as m
    before = m.TOKEN_USAGE["total_tokens"]
    m._track({"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})
    assert m.TOKEN_USAGE["total_tokens"] == before + 30
    assert m.TOKEN_USAGE["prompt_tokens"] >= 10
    assert m.TOKEN_USAGE["completion_tokens"] >= 20


# ── _parse_wait ───────────────────────────────────────────────────────────────

def test_parse_wait_seconds():
    assert _parse_wait("try again in 30s") == pytest.approx(30.0)


def test_parse_wait_minutes_seconds():
    wait = _parse_wait("try again in 1m30.5s")
    assert wait == pytest.approx(90.5)


def test_parse_wait_no_match():
    assert _parse_wait("some other error") == 0.0


# ── cached_llm_call ───────────────────────────────────────────────────────────

def test_cached_llm_call_caches(tmp_path, monkeypatch):
    import core.llm_client as m
    monkeypatch.setattr(m, "LLM_CACHE_FILE", str(tmp_path / "cache.json"))

    call_count = {"n": 0}
    def mock_fn(prompt):
        call_count["n"] += 1
        return "response_text"

    result1 = cached_llm_call("my prompt", mock_fn)
    result2 = cached_llm_call("my prompt", mock_fn)

    assert result1 == "response_text"
    assert result2 == "response_text"
    assert call_count["n"] == 1  # second call should use cache


def test_cached_llm_call_different_prompts(tmp_path, monkeypatch):
    import core.llm_client as m
    monkeypatch.setattr(m, "LLM_CACHE_FILE", str(tmp_path / "cache2.json"))

    call_count = {"n": 0}
    def mock_fn(prompt):
        call_count["n"] += 1
        return f"resp_{call_count['n']}"

    cached_llm_call("prompt A", mock_fn)
    cached_llm_call("prompt B", mock_fn)
    assert call_count["n"] == 2


# ── query_llm ─────────────────────────────────────────────────────────────────

def test_query_llm_returns_response(mock_groq_client):
    result = query_llm("llama-3.1-8b-instant", "test prompt",
                       client=mock_groq_client)
    assert "raw_response" in result
    assert result["raw_response"] == '["Brand A", "Brand B"]'
    assert result["total_tokens"] == 50


def test_query_llm_fatal_error_no_retry(mock_groq_client):
    mock_groq_client.chat.completions.create.side_effect = Exception("invalid api key")
    result = query_llm("model", "prompt", retries=3, client=mock_groq_client)
    # Should stop immediately on fatal error, not retry 3 times
    assert mock_groq_client.chat.completions.create.call_count == 1
    assert result["raw_response"] == ""


def test_query_llm_retries_on_transient_error(mock_groq_client):
    good_choice = MagicMock()
    good_choice.message.content = "ok"
    good_usage = MagicMock()
    good_usage.completion_tokens = 5
    good_usage.prompt_tokens = 5
    good_usage.total_tokens = 10
    good_resp = MagicMock()
    good_resp.choices = [good_choice]
    good_resp.usage = good_usage

    mock_groq_client.chat.completions.create.side_effect = [
        Exception("connection error"),
        good_resp,
    ]
    with patch("time.sleep"):  # don't actually sleep
        result = query_llm("model", "prompt", retries=3, client=mock_groq_client)
    assert result["raw_response"] == "ok"
    assert mock_groq_client.chat.completions.create.call_count == 2


# ── query_provider ─────────────────────────────────────────────────────────────

def test_query_provider_groq_routes_correctly(mock_groq_client):
    with patch("core.llm_client.query_llm",
               return_value={"raw_response": "ok", "total_tokens": 5}) as mock_q:
        result = query_provider("groq", "llama-3.1-8b-instant", "hello")
        mock_q.assert_called_once()
    assert result["raw_response"] == "ok"


def test_query_provider_openrouter_routes_correctly():
    with patch("core.llm_client.query_openrouter",
               return_value="openrouter_response") as mock_or:
        result = query_provider("openrouter", "ignored_model", "hello")
        mock_or.assert_called_once()
    assert result["raw_response"] == "openrouter_response"


def test_query_provider_invalid_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        query_provider("anthropic", "claude", "hello")


# ── _TokenBucket ──────────────────────────────────────────────────────────────

class TestTokenBucket:
    """Unit tests for the token-bucket rate limiter in core/llm_client.py."""

    def test_returns_immediately_when_tokens_available(self):
        """consume_async() should not sleep when bucket has tokens."""
        import asyncio
        from core.llm_client import _TokenBucket

        bucket = _TokenBucket(rpm=60)   # 60 tokens/min -> starts full

        async def _run():
            t0 = time.monotonic()
            await bucket.consume_async()
            return time.monotonic() - t0

        elapsed = asyncio.run(_run())
        assert elapsed < 0.5, f"Expected near-instant, got {elapsed:.2f}s"

    def test_waits_when_bucket_empty(self):
        """consume_async() should sleep > 0 s when bucket is exhausted."""
        import asyncio
        from core.llm_client import _TokenBucket

        # 1 RPM = 1 token/60s -> draining the 1 token forces a wait
        bucket = _TokenBucket(rpm=1)
        bucket._tokens = 0.0    # manually exhaust

        async def _run():
            t0 = time.monotonic()
            try:
                await asyncio.wait_for(bucket.consume_async(), timeout=5.0)
            except asyncio.TimeoutError:
                pass
            return time.monotonic() - t0

        elapsed = asyncio.run(_run())
        assert elapsed > 0.01, f"Expected some wait, got {elapsed:.3f}s"


# ── env-var model constants ───────────────────────────────────────────────────

def test_model_constants_env_override(monkeypatch):
    monkeypatch.setenv("MODEL_FAST", "custom-fast-model")
    import importlib
    import core.llm_client as m
    importlib.reload(m)
    assert m.MODEL_FAST == "custom-fast-model"
    # Clean up: reload without the env var
    monkeypatch.delenv("MODEL_FAST", raising=False)
    importlib.reload(m)
