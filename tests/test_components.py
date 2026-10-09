import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import re
import json
from unittest.mock import MagicMock

from check_citations import check
from finalize_citations import finalize
from tools import with_retry, RetryableError, arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch, _redact_key
from agents import build_subagents, build_lead_agent, LEAD_PROMPT, RESEARCHER_PROMPT, CHECKER_PROMPT, LEAD_LIMITS, SUB_LIMITS
from research import slugify, summarize, build_prompt


def test_check_citations_valid():
    sources = [
        {"n": 1, "id": "2401.00001", "url": "https://arxiv.org/abs/2401.00001", "title": "Paper 1", "date": "2024-01-01", "source": "arxiv"},
        {"n": 2, "id": "2401.00002", "url": "https://huggingface.co/papers/2401.00002", "title": "Paper 2", "date": "2024-01-02", "source": "hf-search"},
    ]
    report = """# Survey Title

## Background
World models learn predictive representations [1]. Recent advances improve sample efficiency [2].

## References

[1] Paper 1. https://arxiv.org/abs/2401.00001
[2] Paper 2. https://huggingface.co/papers/2401.00002
"""
    assert check(report, sources) == []


def test_check_citations_grouped():
    sources = [
        {"n": 1, "id": "1", "url": "https://arxiv.org/abs/1", "title": "P1"},
        {"n": 2, "id": "2", "url": "https://arxiv.org/abs/2", "title": "P2"},
        {"n": 3, "id": "3", "url": "https://arxiv.org/abs/3", "title": "P3"},
    ]
    report = """# Survey

Methods compare [1, 2] and [1-3].

## References

[1] P1. https://arxiv.org/abs/1
[2] P2. https://arxiv.org/abs/2
[3] P3. https://arxiv.org/abs/3
"""
    assert check(report, sources) == []


def test_check_citations_errors():
    # Empty sources
    assert check("# Test\n\n## References", []) == ["no sources in sources.json"]

    # Missing References heading
    sources = [{"n": 1, "url": "https://arxiv.org/abs/1"}]
    assert check("# Test\n\nClaim [1]", sources) == ["missing '## References' section"]

    # Source never cited
    report_uncited = "# Test\n\nNo citations.\n\n## References\n\n[1] P1. https://arxiv.org/abs/1"
    probs = check(report_uncited, sources)
    assert any("never cited" in p for p in probs)

    # Cited missing from sources
    report_ghost = "# Test\n\nClaim [99].\n\n## References\n\n[1] P1. https://arxiv.org/abs/1"
    probs2 = check(report_ghost, sources)
    assert any("[99] cited but missing" in p for p in probs2)


def test_with_retry_behavior():
    attempts = [0]
    def fail_twice():
        attempts[0] += 1
        if attempts[0] < 3:
            raise RetryableError("temporary", retry_after=0.01)
        return "success"

    res = with_retry(fail_twice, attempts=5, base=0.01, cap=0.1)
    assert res == "success"
    assert attempts[0] == 3

    # Non-retryable
    def fail_perm():
        raise KeyError("permanent")
    with pytest.raises(KeyError):
        with_retry(fail_perm, attempts=3)


def test_slugify():
    assert slugify("survey about world model") == "survey-about-world-model"
    assert slugify("../../secret/path") == "secret-path"
    assert slugify("") == "topic"
    assert len(slugify("x" * 100)) == 60


def test_summarize():
    class DummyMsg:
        def __init__(self, tcs, meta):
            self.tool_calls = tcs
            self.usage_metadata = meta

    messages = [
        DummyMsg([{"name": "task"}, {"name": "read_file"}], {"input_tokens": 50, "output_tokens": 20}),
        DummyMsg([{"name": "task"}], {"input_tokens": 100, "output_tokens": 40}),
    ]
    summary = summarize(messages, 15.678, "gpt-4o")
    assert summary["subagent_calls"] == 2
    assert summary["tool_calls"] == {"task": 2, "read_file": 1}
    assert summary["tokens"]["input"] == 150
    assert summary["tokens"]["output"] == 60
    assert summary["elapsed_s"] == 15.7


def test_agents_spec():
    subagents = build_subagents()
    assert len(subagents) == 2
    names = {s["name"] for s in subagents}
    assert "researcher" in names
    assert "citation-checker" in names

    # Check middleware
    for sub in subagents:
        assert len(sub["middleware"]) == 2


def test_redact_key():
    assert _redact_key("Key is abc123xyz", "abc123xyz") == "Key is [REDACTED]"
    assert _redact_key("No secret here", "") == "No secret here"
