"""Tests for config.from_topic() — verifies topic fields propagate into pipeline cfg."""

from __future__ import annotations

from paper_tracker.config import from_topic


_BASE = {
    "paths": {"data_dir": "/tmp/x", "logs_dir": "/tmp/y"},
    "summarizer": {"claude_model": "sonnet"},
    "notify": {"toast": {"enabled": False}, "email": {"enabled": False}},
}


class TestFromTopicPrefilterFields:
    def test_description_propagates(self):
        topic = {"name": "T", "description": "Study of X."}
        cfg = from_topic(topic, _BASE)
        assert cfg["search"]["description"] == "Study of X."

    def test_prefilter_enabled_default_true(self):
        cfg = from_topic({"name": "T"}, _BASE)
        assert cfg["search"]["prefilter_enabled"] is True

    def test_prefilter_enabled_false_propagates(self):
        cfg = from_topic({"name": "T", "prefilter_enabled": False}, _BASE)
        assert cfg["search"]["prefilter_enabled"] is False

    def test_prefilter_criteria_default_empty(self):
        cfg = from_topic({"name": "T"}, _BASE)
        assert cfg["search"]["prefilter_criteria"] == ""

    def test_prefilter_criteria_propagates(self):
        cfg = from_topic({"name": "T", "prefilter_criteria": "Exclude X."}, _BASE)
        assert cfg["search"]["prefilter_criteria"] == "Exclude X."
