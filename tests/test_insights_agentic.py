"""Tests for the agentic multi-stage insights pipeline (insights.generate_agentic).

The LLM is mocked with a prompt-dispatching stub that returns stage-appropriate
output, so we exercise the real orchestration, fallbacks, and grounding logic
without any network/LLM calls.
"""

from __future__ import annotations

import json
import re
from unittest.mock import patch

from paper_tracker import insights


def _paper(aid, title, **kw):
    return {
        "arxiv_id": aid, "paper_id": aid, "title": title,
        "key_insight": kw.get("key_insight", f"insight {aid}"),
        "method": kw.get("method", ""), "contribution": kw.get("contribution", ""),
        "summary": kw.get("summary", f"summary {aid}"),
        "math_concepts": kw.get("math_concepts", []),
        "venue": kw.get("venue", ""), "url": kw.get("url", f"https://arxiv.org/abs/{aid}"),
    }


def _labels_in(prompt: str) -> list[str]:
    return re.findall(r"\[(P\d+)\]", prompt)


def _dispatch(prompt, cfg, *args, **kwargs):
    """Return stage-appropriate output keyed off a unique phrase in each prompt."""
    if "coherent themes" in prompt:                       # Stage 1: outline
        labels = sorted(set(_labels_in(prompt)), key=lambda x: int(x[1:]))
        half = max(1, len(labels) // 2)
        a, b = labels[:half], (labels[half:] or labels[:1])
        return json.dumps({"themes": [
            {"title": "Theme A", "angle": "x", "papers": a},
            {"title": "Theme B", "angle": "y", "papers": b},
        ]})
    if "ONE theme" in prompt:                             # Stage 2: theme synthesis
        labs = _labels_in(prompt)
        return "Theme brief mentioning " + " ".join(f"[{l}]" for l in labs)
    if "SPAN different themes" in prompt:                 # Stage 2.5: connections
        labs = _labels_in(prompt)
        return f"- [{labs[0]}] connects to [{labs[-1]}] via a shared idea"
    if "final cross-paper INSIGHTS memo" in prompt:       # Stage 3: reduce
        labs = sorted(set(_labels_in(prompt)), key=lambda x: int(x[1:]))
        cites = " ".join(f"[{l}]" for l in labs)
        return (f"## Key Trends\nTrend across {cites}.\n\n## Emerging Methods\nM.\n\n"
                "## Connections & Cross-Paper Themes\nC.\n\n"
                "## Research Gaps & Opportunities\nG.\n\n"
                f"## Recommended Reading\n{labs[0]} is must-read.")
    if "auditing an insights memo" in prompt:             # Stage 4: audit
        return '{"ok": true, "issues": []}'
    if "Revise this insights memo" in prompt:             # Stage 5: revise
        return "REVISED MEMO [P1]"
    return ""


# ---------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------

class TestGenerateAgentic:
    def test_empty_returns_none(self, tmp_path):
        assert insights.generate_agentic([], "T", tmp_path, {}) is None

    @patch("paper_tracker.insights.call_cli", side_effect=_dispatch)
    def test_writes_papers_manifest(self, _cli, tmp_path):
        papers = [_paper("2401.00001", "Alpha"), _paper("2401.00002", "Beta")]
        insights.generate_agentic(papers, "T", tmp_path, {})
        mani = json.loads((tmp_path / "insights_papers.json").read_text())
        assert [m["index"] for m in mani] == ["P1", "P2"]   # provenance for brainstorm
        assert mani[0]["title"] == "Alpha" and mani[0]["id"] == "2401.00001"

    @patch("paper_tracker.insights.call_cli", side_effect=_dispatch)
    def test_happy_path_writes_memo_with_references(self, _cli, tmp_path):
        papers = [_paper("2401.00001", "Alpha"), _paper("2401.00002", "Beta")]
        path = insights.generate_agentic(papers, "T", tmp_path, {})
        assert path is not None and path.exists()
        text = path.read_text()
        assert "## Key Trends" in text
        assert "## References" in text
        assert "[P1]" in text and "[P2]" in text
        assert "Alpha" in text and "Beta" in text      # titles in the reference list

    @patch("paper_tracker.insights.call_cli")
    def test_outline_failure_falls_back_to_batches(self, mock_cli, tmp_path):
        def disp(prompt, cfg, *a, **k):
            if "coherent themes" in prompt:
                return "totally not json"              # outline unavailable
            return _dispatch(prompt, cfg, *a, **k)
        mock_cli.side_effect = disp
        papers = [_paper(f"2401.{i:05d}", f"Paper {i}") for i in range(3)]
        path = insights.generate_agentic(papers, "T", tmp_path, {})
        assert path is not None and path.exists()
        assert "## Key Trends" in path.read_text()     # still produced a memo

    @patch("paper_tracker.insights.call_cli")
    def test_reduce_failure_uses_fallback_memo(self, mock_cli, tmp_path):
        def disp(prompt, cfg, *a, **k):
            if "final cross-paper INSIGHTS memo" in prompt:
                return ""                              # reduce fails
            return _dispatch(prompt, cfg, *a, **k)
        mock_cli.side_effect = disp
        papers = [_paper("2401.00001", "Alpha"), _paper("2401.00002", "Beta")]
        path = insights.generate_agentic(papers, "T", tmp_path, {})
        text = path.read_text()
        assert "## Themes" in text                     # fallback memo shape
        assert "Theme brief" in text                   # briefs surface into it

    @patch("paper_tracker.insights.call_cli")
    def test_theme_failure_produces_stub_and_completes(self, mock_cli, tmp_path):
        def disp(prompt, cfg, *a, **k):
            if "ONE theme" in prompt:
                return ""                              # every theme synth fails -> stub
            return _dispatch(prompt, cfg, *a, **k)
        mock_cli.side_effect = disp
        papers = [_paper("2401.00001", "Alpha", key_insight="alpha insight")]
        path = insights.generate_agentic(papers, "T", tmp_path, {})
        assert path is not None and path.exists()
        assert "## Key Trends" in path.read_text()

    @patch("paper_tracker.insights.call_cli", side_effect=_dispatch)
    def test_single_theme_skips_connection_pass(self, mock_cli, tmp_path):
        # one paper -> outline still makes 2 themes here, but verify connection-pass
        # is conditional on >1 theme by forcing a single-theme outline.
        def disp(prompt, cfg, *a, **k):
            if "coherent themes" in prompt:
                return json.dumps({"themes": [{"title": "Solo", "angle": "", "papers": ["P1"]}]})
            return _dispatch(prompt, cfg, *a, **k)
        mock_cli.side_effect = disp
        path = insights.generate_agentic([_paper("2401.00001", "Alpha")], "T", tmp_path, {})
        prompts = [c.args[0] for c in mock_cli.call_args_list]
        assert not any("SPAN different themes" in p for p in prompts)  # no connection pass
        assert path.exists()


# ---------------------------------------------------------------
# Helpers: outline coverage, grounding, references
# ---------------------------------------------------------------

class TestHelpers:
    @patch("paper_tracker.insights.call_cli")
    def test_outline_sweeps_uncovered_papers(self, mock_cli):
        mock_cli.return_value = '{"themes":[{"title":"A","angle":"","papers":["P1"]}]}'
        labeled = [("P1", {"title": "a"}), ("P2", {"title": "b"}), ("P3", {"title": "c"})]
        id_map = dict(labeled)
        themes = insights._outline(labeled, id_map, "T", {})
        covered = {l for t in themes for l in t["labels"]}
        assert covered == {"P1", "P2", "P3"}                 # nothing dropped
        assert any("Additional" in t["title"] for t in themes)

    def test_append_references_only_cited_and_valid(self):
        id_map = {"P1": {"title": "Alpha", "venue": "NeurIPS", "url": "u1"},
                  "P2": {"title": "Beta", "url": ""}}
        memo = "See [P1] and also [P9] which is bogus."
        out = insights._append_references(memo, id_map)
        refs = out.split("## References")[1]
        assert "- [P1] Alpha (NeurIPS) — u1" in refs
        assert "[P9]" not in refs                            # invalid label not referenced
        assert "Beta" not in refs                            # P2 not cited -> omitted

    def test_append_references_none_cited_returns_memo(self):
        assert insights._append_references("no cites", {"P1": {"title": "A"}}) == "no cites"

    @patch("paper_tracker.insights.call_cli")
    def test_audit_revises_on_invalid_citation(self, mock_cli):
        # audit says "ok", but the memo cites a non-existent label -> hard rule forces revise
        def disp(prompt, cfg, *a, **k):
            if "auditing an insights memo" in prompt:
                return '{"ok": true, "issues": []}'
            if "Revise this insights memo" in prompt:
                return "CLEANED MEMO [P1]"
            return ""
        mock_cli.side_effect = disp
        out = insights._audit_and_revise(
            "Memo citing [P1] and [P5].", {"P1": {"title": "A"}}, "T", "[P1] A", {}, lambda m: None)
        assert out == "CLEANED MEMO [P1]"

    @patch("paper_tracker.insights.call_cli", return_value='{"ok": true, "issues": []}')
    def test_audit_clean_keeps_draft_without_revise(self, mock_cli):
        memo = "Clean memo [P1]."
        out = insights._audit_and_revise(memo, {"P1": {"title": "A"}}, "T", "[P1] A", {}, lambda m: None)
        assert out == memo
        mock_cli.assert_called_once()                        # only the audit call, no revise
