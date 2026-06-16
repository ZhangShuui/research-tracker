"""Tests for crossdomain.screen_papers and crossdomain.retrieve."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

from paper_tracker import crossdomain
from paper_tracker import discovery


class TestGatherCrossDomain:
    @patch("paper_tracker.discovery.search_random_era")
    @patch("paper_tracker.discovery.search_broad")
    def test_three_pools_deduped_and_tagged(self, mock_broad, mock_era):
        mock_broad.return_value = [{"arxiv_id": "r1"}]
        # search_random_era is called twice: historical, then wildcard
        mock_era.side_effect = [
            [{"arxiv_id": "h1"}],
            [{"arxiv_id": "w1"}, {"arxiv_id": "r1"}],  # r1 duplicates the recent pool
        ]
        out = discovery.gather_cross_domain_papers(["math.OC"], ["math.AG"])
        assert [p["arxiv_id"] for p in out] == ["r1", "h1", "w1"]  # dedup, pool order
        pools = {p["arxiv_id"]: p["pool"] for p in out}
        assert pools == {"r1": "recent", "h1": "historical", "w1": "wildcard"}


# ---------------------------------------------------------------
# screen_papers
# ---------------------------------------------------------------

class TestScreenPapers:
    def test_empty(self):
        assert crossdomain.screen_papers([], {}) == []

    @patch("paper_tracker.crossdomain.call_cli")
    def test_keeps_and_rejects_by_verdict(self, mock_cli):
        mock_cli.return_value = json.dumps([
            {"id": "2401.111", "keep": True, "domain": "math.OC", "reason": "foundational"},
            {"id": "2401.222", "keep": False, "domain": "cs.CV", "reason": "applied benchmark"},
        ])
        papers = [
            {"arxiv_id": "2401.111", "paper_id": "2401.111", "title": "T1", "abstract": "a"},
            {"arxiv_id": "2401.222", "paper_id": "2401.222", "title": "T2", "abstract": "b"},
        ]
        out = crossdomain.screen_papers(papers, {})
        assert out[0] == {"arxiv_id": "2401.111", "keep": True, "domain": "math.OC", "reason": "foundational"}
        assert out[1]["keep"] is False

    @patch("paper_tracker.crossdomain.call_cli", return_value=None)
    def test_screen_unavailable_keeps_all(self, mock_cli):
        papers = [{"arxiv_id": "2401.111", "title": "T1", "abstract": "a"}]
        out = crossdomain.screen_papers(papers, {})
        assert out[0]["keep"] is True
        assert out[0]["reason"] == "screen unavailable"

    @patch("paper_tracker.crossdomain.call_cli", return_value="garbage not json")
    def test_unparseable_keeps_all(self, mock_cli):
        out = crossdomain.screen_papers([{"arxiv_id": "x", "title": "T"}], {})
        assert out[0]["keep"] is True


# ---------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------

def _corpus(embs: dict[str, list[float]]):
    store = MagicMock()
    store.get_all_embeddings.return_value = list(embs.items())
    store.get_arxiv.side_effect = lambda aid: {
        "arxiv_id": aid, "title": f"corpus {aid}", "authors": "A", "abstract": "abs",
        "url": f"https://arxiv.org/abs/{aid}", "published": "2020-01-01",
        "venue": "math.OC", "key_insight": "bridge reason",
    } if aid in embs else None
    return store


class _FakeRng:
    """Deterministic stand-in for random.Random — picks the first n of the tail."""
    def sample(self, population, n):
        return list(population)[:n]


class TestRetrieve:
    def test_empty_selection(self):
        assert crossdomain.retrieve([], _corpus({"c1": [1.0, 0.0]})) == []

    def test_empty_corpus(self):
        with patch("paper_tracker.rag.embed_texts", return_value=[[1.0, 0.0]]):
            assert crossdomain.retrieve([{"arxiv_id": "s1", "title": "s"}], _corpus({})) == []

    @patch("paper_tracker.rag.embed_texts", return_value=[[1.0, 0.0]])
    def test_ranks_and_excludes_selected(self, _embed):
        corpus = _corpus({"c1": [1.0, 0.0], "c2": [0.0, 1.0], "s1": [1.0, 0.0]})
        out = crossdomain.retrieve([{"arxiv_id": "s1", "title": "sel", "abstract": "x"}], corpus)
        ids = [c["arxiv_id"] for c in out]
        assert ids == ["c1", "c2"]          # c1 most similar; selected s1 excluded
        assert out[0]["source"] == "kb"
        assert out[0]["domain"] == "math.OC"
        assert out[0]["rationale"] == "bridge reason"
        assert out[0]["score"] >= out[1]["score"]

    @patch("paper_tracker.rag.embed_texts", return_value=[[1.0, 0.0], [1.0, 0.0]])
    def test_respects_k(self, _embed):
        corpus = _corpus({f"c{i}": [1.0, 0.0] for i in range(10)})
        out = crossdomain.retrieve([{"arxiv_id": "s", "title": "s"}], corpus, k=3)
        assert len(out) == 3

    @patch("paper_tracker.rag.embed_texts", side_effect=Exception("egress down"))
    def test_embed_failure_returns_empty(self, _embed):
        out = crossdomain.retrieve([{"arxiv_id": "s", "title": "s"}], _corpus({"c1": [1.0, 0.0]}))
        assert out == []

    @patch("paper_tracker.rag.embed_texts", return_value=[[1.0, 0.0]])
    def test_injects_random_picks_from_tail(self, _embed):
        # 6 corpus papers, descending similarity to q=[1,0]; reserve 2 slots for random.
        embs = {
            "c1": [1.0, 0.0], "c2": [0.9, 0.1], "c3": [0.5, 0.5],
            "c4": [0.3, 0.7], "c5": [0.1, 0.9], "c6": [0.0, 1.0],
        }
        out = crossdomain.retrieve(
            [{"arxiv_id": "s", "title": "s"}], _corpus(embs),
            k=4, random_k=2, rng=_FakeRng(),
        )
        assert len(out) == 4
        tops = [c["arxiv_id"] for c in out if not c["random"]]
        randoms = [c["arxiv_id"] for c in out if c["random"]]
        assert tops == ["c1", "c2"]                              # 2 most similar kept
        assert len(randoms) == 2
        assert all(a in {"c3", "c4", "c5", "c6"} for a in randoms)   # drawn from the tail

    @patch("paper_tracker.rag.embed_texts", return_value=[[1.0, 0.0]])
    def test_random_k_zero_is_pure_topk(self, _embed):
        corpus = _corpus({f"c{i}": [1.0, 0.0] for i in range(5)})
        out = crossdomain.retrieve([{"arxiv_id": "s", "title": "s"}], corpus, k=3, random_k=0)
        assert len(out) == 3
        assert all(c["random"] is False for c in out)


class TestSplitIntoSections:
    def test_empty(self):
        assert crossdomain.split_into_sections("") == []

    def test_heading_based_C(self):
        text = "1 Introduction\n" + "a" * 500 + "\n2 Main Results\n" + "b" * 500
        pieces = crossdomain.split_into_sections(text)
        assert len(pieces) == 2
        assert pieces[0]["title"].startswith("1 Introduction")
        assert pieces[1]["title"].startswith("2 Main Results")
        assert "a" in pieces[0]["text"] and "b" in pieces[1]["text"]

    def test_chunk_fallback_B(self):
        pieces = crossdomain.split_into_sections("x" * 20000, chunk_chars=9000, overlap=400)
        assert len(pieces) >= 2
        assert all(p["title"].startswith("Part ") for p in pieces)

    def test_short_single_piece(self):
        pieces = crossdomain.split_into_sections("short text, no headings here")
        assert len(pieces) == 1  # caller treats a single piece as one card (no split)

    def test_caps_pieces(self):
        text = "".join(f"{i} Section Title\n" + "z" * 450 + "\n" for i in range(1, 26))
        pieces = crossdomain.split_into_sections(text, max_pieces=20)
        assert 0 < len(pieces) <= 20


class TestGenerateConceptCard:
    @patch("paper_tracker.crossdomain.call_cli",
           return_value='{"summary":"Studies optimal transport.","math_concepts":["Wasserstein distance","Kantorovich duality"]}')
    def test_extracts_math_content(self, _cli):
        card = crossdomain.generate_concept_card({"title": "OT", "abstract": "..."}, {})
        assert card["summary"].startswith("Studies optimal")
        assert card["math_concepts"] == ["Wasserstein distance", "Kantorovich duality"]

    @patch("paper_tracker.crossdomain.call_cli", return_value="not json")
    def test_unparseable_returns_none(self, _cli):
        assert crossdomain.generate_concept_card({"title": "X", "abstract": "y"}, {}) is None

    def test_no_content_returns_none_without_llm(self):
        # empty title + abstract → no LLM call, None
        assert crossdomain.generate_concept_card({"title": "", "abstract": ""}, {}) is None

    @patch("paper_tracker.crossdomain.call_cli",
           return_value='{"summary":"s","math_concepts":"not a list"}')
    def test_non_list_concepts_coerced(self, _cli):
        card = crossdomain.generate_concept_card({"title": "X", "abstract": "y"}, {})
        assert card["summary"] == "s" and card["math_concepts"] == []

    @patch("paper_tracker.crossdomain.call_cli", return_value='{"summary":"from body"}')
    def test_uses_provided_text(self, mock_cli):
        crossdomain.generate_concept_card({"title": "T", "abstract": "ABSTRACT"}, {}, text="FULL BODY TEXT")
        # the prompt body should be the provided text, not the abstract
        assert "FULL BODY TEXT" in mock_cli.call_args.args[0]
