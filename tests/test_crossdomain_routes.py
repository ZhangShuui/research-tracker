"""Tests for the global cross-domain corpus endpoints."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from paper_tracker.server import app
from paper_tracker.storage import Storage
from paper_tracker import crossdomain

CID = crossdomain.CROSSDOMAIN_ID


@pytest.fixture(autouse=True)
def _server_globals(tmp_path):
    import paper_tracker.server as srv
    srv._registry = MagicMock()
    srv._data_dir = str(tmp_path)
    srv._base_cfg = {
        "paths": {"data_dir": str(tmp_path)},
        "summarizer": {"claude_path": "claude", "claude_model": "opus"},
    }
    srv._scheduler = MagicMock()
    srv._corpus_embed_jobs.clear()
    srv._corpus_import_jobs.clear()
    srv._corpus_card_jobs.clear()
    yield
    srv._registry = None
    srv._scheduler = None


@pytest.fixture()
def client():
    return TestClient(app, raise_server_exceptions=False)


class _Sync:
    def submit(self, fn, *a, **k):
        fn(*a, **k)
        return MagicMock()


def _meta(arxiv_id: str, title: str = "T") -> dict:
    return {
        "arxiv_id": arxiv_id, "title": title, "authors": "A", "abstract": "abs",
        "url": f"https://arxiv.org/abs/{arxiv_id}", "published": "2024-01-01",
        "paper_id": arxiv_id, "doi": f"10.48550/arxiv.{arxiv_id}",
    }


class TestListAndEmbeddings:
    def test_list_empty(self, client):
        r = client.get("/api/crossdomain/papers")
        assert r.status_code == 200
        assert r.json()["total"] == 0

    def test_embeddings_status(self, client):
        r = client.get("/api/crossdomain/embeddings")
        assert r.status_code == 200
        assert "embedding_count" in r.json() and "paper_count" in r.json()


class TestAdd:
    def test_empty_400(self, client):
        assert client.post("/api/crossdomain/papers", json={"text": "  "}).status_code == 400

    @patch("paper_tracker.server._submit_corpus_embed_job")
    @patch("paper_tracker.crossdomain.screen_papers")
    @patch("paper_tracker.sources.arxiv.fetch_many_by_id")
    def test_screens_keeps_and_skips(self, mock_fetch, mock_screen, mock_embed, client, tmp_path):
        mock_fetch.return_value = {"2401.11111": _meta("2401.11111", "P1"),
                                   "2401.22222": _meta("2401.22222", "P2")}
        mock_screen.return_value = [
            {"arxiv_id": "2401.11111", "keep": True, "domain": "math.OC", "reason": "foundational"},
            {"arxiv_id": "2401.22222", "keep": False, "domain": "cs.CV", "reason": "applied"},
        ]
        r = client.post("/api/crossdomain/papers", json={"text": "2401.11111, 2401.22222"})
        assert r.status_code == 201, r.text
        counts = r.json()["counts"]
        assert counts["added"] == 1 and counts["skipped"] == 1
        mock_embed.assert_called_once()

        store = Storage(str(tmp_path), CID)
        try:
            kept = store.get_arxiv("2401.11111")
            assert kept and kept["venue"] == "math.OC" and kept["key_insight"] == "foundational"
            assert kept["source"] == "crossdomain"
            assert store.get_arxiv("2401.22222") is None  # rejected
        finally:
            store.close()

    @patch("paper_tracker.server._submit_corpus_embed_job")
    @patch("paper_tracker.crossdomain.screen_papers")
    @patch("paper_tracker.sources.arxiv.fetch_many_by_id")
    def test_skip_screen_keeps_all(self, mock_fetch, mock_screen, mock_embed, client):
        mock_fetch.return_value = {"2401.11111": _meta("2401.11111")}
        r = client.post("/api/crossdomain/papers",
                        json={"arxiv_ids": ["2401.11111"], "skip_screen": True})
        assert r.json()["counts"]["added"] == 1
        mock_screen.assert_not_called()

    @patch("paper_tracker.server._submit_corpus_embed_job")
    @patch("paper_tracker.crossdomain.screen_papers")
    @patch("paper_tracker.sources.arxiv.fetch_many_by_id")
    def test_duplicate(self, mock_fetch, mock_screen, mock_embed, client, tmp_path):
        store = Storage(str(tmp_path), CID)
        try:
            store.insert_arxiv({**_meta("2401.11111"), "source": "crossdomain", "summary": "",
                                "key_insight": "", "method": "", "contribution": "",
                                "math_concepts": [], "venue": "", "cited_works": [],
                                "quality_score": 0, "citation_count": 0})
        finally:
            store.close()
        mock_fetch.return_value = {}
        r = client.post("/api/crossdomain/papers", json={"text": "2401.11111"})
        assert r.json()["counts"]["duplicate"] == 1
        mock_screen.assert_not_called()  # nothing pending to screen

    @patch("paper_tracker.server._submit_corpus_embed_job")
    @patch("paper_tracker.crossdomain.screen_papers")
    @patch("paper_tracker.server.pdf_source.fetch_pdf_paper")
    def test_pdf_url_ingested(self, mock_pdf, mock_screen, mock_embed, client, tmp_path):
        meta = {
            "arxiv_id": "pdf:abc123", "paper_id": "pdf:abc123", "source": "pdf",
            "title": "Off-arXiv Theorem", "authors": "Euler", "abstract": "deep result",
            "url": "https://maths.ed.ac.uk/foo.pdf", "published": "", "summary": "deep result",
            "doi": "",
        }
        mock_pdf.return_value = meta
        mock_screen.return_value = [{"arxiv_id": "pdf:abc123", "keep": True,
                                     "domain": "math.NT", "reason": "foundational"}]
        r = client.post("/api/crossdomain/papers",
                        json={"text": "https://maths.ed.ac.uk/foo.pdf"})
        assert r.status_code == 201, r.text
        assert r.json()["counts"]["added"] == 1
        mock_pdf.assert_called_once()
        store = Storage(str(tmp_path), CID)
        try:
            p = store.get_arxiv("pdf:abc123")
            assert p and p["title"] == "Off-arXiv Theorem"
            assert p["venue"] == "math.NT" and p["url"] == "https://maths.ed.ac.uk/foo.pdf"
        finally:
            store.close()

    @patch("paper_tracker.server.pdf_source.fetch_pdf_paper", return_value=None)
    def test_pdf_url_unparseable_not_found(self, mock_pdf, client):
        r = client.post("/api/crossdomain/papers", json={"text": "https://x/not-really.pdf"})
        assert r.status_code == 201
        assert r.json()["counts"]["not_found"] == 1


class TestImportSearch:
    @patch("paper_tracker.crossdomain.screen_papers")
    @patch("paper_tracker.server.gather_cross_domain_papers")
    def test_category_path_uses_3pool_gatherer(self, mock_gather, mock_screen, client):
        mock_gather.return_value = [{
            "arxiv_id": "2401.11111", "title": "T", "abstract": "a", "authors": "",
            "url": "", "published": "2024-01-01", "paper_id": "2401.11111", "doi": "",
        }]
        mock_screen.return_value = [{"arxiv_id": "2401.11111", "keep": True,
                                     "domain": "math.OC", "reason": "r"}]
        r = client.post("/api/crossdomain/import-search",
                        json={"categories": ["math.OC"], "max": 10})
        assert r.status_code == 200
        c = r.json()["candidates"][0]
        assert c["keep"] is True and c["domain"] == "math.OC"
        mock_gather.assert_called_once()  # 3-pool gatherer, not plain search_broad

    @patch("paper_tracker.crossdomain.screen_papers")
    @patch("paper_tracker.sources.arxiv.search_by_query")
    @patch("paper_tracker.server.gather_cross_domain_papers")
    def test_keyword_path_uses_search_by_query(self, mock_gather, mock_kw, mock_screen, client):
        mock_kw.return_value = [{
            "arxiv_id": "2401.22222", "title": "T", "abstract": "a", "authors": "",
            "url": "", "published": "2024-01-01", "paper_id": "2401.22222", "doi": "",
        }]
        mock_screen.return_value = [{"arxiv_id": "2401.22222", "keep": True, "domain": "", "reason": ""}]
        r = client.post("/api/crossdomain/import-search", json={"keyword": "optimal transport"})
        assert r.status_code == 200
        mock_kw.assert_called_once()
        mock_gather.assert_not_called()


class TestDelete:
    def test_delete_and_404(self, client, tmp_path):
        store = Storage(str(tmp_path), CID)
        try:
            store.insert_arxiv({**_meta("2401.99999"), "source": "crossdomain", "summary": "",
                                "key_insight": "", "method": "", "contribution": "",
                                "math_concepts": [], "venue": "", "cited_works": [],
                                "quality_score": 0, "citation_count": 0})
        finally:
            store.close()
        assert client.delete("/api/crossdomain/papers/2401.99999").status_code == 204
        assert client.delete("/api/crossdomain/papers/nope").status_code == 404


class TestImportReport:
    def test_report_not_found(self, client):
        import paper_tracker.server as srv
        srv._registry.get_discovery_report.return_value = None
        r = client.post("/api/crossdomain/import-report", json={"report_id": "x"})
        assert r.status_code == 404

    def test_no_papers_400(self, client):
        import paper_tracker.server as srv
        srv._registry.get_discovery_report.return_value = {"papers_json": []}
        r = client.post("/api/crossdomain/import-report", json={"report_id": "r1"})
        assert r.status_code == 400

    @patch("paper_tracker.rag.ensure_embeddings", return_value=0)
    @patch("paper_tracker.crossdomain.screen_papers")
    @patch("paper_tracker.sources.arxiv.fetch_many_by_id")
    def test_imports_kept_papers(self, mock_fetch, mock_screen, mock_emb, client, tmp_path):
        import paper_tracker.server as srv
        srv._registry.get_discovery_report.return_value = {"papers_json": [
            {"arxiv_id": "2401.11111", "title": "a"},
            {"arxiv_id": "2401.22222", "title": "b"},
        ]}
        mock_fetch.return_value = {"2401.11111": _meta("2401.11111"), "2401.22222": _meta("2401.22222")}
        mock_screen.return_value = [
            {"arxiv_id": "2401.11111", "keep": True, "domain": "math.OC", "reason": "r"},
            {"arxiv_id": "2401.22222", "keep": False, "domain": "", "reason": "applied"},
        ]
        with patch.object(srv, "_brainstorm_executor", _Sync()):
            r = client.post("/api/crossdomain/import-report", json={"report_id": "r1"})
        assert r.status_code == 202, r.text

        job = client.get("/api/crossdomain/import-report").json()["job"]
        assert job["status"] == "completed" and job["added"] == 1 and job["skipped"] == 1

        store = Storage(str(tmp_path), CID)
        try:
            assert store.get_arxiv("2401.11111") is not None
            assert store.get_arxiv("2401.22222") is None
        finally:
            store.close()


class TestConceptCards:
    def test_none_needed_completes(self, client):
        r = client.post("/api/crossdomain/concept-cards", json={})
        assert r.status_code == 202
        assert r.json()["total"] == 0

    @patch("paper_tracker.rag.ensure_embeddings", return_value=0)
    @patch("paper_tracker.crossdomain.generate_concept_card")
    def test_generates_for_papers_without_cards(self, mock_gen, mock_emb, client, tmp_path):
        import paper_tracker.server as srv
        store = Storage(str(tmp_path), CID)
        try:
            store.insert_arxiv({**_meta("2401.11111"), "source": "crossdomain",
                                "summary": "abstract here", "key_insight": "", "method": "",
                                "contribution": "", "math_concepts": [], "venue": "",
                                "cited_works": [], "quality_score": 0, "citation_count": 0})
        finally:
            store.close()
        mock_gen.return_value = {"summary": "Core math summary.", "math_concepts": ["Lemma 1"]}

        with patch.object(srv, "_brainstorm_executor", _Sync()):
            r = client.post("/api/crossdomain/concept-cards", json={})
        assert r.status_code == 202, r.text

        job = client.get("/api/crossdomain/concept-cards").json()["job"]
        assert job["status"] == "completed" and job["generated"] == 1

        store = Storage(str(tmp_path), CID)
        try:
            p = store.get_arxiv("2401.11111")
            assert p["summary"] == "Core math summary."
            assert p["math_concepts"] == ["Lemma 1"]
        finally:
            store.close()
        mock_emb.assert_called_once()  # re-embedded after cards

    @patch("paper_tracker.rag.ensure_embeddings", return_value=0)
    @patch("paper_tracker.crossdomain.generate_concept_card",
           return_value={"summary": "section card", "math_concepts": ["c"]})
    @patch("paper_tracker.crossdomain.split_into_sections",
           return_value=[{"title": "1 Intro", "text": "..."}, {"title": "2 Results", "text": "..."}])
    @patch("paper_tracker.sources.pdf.download_and_extract_text", return_value="L" * 20000)
    def test_long_pdf_splits_into_children(self, mock_dl, mock_split, mock_gen, mock_emb,
                                           client, tmp_path):
        import paper_tracker.server as srv
        store = Storage(str(tmp_path), CID)
        try:
            store.insert_arxiv({**_meta("pdf:longbook"), "source": "crossdomain",
                                "url": "https://maths.x/book.pdf", "summary": "abs",
                                "key_insight": "", "method": "", "contribution": "",
                                "math_concepts": [], "venue": "math.AG", "cited_works": [],
                                "quality_score": 0, "citation_count": 0})
        finally:
            store.close()

        with patch.object(srv, "_brainstorm_executor", _Sync()):
            r = client.post("/api/crossdomain/concept-cards", json={})
        assert r.status_code == 202, r.text
        job = client.get("/api/crossdomain/concept-cards").json()["job"]
        assert job["generated"] == 2

        store = Storage(str(tmp_path), CID)
        try:
            assert store.get_arxiv("pdf:longbook") is None          # parent replaced
            c1 = store.get_arxiv("pdf:longbook#01")
            c2 = store.get_arxiv("pdf:longbook#02")
            assert c1 and "— 1 Intro" in c1["title"] and c1["summary"] == "section card"
            assert c2 and "— 2 Results" in c2["title"] and c2["math_concepts"] == ["c"]
        finally:
            store.close()
