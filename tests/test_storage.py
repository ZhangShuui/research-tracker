"""Unit tests for storage.py — delete, update_quality, delete_below_quality."""

import json
import tempfile
from pathlib import Path

import pytest

from paper_tracker.storage import Storage


def _make_paper(arxiv_id: str, quality_score: int = 3, **overrides) -> dict:
    base = {
        "arxiv_id": arxiv_id,
        "title": f"Paper {arxiv_id}",
        "authors": "Author A, Author B",
        "abstract": "An abstract.",
        "url": f"https://arxiv.org/abs/{arxiv_id}",
        "published": "2025-01-01",
        "summary": "A summary.",
        "key_insight": "Something new.",
        "method": "A method.",
        "contribution": "A contribution.",
        "math_concepts": ["concept1"],
        "venue": "NeurIPS 2025",
        "cited_works": ["Doe et al. 2024"],
        "quality_score": quality_score,
    }
    base.update(overrides)
    return base


@pytest.fixture()
def store(tmp_path):
    s = Storage(str(tmp_path))
    yield s
    s.close()


# ---------------------------------------------------------------
# insert_arxiv basics
# ---------------------------------------------------------------

class TestInsertAndRetrieve:
    def test_insert_then_get(self, store: Storage):
        p = _make_paper("2501.00001")
        store.insert_arxiv(p)
        got = store.get_arxiv("2501.00001")
        assert got is not None
        assert got["title"] == "Paper 2501.00001"
        assert got["quality_score"] == 3

    def test_insert_duplicate_ignored(self, store: Storage):
        p = _make_paper("2501.00001")
        store.insert_arxiv(p)
        store.insert_arxiv(p)  # INSERT OR IGNORE
        papers, total = store.get_all_arxiv()
        assert total == 1

    def test_get_nonexistent_returns_none(self, store: Storage):
        assert store.get_arxiv("9999.99999") is None

    def test_math_concepts_stored_as_json(self, store: Storage):
        p = _make_paper("2501.00001", math_concepts=["KL divergence", "ELBO"])
        store.insert_arxiv(p)
        got = store.get_arxiv("2501.00001")
        assert got["math_concepts"] == ["KL divergence", "ELBO"]


# ---------------------------------------------------------------
# delete_arxiv
# ---------------------------------------------------------------

class TestDeleteArxiv:
    def test_delete_existing(self, store: Storage):
        store.insert_arxiv(_make_paper("2501.00001"))
        assert store.delete_arxiv("2501.00001") is True
        assert store.get_arxiv("2501.00001") is None

    def test_delete_nonexistent(self, store: Storage):
        assert store.delete_arxiv("9999.99999") is False

    def test_delete_does_not_affect_other_papers(self, store: Storage):
        store.insert_arxiv(_make_paper("2501.00001"))
        store.insert_arxiv(_make_paper("2501.00002"))
        store.delete_arxiv("2501.00001")
        assert store.get_arxiv("2501.00002") is not None
        _, total = store.get_all_arxiv()
        assert total == 1

    def test_delete_idempotent(self, store: Storage):
        store.insert_arxiv(_make_paper("2501.00001"))
        assert store.delete_arxiv("2501.00001") is True
        assert store.delete_arxiv("2501.00001") is False


# ---------------------------------------------------------------
# update_arxiv_quality
# ---------------------------------------------------------------

class TestUpdateArxivQuality:
    def test_update_existing(self, store: Storage):
        store.insert_arxiv(_make_paper("2501.00001", quality_score=2))
        assert store.update_arxiv_quality("2501.00001", 5) is True
        got = store.get_arxiv("2501.00001")
        assert got["quality_score"] == 5

    def test_update_nonexistent(self, store: Storage):
        assert store.update_arxiv_quality("9999.99999", 5) is False

    def test_update_to_zero(self, store: Storage):
        store.insert_arxiv(_make_paper("2501.00001", quality_score=4))
        store.update_arxiv_quality("2501.00001", 0)
        got = store.get_arxiv("2501.00001")
        assert got["quality_score"] == 0

    def test_update_preserves_other_fields(self, store: Storage):
        store.insert_arxiv(_make_paper("2501.00001", quality_score=2, title="Original Title"))
        store.update_arxiv_quality("2501.00001", 5)
        got = store.get_arxiv("2501.00001")
        assert got["title"] == "Original Title"
        assert got["quality_score"] == 5


# ---------------------------------------------------------------
# delete_arxiv_below_quality
# ---------------------------------------------------------------

class TestDeleteArxivBelowQuality:
    def test_basic_threshold(self, store: Storage):
        store.insert_arxiv(_make_paper("2501.00001", quality_score=1))
        store.insert_arxiv(_make_paper("2501.00002", quality_score=2))
        store.insert_arxiv(_make_paper("2501.00003", quality_score=3))
        store.insert_arxiv(_make_paper("2501.00004", quality_score=4))
        store.insert_arxiv(_make_paper("2501.00005", quality_score=5))

        removed = store.delete_arxiv_below_quality(3)
        assert removed == 2

        _, total = store.get_all_arxiv()
        assert total == 3

        # Papers with score 3, 4, 5 remain
        assert store.get_arxiv("2501.00001") is None
        assert store.get_arxiv("2501.00002") is None
        assert store.get_arxiv("2501.00003") is not None
        assert store.get_arxiv("2501.00004") is not None
        assert store.get_arxiv("2501.00005") is not None

    def test_no_papers_below_threshold(self, store: Storage):
        store.insert_arxiv(_make_paper("2501.00001", quality_score=4))
        store.insert_arxiv(_make_paper("2501.00002", quality_score=5))
        removed = store.delete_arxiv_below_quality(3)
        assert removed == 0
        _, total = store.get_all_arxiv()
        assert total == 2

    def test_all_papers_below_threshold(self, store: Storage):
        store.insert_arxiv(_make_paper("2501.00001", quality_score=1))
        store.insert_arxiv(_make_paper("2501.00002", quality_score=2))
        removed = store.delete_arxiv_below_quality(5)
        assert removed == 2
        _, total = store.get_all_arxiv()
        assert total == 0

    def test_zero_score_not_deleted(self, store: Storage):
        """Papers with quality_score=0 (unscored) should NOT be deleted."""
        store.insert_arxiv(_make_paper("2501.00001", quality_score=0))
        store.insert_arxiv(_make_paper("2501.00002", quality_score=1))
        removed = store.delete_arxiv_below_quality(3)
        assert removed == 1  # only score=1 deleted
        assert store.get_arxiv("2501.00001") is not None  # score=0 kept

    def test_empty_db(self, store: Storage):
        removed = store.delete_arxiv_below_quality(3)
        assert removed == 0

    def test_threshold_1_deletes_nothing(self, store: Storage):
        store.insert_arxiv(_make_paper("2501.00001", quality_score=1))
        store.insert_arxiv(_make_paper("2501.00002", quality_score=2))
        removed = store.delete_arxiv_below_quality(1)
        assert removed == 0


# ---------------------------------------------------------------
# get_all_arxiv search/filter
# ---------------------------------------------------------------

class TestGetAllArxivFilters:
    def test_search_by_title(self, store: Storage):
        store.insert_arxiv(_make_paper("2501.00001", title="Diffusion Models"))
        store.insert_arxiv(_make_paper("2501.00002", title="Language Models"))
        papers, total = store.get_all_arxiv(search="Diffusion")
        assert total == 1
        assert papers[0]["arxiv_id"] == "2501.00001"

    def test_filter_by_venue(self, store: Storage):
        store.insert_arxiv(_make_paper("2501.00001", venue="NeurIPS 2025"))
        store.insert_arxiv(_make_paper("2501.00002", venue="ICML 2025"))
        papers, total = store.get_all_arxiv(venue="NeurIPS")
        assert total == 1

    def test_pagination(self, store: Storage):
        for i in range(5):
            store.insert_arxiv(_make_paper(f"2501.{i:05d}"))
        papers, total = store.get_all_arxiv(limit=2, offset=0)
        assert total == 5
        assert len(papers) == 2
        papers2, _ = store.get_all_arxiv(limit=2, offset=2)
        assert len(papers2) == 2
        # No overlap
        ids1 = {p["arxiv_id"] for p in papers}
        ids2 = {p["arxiv_id"] for p in papers2}
        assert ids1.isdisjoint(ids2)
