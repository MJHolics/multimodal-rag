"""HybridRetriever의 컴포넌트 장애 축소 운영(2026-09-07 추가) 단위 테스트.

실측(실문서 RFP 코퍼스, run_failure_bench.py)은 별도로 하고, 여기서는 라우팅 로직
자체 — "무엇이 죽으면 어느 경로로 빠지는가"가 정확한지만 빠르게 검증한다.
"""
import types

import numpy as np
import pytest

from graph_rag.corpus import Chunk
from graph_rag.retrieval import HybridRetriever


class _StubEmbedder:
    """질의·문서를 원-핫 유사 벡터로 인코딩하는 가짜 임베더 — 결정적이고 빠르다."""

    VOCAB = ["배터리", "모터", "센서"]

    def encode(self, texts, normalize_embeddings=True):
        vecs = []
        for t in texts:
            v = np.array([1.0 if w in t else 0.0 for w in self.VOCAB])
            n = np.linalg.norm(v)
            vecs.append(v / n if n > 0 else v)
        return np.array(vecs)


def _make_chunks():
    return [
        Chunk(chunk_id="c1", doc_name="d", page_num=0, text="배터리 팩 사양 설명"),
        Chunk(chunk_id="c2", doc_name="d", page_num=1, text="모터 제어 로직 설명"),
        Chunk(chunk_id="c3", doc_name="d", page_num=2, text="센서 캘리브레이션 절차"),
    ]


def _raise(self, query):
    raise RuntimeError("simulated outage")


@pytest.fixture
def hybrid():
    return HybridRetriever(_make_chunks(), embedder=_StubEmbedder())


def test_normal_mode_is_hybrid(hybrid):
    hybrid.score("배터리")
    assert hybrid.last_mode == "hybrid"


def test_dense_down_falls_back_to_bm25_only(hybrid):
    hybrid._dense_scores = types.MethodType(_raise, hybrid)
    result = hybrid.retrieve("배터리", top_k=1)
    assert hybrid.last_mode == "bm25_only"
    assert result == ["c1"]  # BM25만으로도 정답 문서는 그대로 1위


def test_bm25_down_falls_back_to_dense_only(hybrid):
    hybrid._bm25_scores = types.MethodType(_raise, hybrid)
    result = hybrid.retrieve("모터", top_k=1)
    assert hybrid.last_mode == "dense_only"
    assert result == ["c2"]


def test_both_down_uses_keyword_fallback(hybrid):
    hybrid._dense_scores = types.MethodType(_raise, hybrid)
    hybrid._bm25_scores = types.MethodType(_raise, hybrid)
    result = hybrid.retrieve("센서 캘리브레이션", top_k=1)
    assert hybrid.last_mode == "keyword_fallback"
    assert result == ["c3"]


def test_keyword_fallback_empty_query_returns_empty(hybrid):
    hybrid._dense_scores = types.MethodType(_raise, hybrid)
    hybrid._bm25_scores = types.MethodType(_raise, hybrid)
    assert hybrid._keyword_scores("???") == {}


def test_score_cache_does_not_leak_across_failure_conditions(hybrid):
    """같은 질의 문자열이라도 실패 조건이 바뀌면 캐시가 아니라 그때 상태로 다시 계산돼야 한다."""
    hybrid.score("배터리")
    assert hybrid.last_mode == "hybrid"
    hybrid._score_cache.clear()
    hybrid._dense_scores = types.MethodType(_raise, hybrid)
    hybrid.score("배터리")
    assert hybrid.last_mode == "bm25_only"
