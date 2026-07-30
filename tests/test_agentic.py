"""반복 검색(agentic)의 불변식 — 순수·결정적, 모델 불필요.

이 파일이 지키는 핵심은 **"반복이 실제로 일을 하는가"를 코드가 스스로 증명하게 만드는 것**이다.
첫 구현은 라운드를 4까지 늘려도 수치가 소수점까지 동일했는데(=반복이 아무 일도 안 함),
그때 이런 테스트가 있었다면 훨씬 빨리 잡았을 것이다. 그래서 세 가지를 고정한다:

  1. rounds=1은 반복 없는 기존 전략과 **정확히 같아야** 한다(대조군의 정당성).
  2. 확장은 **관계를 따라가야** 한다 — 같은 문서에 있다는 이유로 딸려가면 안 된다.
  3. 결과 병합이 뒤 라운드의 성과를 **버리지 않아야** 한다.
"""
from __future__ import annotations

from graph_rag.agentic import IterativeRetriever, RetrievalTrace, _interleave, sweep_rounds
from graph_rag.corpus import build_corpus
from graph_rag.eval_gen import build_generated_queries
from graph_rag.retrieval import GraphRetriever, HybridRetriever


def _mk(rounds: int, **kw) -> IterativeRetriever:
    chunks = build_corpus()
    return IterativeRetriever(
        chunks,
        graph=GraphRetriever(chunks, mode="relation"),
        hybrid=HybridRetriever(chunks, embedder=None),  # BM25 단독 = 모델 불필요
        rounds=rounds,
        **kw,
    )


# ---- 병합 규칙 ----


def test_interleave_gives_each_round_a_share():
    """라운드 로빈 — 각 라운드가 상위 k 안에 자기 몫을 갖는다.

    첫 구현은 점수에 라운드별 감쇠를 곱했는데, 그 탓에 2라운드가 찾아낸 **정답**이
    1라운드의 **오답**보다 낮은 점수를 받아 밀려났다. 그 회귀를 막는다.
    """
    got = _interleave([["a1", "a2", "a3"], ["b1", "b2"]], 4)
    assert got == ["a1", "b1", "a2", "b2"]


def test_interleave_dedupes_and_respects_k():
    got = _interleave([["x", "y"], ["x", "z"]], 3)
    assert got == ["x", "y", "z"]  # 중복 x는 한 번만
    assert len(_interleave([["a", "b", "c"], ["d"]], 2)) == 2


def test_interleave_preserves_first_round_order_when_only_one_round():
    assert _interleave([["a", "b", "c"]], 3) == ["a", "b", "c"]


# ---- 대조군: rounds=1은 반복 없는 것과 같다 ----


def test_single_round_matches_non_iterative_baseline():
    """rounds=1이면 감싼 리트리버 한 번 호출과 결과가 같아야 한다.

    이게 성립해야 "반복이 만든 이득"을 분리해 말할 수 있다.
    """
    chunks = build_corpus()
    graph = GraphRetriever(chunks, mode="relation")
    hybrid = HybridRetriever(chunks, embedder=None)
    it = IterativeRetriever(chunks, graph=graph, hybrid=hybrid, rounds=1, graph_weight=0.5)

    from graph_rag.retrieval import FusedRetriever

    fused = FusedRetriever(hybrid, graph, graph_weight=0.5)
    for q in build_generated_queries(chunks)[:25]:
        seeds = list(q.seed_entities)
        assert it.retrieve(q.question, 3, seeds) == fused.retrieve(q.question, 3, seeds), q.qid


# ---- 확장 규칙: 관계를 따라간다 ----


def test_expansion_follows_relations_not_cooccurrence():
    """다음 시드는 **현재 시드가 참여한 관계의 반대편 끝점**이어야 한다.

    한 청크가 관계를 여럿 담을 수 있다. 예를 들어 [동력 전달 경로]에는
    (구동 모터)-연결-(감속기)와 (인버터)-연결-(고전압 배터리팩)이 함께 있다.
    '감속기'에서 출발했다면 '구동 모터'로 가야지 '고전압 배터리팩'으로 새면 안 된다.
    청크 단위로 엔티티를 긁으면 그게 그래프 탐색이 아니라 동시출현이 된다.
    """
    chunks = build_corpus()
    idx = {c.chunk_id: c for c in chunks}
    it = _mk(2, expand_from_top=1)
    _, tr = it.retrieve_with_trace("감속기에 연결되는 부품은(는) 어디에서 공급하는가?", 3,
                                   ["cmp_reducer"])
    assert len(tr.rounds) >= 2
    first = tr.rounds[0]
    top_chunk = idx[first["retrieved"][0]]
    seeds = set(first["seeds"])
    # 발견된 엔티티는 전부 시드와 관계로 이어져 있어야 한다
    linked = {
        (r.tail if r.head in seeds else r.head)
        for r in top_chunk.states
        if r.head in seeds or r.tail in seeds
    }
    assert first["new_entities"], "확장이 아무것도 못 찾았다"
    assert set(first["new_entities"]) <= linked, (first["new_entities"], linked)


def test_expansion_never_revisits_entities():
    it = _mk(4)
    _, tr = it.retrieve_with_trace("감속기에 연결되는 부품의 사양은 무엇인가?", 3, ["cmp_reducer"])
    seen = set()
    for r in tr.rounds:
        assert not (set(r["new_entities"]) & seen), "이미 본 엔티티를 다시 시드로 삼았다"
        seen |= set(r["new_entities"])


def test_stops_early_when_nothing_new_is_learned():
    """새로 배운 게 없으면 예산이 남아도 멈춘다(낭비 방지)."""
    it = _mk(8)
    _, tr = it.retrieve_with_trace("리튬이온 셀의 사양은 무엇인가?", 3, ["cmp_cell"])
    assert len(tr.rounds) < 8


# ---- 결정성·추적 ----


def test_deterministic():
    a = _mk(3).retrieve("감속기에 연결되는 부품은(는) 어디에서 공급하는가?", 3, ["cmp_reducer"])
    b = _mk(3).retrieve("감속기에 연결되는 부품은(는) 어디에서 공급하는가?", 3, ["cmp_reducer"])
    assert a == b


def test_trace_summary_counts():
    t = RetrievalTrace()
    t.add(["a"], ["c1", "c2"], ["b"])
    t.add(["b"], ["c3"], [])
    assert t.summary() == {"rounds": 2, "total_retrieved": 3, "entities_discovered": 1}


def test_rounds_must_be_positive():
    import pytest

    with pytest.raises(ValueError):
        _mk(0)


# ---- 예산 곡선 ----


def test_sweep_rounds_reports_monotone_budget_and_usage():
    """예산 스윕이 라운드별 성능과 **실제 사용 라운드 수**를 함께 낸다.

    실제 사용량이 예산보다 작을 수 있다(조기 종료). 그 차이를 봐야 "예산을 늘려도
    안 오르는 게 반복이 안 통해서인지, 애초에 돌지도 않아서인지"를 구분할 수 있다.
    """
    chunks = build_corpus()
    mh = [q for q in build_generated_queries(chunks) if q.hops == 2][:8]
    graph = GraphRetriever(chunks, mode="relation")
    hybrid = HybridRetriever(chunks, embedder=None)

    def build(r: int) -> IterativeRetriever:
        return IterativeRetriever(chunks, graph=graph, hybrid=hybrid, rounds=r)

    rows = sweep_rounds(build, mh, 3, max_rounds=3)
    assert [r["rounds"] for r in rows] == [1, 2, 3]
    assert all(0.0 <= r["full_hit"] <= 1.0 for r in rows)
    assert rows[0]["avg_rounds_used"] == 1.0
    assert rows[1]["avg_rounds_used"] <= 2.0  # 조기 종료 가능
