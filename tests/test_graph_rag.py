"""GraphRAG 코어 단위테스트 — 순수·결정적, 모델/DB/네트워크 불필요.

코퍼스·평가셋의 **불변식**을 고정한다. 이게 깨지면 이후 검색 비교 수치가 전부 의미를 잃는다.
"""
from __future__ import annotations

import pytest

from graph_rag.corpus import (
    ENTITY_BY_ID,
    RELATIONS,
    build_corpus,
    chunk_index,
    relation_home,
)
from graph_rag.eval_set import build_queries
from graph_rag.metrics import (
    aggregate,
    full_hit_at_k,
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)


# ---- 코퍼스 불변식 ----

def test_corpus_is_deterministic():
    a = build_corpus()
    b = build_corpus()
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.text for c in a] == [c.text for c in b]


def test_chunk_ids_unique():
    chunks = build_corpus()
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_every_relation_appears_exactly_once():
    """관계는 정확히 한 청크에만 서술된다 — 멀티홉 gold 유도의 전제."""
    chunks = build_corpus()
    home = relation_home(chunks)  # 중복이면 여기서 raise
    assert len(home) == len(RELATIONS)


def test_all_relation_endpoints_are_known_entities():
    for r in RELATIONS:
        assert r.head in ENTITY_BY_ID, r
        assert r.tail in ENTITY_BY_ID, r


def test_chunk_text_mentions_its_entities():
    """청크 본문에 그 청크가 다루는 엔티티 이름이 실제로 등장해야 검색이 성립한다."""
    for c in build_corpus():
        for eid in c.entities:
            assert ENTITY_BY_ID[eid].name in c.text, (c.chunk_id, eid)


# ---- 평가셋 불변식 ----

def test_queries_build_and_gold_exists():
    chunks = build_corpus()
    idx = chunk_index(chunks)
    for q in build_queries(chunks):
        assert q.gold_chunks, q.qid
        for g in q.gold_chunks:
            assert g in idx, (q.qid, g)


def test_multihop_gold_spans_multiple_chunks():
    """멀티홉 질의는 gold가 2개 이상이어야 한다 — 아니면 단일홉과 구분이 없다."""
    chunks = build_corpus()
    multi = [q for q in build_queries(chunks) if q.hops == 2]
    assert multi, "멀티홉 질의가 없다"
    for q in multi:
        assert len(q.gold_chunks) >= 2, (q.qid, q.gold_chunks)


def test_single_hop_gold_is_one_chunk():
    chunks = build_corpus()
    for q in build_queries(chunks):
        if q.hops == 1:
            assert len(q.gold_chunks) == 1, (q.qid, q.gold_chunks)


def test_multihop_questions_do_not_leak_the_bridge_entity():
    """멀티홉 질문은 gold 청크 **전부**에 공통으로 등장하는 엔티티를 이름으로 불러선 안 된다.

    첫 평가셋이 이 규칙을 어겨 실험이 무효였다(2026-07-27). 예: "셀 모듈을 포함하는 **배터리팩**은
    어느 공급사에서?" — 다리인 '배터리팩'을 질문이 알려주면 두 gold 청크가 같은 단어를 공유하게 되어
    플랫 BM25가 그냥 둘 다 찾는다. 그러면 '멀티홉'이 아니라 그냥 키워드 검색 문제다.
    """
    chunks = build_corpus()
    idx = chunk_index(chunks)
    for q in build_queries(chunks):
        if q.hops != 2:
            continue
        common = set(idx[q.gold_chunks[0]].entities)
        for g in q.gold_chunks[1:]:
            common &= set(idx[g].entities)
        leaked = [
            ENTITY_BY_ID[e].name for e in common if ENTITY_BY_ID[e].name in q.question
        ]
        assert not leaked, f"{q.qid}: 다리 엔티티 노출 {leaked} — 멀티홉이 성립하지 않는다"


def test_query_ids_unique():
    chunks = build_corpus()
    qids = [q.qid for q in build_queries(chunks)]
    assert len(qids) == len(set(qids))


# ---- 지표 ----

def test_hit_vs_full_hit_differ_on_partial_retrieval():
    """멀티홉에서 gold 절반만 찾으면 hit=1이지만 full_hit=0 — 이 구분이 실험의 핵심."""
    retrieved = ["a", "x", "y"]
    gold = ("a", "b")
    assert hit_at_k(retrieved, gold, 3) == 1.0
    assert full_hit_at_k(retrieved, gold, 3) == 0.0
    assert recall_at_k(retrieved, gold, 3) == 0.5


def test_full_hit_requires_all_gold_within_k():
    assert full_hit_at_k(["a", "b", "c"], ("a", "b"), 2) == 1.0
    assert full_hit_at_k(["a", "c", "b"], ("a", "b"), 2) == 0.0  # b가 k 밖


def test_reciprocal_rank_uses_best_gold():
    assert reciprocal_rank(["x", "b", "a"], ("a", "b")) == pytest.approx(1 / 2)
    assert reciprocal_rank(["x", "y"], ("a",)) == 0.0


def test_aggregate_shapes():
    res = [(["a", "b"], ("a", "b")), (["x", "y"], ("a", "b"))]
    agg = aggregate(res, k=2)
    assert agg["count"] == 2
    assert agg["full_hit"] == 0.5
    assert agg["hit"] == 0.5


def test_empty_gold_rejected():
    with pytest.raises(ValueError):
        hit_at_k(["a"], [], 1)


# ---- 그래프 백엔드 등가성 ----

def test_kuzu_cypher_matches_in_memory(tmp_path):
    """임베디드 Graph DB(Cypher)와 순수 파이썬 탐색이 같은 결과를 내야 한다.

    이게 깨지면 "Cypher로 그래프 탐색했다"는 주장의 근거가 사라진다.
    """
    kuzu = pytest.importorskip("kuzu")  # noqa: F841
    from graph_rag.graph_store import InMemoryGraphStore, KuzuGraphStore

    mem = InMemoryGraphStore()
    kz = KuzuGraphStore(str(tmp_path / "g.kz"))
    for seed in ["cmp_cell", "sys_bms", "sup_gamma", "cmp_chiller"]:
        for hops in (1, 2):
            assert mem.expand([seed], hops) == kz.expand([seed], hops), (seed, hops)


# ---- distractor 불변식 (2026-07-29 코퍼스 확장) ----

def test_distractors_can_never_be_gold():
    """distractor는 관계를 서술하지 않으므로 어떤 질의의 정답도 될 수 없다.

    이게 깨지면 '정답인데 distractor로 세어 점수가 낮게 나오는' 사고가 조용히 발생한다.
    """
    chunks = build_corpus()
    distractor_ids = {c.chunk_id for c in chunks if not c.states}
    assert distractor_ids, "distractor가 하나도 없다 — 코퍼스 확장이 반영되지 않았다"
    for q in build_queries(chunks):
        assert not (set(q.gold_chunks) & distractor_ids), q.qid


def test_distractors_mention_entities_so_graph_is_also_tempted():
    """distractor 대부분은 엔티티를 언급해야 한다.

    엔티티가 없으면 GraphRetriever가 그 청크를 아예 점수화하지 않는다. 그러면 어휘 검색만
    방해받고 그래프는 무풍지대에 놓여 '그래프가 이겼다'가 코퍼스 설계의 산물이 된다.
    (엔티티 없는 순수 어휘 distractor도 몇 개는 의도적으로 둔다 — 문서 메타·단위 표기 등.)
    """
    chunks = build_corpus()
    distractors = [c for c in chunks if not c.states]
    with_ents = [c for c in distractors if c.entities]
    assert len(with_ents) / len(distractors) >= 0.7


def test_corpus_expansion_kept_original_relations_stable():
    """확장은 append만 해야 한다 — RELATIONS 인덱스를 _CHUNK_PLAN이 참조하기 때문이다.

    앞쪽 25개(1차 실험분)가 그대로여야 1차 결과와의 비교가 성립한다.
    """
    head = RELATIONS[:25]
    assert head[0] == ("sys_bms", "CONTAINS", "cmp_bms_ecu") or (
        head[0].head, head[0].rtype, head[0].tail
    ) == ("sys_bms", "CONTAINS", "cmp_bms_ecu")
    assert (head[24].head, head[24].rtype, head[24].tail) == (
        "cmp_chiller", "SUPPLIED_BY", "sup_beta"
    )
    assert len(RELATIONS) > 25, "확장 관계가 없다"


def test_new_components_use_new_suppliers_only():
    """기존 평가 질의(q10·q15)는 '해당 공급사가 공급하는 부품'이 유일해야 성립한다.

    신규 부품에 alpha/beta/gamma를 붙이면 질문이 중의적이 되어 평가셋이 조용히 깨진다.
    """
    legacy_suppliers = {"sup_alpha", "sup_beta", "sup_gamma"}
    legacy_supplied = {
        "cmp_bms_ecu", "cmp_pack", "cmp_radar", "cmp_camera", "cmp_chiller",
    }
    for r in RELATIONS:
        if r.rtype == "SUPPLIED_BY" and r.tail in legacy_suppliers:
            assert r.head in legacy_supplied, (
                f"{r.head}에 기존 공급사 {r.tail}를 붙이면 q10/q15가 중의적이 된다"
            )


def test_distractors_do_not_change_gold_definitions():
    """distractor를 켜고 끄는 것이 정답 정의를 바꾸면 안 된다(코퍼스 난이도만 달라져야 한다)."""
    lean = build_corpus(with_distractors=False)
    full = build_corpus(with_distractors=True)
    assert len(full) > len(lean)
    gold_lean = {q.qid: q.gold_chunks for q in build_queries(lean)}
    gold_full = {q.qid: q.gold_chunks for q in build_queries(full)}
    assert gold_lean == gold_full


# ---- 관계 인식 그래프 점수 (2026-07-29) ----

def test_relation_mode_ignores_distractors():
    """관계를 서술하지 않는 청크는 relation 모드에서 점수가 나오지 않아야 한다.

    mention 모드는 반대로 점수를 준다 — 이 차이가 distractor 코퍼스에서 결과를 갈랐다.
    """
    from graph_rag.retrieval import GraphRetriever

    chunks = build_corpus()
    distractor_ids = {c.chunk_id for c in chunks if not c.states}
    seeds = ["cmp_pack"]

    rel = GraphRetriever(chunks, mode="relation").score("", seeds)
    assert not (set(rel) & distractor_ids)

    men = GraphRetriever(chunks, mode="mention").score("", seeds)
    assert set(men) & distractor_ids, "mention 모드는 distractor를 집는 게 정상(그게 결함)"


def test_relation_mode_requires_both_endpoints_in_expansion():
    """한쪽 끝점만 확장 집합에 있으면 근거로 인정하지 않는다."""
    from graph_rag.retrieval import GraphRetriever

    chunks = build_corpus()
    idx = chunk_index(chunks)
    g = GraphRetriever(chunks, mode="relation", max_hops=0)  # 시드 자신만
    scores = g.score("", ["spec_voltage"])
    for cid in scores:
        c = idx[cid]
        assert any(r.head in {"spec_voltage"} and r.tail in {"spec_voltage"} for r in c.states)


def test_graph_retriever_rejects_unknown_mode():
    from graph_rag.retrieval import GraphRetriever

    with pytest.raises(ValueError):
        GraphRetriever(build_corpus(), mode="nope")
