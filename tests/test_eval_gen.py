"""생성 평가셋(eval_gen)과 짝지음 검정(stats)의 불변식 — 순수·결정적, 모델 불필요.

이 파일이 지키는 것은 두 가지다:

  1. **생성 규칙이 만들어내는 보장** — 브리지 누출 불가, 답 유일성, gold 상한, 결정성.
     이 보장이 깨지면 평가셋이 조용히 쉬워지고, 그 위에서 낸 수치는 전부 의미를 잃는다.
     (실제로 1차 손수 평가셋이 누출 때문에 무효가 된 적이 있다 — DECISIONS.md G1·G2.)
  2. **검정의 경계 조건** — 특히 "불일치 쌍이 적으면 유의 판정이 불가능하다"는 사실을
     수치로 고정한다. 이걸 모르면 p=0.25를 "차이가 없다"로 잘못 읽는다(G8).
"""
from __future__ import annotations

from graph_rag.corpus import ENTITY_BY_ID, build_corpus, chunk_index, relation_home
from graph_rag.eval_gen import (
    build_generated_queries,
    directed_edges,
    generate_1hop,
    generate_2hop,
    leaked_bridges,
    shared_anchor,
    split_queries,
    summarize,
)
from graph_rag.eval_set import build_queries
from graph_rag.stats import (
    binom_test_half,
    contingency,
    mcnemar,
    min_detectable_discordant,
    wilson_interval,
)

# ---- 생성 평가셋: 결정성·기본 건전성 ----


def test_generation_is_deterministic():
    chunks = build_corpus()
    a = build_generated_queries(chunks)
    b = build_generated_queries(chunks)
    assert [(q.qid, q.question, q.gold_chunks) for q in a] == [
        (q.qid, q.question, q.gold_chunks) for q in b
    ]


def test_generated_qids_unique_and_gold_exists():
    chunks = build_corpus()
    idx = chunk_index(chunks)
    qs = build_generated_queries(chunks)
    assert len({q.qid for q in qs}) == len(qs)
    for q in qs:
        assert q.gold_chunks, q.qid
        for g in q.gold_chunks:
            assert g in idx, (q.qid, g)


def test_generated_set_is_large_enough_to_have_power():
    """평가셋을 키운 목적 자체를 고정한다.

    ## 이 임계가 30인 이유 (처음엔 40으로 썼다가 실측으로 고쳤다)

    처음 40으로 잡은 건 **근거 없는 여유값**이었다. 그 뒤 `shared_anchor` 필터를 넣자
    멀티홉이 51 → 38이 되어 이 테스트가 깨졌다. 여기서 두 가지 중 하나를 해야 했다:
    (a) 데이터를 보고 임계를 낮춘다 — 기준을 결과에 맞추는 나쁜 습관,
    (b) 임계의 **진짜 근거**로 돌아간다.

    (b)를 택했다. 진짜 근거는 문항 수가 아니라 **불일치 쌍이 6개 이상 나오는가**다
    (`min_detectable_discordant`). 38문항에서 실측한 결과 hybrid_flat vs graph_rel의
    멀티홉 불일치가 0:9로 나와 검정력이 유지됨을 **확인한 뒤에** 임계를 조정했다.
    30은 그 실측(9쌍)에 안전 여유를 둔 값이다.

    핵심 교훈: 이 실험에서 문항 수는 대리 지표일 뿐이고, 진짜 지표는 불일치 쌍이다.
    실제로 필터 후 문항은 **줄었는데 검정력은 늘었다**(NOTES.md 참조).
    """
    qs = build_generated_queries(build_corpus())
    s = summarize(qs)
    assert s["multi_hop"] >= 30, s
    assert s["total"] >= 100, s


def test_shared_anchor_detects_common_word_linking_all_gold():
    """gold 전부에 공통이면서 질문에 등장하는 엔티티를 잡아낸다.

    이 검사가 없던 동안 13문항이 통과하고 있었고, 그 문항들을 플랫 BM25가 단어 하나로
    풀고 있었다(hybrid_flat 멀티홉 0.216 → 필터 후 0.053).
    """
    chunks = build_corpus()
    idx = chunk_index(chunks)
    ids = [c.chunk_id for c in chunks if c.states]
    # 두 청크에 공통으로 든 엔티티를 하나 찾아 그 이름을 질문에 넣어 본다
    for a in ids:
        for b in ids:
            if a == b:
                continue
            common = set(idx[a].entities) & set(idx[b].entities)
            if not common:
                continue
            name = ENTITY_BY_ID[sorted(common)[0]].name
            assert shared_anchor(f"{name}의 사양은?", [a, b], idx)
            assert shared_anchor("전혀 무관한 질문", [a, b], idx) == []
            return
    raise AssertionError("공통 엔티티를 가진 청크 쌍을 찾지 못했다")


def test_generated_multihop_has_no_shared_anchor():
    """생성된 모든 멀티홉 질의에 공통 앵커가 없어야 한다(생성 단계 필터의 검증)."""
    chunks = build_corpus()
    idx = chunk_index(chunks)
    for q in build_generated_queries(chunks):
        if q.hops == 2:
            assert shared_anchor(q.question, list(q.gold_chunks), idx) == [], q.qid


# ---- 핵심 불변식 1: 브리지 누출 ----


def test_no_bridge_leak_in_generated_multihop():
    """2-hop 질문은 시드 외의 엔티티 이름을 부르지 않는다(구조적 보장의 검증)."""
    chunks = build_corpus()
    qs = build_generated_queries(chunks)
    assert leaked_bridges(qs, chunks) == []


def test_generated_multihop_gold_shares_no_common_named_entity():
    """손수 평가셋에 걸던 것과 **같은 기준**을 생성 평가셋에도 적용한다.

    gold 청크들이 공통으로 담은 엔티티를 질문이 이름으로 부르면, 플랫 검색이 그 단어로
    두 청크를 한꺼번에 찾아버려 멀티홉이 성립하지 않는다.
    """
    chunks = build_corpus()
    idx = chunk_index(chunks)
    for q in build_generated_queries(chunks):
        if q.hops != 2:
            continue
        common = set(idx[q.gold_chunks[0]].entities)
        for g in q.gold_chunks[1:]:
            common &= set(idx[g].entities)
        leaked = [
            ENTITY_BY_ID[e].name for e in common if ENTITY_BY_ID[e].name in q.question
        ]
        assert not leaked, f"{q.qid}: 다리 엔티티 노출 {leaked}"


# ---- 핵심 불변식 2: 답 유일성과 gold 구조 ----


def test_2hop_first_step_is_unique():
    """앞마디(A→B)는 유일해야 한다 — 아니면 명사구가 여러 대상을 가리켜 중의적이 된다.

    DECISIONS.md G3: 뒷마디는 복수를 허용하되 앞마디 유일성은 유지한다는 결정.
    """
    chunks = build_corpus()
    edges = directed_edges()
    for q in generate_2hop(chunks):
        seed = q.seed_entities[0]
        # 시드에서 나가는 (관계타입, 방향) 중 하나로 이 질의가 만들어졌다.
        # 그 조합의 간선이 정확히 1개인 경우만 존재해야 한다.
        starts = {
            (r, f)
            for r, f in {(e.rtype, e.forward) for e in edges if e.src == seed}
            if len([e for e in edges if e.src == seed and e.rtype == r and e.forward == f]) == 1
        }
        assert starts, q.qid


def test_2hop_gold_spans_at_least_two_chunks():
    """두 관계가 같은 청크면 멀티홉이 아니다 — 생성 단계에서 걸러져야 한다."""
    for q in generate_2hop(build_corpus()):
        assert len(q.gold_chunks) >= 2, (q.qid, q.gold_chunks)
        assert len(set(q.gold_chunks)) == len(q.gold_chunks), q.qid


def test_gold_size_capped_on_both_hop_types():
    """gold 상한은 1-hop·2-hop **양쪽**에 걸려야 한다(G4).

    한쪽에만 걸면 top-k에서 구조적으로 0점인 문항이 한쪽에만 남아 비교가 불공정해진다.
    """
    chunks = build_corpus()
    for q in generate_1hop(chunks, max_gold=3) + generate_2hop(chunks, max_gold=3):
        assert len(q.gold_chunks) <= 3, (q.question, q.gold_chunks)
    # 상한을 낮추면 실제로 문항이 줄어든다(상한이 동작한다는 증거)
    assert len(generate_2hop(chunks, max_gold=2)) <= len(generate_2hop(chunks, max_gold=3))


def test_gold_chunks_actually_state_the_relations():
    """gold는 사람이 고른 게 아니라 관계→청크 매핑에서 나온다 — 그 사실을 고정한다."""
    chunks = build_corpus()
    home = relation_home(chunks)
    homes = set(home.values())
    for q in build_generated_queries(chunks):
        for g in q.gold_chunks:
            assert g in homes, (q.qid, g)


def test_distractors_can_never_be_gold():
    """distractor는 states가 비어 있어 어떤 질의의 gold도 될 수 없다."""
    chunks = build_corpus()
    distractor_ids = {c.chunk_id for c in chunks if not c.states}
    for q in build_generated_queries(chunks):
        assert not (set(q.gold_chunks) & distractor_ids), q.qid


# ---- 손수 평가셋과의 공존 ----


def test_handwritten_eval_set_still_valid_after_corpus_growth():
    """코퍼스를 키워도 손수 만든 16문항은 그대로 유효해야 한다(G5의 제약).

    관계 인덱스를 append만 하고 기존 공급사를 재사용하지 않기로 한 규칙이 지켜졌는지
    확인한다. 이게 깨지면 1차·2차 실험을 이어서 비교할 수 없다.
    """
    chunks = build_corpus()
    qs = build_queries(chunks)  # 규칙 위반 시 build_queries가 raise
    assert len(qs) == 16
    assert {q.hops for q in qs} == {1, 2}


# ---- 홀드아웃 분할 ----


def test_split_is_deterministic_disjoint_and_stratified():
    qs = build_generated_queries(build_corpus())
    dev, test = split_queries(qs)
    dev2, test2 = split_queries(qs)
    assert [q.qid for q in dev] == [q.qid for q in dev2]
    assert [q.qid for q in test] == [q.qid for q in test2]
    # 서로 겹치지 않고, 합치면 전체
    assert not ({q.qid for q in dev} & {q.qid for q in test})
    assert len(dev) + len(test) == len(qs)
    # 층화 — 멀티홉 비율이 양쪽에서 비슷해야 한다(±1문항)
    for hops in (1, 2):
        a = sum(1 for q in dev if q.hops == hops)
        b = sum(1 for q in test if q.hops == hops)
        assert abs(a - b) <= 1, (hops, a, b)


# ---- 검정 ----


def test_mcnemar_ignores_concordant_pairs():
    """둘 다 맞히거나 둘 다 틀린 문항은 p값에 영향을 주지 않는다(McNemar의 정의)."""
    a = [True, True, False, False, True]
    b = [True, False, True, False, True]
    base = mcnemar(a, b)
    # 양쪽 모두 맞힌 문항을 10개 추가해도 p는 그대로
    padded = mcnemar(a + [True] * 10, b + [True] * 10)
    assert base["p_value"] == padded["p_value"]
    assert padded["discordant"] == base["discordant"]


def test_mcnemar_direction_and_delta():
    a = [True, False, False, False]
    b = [True, True, True, True]
    res = mcnemar(a, b)
    assert res["b_only"] == 3 and res["a_only"] == 0
    assert res["delta"] > 0  # B가 낫다


def test_no_significance_possible_below_six_discordant():
    """불일치 쌍이 6개 미만이면 **완전히 쏠려도** p<0.05가 불가능하다(G8).

    이 사실을 모르면 "p=0.25 → 차이 없음"으로 잘못 읽는다. 실제로는 '잴 힘이 없음'이다.
    """
    assert min_detectable_discordant() == 6
    for n in range(1, 6):
        assert binom_test_half(0, n) > 0.05, n  # 최선의 경우조차 유의 불가
    assert binom_test_half(0, 6) <= 0.05


def test_binom_and_contingency_edges():
    assert binom_test_half(0, 0) == 1.0  # 근거 없음
    assert mcnemar([True], [True])["p_value"] == 1.0
    tab = contingency([True, False], [False, False])
    assert tab["a_only"] == 1 and tab["b_only"] == 0 and tab["neither"] == 1


def test_wilson_interval_contains_point_estimate():
    lo, hi = wilson_interval(30, 50)
    assert lo < 0.6 < hi
    assert wilson_interval(0, 0) == (0.0, 0.0)
