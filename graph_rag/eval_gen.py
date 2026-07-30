"""평가셋 **자동 생성** — 그래프에서 질의를 유도한다. 순수·결정적.

## 왜 만들었나 (2026-07-30)

`eval_set.py`는 손으로 쓴 16문항이다. 그 16문항으로 낸 결론이 이랬다:

    hybrid_flat 0.688 → graph_rel 0.812 (불일치 4쌍, McNemar p=0.63)

즉 **차이는 보이는데 판정할 힘이 없다.** 16문항에서 전략 간 불일치가 3~4쌍뿐이면
p값은 무조건 0.2~0.6에 머문다. 이건 "그래프가 안 좋다"가 아니라 **"이 평가셋으로는
아무 말도 할 수 없다"**는 뜻이고, 검정력이 없는 상태에서 수치를 이력서에 쓰면 그건 거짓말이다.
(NOTES.md의 "다음 우선순위 1번"이 이 항목이다.)

문항을 손으로 40개 더 쓰는 건 답이 아니다. 손으로 쓰면 1차 때와 **같은 실수**를 반복한다 —
1차 평가셋은 멀티홉 질문이 다리(bridge) 엔티티를 그대로 노출해서(“셀 모듈을 포함하는
**배터리팩**은 어느 공급사에서?”) 플랫 BM25가 두 gold를 그냥 다 찾았고, 멀티홉이 아니라
키워드 검색 문제였다. 사람이 쓰는 한 이런 누출은 매번 새로 생기고 매번 사후에 잡아야 한다.

그래서 **생성 규칙**으로 옮긴다. 아래 두 성질이 사후 점검이 아니라 **구조적으로 보장**된다:

  1. **브리지 누출 불가** — 2-hop 질문 템플릿은 출발 엔티티 A의 이름만 쓴다. 중간 노드 B는
     템플릿에 들어갈 자리 자체가 없다("A가 포함하는 부품의 **사양**은?" — B를 부르지 않고
     관계로만 가리킨다). 사람이 조심하는 게 아니라 문법상 불가능하다.
  2. **답 유일성** — (A, r1)로 가는 B가 둘 이상이면 답이 확정되지 않아 질의를 **버린다**.
     corpus.py가 주석으로 "신규 부품에는 새 공급사만 붙인다(안 그러면 질문이 중의적이 된다)"고
     사람에게 당부하던 제약을, 생성기가 자동으로 강제한다.

## 정직하게 — 이 방식이 공짜로 얻는 것

질문이 템플릿에서 나오므로 **표면 어휘가 균질하다.** 실사용 질의의 다양성(구어체·약어·오타·
동의어)은 여기 없다. 또 질문은 항상 A의 정식 명칭을 쓰므로 **엔티티 링킹이 실제보다 쉽다**.
따라서 이 평가셋이 늘려주는 것은 *문항 수(=검정력)*이지 *현실성*이 아니다.
현실성은 실 PDF 적용(NOTES 우선순위 3번)에서 따로 확인해야 한다.

한 가지 더: 생성 질의는 gold 청크와 어휘를 공유한다(A 이름이 양쪽에 등장). 이건 어느 한
전략에 유리하게 기운 게 아니라 **플랫 검색에 유리한 쪽**이다 — BM25가 A를 바로 찾는다.
즉 이 평가셋은 그래프에 관대하지 않다.
"""
from __future__ import annotations

from dataclasses import dataclass

from .corpus import ENTITY_BY_ID, RELATIONS, Chunk, Relation, relation_home
from .eval_set import Query

# ---------------------------------------------------------------------------
# 질문 템플릿 — 관계를 "이름 대신 관계로 가리키는" 표현
#
# 정방향: (A)-[r]->(?)  에서 ?를 묻는다
# 역방향: (?)-[r]->(A)  에서 ?를 묻는다
#
# HAS_SPEC의 역방향(사양 → 그 사양을 가진 부품)은 넣지 않았다. "공칭 전압 400V인 부품은?"은
# 질문으로는 성립하지만 사양 문자열을 통째로 질의에 넣게 되어 BM25가 gold를 즉시 찾는다.
# 검색 난이도가 사라지는 질문이라 평가셋에 넣을 가치가 없다.
# ---------------------------------------------------------------------------

_ASK_1HOP_FORWARD = {
    "CONTAINS": "{a}은(는) 어떤 부품으로 구성되는가?",
    "CONNECTS_TO": "{a}은(는) 어떤 부품과 연결되는가?",
    "HAS_SPEC": "{a}의 사양은 무엇인가?",
    "SUPPLIED_BY": "{a}은(는) 어디에서 공급하는가?",
}

_ASK_1HOP_BACKWARD = {
    "CONTAINS": "{a}을(를) 포함하는 상위 항목은 무엇인가?",
    "CONNECTS_TO": "{a}에 연결되는 부품은 무엇인가?",
    "SUPPLIED_BY": "{a}이(가) 공급하는 부품은 무엇인가?",
}

# 2-hop 앞마디: A에서 한 걸음 간 대상을 **이름 없이** 가리키는 명사구
_PHRASE_FORWARD = {
    "CONTAINS": "{a}이(가) 포함하는 부품",
    "CONNECTS_TO": "{a}과(와) 연결된 부품",
    "SUPPLIED_BY": "{a}을(를) 공급하는 회사",
    # HAS_SPEC 정방향은 앞마디로 못 쓴다 — 사양 노드에서 더 뻗는 관계가 없다.
}

_PHRASE_BACKWARD = {
    "CONTAINS": "{a}을(를) 포함하는 상위 항목",
    "CONNECTS_TO": "{a}에 연결되는 부품",
    "SUPPLIED_BY": "{a}이(가) 공급하는 부품",
}

# 2-hop 뒷마디: 앞마디가 가리킨 대상에 대해 무엇을 묻는가
_TAIL_FORWARD = {
    "CONTAINS": "의 구성 부품은 무엇인가?",
    "CONNECTS_TO": "과(와) 연결된 부품은 무엇인가?",
    "HAS_SPEC": "의 사양은 무엇인가?",
    "SUPPLIED_BY": "은(는) 어디에서 공급하는가?",
}

_TAIL_BACKWARD = {
    "CONTAINS": "을(를) 포함하는 상위 항목은 무엇인가?",
    "CONNECTS_TO": "에 연결되는 부품은 무엇인가?",
    "SUPPLIED_BY": "이(가) 공급하는 부품은 무엇인가?",
}


@dataclass(frozen=True)
class Edge:
    """방향을 붙인 간선 — 역방향 순회를 1급으로 다루기 위한 표현.

    `forward=False`면 (tail)에서 출발해 (head)로 거슬러 간다는 뜻이다.
    "알파일렉트로닉스가 공급하는 부품은?"처럼 실제 질의의 상당수가 역방향이라
    정방향만 열거하면 평가셋이 한쪽으로 치우친다.
    """

    src: str
    dst: str
    rtype: str
    forward: bool
    rel: Relation


def directed_edges(relations: list[Relation] | None = None) -> list[Edge]:
    """관계 목록 → 정/역 양방향 간선. 결정적 순서(RELATIONS 정의 순).

    HAS_SPEC 역방향은 위 주석의 이유로 만들지 않는다.
    """
    rels = RELATIONS if relations is None else relations
    out: list[Edge] = []
    for r in rels:
        out.append(Edge(r.head, r.tail, r.rtype, True, r))
        if r.rtype != "HAS_SPEC":
            out.append(Edge(r.tail, r.head, r.rtype, False, r))
    return out


def _steps(edges: list[Edge], src: str, rtype: str, forward: bool) -> list[Edge]:
    """(src, rtype, 방향)으로 나가는 간선 전부. 결정적 순서."""
    return [e for e in edges if e.src == src and e.rtype == rtype and e.forward == forward]


def _unique_step(edges: list[Edge], src: str, rtype: str, forward: bool) -> Edge | None:
    """위와 같되 **정확히 하나**일 때만 준다.

    2-hop의 **앞마디**에만 쓴다. 앞마디가 여럿이면 "A가 포함하는 부품"이라는 명사구가
    여러 대상을 가리켜 **경로 자체가 확정되지 않는다**(진짜 중의적 질문). 이 경우 버린다.

    뒷마디는 다르다 — 아래 generate_2hop 주석 참조.
    """
    cand = _steps(edges, src, rtype, forward)
    return cand[0] if len(cand) == 1 else None


def _name(eid: str) -> str:
    return ENTITY_BY_ID[eid].name


def shared_anchor(question: str, gold: list[str], index: dict[str, Chunk]) -> list[str]:
    """gold 청크 **전부**에 공통으로 등장하면서 질문이 이름으로 부르는 엔티티.

    비어 있어야 멀티홉이 성립한다. 하나라도 있으면 그 단어 하나로 플랫 검색이 gold를
    한꺼번에 찾을 수 있어, 그래프 경로를 타야 할 이유가 사라진다.

    ## 이 검사가 왜 따로 필요한가 (2026-07-30, 테스트가 잡아낸 결함)

    처음엔 `leaked_bridges`(=중간 노드 B의 이름이 질문에 있는가)만 막으면 충분하다고 봤다.
    그런데 단위테스트가 13문항을 잡아냈다. 예:

        "조향각 센서와 연결된 부품은 어디에서 공급하는가?"
          gold1 [조향 연결 구조]   조향각 센서 ↔ 전동식 조향장치
          gold2 [조향 계통 공급사] 전동식 조향장치 공급사 · **조향각 센서** 공급사

    노출된 건 브리지가 아니라 **시드(=출발 엔티티) 자신**이다. 공급사 페이지가 같은 계통
    부품들을 묶어 서술하다 보니 시드가 거기에도 등장한다. 경로는 정상인데 두 gold가
    시드라는 공통 단어를 공유하므로, BM25가 "조향각 센서"만으로 둘 다 끌어온다.
    **누출 경로는 달라도 효과는 1차 사고와 똑같다.**

    그래서 검사 기준을 "브리지가 노출됐는가"에서 **"gold 전부를 잇는 공통 단어가
    질문에 있는가"**로 넓혔다. 시드도 예외가 아니다.
    """
    if not gold:
        return []
    common = set(index[gold[0]].entities)
    for g in gold[1:]:
        common &= set(index[g].entities)
    return sorted(e for e in common if ENTITY_BY_ID[e].name in question)


def generate_1hop(chunks: list[Chunk], max_gold: int = 3) -> list[Query]:
    """1-hop 질의: (A)-[r]->(?) 또는 (?)-[r]->(A).

    답이 여럿인 경우(예: "열관리 시스템은 어떤 부품으로 구성되는가?" → 칠러·펌프)는
    버리지 않고 **gold를 전부** 넣는다. 손으로 쓴 q07·q08과 같은 형태이고, full_hit이
    "근거가 전부 모였는가"를 재는 지표라 복수 정답이 오히려 지표 정의에 맞는다.
    (2-hop과 다른 이유: 1-hop은 답이 여럿이어도 질문 자체가 모호하지 않다.
     "구성 부품들"을 묻는 것이지 특정 하나를 묻는 게 아니다.)
    """
    home = relation_home(chunks)
    edges = directed_edges()
    # (출발 엔티티, 관계타입, 방향) 별로 묶는다
    groups: dict[tuple[str, str, bool], list[Edge]] = {}
    for e in edges:
        groups.setdefault((e.src, e.rtype, e.forward), []).append(e)

    out: list[Query] = []
    for (src, rtype, forward), es in groups.items():
        table = _ASK_1HOP_FORWARD if forward else _ASK_1HOP_BACKWARD
        tmpl = table.get(rtype)
        if tmpl is None:
            continue
        gold: list[str] = []
        for e in es:
            cid = home[(e.rel.head, e.rel.rtype, e.rel.tail)]
            if cid not in gold:
                gold.append(cid)
        # 2-hop과 같은 상한을 건다 — top-k 평가에서 gold > k면 어떤 전략도 0점이라
        # 변별력이 없다. 한쪽에만 상한을 두면 단일홉/멀티홉 비교가 불공정해진다.
        if len(gold) > max_gold:
            continue
        out.append(
            Query(
                qid="",  # 아래 _finalize에서 결정적으로 부여
                question=tmpl.format(a=_name(src)),
                hops=1,
                gold_chunks=tuple(gold),
                seed_entities=(src,),
            )
        )
    return out


def generate_2hop(chunks: list[Chunk], max_gold: int = 3) -> list[Query]:
    """2-hop 질의: (A)-[r1]->(B)-[r2]->(C). B는 질문에 등장하지 않는다.

    ## 앞마디는 유일해야 하고, 뒷마디는 여럿이어도 된다

    처음엔 양쪽 모두 유일성을 요구했더니 멀티홉이 **17문항**밖에 안 나왔다(원인 집계:
    뒷마디 비유일 82건이 지배적). 그 82건의 정체는 "배터리팩의 사양"처럼 답이 여러 개인
    질문이다 — 배터리팩은 전압·용량·온도 사양을 다 갖는다.

    이건 중의적인 질문이 아니라 **답이 복수인 질문**이다. 앞마디("A가 포함하는 부품")가
    가리키는 대상 B는 여전히 하나로 확정되고, 그 B에 대해 "사양들"을 묻는 것이다.
    주 지표 `full_hit`이 "근거가 **전부** 모였는가"를 재므로 복수 정답은 지표 정의에
    오히려 잘 맞는다(1-hop에서 이미 같은 정책을 쓴다).

    반대로 **앞마디**가 여럿이면 명사구 자체가 여러 대상을 가리켜 경로가 확정되지 않는다.
    이건 진짜 중의성이라 계속 버린다.

    ## gold 크기 상한 (max_gold)

    top-k 평가에서 gold가 k보다 많으면 **어떤 전략도 구조적으로 0점**이라 변별력이 없다.
    기본 k=3에 맞춰 gold 3개까지만 채택한다. 이 상한은 평가셋을 쉽게 만들려는 게 아니라
    "재도 아무것도 구분되지 않는 문항"을 빼는 것이다. (상한을 올리려면 k도 같이 올려야 한다.)

    ## 그 외 버리는 경우
      - C가 A와 같다 → "A에 연결된 부품에 연결되는 부품은?"이 A로 되돌아온다(자명).
      - 두 관계가 **같은 청크**에 서술돼 있다 → 한 청크만 찾으면 끝이라 멀티홉이 아니다.
        이 필터가 없으면 "멀티홉인데 플랫 검색으로 풀리는" 질의가 섞여 전략 간 차이를 지운다.
      - 앞마디/뒷마디 템플릿이 없는 조합(사양 노드에서 더 뻗어나가는 경로 등).
    """
    home = relation_home(chunks)
    edges = directed_edges()
    index = {c.chunk_id: c for c in chunks}
    seen: set[tuple[str, str, bool, str, bool]] = set()
    out: list[Query] = []

    for e1 in edges:
        phrase_tbl = _PHRASE_FORWARD if e1.forward else _PHRASE_BACKWARD
        phrase = phrase_tbl.get(e1.rtype)
        if phrase is None:
            continue
        # 앞마디가 성립하려면 A에서 그 관계로 가는 대상이 유일해야 한다
        if _unique_step(edges, e1.src, e1.rtype, e1.forward) is None:
            continue
        b = e1.dst
        g1 = home[(e1.rel.head, e1.rel.rtype, e1.rel.tail)]

        # (관계타입, 방향)으로 묶어 뒷마디를 만든다 — 답이 여럿이면 gold를 전부 넣는다
        for rtype, forward in sorted(
            {(e.rtype, e.forward) for e in edges if e.src == b}
        ):
            tail_tbl = _TAIL_FORWARD if forward else _TAIL_BACKWARD
            tail = tail_tbl.get(rtype)
            if tail is None:
                continue
            hops2 = [e for e in _steps(edges, b, rtype, forward)
                     if e.dst != e1.src and e.dst != b]  # 되돌아오는 가지는 뺀다
            if not hops2:
                continue
            gold = [g1]
            for e2 in hops2:
                g2 = home[(e2.rel.head, e2.rel.rtype, e2.rel.tail)]
                if g2 not in gold:
                    gold.append(g2)
            if len(gold) < 2:
                continue  # 전부 같은 청크 = 멀티홉이 아니다
            if len(gold) > max_gold:
                continue
            key = (e1.src, e1.rtype, e1.forward, rtype, forward)
            if key in seen:
                continue
            question = phrase.format(a=_name(e1.src)) + tail
            # gold 전부를 잇는 공통 단어가 질문에 있으면 멀티홉이 무력화된다(시드도 예외 아님).
            # 위 shared_anchor 주석 참조 — 단위테스트가 잡아낸 13문항이 여기서 걸러진다.
            if shared_anchor(question, gold, index):
                continue
            seen.add(key)
            out.append(
                Query(
                    qid="",
                    question=question,
                    hops=2,
                    gold_chunks=tuple(gold),
                    seed_entities=(e1.src,),
                )
            )
    return out


def _finalize(queries: list[Query], prefix: str) -> list[Query]:
    """결정적 정렬 후 qid 부여 — 실행마다 같은 셋이 나와야 재현이 성립한다."""
    ordered = sorted(queries, key=lambda q: (q.hops, q.question))
    return [
        Query(
            qid=f"{prefix}{i + 1:03d}",
            question=q.question,
            hops=q.hops,
            gold_chunks=q.gold_chunks,
            seed_entities=q.seed_entities,
        )
        for i, q in enumerate(ordered)
    ]


def build_generated_queries(chunks: list[Chunk]) -> list[Query]:
    """그래프에서 유도한 평가셋 전체(1-hop + 2-hop). 순수·결정적."""
    return _finalize(generate_1hop(chunks) + generate_2hop(chunks), "g")


def leaked_bridges(queries: list[Query], chunks: list[Chunk]) -> list[tuple[str, str]]:
    """2-hop 질문이 다리 엔티티 이름을 노출했는지 검사 — 비어 있어야 정상.

    구조적으로 불가능하지만(템플릿에 B 자리가 없다) **불변식은 검사해서 고정한다.**
    1차 평가셋을 무너뜨린 게 정확히 이 누출이었기 때문이다. 템플릿을 나중에 손보다가
    실수로 B를 넣으면 여기서 잡힌다.

    검사 방식: 질문에 등장하는 엔티티 이름 중 시드가 아닌 것이 있으면 누출로 본다.
    """
    bad: list[tuple[str, str]] = []
    for q in queries:
        if q.hops < 2:
            continue
        for eid, ent in ENTITY_BY_ID.items():
            if eid in q.seed_entities:
                continue
            if ent.name in q.question:
                # 시드 이름의 일부로 등장한 경우는 누출이 아니다
                if any(ent.name in _name(s) for s in q.seed_entities):
                    continue
                bad.append((q.qid, eid))
    return bad


def split_queries(queries: list[Query]) -> tuple[list[Query], list[Query]]:
    """평가셋을 dev / test로 **결정적·층화** 분할한다.

    ## 왜 필요한가 (2026-07-30에 발견한 방법론 결함)

    `graph_weight` 스윕은 평가셋 전체에서 최적값을 고른다. 그런데 그 최적값으로 낸 점수의
    유의성을 **같은 평가셋에서** 검정하면, 이미 그 데이터에 맞춰 고른 하이퍼파라미터를
    그 데이터로 다시 정당화하는 셈이 된다(선택 편향). 11개 후보 중 최고를 고른 뒤
    "유의합니다"라고 말하면 p값이 낙관적으로 기운다.

    그래서 가중치는 dev에서 고르고, 검정은 **한 번도 안 본 test에서** 한다.
    (하이브리드의 0.7/0.3처럼 코드에 고정된 값은 이 문제가 없다 — 이 실험 이전에 정해졌다.)

    분할 규칙: hop별로 나눈 뒤 짝수 인덱스=dev, 홀수=test. 무작위가 아니라 **결정적**이고,
    단일홉/멀티홉 비율이 양쪽에서 보존된다(층화). 질의 순서는 이미 `_finalize`에서
    결정적으로 정렬돼 있다.
    """
    dev: list[Query] = []
    test: list[Query] = []
    for hops in sorted({q.hops for q in queries}):
        sub = [q for q in queries if q.hops == hops]
        for i, q in enumerate(sub):
            (dev if i % 2 == 0 else test).append(q)
    return dev, test


def summarize(queries: list[Query]) -> dict:
    """평가셋 요약 — 문항 수와 gold 크기 분포. 검정력을 가늠하는 데 쓴다."""
    single = [q for q in queries if q.hops == 1]
    multi = [q for q in queries if q.hops == 2]
    sizes = [len(q.gold_chunks) for q in queries]
    return {
        "total": len(queries),
        "single_hop": len(single),
        "multi_hop": len(multi),
        "gold_size_min": min(sizes) if sizes else 0,
        "gold_size_max": max(sizes) if sizes else 0,
        "gold_size_mean": round(sum(sizes) / len(sizes), 3) if sizes else 0.0,
    }
