"""실문서 평가셋 — 합성에서 세운 원칙을 실제 조건에 그대로 적용한다.

## 합성과 무엇이 같고 무엇이 다른가

같은 것(원칙은 유지한다):
  - gold는 사람이 고르지 않는다. **그래프에서 유도**한다.
  - 2-hop 질문은 다리(사업명)를 **부르지 않는다**. 관계로만 가리킨다.
  - 답이 여럿이면 버린다(중의성 배제).
  - gold 전부를 잇는 공통 단어가 질문에 있으면 버린다(`shared_anchor`와 같은 기준).

다른 것(실문서이기 때문에):
  - 관계가 **IE의 산물**이다. 틀린 관계면 gold도 틀린다. 표본 감사로 정밀도를 알고 읽어야 한다.
  - 코퍼스가 7,865청크이고 그중 **90%가 관계 없는 방해 문서**다(합성은 72청크·46%).
    top-3가 전체의 0.04%라 훨씬 어렵다.
  - 근거가 문서 곳곳에 흩어져 있고 표기가 흔들린다(합성은 생성 규칙이라 일관됐다).

## 질의 형태

  1-hop  "{사업명}의 사업 예산은 얼마인가?"      gold = 예산 근거 청크
  2-hop  "{기관명}이 발주한 사업의 예산은?"       gold = [기관 근거 청크, 예산 근거 청크]

2-hop이 진짜 멀티홉인 이유: 예산이 적힌 청크에는 **기관명이 없다.** 질의어로는 그 청크에
닿을 수 없고, 기관에서 사업으로 한 번 건너간 다음에야 예산을 찾을 수 있다.
"""
from __future__ import annotations

from graph_rag.eval_set import Query

from .chunking import RealCorpus
from .extract import HAS_BUDGET, HAS_PERIOD, ISSUED_BY, REQUIRES


def _by_type(rc: RealCorpus, rtype: str) -> list[tuple[str, str]]:
    """(head_id, tail_id) 목록 — 청크에 연결된 관계만."""
    return [
        (h, t) for (h, rt, t) in rc.home if rt == rtype
    ]


def build_real_queries(rc: RealCorpus, max_gold: int = 3) -> list[Query]:
    """실문서 그래프에서 질의를 유도한다. 순수·결정적."""
    return [q for q, _ in build_real_queries_with_sources(rc, max_gold)]


def build_real_queries_with_sources(
    rc: RealCorpus, max_gold: int = 3
) -> list[tuple[Query, tuple[str, ...]]]:
    """질의와 **그 질의를 만든 트리플**을 함께 돌려준다.

    감사에서 어떤 관계가 오탐으로 판정되면 그 관계에서 유도된 문항은 gold가 틀린
    것이므로 집계에서 빼야 한다. 그러려면 문항↔관계 대응이 있어야 하는데,
    이걸 나중에 질문 문자열로 되짚으면 같은 유형 관계가 한 청크에 여럿 있을 때 어긋난다.
    생성하는 자리에서 같이 들고 나오는 게 유일하게 안전한 방법이다.
    """
    name = {eid: e.name for eid, e in rc.entities.items()}
    chunk_of = {c.chunk_id: c for c in rc.chunks}

    # 사업 → 기관 (역방향 조회용)
    agency_of: dict[str, str] = {}
    projects_of_agency: dict[str, list[str]] = {}
    for h, t in _by_type(rc, ISSUED_BY):
        agency_of[h] = t
        projects_of_agency.setdefault(t, []).append(h)

    out: list[tuple[Query, tuple[str, ...]]] = []

    # ---- 1-hop: 사업의 속성 ----
    for rtype, ask in (
        (HAS_BUDGET, "{p}의 사업 예산은 얼마인가?"),
        (HAS_PERIOD, "{p}의 사업 기간은 얼마인가?"),
    ):
        for h, t in _by_type(rc, rtype):
            gold = rc.home[(h, rtype, t)]
            out.append(
                (
                    Query(
                        qid="",
                        question=ask.format(p=name[h]),
                        hops=1,
                        gold_chunks=(gold,),
                        seed_entities=(h,),
                    ),
                    (f"{h}|{rtype}|{t}",),
                )
            )

    # ---- 2-hop: 기관 → 사업 → 속성 (사업명을 부르지 않는다) ----
    for rtype, ask in (
        (HAS_BUDGET, "{a}이(가) 발주한 사업의 예산은 얼마인가?"),
        (HAS_PERIOD, "{a}이(가) 발주한 사업의 기간은 얼마인가?"),
        (REQUIRES, "{a}이(가) 발주한 사업은 어떤 기술을 요구하는가?"),
    ):
        pairs: dict[str, list[tuple[str, str]]] = {}
        for h, t in _by_type(rc, rtype):
            a = agency_of.get(h)
            if a:
                pairs.setdefault(a, []).append((h, t))

        for agency, items in pairs.items():
            # 한 기관이 여러 사업을 발주했으면 "그 사업"이 어느 것인지 정해지지 않는다
            if len(projects_of_agency.get(agency, [])) != 1:
                continue
            project = items[0][0]
            issued_gold = rc.home.get((project, ISSUED_BY, agency))
            if issued_gold is None:
                continue

            gold = [issued_gold]
            # 2-hop은 다리(ISSUED_BY)와 속성 관계 둘 다에 의존한다 — 둘 중 하나만
            # 틀려도 gold가 무너지므로 출처에 둘 다 넣는다.
            srcs = [f"{project}|{ISSUED_BY}|{agency}"]
            for h, t in items:
                g = rc.home[(h, rtype, t)]
                srcs.append(f"{h}|{rtype}|{t}")
                if g not in gold:
                    gold.append(g)
            if len(gold) < 2 or len(gold) > max_gold:
                continue

            question = ask.format(a=name[agency])
            if _shared_anchor(question, gold, chunk_of, name):
                continue

            out.append(
                (
                    Query(
                        qid="",
                        question=question,
                        hops=2,
                        gold_chunks=tuple(gold),
                        seed_entities=(agency,),
                    ),
                    tuple(srcs),
                )
            )

    ordered = sorted(out, key=lambda x: (x[0].hops, x[0].question, x[0].gold_chunks))
    return [
        (
            Query(
                qid=f"r{i + 1:04d}",
                question=q.question,
                hops=q.hops,
                gold_chunks=q.gold_chunks,
                seed_entities=q.seed_entities,
            ),
            srcs,
        )
        for i, (q, srcs) in enumerate(ordered)
    ]


def _shared_anchor(question: str, gold: list[str], chunk_of, name) -> bool:
    """gold 전부에 공통이면서 질문이 부르는 엔티티가 있으면 멀티홉이 성립하지 않는다.

    합성 평가셋에서 13문항을 걸러낸 그 기준을 실문서에도 그대로 적용한다
    (거기서 배운 것: 이 검사가 없으면 어휘 검색이 단어 하나로 gold를 다 끌어온다).
    """
    common = set(chunk_of[gold[0]].entities)
    for g in gold[1:]:
        common &= set(chunk_of[g].entities)
    return any(name[e] in question for e in common if e in name)


def summarize_real(queries: list[Query]) -> dict:
    single = sum(1 for q in queries if q.hops == 1)
    multi = sum(1 for q in queries if q.hops == 2)
    sizes = [len(q.gold_chunks) for q in queries]
    return {
        "total": len(queries),
        "single_hop": single,
        "multi_hop": multi,
        "gold_size_mean": round(sum(sizes) / len(sizes), 3) if sizes else 0.0,
    }
