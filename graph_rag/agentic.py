"""반복 검색(iterative retrieval) — 한 번에 못 모으는 근거를 여러 번에 나눠 모은다.

## 왜 만들었나 (2026-07-30)

같은 날 낸 측정이 이랬다(생성 평가셋 135문항, top-3, full_hit):

    멀티홉:  hybrid_flat 0.053 · graph_rel 0.289 · fused_rel 0.263

그래프가 플랫을 이기긴 했지만(McNemar p=0.0039) **절대값은 여전히 낮다.** 이유는 분명하다 —
지금까지의 모든 전략은 **질의 한 번으로 근거를 전부 끌어오려** 했다. 멀티홉 질의의 근거는
정의상 서로 다른 문서에 흩어져 있고, 질의 표면어는 그중 **출발점 쪽 문서에만** 겹친다.

    "감속기에 연결되는 부품은 어디에서 공급하는가?"
      gold1 [동력 전달 경로]   감속기 ↔ 구동 모터      ← 질의어 '감속기'가 여기 있다
      gold2 [구동 계통 공급사] 구동 모터 공급사        ← 질의어가 하나도 없다

gold2를 찾으려면 **'구동 모터'라는 단어를 알아야 하는데, 그건 gold1을 읽어야 나온다.**
한 번의 검색으로는 원리적으로 어렵다. 사람이라면 첫 문서를 읽고 거기서 얻은 단어로 다시 찾는다.

## 이 모듈이 하는 일

그 "다시 찾기"를 그대로 구현한다. 라운드마다:

  1. 현재 시드로 검색한다.
  2. 상위 결과에서 **새 엔티티**를 얻는다(1라운드에서 읽은 문서가 2라운드의 질의가 된다).
  3. 새 엔티티가 있으면 그것을 시드로 삼아 한 번 더 검색한다.
  4. 라운드별 결과를 누적해 최종 순위를 낸다.

## 설계에서 중요하게 잡은 것

- **LLM이 필요 없다.** 흔히 agentic RAG는 질의 분해를 LLM에 시키지만, 여기서는 검색 결과의
  엔티티가 다음 질의가 된다. 그래서 **결정적**이고, 재실행하면 같은 값이 나오며, 키가 없어도 돈다.
  (LLM 분해가 더 유연한 건 맞다. 다만 그건 이 실험이 재는 대상이 아니고, 넣으면 결과의 재현성이
   깨진다. 어디까지가 검색 구조의 힘인지 먼저 보는 게 순서다.)
- **예산을 명시한다.** `rounds`가 곧 추론 예산이다. 늘리면 좋아지는지, 어디서 멈추는지를
  측정할 수 있어야 "에이전트를 붙였다"는 말이 검증된다(`sweep_rounds` 참조).
- **1라운드는 기존 전략과 정확히 같다.** `rounds=1`이면 감싼 리트리버를 그대로 호출한다.
  덕분에 "반복이 만든 이득"만 분리해서 볼 수 있다(대조군이 코드 안에 있다).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import Chunk
from .retrieval import GraphRetriever, HybridRetriever, _normalize, _rank, link_entities


@dataclass
class RetrievalTrace:
    """라운드별로 무엇을 시드로 무엇을 찾았는지 — 관측 없이는 개선도 없다."""

    rounds: list[dict] = field(default_factory=list)

    def add(self, seeds: list[str], retrieved: list[str], new_entities: list[str]) -> None:
        self.rounds.append(
            {
                "seeds": list(seeds),
                "retrieved": list(retrieved),
                "new_entities": list(new_entities),
            }
        )

    def summary(self) -> dict:
        return {
            "rounds": len(self.rounds),
            "total_retrieved": sum(len(r["retrieved"]) for r in self.rounds),
            "entities_discovered": sum(len(r["new_entities"]) for r in self.rounds),
        }


@dataclass
class IterativeRetriever:
    """반복 검색. 감싼 리트리버를 라운드마다 다시 호출하고 결과를 누적한다.

    `graph`는 시드 기반 확장을 위해 필요하고, `hybrid`는 어휘 신호를 섞기 위해 쓴다.
    둘 다 이미 검증된 기존 구현을 그대로 재사용한다(새 점수 체계를 만들지 않는다).
    """

    chunks: list[Chunk]
    graph: GraphRetriever
    hybrid: HybridRetriever | None = None
    rounds: int = 2
    per_round_k: int = 3
    graph_weight: float = 0.5
    # 다음 라운드의 시드를 **상위 몇 개 청크에서** 뽑을지. 아래 '두 번 고친 기록' ② 참조.
    expand_from_top: int = 1

    # ------------------------------------------------------------------
    # 두 번 고친 기록 (2026-07-30) — 첫 구현은 라운드를 늘려도 성능이 **전혀** 안 올랐다
    # (멀티홉 full_hit 0.1842가 rounds 1·2·3·4에서 소수점까지 동일).
    #
    # ① 점수 감쇠(decay)가 2라운드의 성과를 버리고 있었다.
    #    추적을 찍어 보니 반복 자체는 **작동하고 있었다.** 예를 들어
    #    "감속기에 연결되는 부품을 포함하는 상위 항목은?"에서
    #      R1 → drive#p2 (gold1 적중) · R2 → drive#p1 (**gold2 적중**)
    #    인데 최종 결과는 [p2, p3, p4]로 R2가 찾은 p1이 빠졌다. 라운드마다 점수에 0.6을
    #    곱하다 보니, 2라운드가 찾은 **정답**이 1라운드의 **오답**보다 낮은 점수를 받은 것이다.
    #    → 점수 감쇠를 버리고 **라운드 인터리브**로 바꿨다. 각 라운드에서 신규 후보를
    #      번갈아 한 개씩 채운다. 멀티홉은 홉마다 근거가 하나씩 필요하므로 이 구조가 문제에 맞다.
    #    교훈: "늦게 찾은 건 덜 중요하다"는 그럴듯한 가정이었지만, 멀티홉에서는 정확히 틀렸다.
    #    **뒤 라운드에서만 찾을 수 있는 근거가 있다는 게 애초에 반복을 도입한 이유였다.**
    #
    # ② 시드가 오염돼 2라운드 검색 자체가 나빴다. 처음엔 1라운드 상위 k개 청크의 엔티티를
    #    **전부** 다음 시드로 썼는데, 그 안에는 오답 청크의 엔티티도 섞여 있다. 위 예에서
    #    R2가 배터리 문서를 끌어온 게 그 탓이다. → `expand_from_top`으로 **가장 신뢰되는
    #    상위 1개 청크**에서만 시드를 얻는다. 반복 검색은 앞 라운드의 정확도에 기대는 구조라,
    #    시드를 넓게 잡으면 노이즈가 라운드마다 증폭된다.
    #
    # ③ ①②를 고쳐도 수치가 **여전히 0.1842 그대로**였다. 추적을 다시 찍어 보니 진짜 원인은
    #    따로 있었다. 위 예의 1라운드 1위 문서 [동력 전달 경로]는 관계를 **두 개** 담고 있다:
    #      (구동 모터)-연결-(감속기)  ← 질의가 따라가야 할 가지
    #      (인버터)-연결-(고전압 배터리팩)  ← 같은 페이지에 있을 뿐 무관한 가지
    #    그런데 나는 청크의 **모든 엔티티**를 다음 시드로 넘기고 있었다. 그래서 '고전압 배터리팩'이
    #    딸려가 2라운드가 배터리 문서로 새 버렸다.
    #    → 다음 시드를 **현재 시드가 참여한 관계의 반대편 끝점**으로 제한했다.
    #    교훈: 청크 단위로 엔티티를 긁는 건 그래프 탐색이 아니라 **동시출현**이다. 문서에 같이
    #    있다는 것과 관계로 이어져 있다는 건 다르다 — 이건 이 프로젝트가 그래프 점수 함수에서
    #    한 번 배운 교훈(언급 vs 관계)과 정확히 같은 실수였고, 확장 쪽에서 다시 반복한 것이다.
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds >= 1")
        self._index = {c.chunk_id: c for c in self.chunks}

    def _round_scores(self, query: str, seeds: list[str] | None) -> dict[str, float]:
        g = _normalize(self.graph.score(query, seeds))
        if self.hybrid is None:
            return g
        h = _normalize(self.hybrid.score(query))
        keys = set(g) | set(h)
        w = self.graph_weight
        return {k: (1 - w) * h.get(k, 0.0) + w * g.get(k, 0.0) for k in keys}

    def retrieve_with_trace(
        self, query: str, top_k: int, seeds: list[str] | None = None
    ) -> tuple[list[str], RetrievalTrace]:
        trace = RetrievalTrace()
        current = list(seeds) if seeds is not None else link_entities(query)
        seen_entities = set(current)
        per_round: list[list[str]] = []

        for _ in range(self.rounds):
            scores = self._round_scores(query, current if current else None)
            picked = _rank(scores, self.per_round_k)
            per_round.append(picked)

            # 이번 라운드에서 읽은 문서가 다음 라운드의 질의가 된다.
            # 단 **가장 신뢰되는 상위 몇 개**에서, 그리고 **현재 시드가 참여한 관계의
            # 반대편 끝점**만 — 위 ②·③ 참조.
            found: list[str] = []
            frontier = set(current)
            for cid in picked[: self.expand_from_top]:
                for rel in self._index[cid].states:
                    for a, b in ((rel.head, rel.tail), (rel.tail, rel.head)):
                        if a in frontier and b not in seen_entities:
                            seen_entities.add(b)
                            found.append(b)

            trace.add(current, picked, found)
            if not found:
                break  # 새로 배운 게 없으면 더 돌아도 같다
            current = found

        return _interleave(per_round, top_k), trace

    def retrieve(self, query: str, top_k: int, seeds: list[str] | None = None) -> list[str]:
        return self.retrieve_with_trace(query, top_k, seeds)[0]


def _interleave(per_round: list[list[str]], top_k: int) -> list[str]:
    """라운드별 결과를 번갈아 하나씩 뽑아 합친다(라운드 로빈). 순수·결정적.

    점수로 한 줄 세우기를 하지 않는 이유: 라운드마다 점수 체계의 기준(시드)이 달라서
    서로 비교할 수 없다. 대신 **각 라운드가 top_k 안에 자기 몫을 갖도록** 보장한다.
    멀티홉 질의는 홉마다 근거가 하나씩 필요하므로 이 배분이 문제 구조와 맞는다.

    1라운드 결과가 먼저 채워지므로 단일홉(=1라운드로 충분한 질의)의 순위는 그대로 보존된다.
    """
    out: list[str] = []
    seen: set[str] = set()
    depth = max((len(r) for r in per_round), default=0)
    for i in range(depth):
        for rnd in per_round:
            if i < len(rnd) and rnd[i] not in seen:
                seen.add(rnd[i])
                out.append(rnd[i])
                if len(out) == top_k:
                    return out
    return out


def sweep_rounds(
    build, queries, k: int, max_rounds: int = 4
) -> list[dict]:
    """라운드 예산을 늘리며 성능을 잰다 — "에이전트를 붙였다"의 검증.

    `build(rounds)`는 그 예산의 리트리버를 만들어 주는 함수, `queries`는 평가 질의다.
    예산을 늘려도 안 오르면 반복이 실제로는 일을 안 하고 있다는 뜻이므로, 이 곡선이
    주장의 근거가 된다. 반환은 라운드별 full_hit과 평균 라운드 수.
    """
    from .metrics import full_hit_at_k

    out: list[dict] = []
    for r in range(1, max_rounds + 1):
        ret = build(r)
        hits, used = 0.0, 0
        for q in queries:
            got, tr = ret.retrieve_with_trace(q.question, k, list(q.seed_entities))
            hits += full_hit_at_k(got, q.gold_chunks, k)
            used += len(tr.rounds)
        out.append(
            {
                "rounds": r,
                "full_hit": round(hits / len(queries), 4) if queries else 0.0,
                "avg_rounds_used": round(used / len(queries), 2) if queries else 0.0,
            }
        )
    return out
