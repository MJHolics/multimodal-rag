"""검색 전략 — 플랫 하이브리드(현행) vs 그래프 확장(GraphRAG) vs 융합.

## 비교 대상

1. `HybridRetriever` — **현행 파이프라인과 동일**: BGE-M3 dense 70% + BM25 30%
   (`app/retriever.py`의 가중치를 그대로 가져왔다. 다른 값을 쓰면 비교가 아니라 다른 실험이 된다.)
2. `GraphRetriever` — 질의에서 엔티티를 찾아 그래프를 확장하고, 확장된 이웃을 서술하는 청크를 올린다.
3. `FusedRetriever` — 1과 2의 점수를 결합. 실무에서 쓸 법한 형태.

## 시드 엔티티: oracle vs linked

그래프 검색은 "질의 → 시작 엔티티"가 먼저 필요하다. 이 단계 품질이 섞이면
*그래프 탐색 자체*의 이득을 못 잰다. 그래서 두 모드를 둔다:

  - `given`: 평가셋이 지정한 **출발 엔티티 1개**만 시드로 쓴다 → 최소 앵커에서의 탐색력.
  - `linked`: 질의 문자열에서 엔티티명을 매칭해 찾는다 → **엔드투엔드**(링킹 실패 포함).

주의: `given`은 상한이 아니다. 실측(2026-07-27) 결과 `linked`가 `given`보다 **높게** 나왔는데,
질문에 엔티티가 여러 개 언급되면 링킹이 앵커를 더 많이 찾아 확장 범위가 넓어지기 때문이다.
즉 두 모드는 "정답 시드 vs 추정 시드"가 아니라 **"단일 앵커 vs 다중 앵커"** 차이로 읽어야 한다.
(처음엔 given을 oracle=상한으로 불렀는데, 수치가 뒤집혀서 명명을 고쳤다.)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from .corpus import ENTITY_BY_ID, Chunk
from .graph_store import GraphStore, InMemoryGraphStore

# 배합비는 2026-08-10에 `run_hybrid_weight.py`로 재고 정했다.
# 그 전에는 0.7/0.3이 근거 없이 코드에 박혀 있었다(그래프 가중치·agentic 라운드는 dev에서
# 골랐으면서 정작 검색의 토대만 안 잰 상태였다). 실문서 RFP 코퍼스(7,863청크·134문항)에서
# dev 스윕 최적이 0.9였고, 한 번도 안 본 test에서 full_hit 0.627→0.731(+10.5%p),
# McNemar p=0.0156(7건 개선·0건 악화)이라 기본값을 옮겼다.
# 합성 코퍼스(test 8문항)는 불일치 쌍이 1개뿐이라 판정 자체가 불가능하다 — 근거는 실문서 쪽이다.
DENSE_WEIGHT = 0.9  # app/retriever.py와 동일
BM25_WEIGHT = 0.1


def _tok(text: str) -> list[str]:
    """app/retriever.py의 토크나이저와 동일 규칙(한글 2자 이상 + 영숫자)."""
    return re.findall(r"[a-z0-9]+|[가-힣]{2,}", text.lower())


# ---------------------------------------------------------------------------
# 1) 플랫 하이브리드 — 현행 파이프라인 재현
# ---------------------------------------------------------------------------


class HybridRetriever:
    """BGE-M3 dense + BM25 하이브리드. 임베더는 주입식(테스트에서 스텁 가능)."""

    def __init__(self, chunks: list[Chunk], embedder=None,
                 dense_weight: float = DENSE_WEIGHT) -> None:
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.ids = [c.chunk_id for c in chunks]
        # 배합비는 주입 가능하다. 기본값은 모듈 상수이고, run_hybrid_weight.py 가
        # dev 에서 고른 값을 넣어 베이스라인을 튜닝한 상태로 비교할 수 있게 한다.
        # (dense + bm25 = 1 로 두므로 w 하나로 배합이 정해진다 — 기존 0.7/0.3 과 같은 관계.)
        self.dense_weight = dense_weight
        self._bm25 = BM25Okapi([_tok(c.text) for c in chunks])
        self._embedder = embedder
        self._doc_emb = None
        # 질의 점수 캐시. graph_weight 스윕은 같은 질의를 11×2번 다시 매기는데,
        # 그때마다 BGE-M3로 재인코딩하면 실행 시간이 수십 배가 된다.
        # 점수는 질의 문자열의 결정적 함수이므로 캐시해도 결과가 바뀌지 않는다.
        self._score_cache: dict[str, dict[str, float]] = {}
        self.last_mode: str = "hybrid"
        if embedder is not None:
            self._doc_emb = embedder.encode(
                [c.text for c in chunks], normalize_embeddings=True
            )

    def _dense_scores(self, query: str) -> dict[str, float]:
        if self._embedder is None:
            return {}
        q = self._embedder.encode([query], normalize_embeddings=True)
        sims = (self._doc_emb @ q[0])  # 정규화돼 있으므로 내적 = 코사인
        return {self.ids[i]: float(sims[i]) for i in range(len(self.ids))}

    def _bm25_scores(self, query: str) -> dict[str, float]:
        raw = self._bm25.get_scores(_tok(query))
        mx = max(raw) if len(raw) and max(raw) > 0 else 1.0
        return {self.ids[i]: float(raw[i]) / mx for i in range(len(self.ids))}

    def _keyword_scores(self, query: str) -> dict[str, float]:
        """dense·BM25 둘 다 죽었을 때의 마지막 수단.

        사전 계산된 인덱스(임베딩·BM25 통계) 없이, 메모리에 들고 있는 원문 청크를
        그 자리에서 스캔해 질의 토큰과 겹치는 개수로만 점수를 매긴다. 순위 품질은
        하이브리드보다 확실히 떨어지지만(실측 2026-09-07, 아래 클래스 docstring 참고),
        "완전 무응답"보다는 낫다는 것까지만 보장한다.
        """
        q_tokens = set(_tok(query))
        if not q_tokens:
            return {}
        out: dict[str, float] = {}
        for c in self.chunks:
            overlap = len(q_tokens & set(_tok(c.text)))
            if overlap:
                out[c.chunk_id] = float(overlap)
        return out

    def score(self, query: str) -> dict[str, float]:
        cached = self._score_cache.get(query)
        if cached is not None:
            return cached

        dense: dict[str, float] = {}
        dense_ok = True
        try:
            dense = self._dense_scores(query)
        except Exception as e:  # 임베딩 서비스/모델 장애
            logger.warning("[HybridRetriever] dense 실패, 축소 운영: %s", e)
            dense_ok = False

        bm25: dict[str, float] = {}
        bm25_ok = True
        try:
            bm25 = self._bm25_scores(query)
        except Exception as e:  # BM25 인덱스 손상/미로드
            logger.warning("[HybridRetriever] BM25 실패, 축소 운영: %s", e)
            bm25_ok = False

        if not dense_ok and not bm25_ok:
            out = self._keyword_scores(query)
            mode = "keyword_fallback"
        elif not dense:  # 임베더 미주입(구조, 스텁 테스트) 또는 dense 결과 없음 → BM25 단독
            out = bm25
            mode = "bm25_only"
        elif not bm25_ok:  # BM25만 죽음 → dense 단독
            out = dense
            mode = "dense_only"
        else:
            w = self.dense_weight
            out = {
                cid: w * dense.get(cid, 0.0)
                + (1.0 - w) * bm25.get(cid, 0.0)
                for cid in self.ids
            }
            mode = "hybrid"

        self._score_cache[query] = out
        self.last_mode = mode
        return out

    def retrieve(self, query: str, top_k: int) -> list[str]:
        return _rank(self.score(query), top_k)


# ---------------------------------------------------------------------------
# 2) 그래프 확장
# ---------------------------------------------------------------------------


def link_entities(query: str) -> list[str]:
    """질의에서 엔티티를 찾는다 — 표면형 문자열 매칭(결정적).

    합성 코퍼스는 표기가 일관되므로 이 단순 규칙으로 충분하다.
    실 문서에선 별칭·약어·오탈자 정규화가 필요하며, 그 난이도는 여기 반영돼 있지 않다.
    긴 이름부터 매칭해 부분 포함(예: '셀 모듈' ⊃ '셀')로 인한 오탐을 줄인다.
    """
    found: list[str] = []
    for e in sorted(ENTITY_BY_ID.values(), key=lambda x: -len(x.name)):
        if e.name in query and e.eid not in found:
            # 이미 매칭된 더 긴 이름에 포함되는 경우는 건너뛴다
            if any(e.name in ENTITY_BY_ID[f].name for f in found):
                continue
            found.append(e.eid)
    return found


@dataclass
class GraphRetriever:
    """시드 엔티티에서 그래프를 확장하고, 확장 결과를 서술하는 청크를 점수화한다.

    ## 점수 방식 두 가지 (`mode`)

    - `mention`(기본, 1차 실험에서 쓴 것): 청크가 담은 엔티티 중 시드에서 가장 가까운 것의
      거리 d에 대해 1/(1+d) + 겹치는 엔티티 수 소량 가산.
    - `relation`: 청크가 **서술하는 관계** 중 양 끝점이 모두 확장 집합에 있는 것만 세어 점수화.

    ## 왜 두 번째가 생겼나 (2026-07-29)

    코퍼스에 distractor(주의사항·정비절차·구형 사양 — 엔티티는 언급하나 답은 아닌 청크)를
    넣자 `mention` 점수가 무너졌다. "고전양 배터리팩 취급 시 절연 장갑을 착용한다"는 시드
    엔티티를 직접 언급하므로 거리 0 → 점수 1.0으로 **정답 청크와 동점**이 된다.
    즉 이 점수는 *관계를 서술하는가*가 아니라 *엔티티를 언급하는가*를 재고 있었다.
    distractor가 없던 1차 코퍼스에서는 모든 청크가 관계를 하나씩 갖고 있었기 때문에
    두 기준이 우연히 같았고, 그래서 결함이 드러나지 않았다.

    ## `relation` 모드가 공짜로 얻는 것 — 정직하게

    이 점수는 "청크 → 관계" 추출(IE)이 **이미 정확히 돼 있다**고 가정한다. 합성 코퍼스는
    생성 규칙상 그 정보를 공짜로 준다. 실문서라면 별도의 관계 추출 파이프라인이 필요하고
    그 품질이 그대로 상한이 된다. 특히 구형 사양 distractor는 실제 IE라면 "(구형 팩, 전압,
    350V)" 같은 관계를 뽑아낼 텐데, 여기서는 그런 노드를 그래프에 넣지 않았으므로
    자동으로 걸러진다. **즉 relation 모드의 이득에는 IE를 상한으로 가정한 몫이 섞여 있다.**
    """

    chunks: list[Chunk]
    store: GraphStore = None  # type: ignore[assignment]
    max_hops: int = 2
    mode: str = "mention"

    def __post_init__(self) -> None:
        if self.store is None:
            self.store = InMemoryGraphStore()
        if self.mode not in ("mention", "relation"):
            raise ValueError("mode must be 'mention' or 'relation'")

    def score(self, query: str, seeds: list[str] | None = None) -> dict[str, float]:
        seed_ids = seeds if seeds is not None else link_entities(query)
        if not seed_ids:
            return {}
        dist = self.store.expand(seed_ids, self.max_hops)
        if self.mode == "relation":
            return self._score_relation(dist)
        out: dict[str, float] = {}
        for c in self.chunks:
            ds = [dist[e] for e in c.entities if e in dist]
            if not ds:
                continue
            closest = min(ds)
            out[c.chunk_id] = 1.0 / (1.0 + closest) + 0.01 * len(ds)
        return out

    def _score_relation(self, dist: dict[str, int]) -> dict[str, float]:
        """확장 집합 안에서 **양 끝점이 모두 닿는** 관계만 근거로 인정한다.

        관계를 하나도 서술하지 않는 청크(=distractor)는 점수가 나오지 않는다.
        여러 근거를 담은 청크가 위로 오도록 관계별 점수를 더한다.
        """
        out: dict[str, float] = {}
        for c in self.chunks:
            total = 0.0
            for r in c.states:
                if r.head in dist and r.tail in dist:
                    total += 1.0 / (1.0 + min(dist[r.head], dist[r.tail]))
            if total > 0:
                out[c.chunk_id] = total
        return out

    def retrieve(self, query: str, top_k: int, seeds: list[str] | None = None) -> list[str]:
        return _rank(self.score(query, seeds), top_k)


# ---------------------------------------------------------------------------
# 3) 융합
# ---------------------------------------------------------------------------


@dataclass
class FusedRetriever:
    """하이브리드 점수와 그래프 점수를 가중 합. graph_weight는 실측으로 정할 대상."""

    hybrid: HybridRetriever
    graph: GraphRetriever
    graph_weight: float = 0.5

    def retrieve(self, query: str, top_k: int, seeds: list[str] | None = None) -> list[str]:
        h = self.hybrid.score(query)
        g = self.graph.score(query, seeds)
        h = _normalize(h)
        g = _normalize(g)
        keys = set(h) | set(g)
        fused = {
            k: (1 - self.graph_weight) * h.get(k, 0.0) + self.graph_weight * g.get(k, 0.0)
            for k in keys
        }
        return _rank(fused, top_k)


# ---------------------------------------------------------------------------


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """최댓값 1로 스케일. 두 점수 체계의 단위가 달라 그대로 더하면 한쪽이 지배한다."""
    if not scores:
        return {}
    mx = max(scores.values())
    if mx <= 0:
        return {k: 0.0 for k in scores}
    return {k: v / mx for k, v in scores.items()}


def _rank(scores: dict[str, float], top_k: int) -> list[str]:
    """점수 내림차순, 동점은 chunk_id 사전순으로 끊는다(결정적 재현)."""
    return [
        cid for cid, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    ]
