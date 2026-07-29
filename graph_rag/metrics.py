"""검색 지표 — 순수 함수. 멀티홉(gold 복수)을 제대로 다루는 게 핵심.

기존 `03_rag_pipeline.ipynb`의 Hit Rate/MRR는 **gold가 1개**인 단일홉 전제다.
멀티홉 질의는 정답을 구성하는 청크가 2개 이상이고 **둘 다 있어야** 답할 수 있으므로,
"하나만 맞아도 hit"으로 세면 GraphRAG의 이점이 지워지고 플랫 검색이 과대평가된다.
그래서 두 지표를 나눈다:

  - `hit_at_k`      : gold 중 **하나라도** 상위 k에 있으면 1 (관대 — 기존 정의와 호환)
  - `full_hit_at_k` : gold **전부**가 상위 k에 있으면 1 (엄격 — 멀티홉의 실제 답변 가능성)

`full_hit`이 이 실험의 주 지표다. 답을 구성하는 근거가 전부 모여야 생성이 가능하기 때문이다.
MRR은 gold 중 가장 높은 순위를 쓰는 표준 정의를 따른다.
"""
from __future__ import annotations


def hit_at_k(retrieved: list[str], gold: list[str] | tuple[str, ...], k: int) -> float:
    """gold 중 하나라도 상위 k에 있으면 1.0."""
    if not gold:
        raise ValueError("gold가 비어 있다")
    top = retrieved[:k]
    return 1.0 if any(g in top for g in gold) else 0.0


def full_hit_at_k(retrieved: list[str], gold: list[str] | tuple[str, ...], k: int) -> float:
    """gold 전부가 상위 k에 있으면 1.0 — 멀티홉의 '답변 가능' 조건."""
    if not gold:
        raise ValueError("gold가 비어 있다")
    top = set(retrieved[:k])
    return 1.0 if all(g in top for g in gold) else 0.0


def reciprocal_rank(retrieved: list[str], gold: list[str] | tuple[str, ...]) -> float:
    """gold 중 가장 앞선 것의 역순위. 하나도 없으면 0."""
    if not gold:
        raise ValueError("gold가 비어 있다")
    for i, cid in enumerate(retrieved):
        if cid in gold:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(retrieved: list[str], gold: list[str] | tuple[str, ...], k: int) -> float:
    """상위 k가 gold를 얼마나 덮는가(0~1). 멀티홉에서 '절반만 찾음'을 드러낸다."""
    if not gold:
        raise ValueError("gold가 비어 있다")
    top = set(retrieved[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def aggregate(
    results: list[tuple[list[str], tuple[str, ...]]], k: int
) -> dict[str, float]:
    """(검색결과, gold) 목록을 집계. 값은 소수 4자리로 반올림(재현 로그 비교 편의)."""
    n = len(results)
    if n == 0:
        return {"count": 0, "hit": 0.0, "full_hit": 0.0, "mrr": 0.0, "recall": 0.0}
    return {
        "count": n,
        "hit": round(sum(hit_at_k(r, g, k) for r, g in results) / n, 4),
        "full_hit": round(sum(full_hit_at_k(r, g, k) for r, g in results) / n, 4),
        "mrr": round(sum(reciprocal_rank(r, g) for r, g in results) / n, 4),
        "recall": round(sum(recall_at_k(r, g, k) for r, g in results) / n, 4),
    }
