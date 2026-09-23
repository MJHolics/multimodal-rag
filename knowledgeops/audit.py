"""IE 감사 — "정밀도를 곱한 상한"을 실측으로 바꾼다.

## 이 모듈이 생긴 이유

5차(실문서 재측정)는 모든 수치에 단서를 달고 끝났다: **"관계가 IE의 산물이라 gold도
IE에 의존한다. 이 수치는 IE 정밀도를 곱한 상한으로 읽어야 한다."**
NOTES가 남긴 후속 과제는 "표본 감사를 20건에서 늘려 오탐률에 신뢰구간을 붙이면
상한이 아니라 구간으로 말할 수 있다"였다.

**그 계획을 그대로 따르지 않았다.** 평가 문항이 어느 관계에서 유도됐는지 세어 보니
전제가 틀려 있었기 때문이다.

| 관계 | 모집단 | 유도된 문항 |
|---|---:|---|
| HAS_BUDGET | 49 | 1-hop 49 + 2-hop 14 |
| HAS_PERIOD | 71 | 1-hop 69 + 2-hop 39 |
| REQUIRES | 853 | 2-hop 5 |
| ISSUED_BY | 96 | (2-hop의 다리) |

평가에 실제로 쓰인 속성 관계는 **120개뿐**이다. 853개짜리 REQUIRES는 176문항 중
5문항에만 관여한다. 즉 **표본을 뽑아 구간을 추정할 필요가 없다 — 전수 감사가 된다.**
전수를 세면 정밀도는 추정량이 아니라 상수이고, 더 중요하게는 *어느 관계가 오탐인지*를
알게 되므로 그 관계에서 유도된 문항을 **골라내 다시 집계**할 수 있다.

> 원칙: 구간은 전수를 셀 수 없을 때 쓰는 도구다. 셀 수 있으면 세는 게 낫다.
> 모집단 크기를 먼저 보지 않고 "표본 늘려 신뢰구간"으로 직행하면,
> 답을 정확히 알 수 있는 문제에 추정을 쓰게 된다.

REQUIRES처럼 전수가 무리인 것만 표본 + Wilson 구간으로 간다. `plan_audit`이 그 분기를
자동으로 판정하고, 어느 쪽을 골랐는지 근거와 함께 남긴다.

## 순환 주의

IE 결과를 IE로 검증하면 순환이다. 그래서 판정은 **원문 근거 문장**을 읽고 내리며,
판정 결과는 `output/audit_labels.json`에 근거와 함께 남겨 나중에 누구든 다시 볼 수 있게 한다.
"""
from __future__ import annotations

from dataclasses import dataclass

from graph_rag.stats import wilson_interval

# 판정 라벨
OK = "ok"  # 관계가 원문 근거와 맞다
BAD = "bad"  # 오탐 — 숫자·단어는 맞아도 문맥이 다르다


@dataclass(frozen=True)
class AuditPlan:
    """어떻게 감사할지에 대한 결정과 그 근거."""

    rtype: str
    population: int
    mode: str  # "census" | "sample"
    k: int  # 실제로 볼 건수
    reason: str

    @property
    def is_census(self) -> bool:
        return self.mode == "census"


def plan_audit(rtype: str, population: int, budget: int = 150) -> AuditPlan:
    """전수로 셀지 표본을 뽑을지 정한다.

    `budget`은 사람이 근거를 읽고 판정할 수 있는 현실적 상한이다.
    모집단이 그 안이면 전수로 간다 — 표본 오차가 0이고, 개별 오탐을 특정할 수 있다.
    """
    if population <= budget:
        return AuditPlan(
            rtype=rtype,
            population=population,
            mode="census",
            k=population,
            reason=f"모집단 {population} ≤ 예산 {budget} — 전수를 세면 추정이 아니라 상수가 된다",
        )
    return AuditPlan(
        rtype=rtype,
        population=population,
        mode="sample",
        k=budget,
        reason=f"모집단 {population} > 예산 {budget} — 표본 {budget}건에 Wilson 구간을 붙인다",
    )


def systematic_sample(rows: list[dict], k: int) -> list[dict]:
    """정렬 후 균등 간격 추출. 난수를 안 쓰므로 재실행하면 같은 표본이 나온다.

    문서 순서에 주기성이 있으면 계통추출은 편향될 수 있다. 여기서는 정렬 키가
    문서 id라 내용과 무관하고, 재현성을 우선했다.
    """
    if k >= len(rows):
        return list(rows)
    step = len(rows) / k
    return [rows[int(i * step)] for i in range(k)]


def precision(labels: list[str], population: int, mode: str) -> dict:
    """감사 라벨 → 정밀도. 전수면 구간을 만들지 않는다.

    전수 감사에서 Wilson 구간을 붙이는 건 통계적으로 틀렸다. 구간은 모집단을
    표본으로 추정할 때의 불확실성인데, 전수는 추정이 아니다.
    """
    n = len(labels)
    ok = sum(1 for x in labels if x == OK)
    point = ok / n if n else 0.0
    out = {
        "mode": mode,
        "population": population,
        "audited": n,
        "ok": ok,
        "bad": n - ok,
        "precision": round(point, 4),
    }
    if mode == "census":
        out["interval"] = None
        out["note"] = "전수 감사 — 표본 오차 없음. 구간을 붙이지 않는다."
    else:
        lo, hi = wilson_interval(ok, n)
        out["interval"] = [round(lo, 4), round(hi, 4)]
        out["note"] = f"표본 {n}/{population} — Wilson 95% 구간"
    return out


def invalid_triples(records: list[dict]) -> set[str]:
    """오탐으로 판정된 트리플 키 집합."""
    return {r["key"] for r in records if r.get("label") == BAD}


def purify(per_query: list[dict], query_sources: dict[str, list[str]],
           invalid: set[str]) -> dict:
    """오탐 관계에서 유도된 문항을 골라낸다.

    `query_sources[qid]` = 그 문항이 어느 트리플에서 나왔는지(2-hop은 다리 포함).
    하나라도 오탐이면 그 문항의 gold는 틀린 것이므로 집계에서 뺀다.
    """
    kept, dropped = [], []
    for row in per_query:
        srcs = query_sources.get(row["qid"], [])
        if any(s in invalid for s in srcs):
            dropped.append(row)
        else:
            kept.append(row)
    return {"kept": kept, "dropped": dropped}


def score(per_query: list[dict], strategy: str, hops: int | None = None) -> dict:
    """문항별 정오 기록에서 full_hit를 다시 집계한다(검색 재실행 없이)."""
    rows = [r for r in per_query if hops is None or r["hops"] == hops]
    n = len(rows)
    k = sum(1 for r in rows if r["correct"].get(strategy))
    return {"n": n, "correct": k, "full_hit": round(k / n, 4) if n else 0.0}


def compare_before_after(before: list[dict], after: list[dict], strategy: str,
                         hops: int | None = None) -> dict:
    """정화 전후 지표. 차이가 크면 결론이 오탐 위에 서 있었다는 뜻이다."""
    b = score(before, strategy, hops)
    a = score(after, strategy, hops)
    return {
        "strategy": strategy,
        "hops": hops,
        "before": b,
        "after": a,
        "delta": round(a["full_hit"] - b["full_hit"], 4),
        "dropped": b["n"] - a["n"],
    }
