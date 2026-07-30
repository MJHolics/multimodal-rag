"""검색 전략 비교의 통계 — 순수·결정적, 모델·scipy 불필요.

## 왜 필요한가

`run_graph_eval.py`는 전략별 full_hit 집계값만 냈다. 그런데 NOTES.md에 적어둔 결론
"hybrid_flat 0.688 → graph_rel 0.812은 유의하지 않다(p=0.63)"의 **p값을 만든 코드가
이 저장소에 없었다.** 임시로 계산해 문서에만 남긴 수치라 재현이 안 됐다.
집계값만 남기면 개선을 주장할 수도, 반박할 수도 없다 — 그래서 검정을 코드로 내린다.

같은 질의 집합에서 두 전략을 재면 결과는 **짝지어진(paired)** 관측이다.
두 전략이 함께 맞히거나 함께 틀린 질의는 차이에 아무 정보도 주지 않으므로,
McNemar 검정은 불일치 쌍만 본다:

    b = A만 맞힌 질의 수, c = B만 맞힌 질의 수
    귀무가설("두 전략의 실력이 같다") 아래 b ~ Binomial(b+c, 0.5)

질의가 수십 개 규모라 카이제곱 근사 대신 **정확 이항 양측검정**을 쓴다.

구현은 `text2sql-qlora/t2sql/stats.py`와 동일하다(같은 문제를 같은 방식으로 푼다).
두 저장소가 독립이라 공유 패키지를 만들지 않고 이식했다.

## 이 프로젝트에서의 쓰임

정답 벡터는 질의별 `full_hit`(gold 전부가 top-k에 들어왔는가)의 0/1이다.
"검색이 답변 가능한 근거를 다 모았는가"가 이 실험의 성공 정의이기 때문이다.
"""
from __future__ import annotations

import math


def contingency(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """짝지어진 정오 벡터 → 2×2 분할표. a_only/b_only만이 McNemar의 정보다."""
    if len(a_correct) != len(b_correct):
        raise ValueError(f"paired length mismatch: {len(a_correct)} vs {len(b_correct)}")
    both = a_only = b_only = neither = 0
    for a, b in zip(a_correct, b_correct):
        if a and b:
            both += 1
        elif a:
            a_only += 1
        elif b:
            b_only += 1
        else:
            neither += 1
    return {
        "n": len(a_correct),
        "both": both,
        "a_only": a_only,
        "b_only": b_only,
        "neither": neither,
        "discordant": a_only + b_only,
    }


def binom_test_half(k: int, n: int) -> float:
    """P(X가 k만큼 이상 극단 | X~Bin(n, 0.5))의 양측 p값 — 정확 검정.

    대칭 분포라 양측 = 2 × 한쪽 꼬리(1.0으로 절단). n=0이면 근거가 없으므로 p=1.0.
    """
    if n < 0 or not 0 <= k <= max(n, 0):
        raise ValueError("require 0 <= k <= n")
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def mcnemar(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """짝지어진 두 전략의 McNemar 정확검정.

    `delta`는 B−A 정확도 차이(+면 B가 낫다). 불일치 쌍이 0이면 판정 근거가 없다(p=1.0).
    p가 작다고 실무적으로 큰 개선이라는 뜻은 아니다 — delta와 함께 읽는다.
    """
    tab = contingency(a_correct, b_correct)
    b, c = tab["a_only"], tab["b_only"]
    n = tab["n"]
    acc_a = sum(a_correct) / n if n else 0.0
    acc_b = sum(b_correct) / n if n else 0.0
    return {
        **tab,
        "acc_a": round(acc_a, 4),
        "acc_b": round(acc_b, 4),
        "delta": round(acc_b - acc_a, 4),
        "p_value": binom_test_half(min(b, c), b + c),
    }


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """비율의 Wilson 95% 신뢰구간(비대응). 정규근사보다 경계에서 안정적.

    한 전략의 점수를 단독 보고할 때 쓴다. 두 전략 비교에는 mcnemar를 쓸 것 —
    비대응 구간은 짝지음 정보를 버려 보수적으로(넓게) 나온다.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def min_detectable_discordant(p_target: float = 0.05) -> int:
    """"유의"가 나오려면 불일치 쌍이 최소 몇 개여야 하는가 — 검정력의 하한.

    한쪽으로 완전히 쏠린 최선의 경우(b=0)조차 p = 2 / 2^n = 2^(1-n)이다.
    n=4면 p=0.125로 어떤 결과가 나와도 0.05를 못 넘는다. 즉 **불일치 쌍이 6개 미만이면
    유의 판정 자체가 불가능**하다. 평가셋을 키운 이유가 이 숫자다.
    """
    n = 0
    while n < 100:
        n += 1
        if 2.0 ** (1 - n) <= p_target:
            return n
    return n
