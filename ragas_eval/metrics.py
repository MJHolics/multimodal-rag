"""RAGAS 스타일 지표 — 순수 계산 부분(네트워크·LLM 불필요).

지표 정의(RAGAS와 동일 개념):
  - answer_correctness(token F1) : 생성 답과 정답의 토큰 겹침 F1(사실성 근사).
  - context_precision            : 검색된 컨텍스트 중 관련 항목이 상위에 있는가(순위 가중).
  - context_recall               : 정답을 구성하는 진술이 검색 컨텍스트로 뒷받침되는 비율.
  - faithfulness                 : 생성 답의 주장(claim) 중 컨텍스트로 뒷받침되는 비율.
  - answer_relevancy(cosine)     : 답에서 역생성한 질문이 원 질문과 의미적으로 가까운가.
LLM 판정이 필요한 지표(faithfulness 등)는 여기선 *판정 결과(플래그)*를 받아 집계만 한다 →
판정 로직(judge)과 분리해 순수 단위테스트가 가능하다.
"""
from __future__ import annotations

import math
import re


def _tokens(text: str) -> list[str]:
    return re.findall(r"[0-9a-zA-Z가-힣]+", (text or "").lower())


def token_f1(pred: str, gold: str) -> float:
    """답-정답 토큰 멀티셋 겹침 F1(answer correctness의 사실성 성분)."""
    from collections import Counter

    p, g = Counter(_tokens(pred)), Counter(_tokens(gold))
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    prec = overlap / sum(p.values())
    rec = overlap / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def context_precision(relevances: list[int]) -> float:
    """순위 가중 context precision. relevances[k]=1(관련)/0. 관련 항목 위치의 precision@k 평균.

    관련 컨텍스트가 상위에 모일수록 1에 가깝다. 관련 항목이 없으면 0.
    """
    hits = 0
    acc = 0.0
    for k, r in enumerate(relevances, start=1):
        if r:
            hits += 1
            acc += hits / k  # precision@k (이 위치까지의 관련 비율)
    return acc / hits if hits else 0.0


def context_recall(supported_flags: list[int]) -> float:
    """정답 진술 중 컨텍스트로 뒷받침된 비율. supported_flags=[1/0,...]."""
    n = len(supported_flags)
    return sum(supported_flags) / n if n else 0.0


def faithfulness(claim_supported_flags: list[int]) -> float:
    """생성 답의 주장 중 컨텍스트로 뒷받침된 비율(환각의 역지표). 주장이 없으면 1로 본다."""
    n = len(claim_supported_flags)
    return sum(claim_supported_flags) / n if n else 1.0


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return num / (da * db) if da and db else 0.0


def answer_relevancy(question_vec: list[float], gen_question_vecs: list[list[float]]) -> float:
    """원 질문과 '답에서 역생성한 질문들'의 평균 코사인 유사도(답이 질문에 답했는가)."""
    if not gen_question_vecs:
        return 0.0
    return sum(_cosine(question_vec, v) for v in gen_question_vecs) / len(gen_question_vecs)
