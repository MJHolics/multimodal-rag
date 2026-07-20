"""RAGAS 스타일 지표 — 순수 계산부 단위테스트(네트워크·LLM 불필요)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragas_eval.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    token_f1,
)


def test_token_f1():
    assert token_f1("400V 공칭 전압", "공칭 전압 400V") == 1.0   # 같은 토큰 집합
    assert token_f1("", "공칭 전압") == 0.0
    assert 0.0 < token_f1("공칭 전압은 400V", "공칭 전압") < 1.0  # 부분 겹침


def test_context_precision_rank_weighted():
    assert context_precision([1, 0, 0]) == 1.0          # 관련이 1위 → 완벽
    assert context_precision([0, 0, 0]) == 0.0          # 관련 없음
    # 관련이 2위에만: precision@2 = 1/2 → 0.5
    assert abs(context_precision([0, 1]) - 0.5) < 1e-9
    # [1,0,1]: (1/1 + 2/3)/2 = 0.8333
    assert abs(context_precision([1, 0, 1]) - (1 + 2 / 3) / 2) < 1e-9


def test_context_recall_and_faithfulness():
    assert context_recall([1, 1, 0]) == round(2 / 3, 10) or abs(context_recall([1, 1, 0]) - 2 / 3) < 1e-9
    assert faithfulness([1, 1, 1]) == 1.0
    assert faithfulness([]) == 1.0                      # 주장 없으면 1
    assert abs(faithfulness([1, 0]) - 0.5) < 1e-9


def test_answer_relevancy_cosine():
    q = [1.0, 0.0]
    assert abs(answer_relevancy(q, [[1.0, 0.0]]) - 1.0) < 1e-9     # 동일 방향
    assert abs(answer_relevancy(q, [[0.0, 1.0]]) - 0.0) < 1e-9     # 직교
    assert answer_relevancy(q, []) == 0.0


def test_stub_judge_pipeline_runs():
    """stub judge로 faithfulness/recall 파이프라인이 끝까지 도는지(키 불요)."""
    from ragas_eval.judge import StubJudge, compute_context_recall, compute_faithfulness

    j = StubJudge()
    f = compute_faithfulness("공칭 전압은 400V입니다.", ["공칭 전압은 400V이다."], j)
    cr = compute_context_recall("공칭 전압은 400V이다.", ["공칭 전압은 400V이다."], j)
    assert 0.0 <= f <= 1.0 and 0.0 <= cr <= 1.0
