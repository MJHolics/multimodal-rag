"""RAGAS 평가 예시셋 — (질문, 검색 컨텍스트, 컨텍스트 관련성, 생성답, 정답).

실제로는 RAG 파이프라인(retriever+generator)의 출력을 채워 평가한다. 여기 예시는 지표 동작을
보이기 위한 소규모 셋(EV 배터리 매뉴얼 가정). context_relevance는 gold 관련성 라벨.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RagSample:
    question: str
    contexts: list[str]              # 검색된 컨텍스트(순서=검색 순위)
    context_relevance: list[int]     # 각 컨텍스트 관련성(1/0) — context_precision용 gold
    answer: str                      # 생성된 답
    ground_truth: str                # 정답
    relevant_ids: list[int] = field(default_factory=list)


SAMPLES: list[RagSample] = [
    RagSample(
        question="배터리 팩 공칭 전압은?",
        contexts=[
            "배터리 팩 공칭 전압은 400V이다.",
            "충전은 0~40도에서 권장된다.",
            "셀 화학은 NMC를 사용한다.",
        ],
        context_relevance=[1, 0, 0],
        answer="배터리 팩의 공칭 전압은 400V입니다.",
        ground_truth="공칭 전압은 400V이다.",
    ),
    RagSample(
        question="권장 충전 온도 범위는?",
        contexts=[
            "충전은 0~40도에서 권장된다.",
            "배터리 팩 공칭 전압은 400V이다.",
        ],
        context_relevance=[1, 0],
        answer="권장 충전 온도는 0도에서 40도 사이입니다.",
        ground_truth="충전 권장 온도는 0~40도이다.",
    ),
]
