"""실문서 인제스천·IE의 불변식 — 순수·결정적, HWP 원본 없이도 도는 부분만.

실제 HWP 96건 처리는 `build_rfp_graph.py`가 하고, 여기서는 **규칙의 성질**을 고정한다.
특히 표본 검증에서 드러난 오탐 유형을 회귀 테스트로 못박는다 — 그 문장들이 실제
공공조달 RFP에서 그대로 나온 것이라, 규칙을 손대다 다시 열리면 여기서 걸린다.
"""
from __future__ import annotations

from knowledgeops.extract import (
    HAS_BUDGET,
    ISSUED_BY,
    REQUIRES,
    build_document,
    extract_budget,
    extract_period,
    extract_tech,
    extract_tech_with_evidence,
    graph_summary,
    parse_filename,
)
from knowledgeops.hwp import clean

# ---- 파일명 = 1급 신호 ----


def test_parse_filename_splits_agency_and_project():
    a, p = parse_filename("(사)벤처기업협회_2024년 벤처확인종합관리시스템 기능 고도화 용역사업 .hwp")
    assert a == "(사)벤처기업협회"
    assert p.startswith("2024년 벤처확인종합관리시스템")


def test_parse_filename_normalizes_fullwidth_parens():
    """실제 파일명에 전각 괄호가 섞여 있다 — 정규화 안 하면 기관명이 갈라진다."""
    a, _ = parse_filename("(사）한국대학스포츠협의회_KUSF 체육특기자 경기기록 관리시스템 개발.hwp")
    assert a == "(사)한국대학스포츠협의회"


def test_parse_filename_without_convention():
    a, p = parse_filename("규약을안지킨파일.hwp")
    assert a == ""
    assert p == "규약을안지킨파일"


# ---- 예산: 라벨 근접 + 부정 문맥 배제 ----


def test_budget_accepts_labelled_amount():
    text = "다. 사업비 : 157,300,000원(VAT 포함) (단위: 원)"
    got = extract_budget(text)
    assert got and got[0].startswith("157,300,000")


def test_budget_rejects_statutory_threshold():
    """"사업금액 20억원 미만 계약은 대기업 참여 제한" — 법령 기준액이지 이 사업 예산이 아니다.

    실제 RFP에서 나온 문장이고, 가드 전에는 이게 예산으로 잡혔다.
    """
    text = "「소프트웨어 진흥법」제48조제2항에 따라, 사업금액 20억원 미만 계약은 대기업 참여 제한"
    assert extract_budget(text) is None


def test_budget_rejects_example_amount():
    text = "변경 소요 비용 소요비용 예시) 1,046 백만원 변경규모 예시) 2,011 FP"
    assert extract_budget(text) is None


def test_budget_rejects_committee_threshold_table():
    text = "구매요구액 5천5백만원 미만 5천5백만원 이상 ∼ 1억원 미만 위원 수 4명 이상"
    assert extract_budget(text) is None


def test_budget_requires_label_nearby():
    """라벨 없이 금액만 있으면 이 사업의 예산이라고 볼 근거가 없다."""
    assert extract_budget("계약 상대자는 123,456,000원을 참고한다" * 1) is None


# ---- 기술: 부정 문맥 배제 ----


def test_tech_rejects_binding_spring():
    """"제안서는 제본 및 스프링 사용" — 프레임워크가 아니라 제본용 스프링(동음이의)."""
    text = "규격 : A4, 제안서 및 요약본은 제본 및 스프링 사용, 표지 : 백색"
    assert "Spring" not in extract_tech(text)


def test_tech_rejects_negated_requirement():
    text = "사업 특성상 반응형 웹을 통한 모바일 사용자의 접근 수요는 적음"
    assert "반응형웹" not in extract_tech(text)


def test_tech_rejects_security_clause_mention():
    text = "비공개 항공사진·공간정보 등 비공개 정보 유출"
    assert "GIS" not in extract_tech(text)


def test_tech_accepts_real_requirement():
    text = "개발프레임워크 : 전자정부 표준프레임워크 적용, DBMS Oracle 12C"
    got = extract_tech(text)
    assert "Spring" in got and "Oracle" in got


def test_tech_positive_mention_survives_negative_elsewhere():
    """긴 문서에는 두 문맥이 공존한다 — 한 번 부정이 나왔다고 버리면 재현율이 무너진다."""
    text = "제안서는 제본 및 스프링 사용. ... 개발프레임워크는 전자정부 표준프레임워크를 적용한다."
    assert "Spring" in extract_tech(text)


def test_evidence_is_the_position_used_for_judgement():
    """감사 근거는 **판정에 쓰인 바로 그 위치**여야 한다.

    처음엔 판정은 '부정 아닌 언급'으로 하고 근거는 '첫 등장'을 보여줘서,
    멀쩡히 걸러진 문맥이 오탐처럼 찍혔다.
    """
    text = "제안서는 제본 및 스프링 사용. 뒤쪽에서 전자정부 표준프레임워크를 적용한다."
    ev = extract_tech_with_evidence(text)
    assert "Spring" in ev
    assert "제본" not in ev["Spring"]
    assert "표준프레임워크" in ev["Spring"]


# ---- 기간 ----


def test_period_patterns():
    assert extract_period("사업기간: 계약일로부터 90일까지") is None  # 일 단위는 대상 아님
    got = extract_period("나. 사업기간: 계약일로부터 5개월 이내")
    assert got and "5" in got[0]


# ---- 문서 → 그래프 ----


def test_build_document_is_deterministic_and_typed():
    text = "다. 사업비 : 157,300,000원(VAT 포함). 개발프레임워크 : 전자정부 표준프레임워크"
    a = build_document("d1", "기관_사업.hwp", text)
    b = build_document("d1", "기관_사업.hwp", text)
    assert [(t.head, t.rtype, t.tail) for t in a.triples] == [
        (t.head, t.rtype, t.tail) for t in b.triples
    ]
    types = {t.rtype for t in a.triples}
    assert ISSUED_BY in types and HAS_BUDGET in types and REQUIRES in types


def test_every_triple_has_evidence():
    """근거 없는 관계를 만들지 않는다 — 사람이 검증할 수 없는 추출은 쓰지 않는다."""
    text = "다. 사업비 : 157,300,000원. DBMS는 Oracle 12C를 사용한다."
    doc = build_document("d1", "기관_사업.hwp", text)
    for t in doc.triples:
        assert doc.evidence.get(f"{t.head}|{t.rtype}|{t.tail}"), t


def test_graph_summary_counts():
    text = "다. 사업비 : 157,300,000원. DBMS는 Oracle 12C."
    docs = [build_document(f"d{i}", f"기관{i}_사업{i}.hwp", text) for i in range(3)]
    s = graph_summary(docs)
    assert s["documents"] == 3
    assert s["unique_agencies"] == 3
    assert s["by_type"][ISSUED_BY] == 3


# ---- HWP 텍스트 정리 ----


def test_clean_collapses_whitespace_and_blank_lines():
    assert clean("가   나\n\n\n다  \n") == "가 나\n다"
