"""정답을 아는 환각 셋·금액 정규화·숫자 대조 기준선 — 순수 단위테스트(코퍼스·LLM 불필요)."""
import random

from ragas_eval.injected import fmt_korean_units, make_items, numbers_in, parse_won
from ragas_eval.normalize import normalize_amounts
from run_ragas_judge_bench import auroc, lexical_score

GOLD = "사업예산 : 금243,000,000원(금이억사천삼백만 원) (VAT 포함)"


def test_korean_units_roundtrip():
    for n in (243_000_000, 50_000_000, 900_000_000, 1_247_000_000, 157_300_000):
        assert normalize_amounts(fmt_korean_units(n)) == f"{n:,}원"


def test_normalize_leaves_digit_amounts_alone():
    assert normalize_amounts(GOLD) == GOLD


def test_parse_won():
    assert parse_won("243,000,000원") == 243_000_000
    assert parse_won("미정") is None


def test_items_labels_and_hallucination_absent_from_context():
    items = make_items("p", "가 사업", 243_000_000, GOLD, ["다른 문서 1", "다른 문서 2"],
                       "없는기관", "dev", random.Random(0))
    by = {i.variant: i for i in items}
    assert [by[v].hallucinated for v in ("F-exact", "F-para", "H-number", "H-extra")] == [0, 0, 1, 1]
    ctx_nums = numbers_in("\n".join(by["H-number"].contexts))
    assert not (numbers_in(by["H-number"].answer) - {"243000000"}) <= ctx_nums  # 바꾼 금액은 컨텍스트에 없다
    assert "없는기관" not in "\n".join(by["H-extra"].contexts)


def test_lexical_baseline_shape():
    ctx = [GOLD]
    assert lexical_score("예산은 243,000,000원입니다.", ctx) == 1.0
    assert lexical_score("예산은 2억 4,300만 원입니다.", ctx) < 1.0   # 표기만 달라도 놓친다(오경보)
    assert lexical_score("예산은 243,000,000원, 발주는 A기관입니다.", ctx) == 1.0  # 숫자 없는 환각은 못 본다


def test_auroc():
    assert auroc([0.0, 0.0], [1.0, 1.0]) == 1.0
    assert auroc([1.0], [1.0]) == 0.5
