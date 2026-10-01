"""정답을 아는 환각 셋 — RAGAS faithfulness judge가 환각을 실제로 잡는지 재기 위한 데이터.

faithfulness 점수 자체는 judge(LLM)의 판정이다. judge가 틀리면 점수도 틀리는데, 정답 라벨 없이
점수만 보고서는 judge가 맞았는지 알 수 없다. 그래서 **라벨을 구성으로 보장한** 답변을 만든다.

실문서 RFP의 1-hop 예산 질의(`{사업}의 사업 예산은 얼마인가?`)마다, 정답 청크 + 다른 문서 청크 2개를
컨텍스트로 두고 답변 4종을 만든다.

  F-exact   정답 금액을 원문 표기 그대로            → 충실(faithful)
  F-para    정답 금액을 "2억 4,300만 원"처럼 바꿔 씀 → 충실 (표기만 다르다)
  H-number  금액을 다른 값으로 바꿈                  → 환각 (컨텍스트에 없는 금액)
  H-extra   F-exact + 다른 사업의 발주기관을 덧붙임  → 환각 (컨텍스트에 없는 주장 1개 추가)

순수·결정적(시드 고정). 코퍼스 의존부(`build_from_corpus`)만 pickle을 읽는다.
"""
from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass

VARIANTS = ("F-exact", "F-para", "H-number", "H-extra")


@dataclass
class Item:
    item_id: str
    project_id: str
    split: str                 # dev / test (사업 단위로 나눈다 — 같은 사업이 양쪽에 걸치지 않게)
    variant: str
    hallucinated: int          # 1 = 컨텍스트로 뒷받침 안 되는 주장 포함(구성으로 보장)
    question: str
    answer: str
    contexts: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def parse_won(s: str) -> int | None:
    """'243,000,000원' → 243000000."""
    m = re.search(r"(\d[\d,]*)\s*원", s)
    return int(m.group(1).replace(",", "")) if m else None


def fmt_won(n: int) -> str:
    return f"{n:,}원"


def fmt_korean_units(n: int) -> str:
    """243000000 → '2억 4,300만 원'. 만 원 미만은 버리지 않는다(있으면 붙인다)."""
    eok, rest = divmod(n, 100_000_000)
    man, won = divmod(rest, 10_000)
    parts = []
    if eok:
        parts.append(f"{eok:,}억")
    if man:
        parts.append(f"{man:,}만")
    if won:
        parts.append(f"{won:,}")
    return " ".join(parts) + " 원"


def numbers_in(text: str) -> set[str]:
    """텍스트의 숫자 토큰(쉼표 제거). '금243,000,000원' → {'243000000'}."""
    return {t.replace(",", "") for t in re.findall(r"\d[\d,]*\d|\d", text)}


def altered_amount(n: int, rng: random.Random, forbidden: set[str]) -> int:
    """그럴듯하지만 다른 금액 — 컨텍스트에 숫자로 나오지 않는 값만."""
    for _ in range(50):
        f = rng.choice((0.5, 0.6, 0.75, 1.2, 1.5, 2.0))
        m = int(round(n * f / 1_000_000)) * 1_000_000 or 10_000_000
        if m != n and str(m) not in forbidden:
            return m
    raise ValueError(f"대체 금액을 못 찾음: {n}")


def make_items(project_id: str, project_name: str, amount: int, gold_text: str,
               distractors: list[str], other_agency: str, split: str,
               rng: random.Random) -> list[Item]:
    contexts = [gold_text, *distractors]
    ctx_all = "\n".join(contexts)
    forbidden = numbers_in(ctx_all)
    q = f"{project_name}의 사업 예산은 얼마인가?"
    exact = f"{project_name}의 사업 예산은 {fmt_won(amount)}입니다."
    answers = {
        "F-exact": (0, exact),
        "F-para": (0, f"{project_name}의 사업 예산은 {fmt_korean_units(amount)}입니다."),
        "H-number": (1, f"{project_name}의 사업 예산은 {fmt_won(altered_amount(amount, rng, forbidden))}입니다."),
        "H-extra": (1, f"{exact} 이 사업의 발주기관은 {other_agency}입니다."),
    }
    return [
        Item(f"{project_id}|{v}", project_id, split, v, lab, q, a, contexts)
        for v, (lab, a) in answers.items()
    ]


def build_from_corpus(corpus_path: str, seed: int = 0, n_dev: int = 16) -> list[Item]:
    """실 RFP 코퍼스의 예산 관계에서 셋을 만든다. 금액이 정답 청크에 숫자로 실제 있는 것만 쓴다."""
    import pickle

    from knowledgeops.extract import HAS_BUDGET, ISSUED_BY

    rc = pickle.load(open(corpus_path, "rb"))
    chunk_of = {c.chunk_id: c for c in rc.chunks}
    name = {eid: e.name for eid, e in rc.entities.items()}
    agency_of = {}
    for (h, r, t) in rc.home:
        if r == ISSUED_BY:
            agency_of[h] = name[t]

    cands = []
    for (h, r, t), gold_id in sorted(rc.home.items()):
        if r != HAS_BUDGET:
            continue
        amount = parse_won(name[t])
        gold = chunk_of[gold_id]
        if amount is None or str(amount) not in numbers_in(gold.text):
            continue  # 관계 추출이 틀렸거나 표기가 달라 정답 청크로 금액을 확인할 수 없는 건 뺀다
        cands.append((h, amount, gold))

    rng = random.Random(seed)
    rng.shuffle(cands)
    agencies = sorted(set(agency_of.values()))
    items: list[Item] = []
    for i, (h, amount, gold) in enumerate(cands):
        others = [c for c in rc.chunks if c.doc_name != gold.doc_name]
        distractors = [c.text for c in rng.sample(others, 2)]
        ctx_all = gold.text + "\n" + "\n".join(distractors)
        pool = [a for a in agencies if a != agency_of.get(h) and a not in ctx_all]
        split = "dev" if i < n_dev else "test"
        items += make_items(h, name[h], amount, gold.text, distractors, rng.choice(pool), split, rng)
    return items
