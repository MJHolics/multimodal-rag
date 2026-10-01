"""judge 앞단 정규화 — LLM judge가 못 하는 단위 환산을 규칙으로 먼저 해 둔다.

dev에서 7B judge의 오경보 대부분이 "8,000만 원"과 "80,000,000원"을 같은 금액으로 보지 못한 것이었다.
환산은 결정적인 일이라 LLM에 맡길 이유가 없다. 금액 표기("N억 M만 K 원")만 다룬다 — 날짜·기간 같은
다른 표기 차이는 그대로 남는다.
"""
from __future__ import annotations

import re

_AMOUNT = re.compile(
    r"(?:(?P<eok>\d[\d,]*)\s*억)?\s*(?:(?P<man>\d[\d,]*)\s*만)?\s*(?P<won>\d[\d,]*)?\s*원")


def _int(s: str | None) -> int:
    return int(s.replace(",", "")) if s else 0


def normalize_amounts(text: str) -> str:
    """'2억 4,300만 원' → '243,000,000원'. 억·만 단위가 없는 표기('243,000,000원')는 그대로 둔다."""

    def repl(m: re.Match) -> str:
        if not (m.group("eok") or m.group("man")):
            return m.group(0)
        n = _int(m.group("eok")) * 100_000_000 + _int(m.group("man")) * 10_000 + _int(m.group("won"))
        return f"{n:,}원"

    return _AMOUNT.sub(repl, text)
