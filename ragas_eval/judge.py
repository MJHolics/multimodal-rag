"""LLM judge — faithfulness/context_recall에 필요한 claim 분해·근거 판정.

judge는 `(system, user) -> str` 콜러블이면 무엇이든 된다(무료 Gemini/Ollama/Claude). 테스트는
`StubJudge`로 결정적 응답을 주입해 순수하게 검증한다. 실측은 무료 Gemini 키로 GeminiJudge 사용.
"""
from __future__ import annotations

from .metrics import context_recall, faithfulness


def split_claims(answer: str, judge) -> list[str]:
    """답을 검증 가능한 단문 주장(claim)들로 분해(한 줄 1개)."""
    sys = "다음 답변을 검증 가능한 사실 단위로 분해하라. 한 줄에 하나, 군더더기 없이."
    out = judge(sys, answer or "")
    return [ln.strip(" -•\t") for ln in out.splitlines() if ln.strip()]


def _supported(statement: str, contexts: list[str], judge) -> int:
    """statement가 컨텍스트로 뒷받침되면 1, 아니면 0(YES/NO 판정)."""
    sys = "주어진 컨텍스트만 근거로 진술이 뒷받침되면 정확히 YES, 아니면 NO만 출력."
    user = "컨텍스트:\n" + "\n".join(f"- {c}" for c in contexts) + f"\n\n진술: {statement}"
    return 1 if judge(sys, user).strip().upper().startswith("YES") else 0


def compute_faithfulness(answer: str, contexts: list[str], judge) -> float:
    """답의 주장들을 컨텍스트에 대해 검증 → faithfulness."""
    claims = split_claims(answer, judge)
    flags = [_supported(c, contexts, judge) for c in claims]
    return faithfulness(flags)


def compute_context_recall(ground_truth: str, contexts: list[str], judge) -> float:
    """정답을 진술로 분해해 각 진술이 컨텍스트로 뒷받침되는지 → context recall."""
    statements = split_claims(ground_truth, judge)
    flags = [_supported(s, contexts, judge) for s in statements]
    return context_recall(flags)


class StubJudge:
    """결정적 stub judge(테스트용). 분해는 줄 단위, 근거 판정은 키워드 포함으로 흉내."""

    def __call__(self, system: str, user: str) -> str:
        if "분해" in system:                      # split_claims 호출
            text = user
            parts = [p.strip() for p in text.replace("。", ".").split(".") if p.strip()]
            return "\n".join(parts) if parts else text
        # _supported 호출: 진술의 첫 토큰이 컨텍스트에 나오면 YES
        stmt = user.split("진술:")[-1].strip()
        key = stmt.split()[0] if stmt.split() else stmt
        return "YES" if key and key in user.split("진술:")[0] else "NO"
