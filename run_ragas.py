"""RAGAS 스타일 RAG 평가 하네스 — 생성 품질을 4지표로 수치화.

  python run_ragas.py            # 결정적 stub judge로 실행(키 불요, 파이프라인 동작 시연)
  python run_ragas.py --gemini   # 무료 Gemini judge로 실측(GEMINI_API_KEY 필요)

지표: answer_correctness(token F1) · context_precision · context_recall · faithfulness.
순수 지표는 stub로도 의미 있는 값이 나오고, faithfulness/recall은 judge 품질에 따라 달라진다.
"""
from __future__ import annotations

import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from ragas_eval.dataset import SAMPLES  # noqa: E402
from ragas_eval.judge import StubJudge, compute_context_recall, compute_faithfulness  # noqa: E402
from ragas_eval.metrics import context_precision, token_f1  # noqa: E402


def _gemini_judge():
    import os

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def judge(system: str, user: str) -> str:
        r = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=user,
            config=types.GenerateContentConfig(system_instruction=system, temperature=0.0),
        )
        return (r.text or "").strip()

    return judge


def main() -> int:
    judge = _gemini_judge() if "--gemini" in sys.argv else StubJudge()
    mode = "Gemini(실측)" if "--gemini" in sys.argv else "Stub(결정적·키 불요)"
    print(f"=== RAGAS 스타일 RAG 평가 · judge={mode} · {len(SAMPLES)} 샘플 ===\n")

    agg = {"correctness": [], "ctx_precision": [], "ctx_recall": [], "faithfulness": []}
    for s in SAMPLES:
        c = token_f1(s.answer, s.ground_truth)
        cp = context_precision(s.context_relevance)
        cr = compute_context_recall(s.ground_truth, s.contexts, judge)
        f = compute_faithfulness(s.answer, s.contexts, judge)
        agg["correctness"].append(c)
        agg["ctx_precision"].append(cp)
        agg["ctx_recall"].append(cr)
        agg["faithfulness"].append(f)
        print(f"Q: {s.question}")
        print(f"   correctness(F1) {c:.2f} · ctx_precision {cp:.2f} · ctx_recall {cr:.2f} · faithfulness {f:.2f}")

    print("\n=== 평균 ===")
    for k, v in agg.items():
        print(f"   {k:14s}: {sum(v) / len(v):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
