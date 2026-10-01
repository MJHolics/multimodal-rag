"""RAGAS faithfulness judge는 환각을 잡는가 — 정답을 아는 환각 셋으로 judge를 잰다.

  python run_ragas_judge_bench.py --build                          # 셋 생성(results/ragas_judge/injected_set.json)
  python run_ragas_judge_bench.py --judge lexical --tag lexical     # 숫자 대조 기준선(LLM 0회)
  LOCAL_LLM_BASE_URL=http://localhost:8000/v1 LOCAL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ \
    python run_ragas_judge_bench.py --judge own --tag own-7b --repeats 2   # 자체 구현(ragas_eval.judge)
  (공식 ragas 라이브러리는 별도 venv: run_ragas_official.py)

판정 규칙은 튜닝하지 않는다: faithfulness < 1 이면 "환각". 점수 분포는 AUROC로 따로 본다.
결과: results/ragas_judge/judge_<tag>.json (문항별 점수·지연 + 변형별 집계)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from ragas_eval.injected import VARIANTS, build_from_corpus, numbers_in  # noqa: E402
from ragas_eval.judge import compute_faithfulness  # noqa: E402

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "ragas_judge"
SET = OUT / "injected_set.json"


def lexical_score(answer: str, contexts: list[str]) -> float:
    """답의 숫자 토큰 중 컨텍스트에 숫자로 나오는 비율. 숫자가 없으면 1(판정할 게 없다)."""
    nums = numbers_in(answer)
    if not nums:
        return 1.0
    ctx = numbers_in("\n".join(contexts))
    return sum(n in ctx for n in nums) / len(nums)


def local_judge():
    from openai import OpenAI

    client = OpenAI(base_url=os.environ["LOCAL_LLM_BASE_URL"], api_key="EMPTY", timeout=120)
    model = os.environ["LOCAL_LLM_MODEL"]
    calls = {"n": 0}

    def judge(system: str, user: str) -> str:
        calls["n"] += 1
        r = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=512,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return (r.choices[0].message.content or "").strip()

    return judge, calls, model


def auroc(scores_pos: list[float], scores_neg: list[float]) -> float:
    """환각(pos)일수록 점수가 낮아야 한다 → P(score_pos < score_neg), 동점 0.5."""
    if not scores_pos or not scores_neg:
        return float("nan")
    s = 0.0
    for p in scores_pos:
        for n in scores_neg:
            s += 1.0 if p < n else 0.5 if p == n else 0.0
    return s / (len(scores_pos) * len(scores_neg))


def summarize(rows: list[dict], split: str | None) -> dict:
    rs = [r for r in rows if split is None or r["split"] == split]
    out = {"n": len(rs)}
    if not rs:
        return out
    for v in VARIANTS:
        vs = [r for r in rs if r["variant"] == v]
        if vs:
            out[v] = {"flagged": sum(r["flagged"] for r in vs), "n": len(vs),
                      "mean_score": round(sum(r["score"] for r in vs) / len(vs), 3)}
    pos = [r for r in rs if r["hallucinated"]]
    neg = [r for r in rs if not r["hallucinated"]]
    tp = sum(r["flagged"] for r in pos)
    fp = sum(r["flagged"] for r in neg)
    out.update({
        "recall": round(tp / len(pos), 3) if pos else None,
        "false_alarm": round(fp / len(neg), 3) if neg else None,
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "auroc": round(auroc([r["score"] for r in pos], [r["score"] for r in neg]), 3),
        "errors": sum(r.get("error") is not None for r in rs),
    })
    ms = sorted(r["ms"] for r in rs)
    out["p50_ms"] = ms[len(ms) // 2]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    # own: 원래 구현(v1) · own-q: 분해에 질문 제공 · own-norm: 금액 정규화 · own-v2: 둘 다
    ap.add_argument("--judge", choices=["lexical", "own", "own-q", "own-norm", "own-v2"])
    ap.add_argument("--tag")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if a.build:
        items = build_from_corpus(str(ROOT / "output" / "rfp_corpus.pkl"))
        SET.write_text(json.dumps([i.to_dict() for i in items], ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{len(items)} 문항 → {SET}")
        return 0

    items = json.loads(SET.read_text(encoding="utf-8"))
    if a.split != "all":
        items = [i for i in items if i["split"] == a.split]

    judge, calls, model = (None, {"n": 0}, None)
    if a.judge != "lexical":
        judge, calls, model = local_judge()

    runs = []
    for rep in range(a.repeats):
        rows = []
        for it in items:
            t0 = time.perf_counter()
            err = None
            try:
                if a.judge == "lexical":
                    score = lexical_score(it["answer"], it["contexts"])
                else:
                    score = compute_faithfulness(
                        it["answer"], it["contexts"], judge,
                        question=it["question"] if a.judge in ("own-q", "own-v2") else None,
                        normalize=a.judge in ("own-norm", "own-v2"))
            except Exception as e:  # 판정 실패는 점수 없이 기록(환각으로 치지 않는다)
                score, err = 1.0, f"{type(e).__name__}: {e}"[:200]
            rows.append({"item_id": it["item_id"], "split": it["split"], "variant": it["variant"],
                         "hallucinated": it["hallucinated"], "score": round(score, 4),
                         "flagged": int(score < 1.0), "ms": round((time.perf_counter() - t0) * 1000, 1),
                         "error": err})
        runs.append(rows)
        print(f"[rep {rep}] dev {summarize(rows, 'dev')}\n          test {summarize(rows, 'test')}", flush=True)

    flips = 0
    if len(runs) > 1:
        flips = sum(r0["flagged"] != r1["flagged"] for r0, r1 in zip(runs[0], runs[-1]))
    result = {"tag": a.tag, "judge": a.judge, "model": model, "llm_calls": calls["n"],
              "repeats": a.repeats, "flag_flips_between_repeats": flips,
              "summary": {"dev": summarize(runs[0], "dev"), "test": summarize(runs[0], "test")},
              "runs": runs}
    (OUT / f"judge_{a.tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("tag", "llm_calls", "flag_flips_between_repeats", "summary")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
