"""공식 ragas 라이브러리(0.4.x) Faithfulness로 같은 환각 셋을 잰다 — 자체 구현과 나란히 보기 위해.

ragas는 langchain 버전에 민감해 별도 venv에서 돈다(`.venv-ragas`, langchain-community<0.4 고정):
  LOCAL_LLM_BASE_URL=http://localhost:8000/v1 LOCAL_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ \
    .venv-ragas/Scripts/python run_ragas_official.py --tag ragas-7b --repeats 2

판정 규칙·집계는 run_ragas_judge_bench.py와 같다(faithfulness < 1 → 환각). 점수가 NaN(주장 0개)이면
판정 불가로 기록하고 환각으로 치지 않는다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "ragas_judge"
sys.path.insert(0, str(ROOT))
from run_ragas_judge_bench import summarize  # noqa: E402  (순수 집계 함수만 재사용)


async def score_all(items, metric, concurrency: int):
    sem = asyncio.Semaphore(concurrency)

    async def one(it):
        async with sem:
            t0 = time.perf_counter()
            err = None
            try:
                r = await metric.ascore(user_input=it["question"], response=it["answer"],
                                        retrieved_contexts=it["contexts"])
                score = float(r.value)
            except Exception as e:
                score, err = float("nan"), f"{type(e).__name__}: {e}"[:200]
            ms = (time.perf_counter() - t0) * 1000
            nan = math.isnan(score)
            if nan and err is None:
                err = "NaN(주장 0개)"
            return {"item_id": it["item_id"], "split": it["split"], "variant": it["variant"],
                    "hallucinated": it["hallucinated"], "score": 1.0 if nan else round(score, 4),
                    "flagged": 0 if nan else int(score < 1.0), "ms": round(ms, 1), "error": err}

    return await asyncio.gather(*(one(it) for it in items))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--split", choices=["dev", "test", "all"], default="all")
    ap.add_argument("--concurrency", type=int, default=1)
    a = ap.parse_args()

    import ragas
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    model = os.environ["LOCAL_LLM_MODEL"]
    client = AsyncOpenAI(base_url=os.environ["LOCAL_LLM_BASE_URL"], api_key="EMPTY", timeout=180)
    llm = llm_factory(model, client=client, temperature=0.0, max_tokens=1024)
    metric = Faithfulness(llm=llm)

    items = json.loads((OUT / "injected_set.json").read_text(encoding="utf-8"))
    if a.split != "all":
        items = [i for i in items if i["split"] == a.split]

    runs = []
    for rep in range(a.repeats):
        rows = asyncio.run(score_all(items, metric, a.concurrency))
        runs.append(rows)
        print(f"[rep {rep}] dev {summarize(rows, 'dev')}\n          test {summarize(rows, 'test')}", flush=True)

    flips = sum(r0["flagged"] != r1["flagged"] for r0, r1 in zip(runs[0], runs[-1])) if len(runs) > 1 else 0
    result = {"tag": a.tag, "judge": f"ragas {ragas.__version__} Faithfulness", "model": model,
              "repeats": a.repeats, "concurrency": a.concurrency, "flag_flips_between_repeats": flips,
              "summary": {"dev": summarize(runs[0], "dev"), "test": summarize(runs[0], "test")},
              "runs": runs}
    (OUT / f"judge_{a.tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("tag", "flag_flips_between_repeats", "summary")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
