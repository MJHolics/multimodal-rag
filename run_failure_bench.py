"""검색 컴포넌트 장애 대응 실측 — dense·BM25가 하나씩, 그리고 둘 다 죽으면 무슨 일이 나는가.

## 왜 하는가 (2026-09-07)

컴앤휴먼 면접에서 "BM25/BGE-M3 중 하나가 죽으면 어떡하냐"는 질문에 그 자리에서
지어낸 답을 했다 — 코드에는 없는 "하드코딩된 키워드 노출" 같은 장애 대응을 있는 것처럼
말했다. 실제로는 `graph_rag/retrieval.py::HybridRetriever.score()`에 컴포넌트별
try/except와 3단 축소 운영(hybrid → 단일 소스 → 키워드 스캔)을 이번에 추가했고,
이 스크립트는 그 경로를 실제로 태워서 품질을 잰다. 다음에는 지어내지 않고 이 숫자로 답한다.

## 무엇을 비교하는가

`run_hybrid_weight.py`와 같은 실문서 RFP 코퍼스·질의셋(test 분할, gold multi-hop 포함)에
같은 `HybridRetriever` 인스턴스(코퍼스 인코딩 1회)를 두고, `_dense_scores`/`_bm25_scores`를
런타임에 몽키패치로 고장 내 4가지 조건에서 `retrieve()`를 그대로 호출한다:

  1. normal          — dense 0.9 + BM25 0.1 (코드 고정값)
  2. dense_down       — dense 실패 → BM25 단독
  3. bm25_down        — BM25 실패 → dense 단독
  4. both_down        — 둘 다 실패 → 원문 키워드 겹침 스캔(최후 수단)

지표는 `run_hybrid_weight.py`와 동일한 `full_hit@k`(멀티홉 gold 전부 상위 k 안에 있어야 함) ·
`mrr`이고, normal 대비 각 축소 모드를 McNemar로 짝지어 유의성까지 본다.

실행: python run_failure_bench.py
"""
from __future__ import annotations

import json
import pickle
import sys
import time
import types
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from graph_rag.eval_gen import split_queries
from graph_rag.metrics import aggregate, full_hit_at_k
from graph_rag.retrieval import DENSE_WEIGHT, HybridRetriever
from graph_rag.stats import mcnemar
from knowledgeops.eval_real import build_real_queries, summarize_real

ROOT = Path(__file__).parent
CORPUS = ROOT / "output" / "rfp_corpus.pkl"
OUT = ROOT / "output" / "failure_bench.json"
K = 3


def _raise(self, query):
    raise RuntimeError("simulated outage")


def run_condition(hybrid: HybridRetriever, queries, label: str,
                   break_dense: bool, break_bm25: bool):
    orig_dense = HybridRetriever._dense_scores
    orig_bm25 = HybridRetriever._bm25_scores
    hybrid._score_cache.clear()
    if break_dense:
        hybrid._dense_scores = types.MethodType(_raise, hybrid)
    if break_bm25:
        hybrid._bm25_scores = types.MethodType(_raise, hybrid)

    retrieved = [hybrid.retrieve(q.question, K) for q in queries]
    flags = [bool(full_hit_at_k(r, q.gold_chunks, K)) for r, q in zip(retrieved, queries)]
    metrics = aggregate(list(zip(retrieved, (q.gold_chunks for q in queries))), K)
    mode_seen = hybrid.last_mode

    hybrid._dense_scores = types.MethodType(orig_dense, hybrid)
    hybrid._bm25_scores = types.MethodType(orig_bm25, hybrid)

    print(f"  {label:12s} mode={mode_seen:16s} full_hit={metrics['full_hit']:.4f} "
          f"mrr={metrics['mrr']:.4f} hit={metrics['hit']:.4f}")
    return metrics, flags, mode_seen


def main() -> None:
    rc = pickle.load(open(CORPUS, "rb"))
    chunks, queries = rc.chunks, build_real_queries(rc)
    print(f"[queries] {json.dumps(summarize_real(queries), ensure_ascii=False)}")
    dev, test = split_queries(queries)
    print(f"[split] dev {len(dev)}문항 · test {len(test)}문항 (test로 보고)")

    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    embedder = SentenceTransformer("BAAI/bge-m3")
    hybrid = HybridRetriever(chunks, embedder=embedder)
    print(f"[dense] BGE-M3 로드+코퍼스 인코딩 {time.time() - t0:.1f}s · 청크 {len(chunks)}개")

    print(f"\n[조건별 실측] k={K}, dense_weight={DENSE_WEIGHT}(코드 고정값)")
    normal_m, normal_f, _ = run_condition(hybrid, test, "normal", False, False)
    dd_m, dd_f, dd_mode = run_condition(hybrid, test, "dense_down", True, False)
    bd_m, bd_f, bd_mode = run_condition(hybrid, test, "bm25_down", False, True)
    both_m, both_f, both_mode = run_condition(hybrid, test, "both_down", True, True)

    assert dd_mode == "bm25_only" and bd_mode == "dense_only" and both_mode == "keyword_fallback", \
        "모드 라우팅이 기대와 다르다 — score() 로직 확인 필요"

    mc_dense = mcnemar(normal_f, dd_f)
    mc_bm25 = mcnemar(normal_f, bd_f)
    mc_both = mcnemar(normal_f, both_f)

    print("\n[McNemar — normal 대비]")
    print(f"  dense_down: delta {mc_dense['delta']:+.4f} · p {mc_dense['p_value']:.4f} "
          f"· 불일치 {mc_dense['a_only']}/{mc_dense['b_only']}")
    print(f"  bm25_down:  delta {mc_bm25['delta']:+.4f} · p {mc_bm25['p_value']:.4f} "
          f"· 불일치 {mc_bm25['a_only']}/{mc_bm25['b_only']}")
    print(f"  both_down:  delta {mc_both['delta']:+.4f} · p {mc_both['p_value']:.4f} "
          f"· 불일치 {mc_both['a_only']}/{mc_both['b_only']}")

    results = {
        "meta": {
            "k": K,
            "dense_weight_fixed": DENSE_WEIGHT,
            "test_n": len(test),
            "note": "정상 대비 컴포넌트 장애 시뮬레이션(런타임 몽키패치) — 실제 score() 축소 운영 경로를 태움",
        },
        "normal": normal_m,
        "dense_down": dd_m,
        "bm25_down": bd_m,
        "both_down": both_m,
        "mcnemar_vs_normal": {
            "dense_down": mc_dense,
            "bm25_down": mc_bm25,
            "both_down": mc_both,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[json] 저장: {OUT}")


if __name__ == "__main__":
    main()
