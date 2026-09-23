"""rerank의 상한을 잰다 — top-20 후보에 애초에 정답이 다 들어있는 문항이 몇 %인가.

README 한계에 "1단계가 정답을 top-20 밖으로 떨어뜨린 문항은 rerank가 절대 못 구한다"고만
적어 두고 실제 비율은 안 쟀다. full_hit@20(=gold 전부가 top-20 안에 있는가)이 그 상한이다 —
rerank가 아무리 좋아도 이 비율을 못 넘는다.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from graph_rag.eval_gen import split_queries
from graph_rag.metrics import aggregate, full_hit_at_k
from graph_rag.retrieval import DENSE_WEIGHT, HybridRetriever, _rank
from knowledgeops.chunking import corpus_summary
from knowledgeops.eval_real import build_real_queries, summarize_real

ROOT = Path(__file__).parent
CORPUS = ROOT / "output" / "rfp_corpus.pkl"
K_FINAL = 3
N_CANDIDATES = 20


def main() -> None:
    from sentence_transformers import SentenceTransformer

    rc = pickle.load(open(CORPUS, "rb"))
    chunks, queries = rc.chunks, build_real_queries(rc)
    print(f"[corpus] {json.dumps(corpus_summary(rc), ensure_ascii=False)}")
    print(f"[queries] {json.dumps(summarize_real(queries), ensure_ascii=False)}")
    dev, test = split_queries(queries)

    t0 = time.time()
    bge = SentenceTransformer("BAAI/bge-m3")
    hybrid = HybridRetriever(chunks, embedder=bge)
    print(f"[bge-m3] 로드+코퍼스인코딩 {time.time() - t0:.1f}s")

    ids = hybrid.ids

    def retrieve(q, k):
        return _rank(hybrid.score(q.question), k)

    for name, qs in (("dev", dev), ("test", test)):
        at_final = aggregate([(retrieve(q, K_FINAL), q.gold_chunks) for q in qs], K_FINAL)
        at_cand = aggregate([(retrieve(q, N_CANDIDATES), q.gold_chunks) for q in qs], N_CANDIDATES)
        gap = at_cand["full_hit"] - at_final["full_hit"]
        print(f"\n[{name}] full_hit@{K_FINAL}={at_final['full_hit']:.4f} · "
              f"full_hit@{N_CANDIDATES}={at_cand['full_hit']:.4f} "
              f"(rerank 상한, gap={gap:.4f})")

    # 어떤 문항이 top-20에도 못 들어오는지 원인 분리
    test_miss20 = [q for q in test if not full_hit_at_k(retrieve(q, N_CANDIDATES), q.gold_chunks, N_CANDIDATES)]
    print(f"\n[test] top-20에도 정답이 다 안 들어오는 문항: {len(test_miss20)}/{len(test)}"
          f" — rerank로 절대 못 구하는 상한 손실")
    for q in test_miss20[:5]:
        print(f"  - {q.qid} hops={q.hops} gold={q.gold_chunks}")

    result = {
        "meta": {"k_final": K_FINAL, "n_candidates": N_CANDIDATES, "dense_weight": DENSE_WEIGHT},
        "dev": {
            "full_hit_at_final": aggregate([(retrieve(q, K_FINAL), q.gold_chunks) for q in dev], K_FINAL)["full_hit"],
            "full_hit_at_candidates": aggregate([(retrieve(q, N_CANDIDATES), q.gold_chunks) for q in dev], N_CANDIDATES)["full_hit"],
        },
        "test": {
            "full_hit_at_final": aggregate([(retrieve(q, K_FINAL), q.gold_chunks) for q in test], K_FINAL)["full_hit"],
            "full_hit_at_candidates": aggregate([(retrieve(q, N_CANDIDATES), q.gold_chunks) for q in test], N_CANDIDATES)["full_hit"],
            "miss_at_candidates": len(test_miss20),
            "miss_qids": [q.qid for q in test_miss20],
        },
    }
    out = ROOT / "output" / "rerank_upper_bound.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[json] 저장: {out}")


if __name__ == "__main__":
    main()
