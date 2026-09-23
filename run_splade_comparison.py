"""SPLADE(학습된 희소 검색)를 실제로 붙여본다 — BM25 대신 넣으면 나아지는가.

## 이 스크립트가 있는 이유

`run_model_comparison.py`를 쓸 때 SPLADE를 "한국어 지원이 없어서" 스코프아웃하려다,
확인해 보니 `yjoonjang/splade-ko-v1`(skt/A.X-Encoder-base 기반) 한국어 체크포인트가
실제로 있었다 — 틀린 이유였다. 진짜 이유는 인터페이스였다: SPLADE는 `rank_bm25.BM25Okapi`
같은 용어-빈도 점수가 아니라 **5만 차원 학습된 희소 벡터**(MLM 헤드 위에 log(1+relu(x))
max-pooling)를 내고, `sentence_transformers.SparseEncoder`로 인코딩해 내적으로 채점한다.
말로만 "인터페이스가 다르다"고 적고 끝내는 대신, 실제로 그 인터페이스를 만들어 재본다.

## 비교 대상

  1. **SPLADE 단독** vs **BM25 단독** — 둘 다 "어휘 기반" 계열이지만 SPLADE는 학습된
     확장(예: "예산"이 "비용"·"금액"과도 겹치도록 학습됨)이 있고 BM25는 순수 표면형이다.
  2. **BGE-M3 + SPLADE 하이브리드**(dev 스윕) vs **BGE-M3 + BM25 하이브리드**(기존 기본값,
     w=0.9) — sparse 성분을 통째로 교체했을 때 최종 성능이 달라지는가.

프로토콜은 `run_hybrid_weight.py`·`run_model_comparison.py`와 동일: dev에서 가중치를
고르고 test로 보고·검정(McNemar), k=3.

## 실행

    python run_splade_comparison.py
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

import warnings
warnings.filterwarnings("ignore")

from graph_rag.eval_gen import split_queries
from graph_rag.metrics import aggregate, full_hit_at_k
from graph_rag.retrieval import DENSE_WEIGHT, HybridRetriever, _rank
from graph_rag.stats import mcnemar
from knowledgeops.chunking import corpus_summary
from knowledgeops.eval_real import build_real_queries, summarize_real

ROOT = Path(__file__).parent
CORPUS = ROOT / "output" / "rfp_corpus.pkl"
GRID = [round(i * 0.1, 1) for i in range(11)]
K = 3
SPLADE_MODEL = "yjoonjang/splade-ko-v1"


def splade_scores_all(model, texts: list[str], batch_size: int = 8):
    """SPLADE 희소 벡터를 배치로 인코딩해 dense torch 텐서로 반환한다.

    코퍼스가 7,863개라 (7863, 50000) 밀집 행렬은 float32로 약 1.5GB — GPU 메모리에
    올려도 되는 크기라 굳이 희소 연산을 유지하지 않고 내적을 위해 dense로 바꾼다.
    """
    import torch

    parts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        parts.append(emb.to_dense() if emb.is_sparse else emb)
    return torch.cat(parts, dim=0)


def main() -> None:
    from sentence_transformers import SparseEncoder, SentenceTransformer
    import torch

    rc = pickle.load(open(CORPUS, "rb"))
    chunks, queries = rc.chunks, build_real_queries(rc)
    corpus_desc = corpus_summary(rc)
    print(f"[corpus] {json.dumps(corpus_desc, ensure_ascii=False)}")
    print(f"[queries] {json.dumps(summarize_real(queries), ensure_ascii=False)}")
    dev, test = split_queries(queries)
    ids = [c.chunk_id for c in chunks]

    print(f"\n[splade] {SPLADE_MODEL} 로드 중...")
    t0 = time.time()
    splade = SparseEncoder(SPLADE_MODEL)
    load_s = time.time() - t0
    print(f"[splade] 로드 {load_s:.1f}s")

    t0 = time.time()
    doc_emb = splade_scores_all(splade, [c.text for c in chunks])
    q_texts = [q.question for q in queries]
    q_emb = splade_scores_all(splade, q_texts)
    enc_s = time.time() - t0
    print(f"[splade] 인코딩(코퍼스 {len(chunks)}+질의 {len(queries)}) {enc_s:.1f}s"
          f" · nnz/doc 평균 {(doc_emb != 0).sum(dim=1).float().mean().item():.1f}/50000")

    # 질의별 SPLADE 점수(내적) — max 정규화해 BM25/dense와 같은 스케일로 맞춘다.
    sims = (doc_emb.float() @ q_emb.float().T).cpu().numpy()  # (n_chunks, n_queries)
    splade_cache: dict[str, dict[str, float]] = {}
    for qi, q in enumerate(queries):
        raw = sims[:, qi]
        mx = float(raw.max()) if raw.max() > 0 else 1.0
        splade_cache[q.qid] = {ids[i]: float(raw[i]) / mx for i in range(len(ids))}

    del doc_emb, q_emb, sims, splade
    torch.cuda.empty_cache()

    print("\n[bge-m3] BAAI/bge-m3 로드 + 코퍼스 인코딩 중...")
    t0 = time.time()
    bge = SentenceTransformer("BAAI/bge-m3")
    hybrid = HybridRetriever(chunks, embedder=bge)  # BM25 점수도 여기서 같이 얻는다
    print(f"[bge-m3] {time.time() - t0:.1f}s")

    cache = {}
    for q in queries:
        cache[q.qid] = {
            "dense": hybrid._dense_scores(q.question),
            "bm25": hybrid._bm25_scores(q.question),
            "splade": splade_cache[q.qid],
        }

    def retrieve(qid: str, dense_w: float, sparse_key: str) -> list[str]:
        c = cache[qid]
        dense, sparse = c["dense"], c[sparse_key]
        fused = {cid: dense_w * dense.get(cid, 0.0) + (1 - dense_w) * sparse.get(cid, 0.0) for cid in ids}
        return _rank(fused, K)

    def sparse_only(qid: str, sparse_key: str) -> list[str]:
        return _rank(cache[qid][sparse_key], K)

    def evaluate(qs, dense_w, sparse_key):
        return aggregate([(retrieve(q.qid, dense_w, sparse_key), q.gold_chunks) for q in qs], K)

    def evaluate_sparse_only(qs, sparse_key):
        return aggregate([(sparse_only(q.qid, sparse_key), q.gold_chunks) for q in qs], K)

    def flags(qs, dense_w, sparse_key):
        return [bool(full_hit_at_k(retrieve(q.qid, dense_w, sparse_key), q.gold_chunks, K)) for q in qs]

    def flags_sparse_only(qs, sparse_key):
        return [bool(full_hit_at_k(sparse_only(q.qid, sparse_key), q.gold_chunks, K)) for q in qs]

    # ---- 1) SPLADE 단독 vs BM25 단독 ----
    bm25_only_test = evaluate_sparse_only(test, "bm25")
    splade_only_test = evaluate_sparse_only(test, "splade")
    mc_sparse_only = mcnemar(flags_sparse_only(test, "bm25"), flags_sparse_only(test, "splade"))

    # ---- 2) BGE-M3+SPLADE 하이브리드(dev 스윕) vs BGE-M3+BM25(기존 w=0.9) ----
    rows_dev = [{"w": w, **evaluate(dev, w, "splade")} for w in GRID]
    rows_test = [{"w": w, **evaluate(test, w, "splade")} for w in GRID]
    best = max(rows_dev, key=lambda r: (r["full_hit"], -r["w"]))
    best_w = best["w"]
    best_test = next(r for r in rows_test if r["w"] == best_w)

    bm25_hybrid_test = evaluate(test, DENSE_WEIGHT, "bm25")  # 기존 기본값 w=0.9
    mc_hybrid = mcnemar(flags(test, DENSE_WEIGHT, "bm25"), flags(test, best_w, "splade"))

    print("\n" + "=" * 60)
    print("1) 단독 비교 (test, k=3)")
    print(f"   BM25 단독    : full_hit {bm25_only_test['full_hit']:.4f} · mrr {bm25_only_test['mrr']:.4f}")
    print(f"   SPLADE 단독  : full_hit {splade_only_test['full_hit']:.4f} · mrr {splade_only_test['mrr']:.4f}")
    print(f"   McNemar: delta {mc_sparse_only['delta']:+.4f} · p {mc_sparse_only['p_value']:.4f}"
          f" · 불일치 {mc_sparse_only['discordant']}쌍")

    print("\n2) 하이브리드 비교 (test, k=3)")
    print(f"   BGE-M3+BM25   (w={DENSE_WEIGHT}, 기존 기본값): full_hit {bm25_hybrid_test['full_hit']:.4f}"
          f" · mrr {bm25_hybrid_test['mrr']:.4f}")
    print(f"   BGE-M3+SPLADE (w={best_w}, dev 최적)     : full_hit {best_test['full_hit']:.4f}"
          f" · mrr {best_test['mrr']:.4f}")
    print(f"   McNemar: delta {mc_hybrid['delta']:+.4f} · p {mc_hybrid['p_value']:.4f}"
          f" · 불일치 {mc_hybrid['discordant']}쌍")

    results = {
        "meta": {"corpus": corpus_desc, "k": K, "grid": GRID, "split": {"dev": len(dev), "test": len(test)},
                  "splade_model": SPLADE_MODEL, "load_s": load_s, "encode_s": enc_s},
        "sparse_only": {"bm25": bm25_only_test, "splade": splade_only_test, "mcnemar": mc_sparse_only},
        "hybrid": {
            "dev_sweep_splade": rows_dev, "test_sweep_splade": rows_test,
            "best_w_splade": best_w, "best_test_splade": best_test,
            "bm25_hybrid_fixed_test": bm25_hybrid_test,
            "mcnemar_bm25_vs_splade_hybrid": mc_hybrid,
        },
    }
    out = ROOT / "output" / "splade_comparison.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[json] 저장: {out}")


if __name__ == "__main__":
    main()
