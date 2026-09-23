"""검색 모델 교체 실험 — BGE-M3+BM25 이외의 모델을 실측으로 붙여본다.

## 왜 하는가 (2026-09-09)

컴앤휴먼 면접에서 면접관이 우리 프로젝트와 "다른 모델"을 회사에서 쓰고 있다는 뉘앙스를 줬는데,
그 자리에서 비교해본 적이 없어 답이 얕았다. 이 프로젝트는 처음 만든 이후 줄곧 BGE-M3(dense) +
BM25(sparse) 조합 하나만 썼다 — 다른 임베더나 재순위(rerank) 단계를 실제로 붙여서 잰 적이
없다는 뜻이다. 그래서 세 축을 새로 잰다:

  1. **다른 dense 임베더** — BGE-M3 자리에 KURE-v1(한국어 특화, 국내 RAG에서 자주 쓰는 대안)과
     multilingual-e5-large(범용 다국어 대안)를 넣어 같은 프로토콜로 비교한다.
  2. **재순위(rerank) 단계 신설** — 지금까지 이 프로젝트엔 rerank가 아예 없었다. 하이브리드로
     top-20을 뽑고 BAAI/bge-reranker-v2-m3(cross-encoder)로 다시 순위를 매겨 top-3을 추리는
     구조를 처음 붙여서, "2단계(retrieve→rerank)"가 "1단계(retrieve만)"보다 실제로 나은지 잰다.
  3. (참고) 희소 검색 대안 SPLADE는 이번 실행에서는 별도 절(README)로 스코프 아웃 — 이유는 README에 기록.

## 프로토콜 (run_hybrid_weight.py와 동일 원칙)

  - 코퍼스: `output/rfp_corpus.pkl`(실 RFP 문서 7,863청크) — 합성 코퍼스가 아니라 이미 검증에
    쓰던 실문서로 잰다.
  - dense_weight는 0.0~1.0 그리드를 **dev에서만** 스윕해 고르고, **test에서만** 보고·검정한다.
  - 비교는 전부 같은 test 문항(67개)에 대한 짝지은(paired) McNemar로 한다 — 그래야 "이 모델이
    이겼다"는 말에 유의성이 붙는다.
  - k=3 (기존 실험과 동일, `full_hit`이 주 지표 — 멀티홉 gold 전부가 top-3에 있어야 hit).

## 실행

    python run_model_comparison.py                 # 전체(임베더 3종 + rerank) 실행, 수 분~수십분
    python run_model_comparison.py --skip-dense     # rerank 실험만
    python run_model_comparison.py --models bge-m3  # 임베더 목록 지정(비교 대상 좁히기)
"""
from __future__ import annotations

import argparse
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
from graph_rag.retrieval import BM25_WEIGHT, DENSE_WEIGHT, HybridRetriever, _rank
from graph_rag.stats import mcnemar
from knowledgeops.chunking import corpus_summary
from knowledgeops.eval_real import build_real_queries, summarize_real

ROOT = Path(__file__).parent
CORPUS = ROOT / "output" / "rfp_corpus.pkl"
GRID = [round(i * 0.1, 1) for i in range(11)]
K = 3
RERANK_TOPN = 20

# 모델별로 sentence-transformers 인코딩 앞에 붙일 프리픽스.
# e5 계열은 프리픽스가 없으면 성능이 크게 떨어진다고 모델 카드가 명시함 — 빠뜨리면
# "e5가 별로다"가 아니라 "e5를 잘못 썼다"가 되므로 반드시 반영한다.
EMBEDDER_SPECS: dict[str, dict] = {
    "bge-m3": {
        "hf_id": "BAAI/bge-m3",
        "query_prefix": "",
        "passage_prefix": "",
        "note": "현재 코드 기본값(app/retriever.py) — baseline",
    },
    "kure-v1": {
        "hf_id": "nlpai-lab/KURE-v1",
        "query_prefix": "",
        "passage_prefix": "",
        "note": "고려대 NLP&AI Lab, 한국어 검색 특화 파인튜닝(국내 RAG 스택에서 BGE-M3 대안으로 흔히 언급됨)",
    },
    "multilingual-e5-large": {
        "hf_id": "intfloat/multilingual-e5-large",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "note": "범용 다국어 대안, 모델 카드가 query:/passage: 프리픽스를 명시적으로 요구",
    },
}


class PrefixedEmbedder:
    """SentenceTransformer를 감싸 query/passage 프리픽스를 붙인다.

    HybridRetriever는 `embedder.encode(list[str], normalize_embeddings=True)`만
    호출한다(코퍼스 인코딩용). 질의 인코딩은 `_dense_scores`가 별도로 호출하므로,
    질의와 코퍼스에 다른 프리픽스를 붙이려면 두 경로를 구분해야 한다.
    이 클래스는 "기본은 passage(코퍼스용)"로 동작하고, 질의 인코딩 시점에는
    `for_query()`로 얻은 얕은 래퍼를 HybridRetriever의 `_dense_scores`가 쓰게 만든다.
    """

    def __init__(self, model, query_prefix: str, passage_prefix: str):
        self._model = model
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self._mode = "passage"

    def encode(self, texts: list[str], normalize_embeddings: bool = True):
        prefix = self.query_prefix if self._mode == "query" else self.passage_prefix
        return self._model.encode([prefix + t for t in texts], normalize_embeddings=normalize_embeddings)

    def as_query(self) -> "PrefixedEmbedder":
        self._mode = "query"
        return self

    def as_passage(self) -> "PrefixedEmbedder":
        self._mode = "passage"
        return self


def component_scores_prefixed(chunks, queries, embedder: PrefixedEmbedder, bm25_retriever: HybridRetriever):
    """지정 임베더로 질의별 dense 점수를, bm25_retriever로 bm25 점수를 뽑아 캐시한다."""
    embedder.as_passage()
    doc_emb = embedder.encode([c.text for c in chunks], normalize_embeddings=True)
    ids = [c.chunk_id for c in chunks]

    embedder.as_query()
    out = {}
    for q in queries:
        qv = embedder.encode([q.question], normalize_embeddings=True)[0]
        sims = doc_emb @ qv
        dense = {ids[i]: float(sims[i]) for i in range(len(ids))}
        out[q.qid] = {"dense": dense, "bm25": bm25_retriever._bm25_scores(q.question)}
    return out


def retrieve_at(cache, qid, w, ids, k):
    c = cache[qid]
    dense, bm25 = c["dense"], c["bm25"]
    if not dense:
        return _rank(bm25, k)
    fused = {cid: w * dense.get(cid, 0.0) + (1.0 - w) * bm25.get(cid, 0.0) for cid in ids}
    return _rank(fused, k)


def evaluate(cache, queries, w, ids, k):
    return aggregate([(retrieve_at(cache, q.qid, w, ids, k), q.gold_chunks) for q in queries], k)


def correct_flags(cache, queries, w, ids, k):
    return [bool(full_hit_at_k(retrieve_at(cache, q.qid, w, ids, k), q.gold_chunks, k)) for q in queries]


def run_embedder_experiment(name: str, spec: dict, chunks, queries, dev, test) -> dict:
    from sentence_transformers import SentenceTransformer

    print(f"\n{'=' * 60}\n[{name}] {spec['hf_id']} 로드 중...")
    t0 = time.time()
    raw = SentenceTransformer(spec["hf_id"])
    load_s = time.time() - t0
    embedder = PrefixedEmbedder(raw, spec["query_prefix"], spec["passage_prefix"])

    bm25_only = HybridRetriever(chunks, embedder=None)  # bm25 점수만 뽑아 쓰는 용도
    ids = bm25_only.ids

    t0 = time.time()
    cache = component_scores_prefixed(chunks, queries, embedder, bm25_only)
    enc_s = time.time() - t0
    print(f"[{name}] 로드 {load_s:.1f}s + 인코딩(코퍼스 {len(chunks)}+질의 {len(queries)}) {enc_s:.1f}s")

    rows_dev = [{"w": w, **evaluate(cache, dev, w, ids, K)} for w in GRID]
    rows_test = [{"w": w, **evaluate(cache, test, w, ids, K)} for w in GRID]
    best = max(rows_dev, key=lambda r: (r["full_hit"], -r["w"]))
    best_w = best["w"]
    best_test = next(r for r in rows_test if r["w"] == best_w)
    fixed_test = next(r for r in rows_test if r["w"] == DENSE_WEIGHT)

    del raw
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "name": name,
        "hf_id": spec["hf_id"],
        "note": spec["note"],
        "load_s": round(load_s, 1),
        "encode_s": round(enc_s, 1),
        "dev": rows_dev,
        "test": rows_test,
        "best_w": best_w,
        "best_test": best_test,
        "fixed_w_test": fixed_test,
        "test_correct_flags": correct_flags(cache, test, best_w, ids, K),
    }


# ──────────────────────────────────────────────────────────────
# rerank 실험
# ──────────────────────────────────────────────────────────────

def run_rerank_experiment(chunks, queries, dev, test, baseline_cache, baseline_ids) -> dict:
    """BGE-M3+BM25(w=0.9) top-20 → bge-reranker-v2-m3 rerank → top-3."""
    print(f"\n{'=' * 60}\n[rerank] BAAI/bge-reranker-v2-m3 로드 중...")
    text_of = {c.chunk_id: c.text for c in chunks}

    t0 = time.time()
    reranker = None
    backend = None
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
        backend = "sentence_transformers.CrossEncoder"
    except Exception as e:
        print(f"[rerank] CrossEncoder 로드 실패({e}), transformers 직접 로드로 대체")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-v2-m3")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()

        def _score(pairs: list[tuple[str, str]]) -> list[float]:
            with torch.no_grad():
                inputs = tok(pairs, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
                logits = model(**inputs).logits.view(-1).float()
            return logits.cpu().tolist()

        reranker = _score
        backend = "transformers.AutoModelForSequenceClassification"
    load_s = time.time() - t0
    print(f"[rerank] 로드 {load_s:.1f}s ({backend})")

    def rerank_retrieve(qid: str, question: str, w: float, k_final: int, n_cand: int) -> list[str]:
        cand = retrieve_at(baseline_cache, qid, w, baseline_ids, n_cand)
        if not cand:
            return cand
        pairs = [(question, text_of[cid]) for cid in cand]
        if backend == "sentence_transformers.CrossEncoder":
            scores = reranker.predict(pairs)
        else:
            scores = reranker(pairs)
        ranked = [cid for cid, _ in sorted(zip(cand, scores), key=lambda x: -x[1])]
        return ranked[:k_final]

    def eval_rerank(qs):
        results = []
        for q in qs:
            r = rerank_retrieve(q.qid, q.question, DENSE_WEIGHT, K, RERANK_TOPN)
            results.append((r, q.gold_chunks))
        return aggregate(results, K)

    t0 = time.time()
    dev_metrics = eval_rerank(dev)
    test_metrics = eval_rerank(test)
    rerank_s = time.time() - t0

    no_rerank_test_flags = correct_flags(baseline_cache, test, DENSE_WEIGHT, baseline_ids, K)
    rerank_test_flags = [
        bool(full_hit_at_k(rerank_retrieve(q.qid, q.question, DENSE_WEIGHT, K, RERANK_TOPN), q.gold_chunks, K))
        for q in test
    ]
    mc = mcnemar(no_rerank_test_flags, rerank_test_flags)

    return {
        "backend": backend,
        "load_s": round(load_s, 1),
        "rerank_wall_s": round(rerank_s, 1),
        "top_n_candidates": RERANK_TOPN,
        "k_final": K,
        "dense_weight_used_for_candidates": DENSE_WEIGHT,
        "dev": dev_metrics,
        "test_with_rerank": test_metrics,
        "test_no_rerank_baseline": next(
            r for r in [{"w": DENSE_WEIGHT, **evaluate(baseline_cache, test, DENSE_WEIGHT, baseline_ids, K)}]
        ),
        "mcnemar_no_rerank_vs_rerank": mc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["bge-m3", "kure-v1", "multilingual-e5-large"],
                     choices=list(EMBEDDER_SPECS))
    ap.add_argument("--skip-dense", action="store_true")
    ap.add_argument("--skip-rerank", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "output" / "model_comparison.json"))
    args = ap.parse_args()

    rc = pickle.load(open(CORPUS, "rb"))
    chunks, queries = rc.chunks, build_real_queries(rc)
    corpus_desc = corpus_summary(rc)
    print(f"[corpus] {json.dumps(corpus_desc, ensure_ascii=False)}")
    print(f"[queries] {json.dumps(summarize_real(queries), ensure_ascii=False)}")
    dev, test = split_queries(queries)
    print(f"[split] dev {len(dev)} · test {len(test)}")

    results: dict = {
        "meta": {
            "corpus": corpus_desc,
            "split": {"dev": len(dev), "test": len(test)},
            "k": K,
            "grid": GRID,
            "protocol": "dense_weight는 dev에서만 고르고 test로 보고·검정(run_hybrid_weight.py와 동일 원칙)",
        },
        "embedders": {},
    }

    baseline_cache = None
    baseline_ids = None

    if not args.skip_dense:
        for name in args.models:
            spec = EMBEDDER_SPECS[name]
            r = run_embedder_experiment(name, spec, chunks, queries, dev, test)
            results["embedders"][name] = r
            if name == "bge-m3":
                # rerank 실험이 재사용할 baseline 캐시를 별도로 다시 만든다(스코프 분리 목적).
                pass

    # McNemar 교차비교(가능한 쌍): 각 대안 임베더 vs bge-m3, 같은 test 문항 기준
    if "bge-m3" in results["embedders"]:
        base_flags = results["embedders"]["bge-m3"]["test_correct_flags"]
        for name, r in results["embedders"].items():
            if name == "bge-m3":
                continue
            mc = mcnemar(base_flags, r["test_correct_flags"])
            r["mcnemar_vs_bge_m3"] = mc

    if not args.skip_rerank:
        # rerank 실험은 baseline(BGE-M3+BM25 w=0.9) 캐시가 필요 — 이미 있으면 재사용, 없으면 새로 만든다.
        if "bge-m3" in results["embedders"]:
            print("\n[rerank] bge-m3 임베더 재사용 (이미 위에서 로드했던 결과를 다시 계산하지 않고 새로 인코딩)")
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        bge = SentenceTransformer("BAAI/bge-m3")
        hybrid = HybridRetriever(chunks, embedder=bge)
        print(f"[rerank] bge-m3 로드+코퍼스인코딩 {time.time() - t0:.1f}s")
        cache = {}
        for q in queries:
            cache[q.qid] = {"dense": hybrid._dense_scores(q.question), "bm25": hybrid._bm25_scores(q.question)}
        ids = hybrid.ids
        rr = run_rerank_experiment(chunks, queries, dev, test, cache, ids)
        results["rerank"] = rr

    for name, r in results["embedders"].items():
        r.pop("test_correct_flags", None)  # JSON에는 요약만 남긴다(원자료는 재현 가능)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[json] 저장: {out}")

    print("\n" + "=" * 60)
    print("요약 (test, k=3, full_hit)")
    for name, r in results["embedders"].items():
        mc = r.get("mcnemar_vs_bge_m3")
        mc_s = f" · vs bge-m3 McNemar p={mc['p_value']:.4f}" if mc else " (baseline)"
        print(f"  {name:24s} best_w={r['best_w']:.1f}  full_hit={r['best_test']['full_hit']:.4f}"
              f"  mrr={r['best_test']['mrr']:.4f}{mc_s}")
    if "rerank" in results:
        rr = results["rerank"]
        print(f"\n  rerank(top{rr['top_n_candidates']}→bge-reranker-v2-m3→top{rr['k_final']}): "
              f"full_hit={rr['test_with_rerank']['full_hit']:.4f} "
              f"vs no-rerank full_hit={rr['test_no_rerank_baseline']['full_hit']:.4f} "
              f"McNemar p={rr['mcnemar_no_rerank_vs_rerank']['p_value']:.4f}")


if __name__ == "__main__":
    main()
