"""하이브리드 가중치(dense vs BM25) 스윕 — 0.7/0.3에 근거를 붙인다.

## 왜 하는가

`graph_rag/retrieval.py`의 `DENSE_WEIGHT = 0.7 / BM25_WEIGHT = 0.3`은 코드에 고정된 값이고,
`eval_gen.py`의 주석에도 "이 실험 이전에 정해졌다"고 적혀 있다. 즉 **재 보고 고른 값이 아니다.**
그래프 융합 가중치(`graph_weight`)와 agentic 라운드는 dev에서 골랐으면서, 정작 검색의 토대인
어휘·의미 배합비만 근거 없이 박혀 있었다.

여기서 같은 규칙으로 잰다.
  - dense_weight 를 0.0 ~ 1.0 까지 0.1 간격으로 스윕한다 (0.0 = BM25 단독, 1.0 = dense 단독)
  - **가중치는 dev 에서만 고르고**, 보고와 검정은 한 번도 안 본 test 에서 한다
  - 고정값 0.7 과 dev 최적값을 test 에서 McNemar 로 짝지어 비교한다

## 어떻게 빠르게 재는가

가중합은 `w·dense + (1-w)·bm25` 라서, **질의별 dense/bm25 점수를 한 번만 계산해 두면**
11개 가중치를 다시 매기는 데 재인코딩이 필요 없다. 그래서 BGE-M3 인코딩은 코퍼스 1회 +
질의 1회로 끝난다(기존 `HybridRetriever`의 캐시와 같은 발상).

실행:
  python run_hybrid_weight.py --no-dense   # BM25만 (스모크, 스윕 의미 없음)
  python run_hybrid_weight.py              # BGE-M3 + BM25 (실제 조건)
  python run_hybrid_weight.py --synthetic  # 합성 코퍼스(72청크·135문항)로도 확인
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):        # run_real_eval.py 와 같은 이유(cp949 방지)
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


# ──────────────────────────────────────────────────────────────
# 점수 성분을 질의당 한 번만 뽑아 둔다
# ──────────────────────────────────────────────────────────────

def component_scores(hybrid: HybridRetriever, queries) -> dict[str, dict[str, dict[str, float]]]:
    """질의별 dense / bm25 점수를 따로 계산해 캐시한다."""
    out = {}
    for q in queries:
        out[q.qid] = {
            "dense": hybrid._dense_scores(q.question),
            "bm25": hybrid._bm25_scores(q.question),
        }
    return out


def retrieve_at(cache: dict, qid: str, w: float, ids: list[str], k: int) -> list[str]:
    """dense_weight = w 로 재랭킹한다. 재인코딩 없음."""
    c = cache[qid]
    dense, bm25 = c["dense"], c["bm25"]
    if not dense:                       # 임베더가 없으면 BM25 단독
        return _rank(bm25, k)
    fused = {cid: w * dense.get(cid, 0.0) + (1.0 - w) * bm25.get(cid, 0.0) for cid in ids}
    return _rank(fused, k)


def evaluate(cache, queries, w: float, ids: list[str], k: int) -> dict:
    return aggregate([(retrieve_at(cache, q.qid, w, ids, k), q.gold_chunks) for q in queries], k)


def correct_flags(cache, queries, w: float, ids: list[str], k: int) -> list[bool]:
    """McNemar 용 — 질의별 정답 여부(full_hit 기준)."""
    return [bool(full_hit_at_k(retrieve_at(cache, q.qid, w, ids, k), q.gold_chunks, k))
            for q in queries]


# ──────────────────────────────────────────────────────────────
# 그래프
# ──────────────────────────────────────────────────────────────

def plot(rows_dev, rows_test, best_w, out_png: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        matplotlib.rcParams["font.family"] = "Malgun Gothic"
        matplotlib.rcParams["axes.unicode_minus"] = False
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] 건너뜀: {e}")
        return

    ws = [r["w"] for r in rows_dev]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(ws, [r["full_hit"] for r in rows_dev], "o-", label="dev (여기서 고른다)", color="#c66b3d")
    ax.plot(ws, [r["full_hit"] for r in rows_test], "s--", label="test (보고용)", color="#3d7cc6")
    ax.axvline(DENSE_WEIGHT, color="#999", ls=":", lw=1.4)
    ax.text(DENSE_WEIGHT, ax.get_ylim()[0], f" 코드 고정값 {DENSE_WEIGHT}", fontsize=9, color="#666",
            va="bottom")
    ax.axvline(best_w, color="#2e7d32", ls="-.", lw=1.4)
    ax.text(best_w, ax.get_ylim()[1], f"dev 최적 {best_w} ", fontsize=9, color="#2e7d32",
            va="top", ha="right")
    ax.set_xlabel("dense_weight (0 = BM25 단독, 1 = dense 단독)")
    ax.set_ylabel("full hit@k")
    ax.set_title("하이브리드 배합비 스윕 — 어휘와 의미를 얼마씩 섞을 것인가")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    print(f"[plot] 저장: {out_png}")


# ──────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="합성 코퍼스로 검증")
    ap.add_argument("--out", default=str(ROOT / "output" / "hybrid_weight_sweep.json"))
    args = ap.parse_args()
    K = args.k

    if args.synthetic:
        from graph_rag.corpus import build_corpus
        from graph_rag.eval_set import build_queries
        chunks = build_corpus()
        queries = build_queries(chunks)
        corpus_desc = {"corpus": "synthetic", "chunks": len(chunks)}
    else:
        rc = pickle.load(open(CORPUS, "rb"))
        chunks, queries = rc.chunks, build_real_queries(rc)
        corpus_desc = corpus_summary(rc)
        print(f"[queries] {json.dumps(summarize_real(queries), ensure_ascii=False)}")
    print(f"[corpus] {json.dumps(corpus_desc, ensure_ascii=False)}")

    embedder = None
    dense_name = "none(BM25 only)"
    if not args.no_dense:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        embedder = SentenceTransformer("BAAI/bge-m3")
        print(f"[dense] BGE-M3 로드 {time.time() - t0:.1f}s · 청크 {len(chunks)}개 인코딩 중...")
        t0 = time.time()
        hybrid = HybridRetriever(chunks, embedder=embedder)
        dense_name = f"BAAI/bge-m3 (인코딩 {time.time() - t0:.1f}s)"
        print(f"[dense] {dense_name}")
    else:
        hybrid = HybridRetriever(chunks, embedder=None)

    ids = hybrid.ids
    dev, test = split_queries(queries)
    print(f"[split] dev {len(dev)}문항 · test {len(test)}문항")

    t0 = time.time()
    cache = component_scores(hybrid, queries)
    print(f"[score] 질의 {len(queries)}건 성분 점수 계산 {time.time() - t0:.1f}s "
          f"(이후 11개 가중치는 재인코딩 없이 재랭킹)")

    rows_dev, rows_test = [], []
    for w in GRID:
        rows_dev.append({"w": w, **evaluate(cache, dev, w, ids, K)})
        rows_test.append({"w": w, **evaluate(cache, test, w, ids, K)})

    # 가중치는 dev 에서만 고른다. 동점이면 낮은 w(=어휘 쪽)를 택해 보수적으로 간다.
    best = max(rows_dev, key=lambda r: (r["full_hit"], -r["w"]))
    best_w = best["w"]

    fixed_test = next(r for r in rows_test if r["w"] == DENSE_WEIGHT)
    best_test = next(r for r in rows_test if r["w"] == best_w)

    mc = mcnemar(correct_flags(cache, test, DENSE_WEIGHT, ids, K),
                 correct_flags(cache, test, best_w, ids, K))

    print("\n── dev 스윕 (여기서 고른다) ──")
    for r in rows_dev:
        mark = "  ← 최적" if r["w"] == best_w else ("  (코드 고정값)" if r["w"] == DENSE_WEIGHT else "")
        print(f"  w={r['w']:.1f}  full_hit {r['full_hit']:.4f}  mrr {r['mrr']:.4f}{mark}")

    print("\n── test 보고 ──")
    print(f"  코드 고정값 w={DENSE_WEIGHT}: full_hit {fixed_test['full_hit']:.4f} · mrr {fixed_test['mrr']:.4f}")
    print(f"  dev 최적    w={best_w}: full_hit {best_test['full_hit']:.4f} · mrr {best_test['mrr']:.4f}")
    print(f"  McNemar: delta {mc['delta']:+.4f} · p {mc['p_value']:.4f} "
          f"· 불일치 쌍 {mc['a_only']}/{mc['b_only']}")

    results = {
        "meta": {
            "corpus": corpus_desc,
            "dense": dense_name,
            "k": K,
            "grid": GRID,
            "split": {"dev": len(dev), "test": len(test)},
            "protocol": ("가중치는 dev에서만 고르고 test로 보고·검정한다. "
                         "run_real_eval.py의 graph_weight 규칙과 동일."),
            "fixed_in_code": {"dense": DENSE_WEIGHT, "bm25": BM25_WEIGHT},
        },
        "dev": rows_dev,
        "test": rows_test,
        "chosen_w": best_w,
        "fixed_test": fixed_test,
        "best_test": best_test,
        "mcnemar_fixed_vs_chosen": mc,
    }
    out = Path(args.out)
    if args.synthetic:
        out = out.with_name(out.stem + "_synthetic.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[json] 저장: {out}")

    png = ROOT / "output" / ("hybrid_weight_sweep_synthetic.png" if args.synthetic
                             else "hybrid_weight_sweep.png")
    plot(rows_dev, rows_test, best_w, png)


if __name__ == "__main__":
    main()
