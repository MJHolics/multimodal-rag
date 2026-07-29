"""GraphRAG vs 플랫 하이브리드 실측 — 단일홉/멀티홉 분리 비교.

실행:
  python run_graph_eval.py                 # BGE-M3 dense + BM25 (현행 파이프라인과 동일)
  python run_graph_eval.py --no-dense      # BM25 단독(모델 없이 빠른 배선 확인)
  python run_graph_eval.py --k 5

결과는 output/graph_eval.json에 저장. 수치는 전부 결정적(랜덤 없음)이라 재실행하면 같은 값이 나온다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from graph_rag.corpus import build_corpus
from graph_rag.eval_set import build_queries
from graph_rag.metrics import aggregate
from graph_rag.retrieval import FusedRetriever, GraphRetriever, HybridRetriever

OUT = Path(__file__).parent / "output" / "graph_eval.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="top-k")
    ap.add_argument("--no-dense", action="store_true", help="BM25 단독(임베딩 생략)")
    ap.add_argument("--graph-weight", type=float, default=0.5)
    ap.add_argument("--no-distractors", action="store_true",
                    help="distractor 제외(1차 실험 조건 재현 — 코퍼스 난이도 분리 측정)")
    ap.add_argument("--sweep", action="store_true",
                    help="graph_weight 0.0~1.0 스윕 후 최적값 보고(고정값 0.5를 실측으로 대체)")
    ap.add_argument("--out", default=None, help="결과 JSON 경로(기본 output/graph_eval.json)")
    args = ap.parse_args()

    chunks = build_corpus(with_distractors=not args.no_distractors)
    queries = build_queries(chunks)
    K = args.k

    embedder = None
    dense_name = "none(BM25 only)"
    if not args.no_dense:
        from sentence_transformers import SentenceTransformer

        t0 = time.time()
        embedder = SentenceTransformer("BAAI/bge-m3")
        dense_name = f"BAAI/bge-m3 (load {time.time() - t0:.1f}s)"

    hybrid = HybridRetriever(chunks, embedder=embedder)
    graph = GraphRetriever(chunks)                      # mention 점수(1차 실험과 동일)
    graph_rel = GraphRetriever(chunks, mode="relation")  # 관계 인식 점수(2026-07-29 추가)
    fused = FusedRetriever(hybrid, graph, graph_weight=args.graph_weight)
    fused_rel = FusedRetriever(hybrid, graph_rel, graph_weight=args.graph_weight)

    strategies = {
        # 현행 파이프라인 그대로 (Dense 70% + BM25 30%)
        "hybrid_flat": lambda q: hybrid.retrieve(q.question, K),
        # 그래프만 — 시드 1개(평가셋 지정)
        "graph_given": lambda q: graph.retrieve(q.question, K, list(q.seed_entities)),
        # 그래프만 — 질의에서 엔티티 링킹(엔드투엔드)
        "graph_linked": lambda q: graph.retrieve(q.question, K),
        # 그래프만 — 관계 인식 점수
        "graph_rel": lambda q: graph_rel.retrieve(q.question, K),
        # 융합
        "fused": lambda q: fused.retrieve(q.question, K),
        "fused_rel": lambda q: fused_rel.retrieve(q.question, K),
    }

    report: dict = {
        "config": {
            "k": K,
            "dense": dense_name,
            "weights": "dense 0.7 / bm25 0.3 (app/retriever.py와 동일)",
            "graph_weight": args.graph_weight,
            "chunks": len(chunks),
            "distractors": sum(1 for c in chunks if not c.states),
            "queries": len(queries),
        },
        "results": {},
    }

    for name, fn in strategies.items():
        per_split: dict[str, dict] = {}
        for label, hops in [("all", None), ("single_hop", 1), ("multi_hop", 2)]:
            sub = [q for q in queries if hops is None or q.hops == hops]
            per_split[label] = aggregate([(fn(q), q.gold_chunks) for q in sub], K)
        report["results"][name] = per_split

    # 질의별 상세(어디서 이기고 지는지 보려고)
    report["per_query"] = [
        {
            "qid": q.qid,
            "hops": q.hops,
            "question": q.question,
            "gold": list(q.gold_chunks),
            **{name: fn(q) for name, fn in strategies.items()},
        }
        for q in queries
    ]

    # graph_weight 스윕 — 0.5는 처음에 눈대중으로 고른 값이었다. 하이브리드의 0.7/0.3을
    # 실측으로 정했듯 이 값도 실측으로 정한다. 단일홉/멀티홉을 나눠 봐야 하는 이유는
    # 최적 가중치가 두 유형에서 다를 수 있기 때문이다(그래프는 멀티홉에서만 이득).
    if args.sweep:
        report["sweep"] = {}
        for gname, g in [("mention", graph), ("relation", graph_rel)]:
            sweep: list[dict] = []
            for i in range(11):
                w = round(i * 0.1, 1)
                fr = FusedRetriever(hybrid, g, graph_weight=w)
                row: dict = {"graph_weight": w}
                for label, hops in [("all", None), ("single_hop", 1), ("multi_hop", 2)]:
                    sub = [q for q in queries if hops is None or q.hops == hops]
                    m = aggregate(
                        [(fr.retrieve(q.question, K), q.gold_chunks) for q in sub], K
                    )
                    row[label] = m["full_hit"]
                sweep.append(row)
            # 동률이면 그래프 비중이 작은 쪽을 고른다 — 이득이 없는데 복잡도만 늘릴 이유가 없다.
            best = max(sweep, key=lambda r: (r["all"], -r["graph_weight"]))
            report["sweep"][gname] = {"rows": sweep, "best_by_all": best}
            print(f"\n=== graph_weight 스윕 [{gname}] (top-{K}, full_hit) ===")
            print(f"{'w':>5} {'all':>8} {'single':>8} {'multi':>8}")
            for r in sweep:
                mark = "  <= best" if r is best else ""
                print(f"{r['graph_weight']:>5.1f} {r['all']:>8.3f} {r['single_hop']:>8.3f} "
                      f"{r['multi_hop']:>8.3f}{mark}")

    out_path = Path(args.out) if args.out else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 콘솔 요약표
    print(f"\n=== GraphRAG vs 플랫 하이브리드 (top-{K}, dense={dense_name}) ===")
    print(f"{'strategy':<14} {'split':<12} {'full_hit':>9} {'hit':>7} {'mrr':>7} {'recall':>8}")
    for name, splits in report["results"].items():
        for label, m in splits.items():
            print(
                f"{name:<14} {label:<12} {m['full_hit']:>9.3f} {m['hit']:>7.3f} "
                f"{m['mrr']:>7.3f} {m['recall']:>8.3f}"
            )
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
