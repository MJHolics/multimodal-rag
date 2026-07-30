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
from graph_rag.eval_gen import (
    build_generated_queries,
    leaked_bridges,
    split_queries,
    summarize,
)
from graph_rag.eval_set import build_queries
from graph_rag.metrics import aggregate, full_hit_at_k
from graph_rag.agentic import IterativeRetriever, sweep_rounds
from graph_rag.retrieval import FusedRetriever, GraphRetriever, HybridRetriever
from graph_rag.stats import mcnemar, min_detectable_discordant, wilson_interval

OUT = Path(__file__).parent / "output" / "graph_eval.json"

# 짝지어 검정할 전략 쌍 — "무엇을 주장하려는가"를 코드에 박아둔다.
_COMPARISONS = [
    ("hybrid_flat", "graph_rel"),    # 그래프가 현행 하이브리드를 이기는가
    ("graph_linked", "graph_rel"),   # 점수 함수 수정(언급→관계)이 실제 개선인가
    ("hybrid_flat", "fused_rel"),    # 융합이 현행을 이기는가
    ("graph_rel", "fused_rel"),      # 융합이 그래프 단독보다 나은가
    ("fused_rel", "agentic"),        # 반복 검색이 한 번의 검색보다 나은가
    ("hybrid_flat", "agentic"),      # 현행 대비 최종 이득
]

# 이 평가셋에서 **하이퍼파라미터를 고른** 전략. 이들이 낀 비교만 홀드아웃 test에서 검정한다.
#
# 왜 구분하나: 홀드아웃은 공짜가 아니다 — 148문항을 나누면 검정력이 절반이 된다.
# 그런데 선택 편향은 "이 데이터로 뭔가를 골랐을 때"만 생긴다. hybrid_flat(0.7/0.3은 이 실험
# 이전에 정해짐)·graph_rel·graph_linked는 이 평가셋에서 고른 값이 하나도 없으므로
# **전체 148문항에서 검정해도 편향이 없다.** graph_weight를 고른 fused 계열만 test로 보낸다.
# 무차별로 전부 홀드아웃에 넣으면 필요 없는 곳에서까지 검정력을 버리게 된다.
_TUNED = {"fused", "fused_rel", "agentic"}


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
    ap.add_argument("--holdout", action="store_true",
                    help="graph_weight는 dev에서 고르고 검정은 test에서 — 선택 편향 차단")
    ap.add_argument("--auto", action="store_true",
                    help="손수 쓴 16문항 대신 **그래프에서 생성한** 평가셋을 쓴다(검정력 확보)")
    args = ap.parse_args()

    chunks = build_corpus(with_distractors=not args.no_distractors)
    if args.auto:
        queries = build_generated_queries(chunks)
        leaks = leaked_bridges(queries, chunks)
        if leaks:  # 구조적으로 불가능하지만 템플릿을 손대면 깨질 수 있다
            raise SystemExit(f"평가셋에 다리 엔티티 누출: {leaks[:5]}")
        print(f"[eval_gen] 생성 평가셋 {summarize(queries)}")
    else:
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

    # ------------------------------------------------------------------
    # 홀드아웃 — graph_weight를 **dev에서 고르고** 검정은 test에서 한다.
    #
    # 스윕이 최적 가중치를 골라 놓고 그 가중치의 유의성을 같은 평가셋에서 주장하면,
    # 11개 후보 중 최고를 고른 뒤 그 데이터로 다시 정당화하는 셈이라 p값이 낙관적으로 기운다.
    # (하이브리드의 0.7/0.3은 이 실험 이전에 정해진 값이라 해당 없음.)
    # ------------------------------------------------------------------
    holdout: dict | None = None
    if args.holdout:
        if not args.auto:
            raise SystemExit("--holdout은 --auto(생성 평가셋)와 함께 써야 한다")
        dev, test = split_queries(queries)
        picked: dict[str, float] = {}
        print(f"\n=== 홀드아웃: dev {len(dev)}문항에서 graph_weight 선택 ===")
        for gname, g in [("mention", graph), ("relation", graph_rel)]:
            rows = []
            for i in range(11):
                w = round(i * 0.1, 1)
                fr = FusedRetriever(hybrid, g, graph_weight=w)
                m = aggregate(
                    [(fr.retrieve(q.question, K), q.gold_chunks) for q in dev], K
                )
                rows.append({"graph_weight": w, "full_hit": m["full_hit"]})
            # 동률이면 그래프 비중이 작은 쪽 — 이득이 없는데 복잡도만 늘릴 이유가 없다
            best = max(rows, key=lambda r: (r["full_hit"], -r["graph_weight"]))
            picked[gname] = best["graph_weight"]
            print(f"  {gname}: w={best['graph_weight']} (dev full_hit {best['full_hit']:.3f})")
        test_ids = {q.qid for q in test}
        holdout = {
            "dev_n": len(dev), "test_n": len(test), "picked": picked,
            "note": "가중치는 dev에서 선택. 튜닝된 fused 계열이 낀 검정만 test에서, "
                    "튜닝 없는 전략끼리는 전체에서 잰다(불필요한 검정력 손실 방지).",
        }
        graph_w_mention = picked["mention"]
        graph_w_relation = picked["relation"]
    else:
        test_ids = None
        graph_w_mention = graph_w_relation = args.graph_weight

    fused = FusedRetriever(hybrid, graph, graph_weight=graph_w_mention)
    fused_rel = FusedRetriever(hybrid, graph_rel, graph_weight=graph_w_relation)

    # 반복 검색 — 라운드 예산도 하이퍼파라미터이므로 dev에서 고른다(선택 편향 차단).
    def _mk_iter(r: int) -> IterativeRetriever:
        return IterativeRetriever(chunks, graph=graph_rel, hybrid=hybrid,
                                  rounds=r, graph_weight=graph_w_relation)

    if args.holdout:
        curve = sweep_rounds(_mk_iter, [q for q in dev if q.hops == 2], K, max_rounds=4)
        best_r = max(curve, key=lambda x: (x["full_hit"], -x["rounds"]))["rounds"]
        print(f"  agentic rounds={best_r} (dev 멀티홉 곡선 {[(c['rounds'], c['full_hit']) for c in curve]})")
        holdout["picked"]["agentic_rounds"] = best_r
    else:
        best_r = 2
    agentic = _mk_iter(best_r)

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
        # 반복 검색(agentic) — 라운드마다 앞서 읽은 문서에서 다음 시드를 얻는다
        "agentic": lambda q: agentic.retrieve(q.question, K, list(q.seed_entities)),
    }

    report: dict = {
        "config": {
            "k": K,
            "dense": dense_name,
            "weights": "dense 0.7 / bm25 0.3 (app/retriever.py와 동일)",
            "graph_weight": {"mention": graph_w_mention, "relation": graph_w_relation},
            "chunks": len(chunks),
            "distractors": sum(1 for c in chunks if not c.states),
            "queries": len(queries),
            "query_source": "generated(eval_gen)" if args.auto else "handwritten(eval_set)",
        },
        "holdout": holdout,
        "results": {},
    }

    # 전략별 검색 결과를 **한 번만** 계산해 재사용한다(집계·질의별 상세·짝지음 검정이
    # 같은 결과를 봐야 한다. 따로 호출하면 미묘한 불일치가 생길 여지가 있다).
    retrieved: dict[str, list[list[str]]] = {
        name: [fn(q) for q in queries] for name, fn in strategies.items()
    }
    # 질의별 성공 여부(full_hit 0/1) — McNemar의 입력
    correct: dict[str, list[bool]] = {
        name: [
            full_hit_at_k(r, q.gold_chunks, K) == 1.0
            for r, q in zip(rows, queries)
        ]
        for name, rows in retrieved.items()
    }

    for name, rows in retrieved.items():
        per_split: dict[str, dict] = {}
        for label, hops in [("all", None), ("single_hop", 1), ("multi_hop", 2)]:
            pairs = [
                (r, q.gold_chunks)
                for r, q in zip(rows, queries)
                if hops is None or q.hops == hops
            ]
            per_split[label] = aggregate(pairs, K)
        report["results"][name] = per_split

    # 홀드아웃일 때: fused 계열의 전체-셋 점수는 **가중치를 고른 데이터에서 잰 값**이라
    # 낙관적이다. test 전용 점수를 따로 남겨 둘을 구분할 수 있게 한다.
    if test_ids is not None:
        report["results_test"] = {}
        for name, rows in retrieved.items():
            per_split = {}
            for label, hops in [("all", None), ("single_hop", 1), ("multi_hop", 2)]:
                pairs = [
                    (r, q.gold_chunks)
                    for r, q in zip(rows, queries)
                    if (hops is None or q.hops == hops) and q.qid in test_ids
                ]
                per_split[label] = aggregate(pairs, K)
            report["results_test"][name] = per_split

    # 질의별 상세(어디서 이기고 지는지 보려고)
    report["per_query"] = [
        {
            "qid": q.qid,
            "hops": q.hops,
            "question": q.question,
            "gold": list(q.gold_chunks),
            **{name: retrieved[name][i] for name in strategies},
        }
        for i, q in enumerate(queries)
    ]

    # ------------------------------------------------------------------
    # 짝지음 유의성 검정 — 집계값 차이가 우연인지 판정한다.
    #
    # 이걸 코드로 내리는 이유: NOTES.md에 남긴 1차 p값들은 임시 계산이라 재현이 안 됐다.
    # 집계값만 남기면 개선을 주장할 수도, 반박할 수도 없다.
    # ------------------------------------------------------------------
    need = min_detectable_discordant()
    report["significance"] = {"min_discordant_for_p05": need, "tests": {}}
    print(f"\n=== 짝지음 검정 (McNemar 정확검정, full_hit@{K}, n={len(queries)}) ===")
    print(f"(유의 판정이 가능하려면 불일치 쌍이 최소 {need}개 필요)")
    print(f"{'비교':<30} {'n':>4} {'A':>7} {'B':>7} {'Δ':>8} {'불일치':>9} {'p':>9}  판정")
    for a, b in _COMPARISONS:
        # 튜닝된 전략이 끼면 홀드아웃 test에서만, 아니면 전체 평가셋에서 검정
        tuned = bool(_TUNED & {a, b})
        on_test = tuned and test_ids is not None
        for label, hops in [("all", None), ("multi_hop", 2)]:
            idx = [
                i for i, q in enumerate(queries)
                if (hops is None or q.hops == hops)
                and (not on_test or q.qid in test_ids)
            ]
            va = [correct[a][i] for i in idx]
            vb = [correct[b][i] for i in idx]
            res = mcnemar(va, vb)
            res["wilson_a"] = wilson_interval(sum(va), len(va))
            res["wilson_b"] = wilson_interval(sum(vb), len(vb))
            # 불일치 쌍이 부족하면 p값을 읽는 것 자체가 무의미하다 — 명시적으로 구분한다.
            if res["discordant"] < need:
                verdict = f"판정불가(불일치 {res['discordant']}<{need})"
            elif res["p_value"] < 0.05:
                verdict = "유의"
            else:
                verdict = "유의하지 않음"
            res["verdict"] = verdict
            res["eval_on"] = "holdout_test" if on_test else "full"
            report["significance"]["tests"][f"{a}→{b}[{label}]"] = res
            mark = "*" if on_test else " "
            print(
                f"{a + '→' + b + '[' + label + ']':<29}{mark} {len(idx):>4} "
                f"{res['acc_a']:>7.3f} {res['acc_b']:>7.3f} {res['delta']:>+8.3f} "
                f"{res['a_only']:>4}:{res['b_only']:<4} {res['p_value']:>9.4f}  {verdict}"
            )
    if test_ids is not None:
        print("  * = 홀드아웃 test 전용(그 전략이 이 평가셋에서 가중치를 골랐기 때문)")

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
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
