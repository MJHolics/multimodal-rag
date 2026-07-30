"""실문서(공공조달 RFP 96건) 검색 전략 비교 — 합성 코퍼스 결론이 실제 조건에서도 성립하는가.

실행:
  python run_real_eval.py --no-dense     # BM25 단독(빠른 확인)
  python run_real_eval.py                # BGE-M3 + BM25 (실제 파이프라인 조건)

합성 코퍼스(72청크·135문항)에서 낸 결론을 실문서(7,865청크·176문항)에서 다시 잰다.
바뀐 조건: ①관계가 IE의 산물이라 불완전 ②방해 문서가 90% ③표기가 흔들린다.
수치는 결정적이라 재실행하면 같은 값이 나온다.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

from graph_rag.agentic import IterativeRetriever, sweep_rounds
from graph_rag.eval_gen import split_queries
from graph_rag.metrics import aggregate, full_hit_at_k
from graph_rag.retrieval import FusedRetriever, GraphRetriever, HybridRetriever
from graph_rag.stats import mcnemar, min_detectable_discordant
from knowledgeops.chunking import corpus_summary
from knowledgeops.eval_real import build_real_queries, summarize_real

ROOT = Path(__file__).parent
CORPUS = ROOT / "output" / "rfp_corpus.pkl"

_COMPARISONS = [
    ("hybrid_flat", "graph_rel"),
    ("hybrid_flat", "fused_rel"),
    ("graph_rel", "fused_rel"),
    ("fused_rel", "agentic"),
]
_TUNED = {"fused", "fused_rel", "agentic"}


def _link_entities_factory(rc):
    """실문서용 엔티티 링킹 — 질의 문자열에서 엔티티 표면형을 찾는다.

    `graph_rag.retrieval.link_entities`는 합성 코퍼스의 전역 ENTITY_BY_ID를 본다.
    실문서는 엔티티 집합이 다르므로 여기서 만들어 주입한다. 긴 이름부터 매칭하는
    규칙은 동일하다(부분 포함 오탐 방지).
    """
    items = sorted(rc.entities.values(), key=lambda e: -len(e.name))

    def link(query: str) -> list[str]:
        found: list[str] = []
        for e in items:
            if len(e.name) >= 2 and e.name in query:
                if any(e.name in rc.entities[f].name for f in found):
                    continue
                found.append(e.eid)
        return found

    return link


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "output" / "real_eval.json"))
    args = ap.parse_args()

    rc = pickle.load(open(CORPUS, "rb"))
    queries = build_real_queries(rc)
    K = args.k
    print(f"[corpus] {json.dumps(corpus_summary(rc), ensure_ascii=False)}")
    print(f"[queries] {json.dumps(summarize_real(queries), ensure_ascii=False)}")

    embedder = None
    dense_name = "none(BM25 only)"
    if not args.no_dense:
        from sentence_transformers import SentenceTransformer

        t0 = time.time()
        embedder = SentenceTransformer("BAAI/bge-m3")
        print(f"[dense] BGE-M3 로드 {time.time() - t0:.1f}s · 청크 {len(rc.chunks)}개 인코딩 중...")
        t0 = time.time()
        hybrid = HybridRetriever(rc.chunks, embedder=embedder)
        dense_name = f"BAAI/bge-m3 (인코딩 {time.time() - t0:.1f}s)"
        print(f"[dense] {dense_name}")
    else:
        hybrid = HybridRetriever(rc.chunks, embedder=None)

    # 그래프 검색은 실문서 엔티티/관계를 봐야 한다 — 링킹 함수를 갈아 끼운다
    import graph_rag.retrieval as R

    R.link_entities = _link_entities_factory(rc)
    from graph_rag.graph_store import InMemoryGraphStore

    store = InMemoryGraphStore(rc.relations)
    graph_rel = GraphRetriever(rc.chunks, store=store, mode="relation")

    dev, test = split_queries(queries)
    test_ids = {q.qid for q in test}

    # 융합 가중치·라운드 예산은 dev에서 고른다(선택 편향 차단 — 합성 실험과 같은 규칙)
    rows = []
    for i in range(11):
        w = round(i * 0.1, 1)
        fr = FusedRetriever(hybrid, graph_rel, graph_weight=w)
        m = aggregate([(fr.retrieve(q.question, K), q.gold_chunks) for q in dev], K)
        rows.append({"w": w, "full_hit": m["full_hit"]})
    best_w = max(rows, key=lambda r: (r["full_hit"], -r["w"]))["w"]
    print(f"[holdout] graph_weight={best_w} (dev {len(dev)}문항)")

    def _mk(r: int) -> IterativeRetriever:
        return IterativeRetriever(rc.chunks, graph=graph_rel, hybrid=hybrid,
                                  rounds=r, graph_weight=best_w)

    curve = sweep_rounds(_mk, [q for q in dev if q.hops == 2], K, max_rounds=4)
    best_r = max(curve, key=lambda x: (x["full_hit"], -x["rounds"]))["rounds"]
    print(f"[holdout] agentic rounds={best_r} · dev 곡선 "
          f"{[(c['rounds'], c['full_hit']) for c in curve]}")

    fused_rel = FusedRetriever(hybrid, graph_rel, graph_weight=best_w)
    agentic = _mk(best_r)

    strategies = {
        "hybrid_flat": lambda q: hybrid.retrieve(q.question, K),
        "graph_rel": lambda q: graph_rel.retrieve(q.question, K),
        "fused_rel": lambda q: fused_rel.retrieve(q.question, K),
        "agentic": lambda q: agentic.retrieve(q.question, K, list(q.seed_entities)),
    }

    retrieved = {n: [f(q) for q in queries] for n, f in strategies.items()}
    correct = {
        n: [full_hit_at_k(r, q.gold_chunks, K) == 1.0 for r, q in zip(rows_, queries)]
        for n, rows_ in retrieved.items()
    }

    report = {
        "config": {
            "k": K, "dense": dense_name, "graph_weight": best_w,
            "agentic_rounds": best_r,
            "corpus": corpus_summary(rc), "queries": summarize_real(queries),
        },
        "sweep": {"graph_weight": rows, "agentic_rounds": curve},
        "results": {},
        "significance": {},
    }

    print(f"\n=== 실문서 검색 비교 (top-{K}, dense={dense_name}) ===")
    print(f"{'strategy':<14} {'split':<12} {'full_hit':>9} {'hit':>7} {'mrr':>7} {'recall':>8}")
    for n, rows_ in retrieved.items():
        per = {}
        for label, hops in [("all", None), ("single_hop", 1), ("multi_hop", 2)]:
            pairs = [(r, q.gold_chunks) for r, q in zip(rows_, queries)
                     if hops is None or q.hops == hops]
            per[label] = aggregate(pairs, K)
            m = per[label]
            print(f"{n:<14} {label:<12} {m['full_hit']:>9.3f} {m['hit']:>7.3f} "
                  f"{m['mrr']:>7.3f} {m['recall']:>8.3f}")
        report["results"][n] = per

    need = min_detectable_discordant()
    print(f"\n=== 짝지음 검정 (McNemar, full_hit@{K}) — 유의 판정에 불일치 {need}쌍 필요 ===")
    print(f"{'비교':<28} {'n':>4} {'A':>7} {'B':>7} {'Δ':>8} {'불일치':>9} {'p':>9}  판정")
    for a, b in _COMPARISONS:
        on_test = bool(_TUNED & {a, b})
        for label, hops in [("all", None), ("multi_hop", 2)]:
            idx = [i for i, q in enumerate(queries)
                   if (hops is None or q.hops == hops)
                   and (not on_test or q.qid in test_ids)]
            va = [correct[a][i] for i in idx]
            vb = [correct[b][i] for i in idx]
            res = mcnemar(va, vb)
            if res["discordant"] < need:
                verdict = f"판정불가({res['discordant']}<{need})"
            elif res["p_value"] < 0.05:
                verdict = "유의"
            else:
                verdict = "유의하지 않음"
            res["verdict"] = verdict
            res["eval_on"] = "holdout_test" if on_test else "full"
            report["significance"][f"{a}→{b}[{label}]"] = res
            mark = "*" if on_test else " "
            print(f"{a + '→' + b + '[' + label + ']':<27}{mark} {len(idx):>4} "
                  f"{res['acc_a']:>7.3f} {res['acc_b']:>7.3f} {res['delta']:>+8.3f} "
                  f"{res['a_only']:>4}:{res['b_only']:<4} {res['p_value']:>9.4f}  {verdict}")
    print("  * = 홀드아웃 test 전용")

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
