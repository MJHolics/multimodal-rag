"""IE 감사 → 평가셋 정화 → 재집계.

5차는 모든 수치에 "IE 정밀도를 곱한 상한"이라는 단서를 달고 끝났다. 이 러너가 그 단서를 없앤다.

  1. 평가에 실제로 쓰인 속성 관계(HAS_BUDGET 49 · HAS_PERIOD 71)를 **전수** 감사한다.
     라벨은 `output/audit_labels.json`에 근거 문장과 함께 고정돼 있다.
  2. 오탐으로 판정된 관계에서 유도된 문항을 걸러낸다(2-hop은 다리가 틀려도 걸러진다).
  3. `real_eval.json`의 문항별 정오 기록으로 **검색을 다시 돌리지 않고** 재집계한다.

실행:
    python run_real_eval.py    # 먼저 per_query 기록을 만들어야 한다
    python run_audit.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

from graph_rag.stats import mcnemar, min_detectable_discordant
from knowledgeops.audit import (
    compare_before_after,
    invalid_triples,
    plan_audit,
    precision,
    purify,
)
from knowledgeops.eval_real import build_real_queries_with_sources

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
OUT = ROOT / "output"
STRATEGIES = ["hybrid_flat", "graph_rel", "fused_rel", "agentic"]
# 감사 대상 관계의 실제 모집단(라벨 건수와 다를 수 있다 — 표본 감사인 경우)
POPULATION = {"HAS_BUDGET": 49, "HAS_PERIOD": 71, "REQUIRES": 853}


def main() -> None:
    labels = json.loads((OUT / "audit_labels.json").read_text(encoding="utf-8"))
    report_in = json.loads((OUT / "real_eval.json").read_text(encoding="utf-8"))
    if "per_query" not in report_in:
        raise SystemExit("real_eval.json에 per_query가 없다 — run_real_eval.py를 먼저 돌릴 것")
    rc = pickle.loads((OUT / "rfp_corpus.pkl").read_bytes())

    recs = labels["records"]
    out: dict = {"precision": {}, "purify": {}, "metrics": [], "significance": {}}

    # ---- 1. 정밀도 (전수) ----
    print("=== IE 정밀도 — 평가에 쓰인 관계 전수 감사 ===")
    print(f"{'관계':<12} {'모집단':>6} {'감사':>5} {'정확':>5} {'오탐':>5} {'정밀도':>8}  방식")
    for rtype in ("HAS_BUDGET", "HAS_PERIOD", "REQUIRES"):
        rows = [r for r in recs if r["rtype"] == rtype]
        # REQUIRES는 모집단(853)이 감사 예산을 넘어 표본만 라벨돼 있다.
        pop = POPULATION.get(rtype, len(rows))
        plan = plan_audit(rtype, pop)
        p = precision([r["label"] for r in rows], pop, plan.mode)
        out["precision"][rtype] = p | {"plan": plan.reason}
        ci = "" if p["interval"] is None else f"  95% CI [{p['interval'][0]:.3f}, {p['interval'][1]:.3f}]"
        print(f"{rtype:<12} {p['population']:>6} {p['audited']:>5} {p['ok']:>5} "
              f"{p['bad']:>5} {p['precision']:>8.3f}  {plan.mode}{ci}")

    # 오탐 유형 분포 — 원인을 모르면 고칠 수 없다
    kinds: dict[str, int] = {}
    for r in recs:
        if r["label"] == "bad":
            kinds[r["reason"]] = kinds.get(r["reason"], 0) + 1
    out["bad_kinds"] = dict(sorted(kinds.items(), key=lambda x: -x[1]))
    print("\n오탐 유형:", ", ".join(f"{k} {v}" for k, v in out["bad_kinds"].items()))

    # ---- 2. 정화 ----
    invalid = {r["rc_key"] for r in recs if r["label"] == "bad"}
    sources = {q.qid: list(s) for q, s in build_real_queries_with_sources(rc)}
    per_q = report_in["per_query"]
    split = purify(per_q, sources, invalid)
    kept, dropped = split["kept"], split["dropped"]
    out["purify"] = {
        "invalid_triples": len(invalid),
        "queries_before": len(per_q),
        "queries_after": len(kept),
        "dropped": len(dropped),
        "dropped_multi_hop": sum(1 for r in dropped if r["hops"] == 2),
        "after_single_hop": sum(1 for r in kept if r["hops"] == 1),
        "after_multi_hop": sum(1 for r in kept if r["hops"] == 2),
    }
    print(f"\n=== 정화 === 오탐 관계 {len(invalid)}개 → 문항 {len(per_q)} → {len(kept)} "
          f"(제외 {len(dropped)}, 그중 멀티홉 {out['purify']['dropped_multi_hop']})")

    # ---- 3. 재집계 ----
    print(f"\n{'strategy':<14} {'split':<12} {'전(n)':>10} {'후(n)':>10} {'Δ':>8}")
    for s in STRATEGIES:
        for label, hops in [("all", None), ("single_hop", 1), ("multi_hop", 2)]:
            c = compare_before_after(per_q, kept, s, hops)
            out["metrics"].append(c)
            print(f"{s:<14} {label:<12} {c['before']['full_hit']:>6.3f}({c['before']['n']:>3})"
                  f" {c['after']['full_hit']:>6.3f}({c['after']['n']:>3}) {c['delta']:>+8.3f}")

    # ---- 4. 정화 후에도 결론이 서는가 ----
    need = min_detectable_discordant()
    print(f"\n=== 정화 후 짝지음 검정 (불일치 {need}쌍 필요) ===")
    for a, b in [("hybrid_flat", "graph_rel"), ("hybrid_flat", "fused_rel"),
                 ("graph_rel", "fused_rel")]:
        for label, hops in [("all", None), ("multi_hop", 2)]:
            rows = [r for r in kept if hops is None or r["hops"] == hops]
            res = mcnemar([r["correct"][a] for r in rows],
                          [r["correct"][b] for r in rows])
            if res["discordant"] < need:
                res["verdict"] = f"판정불가({res['discordant']}<{need})"
            else:
                res["verdict"] = "유의" if res["p_value"] < 0.05 else "유의하지 않음"
            out["significance"][f"{a}→{b}[{label}]"] = res
            print(f"{a}→{b}[{label}]".ljust(34)
                  + f"n={res['n']:>3} {res['acc_a']:>6.3f}→{res['acc_b']:>6.3f} "
                  f"불일치 {res['a_only']}:{res['b_only']:<3} p={res['p_value']:.4f}  {res['verdict']}")

    (OUT / "audit_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {OUT / 'audit_report.json'}")


if __name__ == "__main__":
    main()
