"""model_comparison.json을 그래프로 — full_hit·MRR을 나란히, 판정불가는 표시로 구분.

run_hybrid_weight.py의 plot() 관례를 따른다. 막대에 유의성 표시를 붙여 "차이가 없다"와
"판정할 힘이 없다"를 시각적으로도 구분한다(README/DECISIONS.md G32와 같은 원칙).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
MIN_DISCORDANT = 6  # graph_rag/stats.min_detectable_discordant()


def main() -> None:
    d = json.loads((ROOT / "output" / "model_comparison.json").read_text(encoding="utf-8"))
    emb = d["embedders"]
    rerank = d["rerank"]
    splade = json.loads((ROOT / "output" / "splade_comparison.json").read_text(encoding="utf-8"))

    labels = ["BGE-M3+BM25\n(기본값)", "KURE-v1+BM25", "multilingual-e5\n-large+BM25",
              "BGE-M3+BM25\n→rerank", "SPLADE\n단독", "BGE-M3+SPLADE\n(dev 최적)"]
    full_hit = [
        emb["bge-m3"]["best_test"]["full_hit"],
        emb["kure-v1"]["best_test"]["full_hit"],
        emb["multilingual-e5-large"]["best_test"]["full_hit"],
        rerank["test_with_rerank"]["full_hit"],
        splade["sparse_only"]["splade"]["full_hit"],
        splade["hybrid"]["best_test_splade"]["full_hit"],
    ]
    mrr = [
        emb["bge-m3"]["best_test"]["mrr"],
        emb["kure-v1"]["best_test"]["mrr"],
        emb["multilingual-e5-large"]["best_test"]["mrr"],
        rerank["test_with_rerank"]["mrr"],
        splade["sparse_only"]["splade"]["mrr"],
        splade["hybrid"]["best_test_splade"]["mrr"],
    ]
    # 판정 라벨: 검정력 하한(6쌍) 기준으로 유의/판정불가/유의(나쁨) 구분
    discordant = [
        None,
        emb["kure-v1"]["mcnemar_vs_bge_m3"]["discordant"],
        emb["multilingual-e5-large"]["mcnemar_vs_bge_m3"]["discordant"],
        rerank["mcnemar_no_rerank_vs_rerank"]["discordant"],
        splade["sparse_only"]["mcnemar"]["discordant"],
        splade["hybrid"]["mcnemar_bm25_vs_splade_hybrid"]["discordant"],
    ]
    p_values = [
        None,
        emb["kure-v1"]["mcnemar_vs_bge_m3"]["p_value"],
        emb["multilingual-e5-large"]["mcnemar_vs_bge_m3"]["p_value"],
        rerank["mcnemar_no_rerank_vs_rerank"]["p_value"],
        splade["sparse_only"]["mcnemar"]["p_value"],
        splade["hybrid"]["mcnemar_bm25_vs_splade_hybrid"]["p_value"],
    ]

    def verdict(i: int) -> str:
        if discordant[i] is None:
            return "baseline"
        ref = "vs BM25단독" if i == 4 else "vs 기본값"
        if discordant[i] < MIN_DISCORDANT:
            return f"판정불가({ref})\n(불일치{discordant[i]}<{MIN_DISCORDANT})"
        sig = "유의" if p_values[i] < 0.05 else "유의하지 않음"
        return f"{sig}({ref})\n(p={p_values[i]:.4f})"

    x = range(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

    colors = ["#666666", "#3d7cc6", "#c0392b", "#2e7d32", "#8e44ad", "#8e44ad"]

    ax = axes[0]
    bars = ax.bar(x, full_hit, color=colors)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("full_hit@3 (test 67문항)")
    ax.set_title("주 지표 — 멀티홉 gold 전부 확보율")
    ax.set_ylim(0, 0.9)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{full_hit[i]:.3f}",
                ha="center", fontsize=9, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, 0.03, verdict(i), ha="center", fontsize=6.6,
                color="white")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    bars = ax.bar(x, mrr, color=colors)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MRR (test 67문항)")
    ax.set_title("순위 지표 — 정답이 얼마나 상위에 오는가")
    ax.set_ylim(0, 0.9)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{mrr[i]:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("다른 임베더·rerank 실측 — \"판정불가\"와 \"유의하게 나쁨\"은 다른 결론이다", fontsize=11)
    fig.tight_layout()
    out = ROOT / "output" / "model_comparison.png"
    fig.savefig(out, dpi=140)
    print(f"[plot] 저장: {out}")


if __name__ == "__main__":
    main()
