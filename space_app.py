"""TechDocRAG — 검색 전략 비교 데모 (Hugging Face Space 진입점).

## 이 데모가 보여주려는 것

"RAG 만들어봤습니다"는 누구나 쓴다. 여기서 보여주는 건 **검색 전략을 어떻게 비교·검증하는가**다.
같은 질문에 세 가지 검색이 각각 무엇을 가져오는지 나란히 놓고, 정답/오답/함정 문서를 표시한다.

  - 플랫 하이브리드 — BGE-M3 dense 70% + BM25 30% (현행 파이프라인)
  - 그래프(엔티티 언급) — 시드 엔티티에서 그래프를 확장, 확장 집합을 *언급하는* 청크를 올림
  - 그래프(관계 인식) — 확장 집합 안에서 **양 끝점이 모두 닿는 관계를 서술하는** 청크만 올림
  - 융합 — 하이브리드 + 관계 인식 그래프

## 함정 문서(distractor)가 핵심이다

코퍼스 51개 중 24개는 **엔티티는 언급하지만 답은 아닌** 문서다(안전 주의사항·정비 절차·
구형 모델 사양·용어집). 처음엔 이런 문서 없이 15개로만 쟀는데, 그때는 그래프 융합이 이겼다.
함정을 넣자 결과가 뒤집혔고, 원인이 "내 그래프 점수가 관계가 아니라 엔티티 언급을 재고 있었다"
는 것이었다. 이 데모는 그 장면을 직접 눌러볼 수 있게 만든 것이다.

## 배포 제약과 대응

무료 CPU Space라 BGE-M3(2GB+) 콜드 스타트가 느리다. 그래서 문서·예제질의 임베딩을 미리
계산해 싣고(demo_assets/), 자유 질의를 입력할 때만 모델을 늦게 로드한다.
모델을 못 쓰는 환경에서는 BM25 단독으로 자동 강등하고 그 사실을 화면에 표시한다.

로컬 실행: python space_app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import numpy as np

from graph_rag.corpus import build_corpus
from graph_rag.eval_set import build_queries
from graph_rag.retrieval import BM25_WEIGHT, DENSE_WEIGHT, GraphRetriever, _rank, _tok

ASSETS = Path(__file__).parent / "demo_assets"
TOP_K = 3

CHUNKS = build_corpus()
QUERIES = {q.question: q for q in build_queries(CHUNKS)}
BY_ID = {c.chunk_id: c for c in CHUNKS}

_index = json.loads((ASSETS / "index.json").read_text(encoding="utf-8"))
_emb = np.load(ASSETS / "emb.npz")
DOC_EMB = _emb["doc"].astype(np.float32)
Q_EMB = {q["question"]: _emb["query"][i].astype(np.float32)
         for i, q in enumerate(_index["queries"])}
IDS = [c["chunk_id"] for c in _index["chunks"]]

_model = None
_model_failed = False


def _dense_query_vector(question: str):
    """예제 질의는 사전 계산분을 쓰고, 자유 질의만 모델을 늦게 로드한다."""
    global _model, _model_failed
    if question in Q_EMB:
        return Q_EMB[question], "precomputed"
    if _model_failed:
        return None, "unavailable"
    try:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(_index["model"])
        v = _model.encode([question], normalize_embeddings=True)[0]
        return v.astype(np.float32), "encoded"
    except Exception:  # 모델 없음/메모리 부족 → BM25 단독으로 강등(정직하게 표시)
        _model_failed = True
        return None, "unavailable"


# --- 검색기 (사전 계산 임베딩을 쓰도록 HybridRetriever를 얇게 재구현) ---
from rank_bm25 import BM25Okapi  # noqa: E402

_bm25 = BM25Okapi([_tok(c.text) for c in CHUNKS])
_graph_mention = GraphRetriever(CHUNKS, mode="mention")
_graph_relation = GraphRetriever(CHUNKS, mode="relation")


def _hybrid_scores(question: str) -> tuple[dict[str, float], str]:
    raw = _bm25.get_scores(_tok(question))
    mx = max(raw) if len(raw) and max(raw) > 0 else 1.0
    bm = {IDS[i]: float(raw[i]) / mx for i in range(len(IDS))}
    qv, mode = _dense_query_vector(question)
    if qv is None:
        return bm, mode
    sims = DOC_EMB @ qv
    dense = {IDS[i]: float(sims[i]) for i in range(len(IDS))}
    return (
        {cid: DENSE_WEIGHT * dense[cid] + BM25_WEIGHT * bm[cid] for cid in IDS},
        mode,
    )


def _normalize(s: dict[str, float]) -> dict[str, float]:
    if not s:
        return {}
    mx = max(s.values())
    return {k: v / mx for k, v in s.items()} if mx > 0 else {k: 0.0 for k in s}


def _render(title: str, ids: list[str], gold: set[str], note: str = "") -> str:
    lines = [f"### {title}"]
    if note:
        lines.append(f"<sub>{note}</sub>\n")
    if not ids:
        lines.append("_결과 없음 — 질의에서 그래프 시드 엔티티를 찾지 못했습니다._")
        return "\n".join(lines)
    for rank, cid in enumerate(ids, 1):
        c = BY_ID[cid]
        if gold and cid in gold:
            badge = "✅ 정답"
        elif not c.states:
            badge = "⚠️ 함정 문서"
        elif gold:
            badge = "· 오답"
        else:
            # 평가셋 밖 질의라 정답이 정의돼 있지 않다. 오답이라 단정하면 안 된다.
            badge = "· 검색됨"
        body = c.text if len(c.text) <= 110 else c.text[:110] + "…"
        lines.append(f"**{rank}. {badge}** &nbsp; `{cid}`  \n{body}\n")
    if gold:
        found = len([i for i in ids if i in gold])
        lines.append(f"\n**정답 {found}/{len(gold)}개 확보** "
                     f"{'— 완전 검색 성공' if found == len(gold) else '— 근거 부족'}")
    return "\n".join(lines)


def search(question: str, graph_weight: float):
    question = (question or "").strip()
    if not question:
        empty = "_질문을 입력하거나 아래 예제를 눌러 보세요._"
        return empty, empty, empty, empty, ""

    q = QUERIES.get(question)
    gold = set(q.gold_chunks) if q else set()

    hyb, dense_mode = _hybrid_scores(question)
    men = _graph_mention.score(question)
    rel = _graph_relation.score(question)

    hn, rn = _normalize(hyb), _normalize(rel)
    keys = set(hn) | set(rn)
    fused = {k: (1 - graph_weight) * hn.get(k, 0.0) + graph_weight * rn.get(k, 0.0)
             for k in keys}

    if dense_mode == "unavailable":
        status = ("이 배포본에는 임베딩 모델(2GB+)을 싣지 않았습니다 — 무료 CPU에서 콜드 스타트를 "
                  "짧게 유지하려는 선택입니다. **예제 질의는 사전 계산된 BGE-M3 임베딩**으로 돌고, "
                  "자유 질의는 **BM25 + 그래프**로 검색합니다.")
    elif dense_mode == "precomputed":
        status = "사전 계산된 임베딩 사용 (BGE-M3, 즉시 응답)"
    else:
        status = "자유 질의 — BGE-M3로 실시간 인코딩"

    if q:
        status += (f" &nbsp;|&nbsp; 평가셋 질의 `{q.qid}` "
                   f"({'멀티홉' if q.hops == 2 else '단일홉'}, 정답 청크 {len(gold)}개)")
    else:
        status += " &nbsp;|&nbsp; 평가셋 밖 질의라 정답 표시가 없습니다."

    return (
        _render("① 플랫 하이브리드", _rank(hyb, TOP_K), gold,
                "BGE-M3 dense 70% + BM25 30% — 현행 파이프라인"),
        _render("② 그래프 · 엔티티 언급", _rank(men, TOP_K), gold,
                "확장 집합을 <b>언급</b>하는 청크. 함정 문서에 약하다"),
        _render("③ 그래프 · 관계 인식", _rank(rel, TOP_K), gold,
                "확장 집합 안에서 <b>관계를 서술</b>하는 청크만"),
        _render(f"④ 융합 (①+③, graph_weight={graph_weight:.1f})",
                _rank(fused, TOP_K), gold, "커버리지는 ③과 같고 <b>순위</b>가 개선된다"),
        status,
    )


INTRO = """
# TechDocRAG — 검색 전략 비교

기술문서 RAG에서 **플랫 하이브리드 검색과 그래프 검색이 각각 무엇을 가져오는지** 나란히 비교하는 데모입니다.
코퍼스는 EV 배터리·ADAS·충전·구동 계통 기술문서 **51개 청크**이고, 그중 **24개는 함정 문서**입니다 —
부품 이름은 나오지만 답은 없는 안전 주의사항·정비 절차·구형 모델 사양·용어집입니다.

**이 데모의 요점:** 함정 문서가 없을 때는 그래프 융합이 이겼습니다(멀티홉 0.625 → 0.750).
함정을 넣자 **오히려 나빠졌고**(0.375 → 0.250), 원인은 그래프 점수가 *관계를 서술하는가*가 아니라
*엔티티를 언급하는가*를 재고 있었기 때문이었습니다. ②와 ③을 비교해 보시면 그 차이가 보입니다.
"""

FOOTER = """
---
### 실측 (top-3, full_hit — 정답 청크를 **전부** 찾아야 1점, 질의 16개)

| 전략 | 전체 | 단일홉 | 멀티홉 | MRR |
|---|---:|---:|---:|---:|
| ① 플랫 하이브리드 | 0.688 | 1.000 | 0.375 | 0.833 |
| ② 그래프 · 엔티티 언급 | 0.438 | 0.625 | 0.250 | 0.594 |
| ③ 그래프 · 관계 인식 | **0.812** | 1.000 | **0.625** | 0.823 |
| ④ 융합 (w=0.5) | **0.812** | 1.000 | **0.625** | **0.927** |

### 정직한 범위

- **"그래프가 하이브리드보다 낫다"고 주장하지 않습니다.** 짝지어 검정하면 ①→③은 불일치 쌍이
  4개뿐이라 **p=0.63으로 유의하지 않습니다**(질의 16개는 검정력이 부족합니다).
  유의한 것은 ②→③, 즉 **제 점수 함수의 결함을 고친 것**입니다(p=0.031).
- ④ 융합은 정답을 더 찾는 게 아니라 **순서를 정리합니다** — 커버리지는 ③과 동일하고 MRR만 오릅니다.
- ③은 "청크 → 관계 추출(IE)"이 이미 정확하다고 가정합니다. 합성 코퍼스는 그 정보를 공짜로 주므로,
  실문서에서는 IE 품질이 그대로 상한이 됩니다.
- 코퍼스는 그래프에서 생성한 합성 문서입니다. 표기가 일관돼 엔티티 링킹이 실제보다 쉽습니다.

코드: [github.com/MJHolics/multimodal_rag](https://github.com/MJHolics/multimodal_rag)
"""


# 첫 화면을 비워두지 않는다. 링크를 연 사람이 아무것도 입력하지 않아도 이 데모의 핵심 장면이
# 바로 보여야 한다. q09를 고른 이유: ①플랫과 ②엔티티언급은 근거 2개 중 1개만 찾고
# (②는 top-3에 함정 문서가 2개 들어온다), ③관계 인식만 2개를 다 찾는다 — 대비가 가장 선명하다.
DEFAULT_Q = "셀 모듈을 포함하는 상위 부품은 어느 회사에서 공급하는가?"


def build_app() -> gr.Blocks:
    init = search(DEFAULT_Q, 0.5)

    with gr.Blocks(title="TechDocRAG — 검색 전략 비교") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            box = gr.Textbox(
                label="질문",
                value=DEFAULT_Q,
                placeholder="예: 배터리 칠러와 연결된 부품의 용량은 얼마인가?",
                scale=4,
            )
            btn = gr.Button("검색", variant="primary", scale=1)
        weight = gr.Slider(0.0, 1.0, value=0.5, step=0.1,
                           label="융합 가중치 (그래프 비중)")
        status = gr.Markdown(value=init[4])
        with gr.Row():
            out1 = gr.Markdown(value=init[0])
            out2 = gr.Markdown(value=init[1])
        with gr.Row():
            out3 = gr.Markdown(value=init[2])
            out4 = gr.Markdown(value=init[3])

        multi = [q.question for q in QUERIES.values() if q.hops == 2][:5]
        single = [q.question for q in QUERIES.values() if q.hops == 1][:3]
        gr.Examples(
            examples=[[m] for m in multi + single],
            inputs=box,
            label="예제 질의 (위 5개는 멀티홉 — 근거가 여러 문서에 흩어져 있습니다)",
        )
        gr.Markdown(FOOTER)

        outputs = [out1, out2, out3, out4, status]
        btn.click(search, [box, weight], outputs)
        box.submit(search, [box, weight], outputs)
        weight.change(search, [box, weight], outputs)
    return demo


if __name__ == "__main__":
    build_app().launch()
