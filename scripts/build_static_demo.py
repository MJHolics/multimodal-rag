"""정적 데모 데이터 생성 — 브라우저에서 돌릴 수 있게 코퍼스·그래프·결과를 JSON으로 굳힌다.

## 왜 정적인가

Hugging Face가 Gradio Space를 무료 CPU에서 못 돌리게 바꿨다(PRO 필요). Static Space는 무료다.
이 데모는 코퍼스 51청크에 **결정적 계산**뿐이라 서버가 필요 없다 — 브라우저에서 전부 된다.
오히려 이득이다: 콜드 스타트 0, 빌드 0, GitHub Pages로도 그대로 나간다.

## 무엇을 미리 계산하고 무엇을 브라우저에 맡기나

  - **예제 질의 16개의 결과** → 여기서 파이썬으로 계산해 넣는다. 즉 화면에 보이는 순위는
    `run_graph_eval.py`가 낸 실측과 **바이트 단위로 같다**(BM25 구현 차이로 흔들릴 여지 없음).
  - **자유 질의** → 브라우저가 BM25 + 그래프로 계산한다. dense는 질의 임베딩이 필요한데
    모델을 브라우저에 올릴 수 없으므로 빠진다. UI에 그 사실을 표시한다.

실행: python -m scripts.build_static_demo
"""
from __future__ import annotations

import json
from pathlib import Path

from graph_rag.corpus import ENTITY_BY_ID, build_corpus
from graph_rag.eval_set import build_queries
from graph_rag.retrieval import (
    FusedRetriever,
    GraphRetriever,
    HybridRetriever,
    _rank,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "static" / "data.json"
TOP_K = 3


class _PrecomputedEmbedder:
    """사전 계산된 임베딩을 SentenceTransformer 인터페이스로 감싼다.

    예제 질의만 쓰므로 encode에 들어오는 문자열이 사전 계산 목록에 반드시 있어야 한다.
    없으면 조용히 틀리는 대신 즉시 실패시킨다.
    """

    def __init__(self, doc_texts: list[str], doc_emb, q_map: dict) -> None:
        self._doc = dict(zip(doc_texts, doc_emb))
        self._q = q_map

    def encode(self, texts, normalize_embeddings=True):
        import numpy as np

        out = []
        for t in texts:
            v = self._doc.get(t)
            if v is None:
                v = self._q.get(t)
            if v is None:
                raise KeyError(f"사전 계산에 없는 텍스트: {t[:40]}")
            out.append(v)
        return np.asarray(out, dtype="float32")


def main() -> None:
    import numpy as np

    chunks = build_corpus()
    queries = build_queries(chunks)

    assets = ROOT / "demo_assets"
    idx = json.loads((assets / "index.json").read_text(encoding="utf-8"))
    emb = np.load(assets / "emb.npz")
    doc_emb = emb["doc"].astype("float32")
    q_emb = emb["query"].astype("float32")
    q_map = {q["question"]: q_emb[i] for i, q in enumerate(idx["queries"])}

    embedder = _PrecomputedEmbedder([c.text for c in chunks], doc_emb, q_map)
    hybrid = HybridRetriever(chunks, embedder=embedder)
    g_men = GraphRetriever(chunks, mode="mention")
    g_rel = GraphRetriever(chunks, mode="relation")
    fused = FusedRetriever(hybrid, g_rel, graph_weight=0.5)

    data = {
        "top_k": TOP_K,
        "entities": {
            e.eid: {"name": e.name, "etype": e.etype} for e in ENTITY_BY_ID.values()
        },
        "chunks": [
            {
                "id": c.chunk_id,
                "doc": c.doc_name,
                "text": c.text,
                "entities": c.entities,
                # 관계를 (head, tail)로만 싣는다 — 브라우저 점수 계산에 rtype은 안 쓴다.
                "rels": [[r.head, r.tail] for r in c.states],
                "distractor": not c.states,
            }
            for c in chunks
        ],
        # 그래프 확장용 인접 리스트(무방향 — expand가 무방향으로 도는 것과 맞춘다)
        "adj": _adjacency(chunks),
        "queries": [
            {
                "qid": q.qid,
                "question": q.question,
                "hops": q.hops,
                "gold": list(q.gold_chunks),
                "results": {
                    "hybrid": hybrid.retrieve(q.question, TOP_K),
                    "mention": g_men.retrieve(q.question, TOP_K),
                    "relation": g_rel.retrieve(q.question, TOP_K),
                    "fused": fused.retrieve(q.question, TOP_K),
                },
            }
            for q in queries
        ],
    }

    # 스윕 결과도 함께 실어 슬라이더가 사전 계산값을 그대로 보여주게 한다.
    sweep = {}
    for i in range(11):
        w = round(i * 0.1, 1)
        fr = FusedRetriever(hybrid, g_rel, graph_weight=w)
        sweep[f"{w:.1f}"] = {q.qid: fr.retrieve(q.question, TOP_K) for q in queries}
    data["fused_sweep"] = sweep

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"chunks={len(chunks)} queries={len(queries)} → {OUT} ({kb:.0f} KB)")


def _adjacency(chunks) -> dict[str, list[str]]:
    from graph_rag.corpus import RELATIONS

    adj: dict[str, list[str]] = {}
    for r in RELATIONS:
        adj.setdefault(r.head, []).append(r.tail)
        adj.setdefault(r.tail, []).append(r.head)
    return adj


if __name__ == "__main__":
    main()
