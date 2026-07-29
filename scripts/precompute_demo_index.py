"""데모용 인덱스 사전 계산 — 청크/예제질의 임베딩을 파일로 굳힌다.

## 왜 사전 계산인가

배포 대상이 Hugging Face 무료 CPU Space다. BGE-M3는 2GB가 넘어 콜드 스타트에 1~2분이 걸린다.
채용담당자가 링크를 열었을 때 첫 화면이 늦으면 그냥 닫는다. 그래서:

  - 문서 임베딩(51×1024)과 **예제 질의 임베딩**을 미리 계산해 저장한다 → 예제는 모델 없이 즉시.
  - 자유 질의를 입력했을 때만 모델을 늦게 로드한다(그때 한 번 기다린다).

즉 "빠른 첫인상"과 "임의 질의 지원"을 둘 다 가져가는 절충이다.

실행: python -m scripts.precompute_demo_index
산출: demo_assets/index.json (청크 메타·gold·distractor 표시) + demo_assets/emb.npz (float16)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from graph_rag.corpus import build_corpus
from graph_rag.eval_set import build_queries

OUT = Path(__file__).resolve().parent.parent / "demo_assets"


def main() -> None:
    from sentence_transformers import SentenceTransformer

    chunks = build_corpus()
    queries = build_queries(chunks)

    model = SentenceTransformer("BAAI/bge-m3")
    doc_emb = model.encode(
        [c.text for c in chunks], normalize_embeddings=True, show_progress_bar=True
    )
    q_emb = model.encode(
        [q.question for q in queries], normalize_embeddings=True, show_progress_bar=True
    )

    index = {
        "model": "BAAI/bge-m3",
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "doc_name": c.doc_name,
                "page_num": c.page_num,
                "text": c.text,
                "entities": c.entities,
                # 관계는 (head, rtype, tail)로 평탄화 — 데모는 graph_rag를 import하므로
                # 사실 재계산할 수 있지만, 어떤 청크가 distractor인지 UI에서 바로 쓰려고 남긴다.
                "n_relations": len(c.states),
                "is_distractor": not c.states,
            }
            for c in chunks
        ],
        "queries": [
            {
                "qid": q.qid,
                "question": q.question,
                "hops": q.hops,
                "gold": list(q.gold_chunks),
                "seeds": list(q.seed_entities),
            }
            for q in queries
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # float16으로 저장 — 코사인 유사도 순위에 영향이 없을 정도의 정밀도이고 파일이 절반이 된다.
    np.savez_compressed(
        OUT / "emb.npz",
        doc=doc_emb.astype(np.float16),
        query=q_emb.astype(np.float16),
    )

    size = (OUT / "emb.npz").stat().st_size / 1024
    print(f"chunks={len(chunks)} queries={len(queries)} dim={doc_emb.shape[1]}")
    print(f"저장: {OUT}/index.json, {OUT}/emb.npz ({size:.0f} KB)")


if __name__ == "__main__":
    main()
