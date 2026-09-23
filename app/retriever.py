"""하이브리드 검색 + Qwen2.5-VL 답변 생성"""
import re
import pickle
import base64
import logging
import urllib.request
from pathlib import Path
from dataclasses import dataclass

import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
VECTOR_DIR = ROOT / "vector_db"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = ROOT / "models"

# 싱글톤
_embedder = None
_collection = None
_bm25_data = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("BAAI/bge-m3", cache_folder=str(MODEL_DIR))
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(VECTOR_DIR))
        _collection = client.get_collection("tech_docs")
    return _collection


def _get_bm25():
    global _bm25_data
    if _bm25_data is None:
        p = OUTPUT_DIR / "bm25_index.pkl"
        if p.exists():
            with open(p, "rb") as f:
                _bm25_data = pickle.load(f)
    return _bm25_data


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    image_path: str
    page_num: int
    doc_name: str
    content_type: str
    hybrid_score: float
    dense_score: float
    bm25_score: float
    retrieval_mode: str = "hybrid"


def _tok(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[가-힣]{2,}", text.lower())


def _dense_scores(query: str, top_k: int) -> dict[str, float]:
    emb = _get_embedder()
    col = _get_collection()
    q_emb = emb.encode([query], normalize_embeddings=True)
    dr = col.query(
        query_embeddings=q_emb.tolist(),
        n_results=min(top_k * 2, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    return {
        dr["ids"][0][i]: 1 - dr["distances"][0][i]
        for i in range(len(dr["ids"][0]))
    }


def _bm25_scores(query: str) -> dict[str, float]:
    bd = _get_bm25()
    if not bd:
        return {}
    raw = bd["bm25"].get_scores(_tok(query))
    mx = max(raw) if max(raw) > 0 else 1
    return {bd["chunk_ids"][i]: float(raw[i]) / mx for i in range(len(bd["chunk_ids"]))}


def _keyword_fallback(query: str, top_k: int) -> dict[str, float]:
    """dense(임베더)·BM25 계산 둘 다 죽었을 때의 마지막 수단.

    질의 토큰과 겹치는 원문을 그 자리에서 스캔해 점수화한다(실측·한계는
    graph_rag/retrieval.py의 `HybridRetriever._keyword_scores` 참고 — 같은 로직).
    **주의**: 원문을 ChromaDB `col.get()`으로 읽으므로 ChromaDB 자체(저장소)가
    아니라 임베딩 계산·BM25 계산 계층의 장애까지만 방어한다 — 저장소 자체가
    죽으면 이 폴백도 같이 죽는다. `graph_rag` 쪽 벤치(원문을 별도 파이썬 리스트로
    들고 있어 저장소 무관)가 진짜 최후 방어선이고, 이쪽은 그 로직을 서빙 코드에
    그대로 옮겨 온 것이다.
    """
    col = _get_collection()
    q_tokens = set(_tok(query))
    if not q_tokens:
        return {}
    all_docs = col.get(include=["documents"])
    out = {}
    for cid, text in zip(all_docs["ids"], all_docs["documents"]):
        overlap = len(q_tokens & set(_tok(text or "")))
        if overlap:
            out[cid] = float(overlap)
    return out


def search(query: str, top_k: int = 2) -> list[RetrievedChunk]:
    dense, dense_ok = {}, True
    try:
        dense = _dense_scores(query, top_k)
    except Exception as e:
        logger.warning("[retriever] dense 실패, 축소 운영: %s", e)
        dense_ok = False

    bm25_scores, bm25_ok = {}, True
    try:
        bm25_scores = _bm25_scores(query)
    except Exception as e:
        logger.warning("[retriever] BM25 실패, 축소 운영: %s", e)
        bm25_ok = False

    if not dense_ok and not bm25_ok:
        hybrid = _keyword_fallback(query, top_k)
        mode = "keyword_fallback"
    elif not dense_ok:
        hybrid = dict(bm25_scores)
        mode = "bm25_only"
    elif not bm25_ok or not bm25_scores:
        hybrid = dict(dense)
        mode = "dense_only"
    else:
        all_ids = set(dense) | set(k for k, v in bm25_scores.items() if v > 0)
        hybrid = {
            # 0.9/0.1 — run_hybrid_weight.py 의 dev 스윕으로 정한 값(graph_rag/retrieval.py 와 동일)
            cid: 0.9 * dense.get(cid, 0) + 0.1 * bm25_scores.get(cid, 0)
            for cid in all_ids
        }
        mode = "hybrid"

    ranked = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:top_k]

    ids = [r[0] for r in ranked]
    col = _get_collection()
    meta_r = col.get(ids=ids, include=["metadatas", "documents"])
    meta_m = {meta_r["ids"][i]: meta_r["metadatas"][i] for i in range(len(meta_r["ids"]))}
    doc_m = {meta_r["ids"][i]: meta_r["documents"][i] for i in range(len(meta_r["ids"]))}

    results = []
    for cid, h in ranked:
        m = meta_m.get(cid, {})
        results.append(
            RetrievedChunk(
                chunk_id=cid,
                text=doc_m.get(cid, ""),
                image_path=m.get("image_path", ""),
                page_num=m.get("page_num", 0),
                doc_name=m.get("doc_name", ""),
                content_type=m.get("content_type", ""),
                hybrid_score=round(h, 4),
                dense_score=round(dense.get(cid, 0), 4),
                bm25_score=round(bm25_scores.get(cid, 0), 4),
                retrieval_mode=mode,
            )
        )
    return results


def _check_ollama() -> bool:
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False


def generate_answer(
    question: str,
    results: list[RetrievedChunk],
    ollama_model: str = "qwen2.5vl:7b",
) -> tuple[str, str]:
    if not results:
        return "관련 문서를 찾을 수 없습니다.", "rule-based"

    context = "\n\n---\n\n".join(
        f"[문서 {i+1}: {r.doc_name} / 페이지 {r.page_num+1}]\n{r.text}"
        for i, r in enumerate(results)
    )

    if not _check_ollama():
        top = results[0]
        return (
            f"[{top.doc_name} / 페이지 {top.page_num+1}] 관련 내용:\n\n"
            f"{top.text[:600]}\n\n(※ Ollama 미연결 — Rule-based 폴백)"
        ), "rule-based"

    try:
        import ollama

        imgs = []
        for r in results[:2]:
            if r.image_path and Path(r.image_path).exists():
                with open(r.image_path, "rb") as f:
                    imgs.append(base64.b64encode(f.read()).decode())

        msg = {
            "role": "user",
            "content": (
                f"다음 문서를 참고하여 질문에 답하세요.\n\n"
                f"[참고 문서]\n{context}\n\n[질문]\n{question}\n\n[답변]"
            ),
        }
        if imgs:
            msg["images"] = imgs

        resp = ollama.chat(
            model=ollama_model,
            messages=[
                {
                    "role": "system",
                    "content": "당신은 EV 기술 문서 전문가입니다. 제공된 문서와 이미지를 기반으로 정확하게 답하세요.",
                },
                msg,
            ],
            options={"temperature": 0.1},
        )
        return resp["message"]["content"], ollama_model
    except Exception as e:
        top = results[0]
        return (
            f"[{top.doc_name} / 페이지 {top.page_num+1}]\n{top.text[:600]}"
            f"\n\n(Ollama 오류: {e})"
        ), "rule-based"
