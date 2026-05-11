"""TechDocRAG FastAPI 서버"""
import time
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import QueryRequest, QueryResponse, SourceInfo, IngestResponse, HealthResponse
import ingestion
import retriever as ret

app = FastAPI(
    title="TechDocRAG API",
    description="멀티모달 기술 문서 RAG — PDF(텍스트+이미지+표) 검색 + Qwen2.5-VL 답변",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """서버 상태 및 인덱스 현황 확인"""
    try:
        col = ret._get_collection()
        count = col.count()
    except Exception:
        count = 0
    return HealthResponse(
        status="healthy",
        indexed_docs=count,
        ollama_available=ret._check_ollama(),
        version="1.0.0",
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest(file: UploadFile = File(...)):
    """
    PDF 파일을 업로드하여 인덱싱합니다.
    - 페이지 이미지 변환 (200 DPI)
    - 텍스트 + 표 추출
    - BGE-M3 임베딩 → ChromaDB 저장
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 지원합니다.")

    t0 = time.time()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        chunks = ingestion.ingest_pdf(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if not chunks:
        raise HTTPException(status_code=422, detail="PDF에서 텍스트를 추출할 수 없습니다.")

    import chromadb
    import re
    import pickle
    from rank_bm25 import BM25Okapi

    ROOT = Path(__file__).parent.parent
    embedder = ret._get_embedder()
    client = chromadb.PersistentClient(path=str(ROOT / "vector_db"))

    try:
        col = client.get_collection("tech_docs")
    except Exception:
        col = client.create_collection("tech_docs", metadata={"hnsw:space": "cosine"})

    texts = [c.text for c in chunks]
    embs = embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    col.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embs.tolist(),
        documents=texts,
        metadatas=[
            {
                "doc_name": c.doc_name,
                "page_num": c.page_num,
                "image_path": c.image_path,
                "content_type": c.content_type,
                "has_table": str(c.has_table),
                "char_count": c.char_count,
            }
            for c in chunks
        ],
    )

    def tok(t):
        return re.findall(r"[a-z0-9]+|[가-힣]{2,}", t.lower())

    corpus = [tok(c.text) for c in chunks]
    bm25 = BM25Okapi(corpus)
    bm25_save = {
        "bm25": bm25,
        "tokenized_corpus": corpus,
        "chunk_ids": [c.chunk_id for c in chunks],
    }
    with open(ROOT / "output" / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_save, f)

    ret._collection = None
    ret._bm25_data = None

    return IngestResponse(
        doc_name=chunks[0].doc_name,
        pages_indexed=len(chunks),
        elapsed_sec=round(time.time() - t0, 2),
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def query(req: QueryRequest):
    """
    질문을 받아 관련 문서를 검색하고 Qwen2.5-VL로 답변을 생성합니다.
    - 하이브리드 검색 (BGE-M3 70% + BM25 30%)
    - 페이지 이미지를 Vision LLM에 주입
    - Ollama 미연결 시 Rule-based 폴백
    """
    results = ret.search(req.question, top_k=req.top_k)
    if not results:
        raise HTTPException(status_code=404, detail="관련 문서를 찾을 수 없습니다.")

    t0 = time.time()
    answer, model_used = ret.generate_answer(req.question, results)

    return QueryResponse(
        answer=answer,
        sources=[
            SourceInfo(chunk_id=r.chunk_id, page_num=r.page_num, score=r.hybrid_score)
            for r in results
        ],
        model_used=model_used,
        elapsed_sec=round(time.time() - t0, 2),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
