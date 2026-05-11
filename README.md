# TechDocRAG — 멀티모달 기술 문서 RAG

PDF 기술 문서(텍스트 + 표 + 다이어그램)를 업로드하면 자연어 질문에 **페이지 이미지와 함께** 답변하는 RAG 시스템.

## 아키텍처

```
PDF (텍스트 + 표 + 다이어그램)
  ↓  PyMuPDF + pdfplumber
PageChunk → BGE-M3 (1024-dim) → ChromaDB
            BM25Okapi          → pkl 인덱스
  ↓  하이브리드 검색 (Dense 70% + BM25 30%)
Top-K RetrievedChunk
  ↓  Qwen2.5-VL 7B (Ollama) / Rule-based 폴백
구조화된 답변 + 출처 페이지 이미지
```

## 기술 스택

| 역할 | 기술 |
|------|------|
| PDF 파싱 | PyMuPDF (페이지 이미지 200 DPI) + pdfplumber (표 추출) |
| 텍스트 임베딩 | BAAI/bge-m3 (1024-dim, 한/영 다국어) |
| 키워드 인덱스 | BM25Okapi (한국어 정규식 토크나이저) |
| 벡터 스토어 | ChromaDB PersistentClient (cosine space) |
| 하이브리드 검색 | Dense 70% + BM25 30% 앙상블 |
| Vision LLM | Qwen2.5-VL 7B via Ollama (Rule-based 폴백) |
| API 서버 | FastAPI + uvicorn |
| 데모 UI | Streamlit |
| 배포 | Docker + docker-compose |

## 노트북 구성

| 노트북 | 내용 |
|--------|------|
| `01_document_processing.ipynb` | PyMuPDF 이미지 변환 + pdfplumber 표 추출 + PageChunk 설계 |
| `02_multimodal_embedding.ipynb` | BGE-M3 임베딩 + BM25 인덱스 + ChromaDB 인덱싱 |
| `03_rag_pipeline.ipynb` | 하이브리드 검색 + Qwen2.5-VL 답변 생성 + Hit Rate/MRR 평가 |
| `04_full_integration.ipynb` | FastAPI 서버 + Streamlit UI + 벤치마크 + Docker 구성 |

## API 엔드포인트

```
GET  /health    서버 상태 및 인덱스 현황
POST /ingest    PDF 업로드 → 자동 인덱싱
POST /query     하이브리드 검색 + Vision LLM 답변
```

### 예시

```bash
# 헬스체크
curl http://localhost:8000/health

# PDF 인덱싱
curl -X POST http://localhost:8000/ingest \
  -F "file=@data/docs/ev_battery_manual.pdf"

# 질문
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "배터리 팩 공칭 전압은?", "top_k": 2}'
```

## 실행 방법

### 로컬 (conda 환경)

```bash
conda activate multimodal_rag

# FastAPI 서버
cd app && uvicorn main:app --port 8000 --reload

# Streamlit UI (별도 터미널)
streamlit run app/streamlit_app.py
```

### Docker

```bash
docker-compose up --build
```

Swagger UI: http://localhost:8000/docs

## 벤치마크 결과

하이브리드 검색 응답시간 (25회 측정, BGE-M3 + BM25):

| 지표 | 값 |
|------|----|
| 평균 | ~15 ms |
| P50 | ~12 ms |
| P95 | ~28 ms |

실시간 검색 기준(200ms) 대비 **10배 이상 여유**.

## 환경 설정

```bash
conda create -n multimodal_rag python=3.11 -y
conda activate multimodal_rag
pip install -r requirements.txt
```

Ollama Vision LLM (선택):
```bash
ollama pull qwen2.5vl:7b
```
