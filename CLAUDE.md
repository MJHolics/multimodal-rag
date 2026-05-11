# TechDocRAG — 멀티모달 기술 문서 RAG 프로젝트 컨텍스트

## 목표
PDF 기술 문서(다이어그램 + 표 + 텍스트)를 이미지 + 텍스트 양쪽으로 처리하여,
Vision LLM이 그림과 표까지 이해하고 답변하는 멀티모달 RAG 시스템 구축.

## 기존 RAG와 차별점
- **기존 RAG**: 텍스트만 추출 → LLM에 텍스트 제공 → 다이어그램/표 정보 손실
- **이 프로젝트**: 텍스트 + 페이지 이미지 동시 처리 → Qwen2.5-VL(Vision LLM)이 그림·표까지 이해해서 답변

## 사용자 배경
- LLM/RAG/PEFT 경험 (강점)
- CV 경험: YOLO/SAM/DepthAnything, 자율주행 파이프라인
- FastAPI + Docker 경험
- LangGraph 에이전트 경험
- 목표: 현대차/모빌리티 AI 취업 포트폴리오

## 기술 스택
- **PyMuPDF (fitz)** — PDF 텍스트 추출 + 페이지 이미지 렌더링
- **pdfplumber** — 표(table) 정밀 추출
- **BAAI/bge-m3** — 다국어 텍스트 임베딩 (한/영 동시 지원)
- **rank-bm25** — 키워드 기반 BM25 검색
- **ChromaDB** — 벡터 저장소 (텍스트 + 이미지 경로 메타데이터)
- **Ollama + Qwen2.5-VL** — 로컬 Vision LLM (이미지 + 텍스트 → 답변, 무료)
- **FastAPI** — REST API 서빙
- **Streamlit** — 데모 UI
- **Docker** — 배포

## 시스템 아키텍처

```
[인제스천 파이프라인]
PDF 문서 입력
    ↓
PyMuPDF → 페이지별 이미지 렌더링 (PNG) + 텍스트 추출
    ↓
pdfplumber → 표(table) 구조 추출
    ↓
텍스트 청킹 (페이지 단위 + overlap)
    ↓
BGE-M3 임베딩 + BM25 인덱스 구축
    ↓
ChromaDB 저장 (텍스트 벡터 + 페이지 이미지 경로 메타데이터)

[쿼리 파이프라인]
사용자 질문 (텍스트)
    ↓
하이브리드 검색: BGE-M3 밀집 검색 70% + BM25 30%
    ↓
Top-K 청크 + 해당 페이지 이미지 로드
    ↓
Qwen2.5-VL (Ollama): 텍스트 + 이미지들 → 답변 생성
    ↓
구조화된 응답 (답변 + 출처 페이지 + 이미지)
```

## 4단계 노트북 로드맵

### Notebook 01 — PDF 문서 처리
- PyMuPDF로 PDF 페이지 → PNG 이미지 변환
- 텍스트 추출 (일반 텍스트 + 폰트 정보)
- pdfplumber로 표(table) 구조 추출 → 마크다운 변환
- 페이지별 컨텐츠 타입 분류 (텍스트 heavy / 이미지 heavy / 표 heavy)
- 청킹 전략: 페이지 단위 + 의미 단위 분할

### Notebook 02 — 멀티모달 임베딩 & 인덱싱
- BGE-M3 텍스트 임베딩 (한/영 다국어)
- BM25 인덱스 구축
- ChromaDB 컬렉션 설계:
  - text_chunks: 텍스트 벡터 + 메타데이터(page_num, image_path, doc_name)
- 이미지 경로를 메타데이터로 연결 (ChromaDB에 경로만 저장, 이미지는 파일 시스템)
- 인덱싱 성능 벤치마크

### Notebook 03 — RAG 파이프라인
- 하이브리드 검색 구현 (BGE-M3 + BM25 앙상블)
- 검색 결과에서 이미지 로드 및 Ollama Qwen2.5-VL 에 전달
- 프롬프트 엔지니어링:
  - 텍스트 컨텍스트 + 이미지 동시 주입
  - 한국어 기술 문서 특화 프롬프트
- Rule-based 폴백 (Ollama 미설치 시)
- 검색 정확도 평가 (Hit Rate, MRR)

### Notebook 04 — 풀 파이프라인 통합
- 전체 인제스천 → 검색 → 생성 파이프라인 E2E 테스트
- FastAPI 서빙 (POST /ingest, POST /query, GET /health)
- Streamlit 데모 UI (PDF 업로드 → 질문 → 답변 + 출처 이미지 표시)
- 처리 속도 / 메모리 프로파일링
- 샘플 기술 문서로 실제 동작 시연

## 파일 구조
```
multimodal_rag/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
├── notebooks/
│   ├── 01_document_processing.ipynb
│   ├── 02_multimodal_embedding.ipynb
│   ├── 03_rag_pipeline.ipynb
│   └── 04_full_integration.ipynb
├── app/
│   ├── main.py           # FastAPI 엔트리포인트
│   ├── ingestion.py      # PDF 처리 + 인덱싱
│   ├── retriever.py      # 하이브리드 검색
│   ├── generator.py      # Qwen2.5-VL 답변 생성
│   └── schemas.py        # Pydantic 모델
├── data/
│   └── docs/             # 샘플 PDF 문서
├── models/               # 다운로드된 임베딩 모델
├── vector_db/            # ChromaDB 영구 저장
├── output/               # 처리된 페이지 이미지
├── Dockerfile
├── docker-compose.yml
└── TROUBLESHOOTING.md
```

## 환경
- Python: Anaconda (신규 `multimodal_rag` 커널)
- LLM: Ollama 로컬 (Qwen2.5-VL 7B — 이미 설치되어 있음)
- GPU: RTX 4080 Super (16GB VRAM) — BGE-M3 + Qwen2.5-VL 동시 실행 가능
- API 키: 불필요 (완전 로컬)

## 핵심 구현 포인트

### PDF → 이미지 변환
```python
import fitz  # PyMuPDF
doc = fitz.open("document.pdf")
for page_num, page in enumerate(doc):
    mat = fitz.Matrix(2.0, 2.0)  # 2x 해상도
    pix = page.get_pixmap(matrix=mat)
    pix.save(f"output/page_{page_num:03d}.png")
```

### BGE-M3 임베딩
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
embeddings = model.encode(texts, normalize_embeddings=True)
```

### Qwen2.5-VL + 이미지 (Ollama)
```python
import ollama, base64
with open(image_path, 'rb') as f:
    img_data = base64.b64encode(f.read()).decode()
response = ollama.chat(
    model='qwen2.5-vl:7b',
    messages=[{
        'role': 'user',
        'content': prompt,
        'images': [img_data]
    }]
)
```

## 노트북 작성 규칙
- 한글 폰트 설정 (첫 셀):
  ```python
  import matplotlib
  matplotlib.rcParams['font.family'] = 'Malgun Gothic'
  matplotlib.rcParams['axes.unicode_minus'] = False
  ```
- 각 셀마다 목적 주석 포함
- Ollama 미설치 시 Rule-based 폴백으로 자동 동작

## 포트폴리오 포인트
- **차별화**: 텍스트만 처리하는 기존 RAG와 달리, 다이어그램·표·수식이 포함된 기술 문서를 Vision LLM으로 이해
- **현대차 연관성**: 자동차 정비 매뉴얼, EV 배터리 스펙, ADAS 기술 문서 검색에 직접 적용 가능
- **완전 로컬**: Ollama로 API 비용 0원
- **기술 스택**: RAG + Vision + 하이브리드 검색 + Docker 배포

## 진행 상황
- [ ] 프로젝트 구조 생성
- [ ] Notebook 01 — PDF 문서 처리
- [ ] Notebook 02 — 멀티모달 임베딩 & 인덱싱
- [ ] Notebook 03 — RAG 파이프라인
- [ ] Notebook 04 — 풀 파이프라인 통합
- [ ] FastAPI 서빙
- [ ] Streamlit 데모 UI
- [ ] Docker 배포
