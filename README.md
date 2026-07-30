---
title: "TechDocRAG — 검색 전략 비교"
emoji: 🔎
colorFrom: indigo
colorTo: gray
sdk: gradio
app_file: space_app.py
pinned: false
short_description: 기술문서 RAG에서 플랫 하이브리드 vs 그래프 검색을 함정 문서와 함께 비교
---

# TechDocRAG — 멀티모달 기술 문서 RAG

PDF 기술 문서(텍스트 + 표 + 다이어그램)를 업로드하면 자연어 질문에 **페이지 이미지와 함께** 답변하는 RAG 시스템.

## 검색 전략 비교 데모 (LLM 불필요, 무료 CPU에서 동작)

![검색 전략 비교 데모](docs/img/techdocrag_demo.png)

같은 질문에 **플랫 하이브리드 · 그래프(엔티티 언급) · 그래프(관계 인식) · 융합**이 각각 무엇을
가져오는지 나란히 보여준다. 코퍼스 72개 중 33개는 **함정 문서**다 — 부품 이름은 나오지만 답은 없는
안전 주의사항·정비 절차·구형 모델 사양·용어집.

위 화면이 이 데모의 요점이다. 멀티홉 질의에서 플랫 하이브리드와 엔티티 언급 그래프는 근거 2개 중
1개만 찾고 top-3에 함정 문서를 2개 올리는 반면, **관계 인식 그래프만 2개를 다 찾는다.**

실행: `python space_app.py` · 배포 절차: [`DEPLOY_SPACE.md`](DEPLOY_SPACE.md)

### GraphRAG vs 플랫 하이브리드 — 실측 (2026-07-30, 3차)

평가셋을 **그래프에서 생성**해(135문항 = 단일홉 97 + 멀티홉 38) 검정력을 확보하고,
전략 차이를 **McNemar 정확검정**으로 판정했다. top-3, dense=BAAI/bge-m3, 지표는 `full_hit`
(정답을 구성하는 청크가 **전부** 상위 k에 들어야 1점 — 멀티홉은 근거가 다 모여야 답이 된다).

| 전략 | 전체 | 단일홉 | **멀티홉** |
|---|---:|---:|---:|
| hybrid_flat (현행: dense 0.7 + BM25 0.3) | 0.733 | **1.000** | 0.053 |
| graph_linked (엔티티 언급 점수) | 0.644 | 0.866 | 0.079 |
| **graph_rel (관계 인식 점수)** | 0.711 | 0.876 | **0.289** |
| **fused_rel (융합, w=0.2)** | **0.793** | **1.000** | 0.263 |

**멀티홉에서 그래프가 플랫을 이긴다: 0.053 → 0.289, 불일치 0:9, McNemar p=0.0039(유의).**
반대로 **단일홉에서는 그래프가 진다**(1.000 vs 0.876) — 답이 한 청크에 있으면 어휘 검색으로
충분하고 그래프는 노이즈만 더한다. 전체로 합치면 차이가 사라진다(p=0.66). 즉 **"GraphRAG가
낫다"는 뭉뚱그린 주장은 이 실험에서 성립하지 않고, 이득은 멀티홉에 한정된다.**
그래서 실무 답은 융합(`fused_rel`)이고, `graph_weight`는 **홀드아웃 dev에서 골랐다**(0.2).

**측정 방법 자체가 결론을 바꾼 사례 두 건**(자세히는 [`graph_rag/NOTES.md`](graph_rag/NOTES.md)):
- 단위테스트가 생성기 결함을 잡아 멀티홉을 51 → **38문항으로 줄였더니 검정력이 오히려 생겼다**
  (p 0.238 → 0.0039). 빠진 13문항은 **시드 단어 하나로 플랫 BM25가 풀던 것들**이라 그래프의
  이점을 가리고 있었다.
- "유의하지 않음"과 **"판정불가"**를 구분한다 — 불일치 쌍이 6개 미만이면 완전히 쏠려도
  p<0.05가 불가능하므로 그건 결론이 아니라 잴 힘이 없다는 뜻이다.

재현: `python run_graph_eval.py --auto --holdout` (결정적 — 랜덤 없음)
판단의 이력(왜 그렇게 설계했나): [`graph_rag/DECISIONS.md`](graph_rag/DECISIONS.md)

### KnowledgeOps — 실문서(공공조달 RFP 96건)로 옮겨 재측정

위 실험은 합성 코퍼스였다. 그 가정을 걷어내고 **실제 HWP 원본 96건**(각 4만~10만 자)으로
같은 비교를 다시 했다. 관계는 더 이상 주어지지 않고 **직접 추출(IE)**한다.

| | 합성 | **실문서** |
|---|---:|---:|
| 청크 | 72 | **7,863** |
| 방해 문서 | 46% | **90%** |
| 관계 | 73(생성 규칙) | **1,069(IE 추출)** |

**실측 (top-3, BGE-M3+BM25, full_hit)**

| 전략 | 전체 | 단일홉 | **멀티홉** |
|---|---:|---:|---:|
| hybrid_flat | 0.398 | 0.593 | **0.000** |
| graph_rel | 0.364 | 0.364 | **0.362** |
| **fused_rel** | **0.545** | **0.652** | 0.328 |

**합성의 결론이 뒤집혔다.** 합성에서는 "dense가 그래프의 이점을 상당히 흡수한다"였는데
(멀티홉 hybrid 0.053), **실문서에서는 hybrid가 0.000이다**(불일치 0:21, p<0.0001).
질의는 기관명으로 시작하는데 예산이 적힌 청크에는 기관명이 없고 의미적으로도 가깝지 않다 —
그래프가 없으면 멀티홉이 **원리적으로 불가능**하다. 실험 조건이 결론을 얼마나 좌우하는지
보여주는 사례다. (반복 검색이 이득 없다는 결론은 양쪽에서 동일하게 유지됐다.)

**IE 품질이 곧 상한이다 — 그래서 감사한다.** 추출 직후 표본 20건을 눈으로 읽었더니 절반이
오탐이었다("사업금액 20억원 **미만** 계약은 대기업 제한"을 예산으로, "제본 및 **스프링** 사용"을
프레임워크로). 문맥 가드(라벨 근접 + 부정 배제)로 예산 표본 6/6까지 올렸고, 커버리지가
88→49건으로 준 것은 감수했다 — **틀린 관계는 gold를 오염시켜 평가 전체를 무의미하게** 만들기
때문이다.

**재현 절차** (문서는 저장소에 없다 — 경로만 바꾸면 어떤 HWP 묶음에서도 돈다)

```
# 1) 문서 → 지식그래프. `기관명_사업명.hwp` 규약만 지키면 된다
python build_rfp_graph.py --src "경로/to/hwp_dir" --audit HAS_BUDGET

# 2) 검색 전략 비교 + 통계 검정 (결정적 — 재실행하면 같은 값)
python run_real_eval.py

# 3) 4탭 콘솔: 문서→그래프 · 검색 비교 · 평가 리포트 · IE 감사
python knowledgeops_app.py
python knowledgeops_app.py --demo   # HWP 없이 합성 코퍼스로 동작 확인
```

RFP 원본(약 164MB)은 저장소에 넣지 않았다. 대신 **측정 결과 JSON은 커밋해 뒀으므로**
문서가 없어도 리포트 탭과 위 수치는 확인할 수 있고, 콘솔은 `--demo`로 뜬다.

정직한 한계: 실문서 수치는 **IE 정밀도를 곱한 상한**이라 합성 수치와 직접 비교하면 안 된다 ·
요구 기술은 사전 매칭이라 사전 크기가 재현율 상한 · 합성 코퍼스의 생성 질의는 어휘가 균질하다 ·
**Neo4j는 코드만 있고 실행 미검증**(Kuzu 임베디드 Graph DB로 Cypher 탐색은 검증됨).

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

## RAG 생성 품질 평가 (RAGAS 스타일)

검색 품질(Hit Rate/MRR)을 넘어, **생성 답의 품질**을 RAGAS 지표로 수치화한다 — 라이브러리 래핑이
아니라 지표를 직접 구현(`ragas_eval/`). 순수 계산부는 오프라인 단위테스트되고, LLM 판정이 필요한
지표는 judge를 주입한다(테스트는 stub, 실측은 무료 Gemini).

| 지표 | 의미 | 계산 |
|---|---|---|
| answer_correctness | 답 vs 정답 토큰 F1(사실성) | 순수 |
| context_precision | 관련 컨텍스트가 상위에 있는가 | 순수(순위 가중) |
| context_recall | 정답 진술이 컨텍스트로 뒷받침되는 비율 | LLM judge |
| faithfulness | 답의 주장이 컨텍스트로 뒷받침되는 비율(환각 역지표) | LLM judge |

```bash
python run_ragas.py            # 결정적 stub judge(키 불요, 파이프라인 동작 시연)
python run_ragas.py --gemini   # 무료 Gemini judge로 실측(GEMINI_API_KEY)
python -m pytest tests/test_ragas.py   # 순수 지표 단위테스트
```

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
