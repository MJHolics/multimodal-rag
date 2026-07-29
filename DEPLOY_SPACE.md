# Hugging Face Space 배포 — TechDocRAG 검색 전략 비교 데모

> 진입점 `space_app.py` · 무료 CPU Space 기준 · LLM 불필요
> 왜 배포하는가: 타겟 공고에서 **RAG가 가장 많이 요구되는데(10회) 정작 눌러볼 수 있는 RAG 데모가
> 없었다.** 서류에서 "RAG 해봤다"는 문장보다 링크 하나가 강하다.

## 왜 이 데모가 무료 CPU에서 도는가

LLM 생성을 뺐다. 보여주는 건 **검색 전략 비교**이고, 생성은 이미 로컬 파이프라인(Ollama +
Qwen2.5-VL)에 있다. 임베딩은 미리 계산해 `demo_assets/`에 실었으므로(124KB) 예제 질의는
모델 없이 즉시 응답한다. 자유 질의를 입력할 때만 BGE-M3를 늦게 로드하고, 로드에 실패하면
BM25 단독으로 강등하면서 그 사실을 화면에 표시한다.

## 파일

| 파일 | 역할 |
|---|---|
| `space_app.py` | Gradio 진입점 |
| `demo_assets/index.json` · `emb.npz` | 사전 계산된 청크·예제질의 임베딩(124KB) |
| `graph_rag/` | 코퍼스·검색기(순수 파이썬, 모델 불필요) |
| `requirements-space.txt` | Space용 최소 의존성 |

재생성이 필요하면: `python -m scripts.precompute_demo_index`
(코퍼스나 예제 질의를 바꾸면 반드시 다시 돌려야 한다 — 임베딩이 어긋나면 조용히 틀린다.)

## 배포 절차

Space는 루트에 `requirements.txt`와 front-matter가 있는 `README.md`를 요구한다. 본 레포의
README/requirements는 로컬 풀파이프라인용이라 그대로 쓰면 안 되므로, **배포 전용 브랜치**를
만들어 그 브랜치에서만 바꿔 넣는다(inspection-copilot을 `deploy-lfs` 브랜치로 올린 것과 같은 방식).

```bash
cd LLM활용/multimodal_rag

# 1) 배포 브랜치
git checkout -b deploy-space

# 2) Space용 requirements로 교체
cp requirements-space.txt requirements.txt

# 3) README 맨 위에 front-matter 추가 (아래 블록을 그대로 앞에 붙인다)
#    ※ 이 브랜치에서만 한다. master의 README는 건드리지 않는다.

git add -A && git commit -m "deploy: TechDocRAG 검색 전략 비교 데모 (HF Space)"

# 4) Space 원격 등록 후 푸시
hf auth login --add-to-git-credential        # 쓰기 토큰
git remote add space https://huggingface.co/spaces/appleholics/techdocrag-retrieval
git push space deploy-space:main
```

### README.md 맨 위에 붙일 front-matter

```yaml
---
title: TechDocRAG — 검색 전략 비교
emoji: 🔎
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.0.0
app_file: space_app.py
pinned: false
---
```

## 배포 후 확인할 것

- [ ] 첫 화면이 모델 로딩 없이 뜨는가(예제 질의 클릭 → 즉시 응답)
- [ ] 자유 질의 첫 입력에서 모델 로드가 되는가, 실패 시 BM25 강등 문구가 뜨는가
- [ ] 멀티홉 예제(q11 "배터리 칠러와 연결된 부품의 용량")에서 ②가 함정 문서를 1위로 집는가
      — 이 데모의 핵심 장면이다
- [ ] 이력서·포트폴리오의 링크 갱신
