"""TechDocRAG Streamlit 데모 UI"""
import streamlit as st
import requests
from pathlib import Path

API_URL = "http://localhost:8000"

st.set_page_config(page_title="TechDocRAG", page_icon="📄", layout="wide")
st.title("📄 TechDocRAG — 기술 문서 멀티모달 RAG")
st.caption("PDF 문서(표·다이어그램 포함)를 업로드하고 자연어로 질문하세요.")

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    top_k = st.slider("검색 페이지 수 (top_k)", 1, 5, 2)
    st.divider()

    try:
        h = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("서버 연결됨")
        st.metric("인덱싱된 페이지", h["indexed_docs"])
        st.metric("Ollama", "✓ 연결됨" if h["ollama_available"] else "✗ 폴백 모드")
    except Exception:
        st.error("서버 연결 실패 — FastAPI 서버를 먼저 실행하세요")
        st.code("cd app && uvicorn main:app --port 8000")

    st.divider()

    st.header("📂 문서 업로드")
    pdf_file = st.file_uploader("PDF 파일 선택", type=["pdf"])
    if pdf_file and st.button("인덱싱 시작", use_container_width=True):
        with st.spinner("인덱싱 중..."):
            resp = requests.post(
                f"{API_URL}/ingest",
                files={"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")},
            )
        if resp.status_code == 200:
            d = resp.json()
            st.success(f"완료: {d['pages_indexed']}페이지 ({d['elapsed_sec']}초)")
        else:
            st.error(f"오류: {resp.text}")

# 메인 영역
st.header("💬 질문하기")
question = st.text_input(
    "질문을 입력하세요",
    placeholder="예: 배터리 팩의 공칭 전압은? / DTC 코드 P0A1E는 무엇인가요?",
)

if st.button("검색 + 답변 생성", type="primary", use_container_width=True) and question:
    with st.spinner("하이브리드 검색 + AI 답변 생성 중..."):
        try:
            resp = requests.post(
                f"{API_URL}/query",
                json={"question": question, "top_k": top_k},
                timeout=120,
            )
        except requests.exceptions.ConnectionError:
            st.error("서버에 연결할 수 없습니다.")
            st.stop()

    if resp.status_code == 200:
        data = resp.json()

        st.subheader("🤖 AI 답변")
        st.markdown(data["answer"])
        st.caption(f"모델: `{data['model_used']}` | 응답시간: {data['elapsed_sec']}초")

        st.divider()

        st.subheader("📑 출처 페이지")
        cols = st.columns(len(data["sources"]))
        for col, src in zip(cols, data["sources"]):
            with col:
                st.markdown(
                    f"**{src['chunk_id']}**  \n페이지 {src['page_num']+1} | score `{src['score']:.3f}`"
                )
                doc = src["chunk_id"].split("_p")[0]
                pnum = int(src["chunk_id"].split("_p")[1])
                ipath = Path(__file__).parent.parent / "output" / doc / f"page_{pnum:03d}.png"
                if ipath.exists():
                    st.image(str(ipath), use_container_width=True)
                else:
                    st.info("이미지를 찾을 수 없습니다.")
    elif resp.status_code == 404:
        st.warning("관련 문서를 찾을 수 없습니다. PDF를 먼저 업로드하고 인덱싱해주세요.")
    else:
        st.error(f"오류 {resp.status_code}: {resp.text}")

with st.expander("💡 예시 질문"):
    examples = [
        "배터리 팩의 공칭 전압과 용량은 얼마인가요?",
        "DTC 코드 P0A1E는 어떤 문제를 나타내나요?",
        "SOC는 어떻게 추정하나요?",
        "고전압 작업 시 안전 절차를 알려주세요.",
        "셀 밸런싱이란 무엇인가요?",
    ]
    for ex in examples:
        st.markdown(f"- {ex}")
