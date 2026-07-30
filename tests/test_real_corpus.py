"""실문서 청킹·관계연결·평가셋의 불변식 — 순수·결정적, HWP 원본 불필요.

여기서 지키는 것은 **실문서로 넘어오면서 새로 생긴 위험**들이다:
  - 근거 문장이 청크 경계에 걸려 gold를 못 만드는 경우
  - IE 관계와 청크의 연결이 끊기는 경우
  - 평가셋이 조용히 쉬워지는 경우(사업명 누출·중의성)
  - **파싱 단계의 조용한 손상**(짝 없는 서로게이트)이 나중 단계에서 터지는 경우
"""
from __future__ import annotations

from graph_rag.corpus import Chunk
from knowledgeops.chunking import build_real_corpus, corpus_summary, split_text
from knowledgeops.eval_real import build_real_queries, summarize_real
from knowledgeops.extract import Document, Triple
from knowledgeops.hwp import _decode_para

# ---- 청킹 ----


def test_split_respects_size_and_covers_all_lines():
    text = "\n".join(f"{i}번째 줄 내용입니다" * 3 for i in range(60))
    parts = split_text(text, size=300, overlap=50)
    assert len(parts) > 1
    joined = "".join(parts)
    for i in range(60):
        assert f"{i}번째 줄" in joined, i


def test_split_is_deterministic():
    text = "\n".join(f"줄{i}" for i in range(200))
    assert split_text(text, 200, 30) == split_text(text, 200, 30)


def test_split_overlap_bridges_boundary():
    """겹침이 없으면 경계에 걸친 근거 문장을 어느 청크에서도 못 찾는다."""
    text = "\n".join(f"문단{i} 내용" for i in range(40))
    with_overlap = split_text(text, 100, 40)
    no_overlap = split_text(text, 100, 0)
    assert sum(len(p) for p in with_overlap) > sum(len(p) for p in no_overlap)


def test_split_rejects_bad_size():
    import pytest

    with pytest.raises(ValueError):
        split_text("가나다", 0)


# ---- 관계 → 청크 연결 ----


def _doc(text: str, triples, evidence) -> Document:
    d = Document(doc_id="d0", agency="기관", project="사업", text=text)
    d.triples = triples
    d.evidence = evidence
    return d


def test_relation_links_to_chunk_containing_its_evidence():
    body = "\n".join(["머리말 " * 20, "다. 사업비 : 157,300,000원(VAT 포함)", "꼬리말 " * 20])
    doc = _doc(
        body,
        [Triple("사업", "HAS_BUDGET", "157,300,000원")],
        {"사업|HAS_BUDGET|157,300,000원": "다. 사업비 : 157,300,000원(VAT 포함)"},
    )
    rc = build_real_corpus([doc], size=200)
    assert rc.unlinked == []
    (key,) = list(rc.home)
    cid = rc.home[key]
    target = next(c for c in rc.chunks if c.chunk_id == cid)
    assert "157,300,000원" in target.text


def test_linked_chunk_carries_relation_and_entities():
    doc = _doc(
        "다. 사업비 : 100,000,000원",
        [Triple("사업", "HAS_BUDGET", "100,000,000원")],
        {"사업|HAS_BUDGET|100,000,000원": "다. 사업비 : 100,000,000원"},
    )
    rc = build_real_corpus([doc])
    linked = [c for c in rc.chunks if c.states]
    assert linked and linked[0].entities  # 검색이 쓸 수 있는 형태여야 한다


def test_issued_by_falls_back_to_first_chunk():
    """파일명에서 온 관계는 본문에 근거가 없을 수 있다 — 첫 청크(개요)로 연결한다."""
    doc = _doc(
        "\n".join(["본문 " * 30] * 10),
        [Triple("사업", "ISSUED_BY", "기관")],
        {"사업|ISSUED_BY|기관": "[파일명] 기관_사업.hwp"},
    )
    rc = build_real_corpus([doc], size=200)
    assert rc.unlinked == []
    assert rc.home[list(rc.home)[0]].endswith("#c000")


def test_corpus_summary_reports_link_rate():
    doc = _doc(
        "다. 사업비 : 100,000,000원",
        [Triple("사업", "HAS_BUDGET", "100,000,000원")],
        {"사업|HAS_BUDGET|100,000,000원": "다. 사업비 : 100,000,000원"},
    )
    s = corpus_summary(build_real_corpus([doc]))
    assert s["link_rate"] == 1.0
    assert s["relations"] == 1


# ---- 평가셋 ----


def _two_docs() -> list[Document]:
    # 예산 문장이 **첫 청크 밖**에 오도록 앞머리를 길게 둔다.
    # 기관 근거(첫 청크)와 예산 근거가 같은 청크면 멀티홉이 성립하지 않고,
    # 생성기가 그 질의를 정상적으로 버린다(그 동작 자체는 다른 테스트에서 확인).
    head = "\n".join(["사업 개요 서술 문장입니다" for _ in range(20)])
    tail = "\n".join(["기타 참고 사항 문장입니다" for _ in range(20)])
    body_a = "\n".join([head, "다. 사업비 : 111,000,000원", tail])
    body_b = "\n".join([head, "다. 사업비 : 222,000,000원", tail])
    a = Document("d1", "기관가", "사업가", body_a)
    a.triples = [Triple("사업가", "ISSUED_BY", "기관가"),
                 Triple("사업가", "HAS_BUDGET", "111,000,000원")]
    a.evidence = {"사업가|ISSUED_BY|기관가": "[파일명] 기관가_사업가.hwp",
                  "사업가|HAS_BUDGET|111,000,000원": "다. 사업비 : 111,000,000원"}
    b = Document("d2", "기관나", "사업나", body_b)
    b.triples = [Triple("사업나", "ISSUED_BY", "기관나"),
                 Triple("사업나", "HAS_BUDGET", "222,000,000원")]
    b.evidence = {"사업나|ISSUED_BY|기관나": "[파일명] 기관나_사업나.hwp",
                  "사업나|HAS_BUDGET|222,000,000원": "다. 사업비 : 222,000,000원"}
    return [a, b]


def test_real_queries_are_deterministic_and_typed():
    rc = build_real_corpus(_two_docs(), size=200)
    qs1 = build_real_queries(rc)
    qs2 = build_real_queries(rc)
    assert [(q.qid, q.question, q.gold_chunks) for q in qs1] == [
        (q.qid, q.question, q.gold_chunks) for q in qs2
    ]
    assert {q.hops for q in qs1} <= {1, 2}


def test_multihop_query_does_not_name_the_project():
    """다리(사업명)를 부르면 어휘 검색이 그냥 찾는다 — 합성에서 배운 그 규칙."""
    rc = build_real_corpus(_two_docs(), size=200)
    for q in build_real_queries(rc):
        if q.hops == 2:
            assert "사업가" not in q.question and "사업나" not in q.question


def test_multihop_gold_spans_two_chunks():
    rc = build_real_corpus(_two_docs(), size=200)
    multi = [q for q in build_real_queries(rc) if q.hops == 2]
    assert multi
    for q in multi:
        assert len(q.gold_chunks) >= 2
        assert len(set(q.gold_chunks)) == len(q.gold_chunks)


def test_multihop_skipped_when_agency_has_multiple_projects():
    """한 기관이 여러 사업을 발주하면 '그 사업'이 정해지지 않는다 → 질의를 만들지 않는다."""
    docs = _two_docs()
    docs[1].triples[0] = Triple("사업나", "ISSUED_BY", "기관가")  # 같은 기관이 둘 발주
    docs[1].evidence["사업나|ISSUED_BY|기관가"] = "[파일명] 기관가_사업나.hwp"
    rc = build_real_corpus(docs, size=200)
    for q in build_real_queries(rc):
        if q.hops == 2:
            assert "기관가" not in q.question


def test_summarize_real_counts():
    rc = build_real_corpus(_two_docs(), size=200)
    s = summarize_real(build_real_queries(rc))
    assert s["total"] == s["single_hop"] + s["multi_hop"]


# ---- 파싱 손상 방지 ----


def test_decode_para_joins_surrogate_pairs():
    """BMP 밖 문자는 서로게이트 **쌍**으로 저장된다 — 합쳐야 원래 문자가 된다.

    각 단위를 따로 chr()하면 짝 없는 서로게이트가 남고, 그 문자열은 인코딩이 안 된다.
    실제로 이 손상이 청크 7,863개 중 260개에 있었고, 임베딩 단계에서야 터졌다.
    """
    payload = (0xD83D).to_bytes(2, "little") + (0xDE00).to_bytes(2, "little")
    got = _decode_para(payload)
    assert got == "\U0001F600"
    got.encode("utf-8")  # 인코딩 가능해야 한다(여기서 죽으면 임베딩이 죽는다)


def test_decode_para_drops_lone_surrogate():
    payload = (0xD800).to_bytes(2, "little") + ord("가").to_bytes(2, "little")
    got = _decode_para(payload)
    assert got == "가"
    got.encode("utf-8")


def test_decode_para_keeps_normal_text_and_breaks():
    payload = b"".join(
        c.to_bytes(2, "little") for c in [ord("가"), 13, ord("나")]
    )
    assert _decode_para(payload) == "가\n나"
