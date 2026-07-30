"""실문서를 검색 단위(청크)로 쪼개고, IE가 찾은 근거를 청크에 연결한다.

## 왜 이 연결이 핵심인가

합성 코퍼스에서는 "이 청크가 이 관계를 서술한다"가 생성 규칙으로 **주어졌다**(`Chunk.states`).
실문서에는 그런 게 없다. 대신 IE가 관계를 뽑을 때 **근거 문장**을 함께 남겨 뒀으므로
(`Document.evidence`), 그 문장이 실제로 들어 있는 청크를 찾으면 같은 구조가 복원된다.

    관계 → (IE가 남긴) 근거 문장 → 그 문장을 포함한 청크

이 연결이 있어야 `graph_rag`의 검색 전략들을 **코드 수정 없이 그대로** 실문서에 쓸 수 있다.
평가의 gold도 여기서 나온다: "그 관계를 답하려면 이 청크를 찾아야 한다".

## 정직한 주의 — 이건 IE를 정답으로 가정한다

IE가 틀린 관계를 뽑았다면 gold도 틀린다. 표본 감사에서 예산 오탐을 문맥 가드로 닫았지만
0이 되진 않았다. 그래서 실문서 수치는 **IE 정밀도를 곱한 상한**으로 읽어야 하고,
합성 코퍼스 수치와 직접 비교하면 안 된다(합성은 IE가 완벽한 조건이다).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from graph_rag.corpus import Chunk, Entity, Relation

from .extract import Document, Triple

# 검색 단위 크기. RFP 본문은 4만~10만 자라 통째로는 검색 단위가 될 수 없다.
# 1,000자는 `app/retriever.py`가 쓰는 페이지 청크와 비슷한 규모로 잡았다.
CHUNK_SIZE = 1000
OVERLAP = 150


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """줄 경계를 지키며 자른다. 순수·결정적.

    문자 수로 딱 자르면 근거 문장이 두 청크에 걸쳐 **어느 쪽에서도 온전히 안 보이는** 일이
    생긴다(그러면 gold를 연결할 수 없다). 그래서 줄 단위로 모으고, overlap으로 경계를 덮는다.
    """
    if size <= 0:
        raise ValueError("size > 0")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    chunks: list[str] = []
    buf: list[str] = []
    length = 0
    for ln in lines:
        if length + len(ln) > size and buf:
            chunks.append("\n".join(buf))
            # overlap만큼 뒤에서부터 되감아 다음 청크의 머리로 쓴다
            back, tail = 0, []
            for prev in reversed(buf):
                if back >= overlap:
                    break
                tail.insert(0, prev)
                back += len(prev)
            buf, length = tail, back
        buf.append(ln)
        length += len(ln)
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _norm(s: str) -> str:
    """근거 문장 매칭용 정규화 — 공백 차이로 매칭이 깨지는 걸 막는다."""
    return re.sub(r"\s+", "", s)


@dataclass
class RealCorpus:
    """실문서에서 만든 검색 코퍼스 + 지식그래프.

    `graph_rag`가 기대하는 형태(Chunk/Entity/Relation)로 맞춰 내보내므로
    기존 검색·평가 코드를 그대로 재사용할 수 있다.
    """

    chunks: list[Chunk] = field(default_factory=list)
    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    # 관계 → 그 관계를 서술하는 청크 id (합성 코퍼스의 relation_home과 같은 역할)
    home: dict[tuple[str, str, str], str] = field(default_factory=dict)
    unlinked: list[tuple[str, str, str]] = field(default_factory=list)


def _eid(kind: str, name: str) -> str:
    """엔티티 id — 이름이 곧 표면형이라 종류 접두어로만 구분한다."""
    return f"{kind}:{name}"


def build_real_corpus(docs: list[Document], size: int = CHUNK_SIZE) -> RealCorpus:
    """문서들 → 청크 + 그래프 + 관계-청크 연결. 순수 함수."""
    rc = RealCorpus()

    for doc in docs:
        pieces = split_text(doc.text, size)
        norm_pieces = [_norm(p) for p in pieces]
        for i, piece in enumerate(pieces):
            rc.chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#c{i:03d}",
                    doc_name=doc.doc_id,
                    page_num=i,
                    text=piece,
                    states=[],
                    entities=[],
                )
            )

        base = len(rc.chunks) - len(pieces)
        for t in doc.triples:
            head_id = _eid("project", t.head)
            tail_kind = {
                "ISSUED_BY": "agency",
                "REQUIRES": "tech",
                "HAS_BUDGET": "budget",
                "HAS_PERIOD": "period",
            }[t.rtype]
            tail_id = _eid(tail_kind, t.tail)
            rc.entities.setdefault(head_id, Entity(head_id, t.head, "project"))
            rc.entities.setdefault(tail_id, Entity(tail_id, t.tail, tail_kind))
            rel = Relation(head_id, t.rtype, tail_id)
            rc.relations.append(rel)

            # 근거 문장이 들어 있는 청크를 찾는다
            ev = _norm(doc.evidence.get(f"{t.head}|{t.rtype}|{t.tail}", ""))
            idx = _locate(ev, norm_pieces)
            if idx is None and t.rtype == "REQUIRES":
                # 기술 근거는 매우 짧을 수 있다("RESTful", "SSO/LDAP 서버").
                # 근거 문장으로 못 찾으면 **그 기술의 표면형**으로 청크를 찾는다.
                # 이건 IE가 이미 그 문서에서 그 기술을 인정했다는 사실을 이용하는 것이라
                # 새 판단을 만들지 않는다(위치만 복원한다).
                idx = _locate_surface(t.tail, norm_pieces)
            if idx is None:
                # 파일명에서 온 관계(ISSUED_BY)는 본문에 근거가 없을 수 있다 →
                # 문서의 첫 청크(사업 개요)를 근거로 둔다. 실제로 표지·개요에 기관이 적힌다.
                idx = 0 if t.rtype == "ISSUED_BY" else None
            if idx is None:
                rc.unlinked.append((t.head, t.rtype, t.tail))
                continue

            cid = rc.chunks[base + idx].chunk_id
            key = (head_id, t.rtype, tail_id)
            rc.home.setdefault(key, cid)
            ch = rc.chunks[base + idx]
            ch.states.append(rel)
            for e in (head_id, tail_id):
                if e not in ch.entities:
                    ch.entities.append(e)

    return rc


def _locate(needle: str, haystacks: list[str]) -> int | None:
    """근거 문장이 들어 있는 첫 청크 인덱스. 짧은 근거는 오탐이 많아 버린다."""
    if len(needle) < 8:
        return None
    for i, h in enumerate(haystacks):
        if needle in h:
            return i
    # overlap 경계에 걸린 경우를 위해 앞부분만으로 한 번 더
    probe = needle[: max(8, len(needle) // 2)]
    for i, h in enumerate(haystacks):
        if probe in h:
            return i
    return None


def _locate_surface(canon: str, haystacks: list[str]) -> int | None:
    """기술의 표면형(별칭 포함)이 등장하는 첫 청크. 근거 문장이 너무 짧을 때의 보조 경로."""
    from .extract import TECH_ALIASES

    for form in TECH_ALIASES.get(canon, (canon,)):
        probe = _norm(form)
        if len(probe) < 2:
            continue
        for i, h in enumerate(haystacks):
            if probe in h:
                return i
    return None


def corpus_summary(rc: RealCorpus) -> dict:
    linked = len(rc.home)
    return {
        "chunks": len(rc.chunks),
        "entities": len(rc.entities),
        "relations": len(rc.relations),
        "linked_relations": linked,
        "unlinked_relations": len(rc.unlinked),
        "link_rate": round(linked / len(rc.relations), 4) if rc.relations else 0.0,
        "chunks_with_relation": sum(1 for c in rc.chunks if c.states),
        "distractor_chunks": sum(1 for c in rc.chunks if not c.states),
    }
