"""Neo4j 백엔드 테스트.

`graph_store.py`의 모듈 주석은 오래전부터 "Neo4j 테스트는 서버가 없으면 skip된다"고 적고
있었는데, **실제로는 Neo4j 테스트가 하나도 없었다**(2026-07-31 확인). 문서가 코드보다 앞서
있었던 것이라 채운다. 없는 걸 있다고 적어 두는 것도 과장이다.

두 종류로 나눈다:
  - 서버 없이 검증 가능한 것 — 설정 해석, 쿼리 안전성, 실패 시 조용히 폴백하지 않는지.
  - 서버가 있어야만 검증 가능한 것 — 실제 Cypher 결과가 순수 구현과 같은지.
    후자는 `NEO4J_URI`/`NEO4J_PASSWORD`가 없으면 skip한다. **안 돌린 걸 돌린 척하지 않는다.**
"""
from __future__ import annotations

import os

import pytest

from graph_rag.corpus import RELATIONS
from graph_rag.graph_store import InMemoryGraphStore, Neo4jGraphStore

HAS_SERVER = bool(os.environ.get("NEO4J_URI") and os.environ.get("NEO4J_PASSWORD"))
needs_server = pytest.mark.skipif(
    not HAS_SERVER, reason="NEO4J_URI/NEO4J_PASSWORD 없음 — AuraDB 접속 정보를 넣으면 돈다"
)


# ---------- 서버 없이 ----------

def test_from_env_refuses_to_fall_back_silently(monkeypatch):
    """자격증명이 없으면 예외. 조용히 로컬로 폴백하면 'AuraDB로 돌렸다'가 거짓이 된다."""
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
    with pytest.raises(RuntimeError) as e:
        Neo4jGraphStore.from_env()
    assert "NEO4J_URI" in str(e.value)


def test_from_env_reads_all_settings(monkeypatch):
    """AuraDB는 TLS 스킴(neo4j+s://)과 database 지정이 필요하다 — 설정 경로를 고정한다."""
    monkeypatch.setenv("NEO4J_URI", "neo4j+s://demo.databases.neo4j.io")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")
    monkeypatch.setenv("NEO4J_DATABASE", "neo4j")
    captured = {}

    class _FakeDriver:
        def session(self, **kw):
            raise AssertionError("이 테스트는 연결하지 않는다")

        def close(self):
            pass

    def _fake_driver(uri, auth):
        captured["uri"] = uri
        captured["auth"] = auth
        return _FakeDriver()

    import neo4j

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", staticmethod(_fake_driver))
    store = Neo4jGraphStore.from_env(relations=RELATIONS)
    assert captured["uri"].startswith("neo4j+s://")
    assert captured["auth"] == ("neo4j", "pw")
    assert store._database == "neo4j"


def test_expand_rejects_negative_hops(monkeypatch):
    """가변길이 상한은 Cypher에 **문자열 보간**된다(파라미터 불가) — int 검증이 유일한 방어선."""
    monkeypatch.setenv("NEO4J_URI", "neo4j+s://demo.databases.neo4j.io")
    monkeypatch.setenv("NEO4J_PASSWORD", "pw")

    class _FakeDriver:
        def session(self, **kw):
            raise AssertionError("검증에서 막혀야 하므로 세션까지 가면 안 된다")

        def close(self):
            pass

    import neo4j

    monkeypatch.setattr(neo4j.GraphDatabase, "driver", staticmethod(lambda uri, auth: _FakeDriver()))
    store = Neo4jGraphStore.from_env(relations=RELATIONS)
    with pytest.raises(ValueError):
        store.expand(["a"], -1)
    with pytest.raises((ValueError, TypeError)):
        store.expand(["a"], "2; MATCH (n) DETACH DELETE n")  # type: ignore[arg-type]


def test_load_batches_by_relation_type():
    """관계마다 왕복하면 원격(AuraDB)에서 지연이 곱해진다 — 타입별 UNWIND 묶음인지 확인."""
    import inspect

    src = inspect.getsource(Neo4jGraphStore.load)
    assert "UNWIND" in src
    assert "batch" in src


# ---------- 서버가 있어야 ----------

@needs_server
def test_neo4j_matches_pure_implementation_on_expand():
    """실제 Cypher 결과가 순수 구현과 같은가 — 이게 '실행 검증'의 내용이다."""
    store = Neo4jGraphStore.from_env(relations=RELATIONS)
    try:
        store.load()
        mem = InMemoryGraphStore(RELATIONS)
        seeds = sorted({r.head for r in RELATIONS})[:5]
        for s in seeds:
            assert store.expand([s], 2) == mem.expand([s], 2), f"seed={s}"
    finally:
        store.close()


@needs_server
def test_neo4j_matches_pure_implementation_on_neighbors():
    store = Neo4jGraphStore.from_env(relations=RELATIONS)
    try:
        store.load()
        mem = InMemoryGraphStore(RELATIONS)
        for s in sorted({r.head for r in RELATIONS})[:5]:
            assert {tuple(x) for x in store.neighbors(s)} == {
                tuple(x) for x in mem.neighbors(s)
            }, f"seed={s}"
    finally:
        store.close()


@needs_server
def test_load_is_idempotent():
    """두 번 적재해도 그래프가 불어나지 않아야 한다(MERGE + 선삭제)."""
    store = Neo4jGraphStore.from_env(relations=RELATIONS)
    try:
        store.load()
        first = store.expand([sorted({r.head for r in RELATIONS})[0]], 2)
        store.load()
        assert store.expand([sorted({r.head for r in RELATIONS})[0]], 2) == first
    finally:
        store.close()
