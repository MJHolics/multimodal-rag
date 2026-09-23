"""지식그래프 저장소 — 백엔드 3종을 같은 인터페이스로.

## 왜 여러 개인가

- **InMemoryGraphStore**: 순수 파이썬, 외부 의존 0. 단위테스트와 실험 재현의 기준.
- **KuzuGraphStore**: **임베디드 그래프 DB(Cypher)**. 서버·Docker 없이 파일 하나로 돌아
  CI에서도 실제 Cypher 경로가 검증된다. 이 프로젝트에서 실제로 실행되는 Graph DB 경로다.
- **Neo4jGraphStore**: 서버형 Graph DB. 같은 연산을 Neo4j Cypher로 수행한다.

정직한 상태(2026-07-31 갱신): 이 개발 환경에는 Docker도 Java도 없어 **로컬 Neo4j 서버는
띄우지 못한다.** 그래서 실행 검증은 Kuzu로 했다(둘 다 Cypher).
Neo4j 경로는 **Neo4j AuraDB 무료 인스턴스**로 검증할 수 있게 준비돼 있다 —
`Neo4jGraphStore.from_env()` + `verify_neo4j.py`. 접속 정보가 없으면 테스트는 skip된다.

⚠️ 2026-07-31 이전 이 주석은 "Neo4j 테스트는 서버가 없으면 skip된다"고 적고 있었으나
**Neo4j 테스트 자체가 하나도 없었다.** 문서가 코드보다 앞서 있었던 셈이라 테스트를 채웠다
(`tests/test_neo4j.py`). 없는 걸 있다고 적어 둔 것도 과장이다.

세 백엔드가 **같은 결과**를 내는지는 테스트로 고정한다.
이 구조가 아니면 "그래프 DB 붙였다"가 동작 검증 없는 주장이 된다.
"""
from __future__ import annotations

from collections import deque
from typing import Protocol

from .corpus import RELATIONS, Relation


class GraphStore(Protocol):
    """검색이 그래프에 요구하는 최소 연산."""

    def neighbors(self, eid: str) -> list[tuple[str, str, str]]:
        """eid에 인접한 (head, rtype, tail) 관계들. 방향 무관(검색은 양방향 탐색)."""
        ...

    def expand(self, seeds: list[str], max_hops: int) -> dict[str, int]:
        """시드에서 max_hops 이내로 도달하는 엔티티 → 최소 홉수."""
        ...


class InMemoryGraphStore:
    """인접리스트 기반. 결정적(삽입 순서 유지)."""

    def __init__(self, relations: list[Relation] | None = None) -> None:
        self._rels = list(relations if relations is not None else RELATIONS)
        self._adj: dict[str, list[tuple[str, str, str]]] = {}
        for r in self._rels:
            t = (r.head, r.rtype, r.tail)
            self._adj.setdefault(r.head, []).append(t)
            self._adj.setdefault(r.tail, []).append(t)

    def neighbors(self, eid: str) -> list[tuple[str, str, str]]:
        return list(self._adj.get(eid, []))

    def expand(self, seeds: list[str], max_hops: int) -> dict[str, int]:
        """BFS. 시드는 거리 0. 같은 엔티티는 최단 거리만 남긴다."""
        dist: dict[str, int] = {}
        q: deque[tuple[str, int]] = deque()
        for s in seeds:
            if s not in dist:
                dist[s] = 0
                q.append((s, 0))
        while q:
            eid, d = q.popleft()
            if d >= max_hops:
                continue
            for head, _rtype, tail in self._adj.get(eid, []):
                for nxt in (head, tail):
                    if nxt not in dist:
                        dist[nxt] = d + 1
                        q.append((nxt, d + 1))
        return dist


class KuzuGraphStore:
    """임베디드 그래프 DB(Kuzu) 백엔드 — 탐색을 실제 **Cypher**로 수행.

    Kuzu는 SQLite처럼 파일 기반이라 서버가 필요 없다. 스키마가 고정형(노드/엣지 테이블 선언)이라
    관계 타입마다 REL TABLE을 만든다 — Neo4j의 스키마리스와 다른 지점이고, 그래서
    `expand`의 가변길이 탐색도 관계 테이블 전체를 대상으로 쓴다.
    """

    #: 코퍼스가 쓰는 관계 타입(REL TABLE 이름이 된다)
    RTYPES = ("CONTAINS", "CONNECTS_TO", "HAS_SPEC", "SUPPLIED_BY")

    def __init__(self, db_path: str, relations: list[Relation] | None = None) -> None:
        import kuzu

        self._rels = list(relations if relations is not None else RELATIONS)
        self._db = kuzu.Database(db_path)
        self._conn = kuzu.Connection(self._db)
        self._create_schema()
        self._load()

    def _create_schema(self) -> None:
        self._conn.execute("CREATE NODE TABLE IF NOT EXISTS Entity(eid STRING, PRIMARY KEY(eid))")
        for rt in self.RTYPES:
            self._conn.execute(
                f"CREATE REL TABLE IF NOT EXISTS {rt}(FROM Entity TO Entity)"
            )

    def _load(self) -> None:
        eids: list[str] = []
        for r in self._rels:
            for e in (r.head, r.tail):
                if e not in eids:
                    eids.append(e)
        for e in eids:
            self._conn.execute("MERGE (:Entity {eid: $e})", {"e": e})
        for r in self._rels:
            self._conn.execute(
                f"MATCH (a:Entity {{eid:$h}}), (b:Entity {{eid:$t}}) "
                f"MERGE (a)-[:{r.rtype}]->(b)",
                {"h": r.head, "t": r.tail},
            )

    def neighbors(self, eid: str) -> list[tuple[str, str, str]]:
        res = self._conn.execute(
            "MATCH (a:Entity {eid:$e})-[r]-(b:Entity) "
            "RETURN a.eid, label(r), b.eid",
            {"e": eid},
        )
        out: list[tuple[str, str, str]] = []
        while res.has_next():
            a, rt, b = res.get_next()
            # 무방향 조회라 (a,b)가 뒤집혀 나올 수 있다 — 원 방향은 관계 정의로 되돌린다.
            out.append((a, rt, b))
        return out

    def expand(self, seeds: list[str], max_hops: int) -> dict[str, int]:
        hops = int(max_hops)
        if hops < 0:
            raise ValueError("max_hops는 0 이상이어야 한다")
        dist: dict[str, int] = {s: 0 for s in seeds}
        if hops == 0:
            return dist
        res = self._conn.execute(
            f"MATCH p = (a:Entity)-[*1..{hops}]-(b:Entity) "
            "WHERE a.eid IN $seeds "
            "RETURN b.eid, min(length(p))",
            {"seeds": list(seeds)},
        )
        while res.has_next():
            eid, d = res.get_next()
            d = int(d)
            if eid not in dist or d < dist[eid]:
                dist[eid] = d
        return dist


class Neo4jGraphStore:
    """Neo4j 백엔드 — 같은 연산을 Cypher로 수행.

    사용 전 `load()`로 그래프를 적재한다(멱등: 기존 :Entity 노드를 지우고 다시 씀).
    연결 실패 시 예외를 그대로 올린다 — 조용히 in-memory로 폴백하면 "Neo4j로 돌렸다"가
    거짓이 될 수 있어서다.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "testpassword",
        relations: list[Relation] | None = None,
        database: str | None = None,
    ) -> None:
        from neo4j import GraphDatabase  # 지연 import — 드라이버 없이도 모듈 로드 가능

        self._rels = list(relations if relations is not None else RELATIONS)
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    @classmethod
    def from_env(cls, relations: list[Relation] | None = None) -> "Neo4jGraphStore":
        """환경변수로 접속한다 — 로컬 Docker든 **Neo4j AuraDB 무료 인스턴스**든 같은 코드.

            NEO4J_URI       AuraDB는 neo4j+s://xxxx.databases.neo4j.io (TLS 필수)
            NEO4J_USER      기본 neo4j
            NEO4J_PASSWORD  인스턴스 생성 시 한 번만 보여주는 값
            NEO4J_DATABASE  AuraDB 무료는 neo4j 고정

        자격증명이 없으면 `RuntimeError`를 낸다. 조용히 로컬로 폴백하면
        "AuraDB에서 돌렸다"가 거짓이 될 수 있어서다(클래스 docstring과 같은 이유).
        """
        import os

        uri = os.environ.get("NEO4J_URI")
        password = os.environ.get("NEO4J_PASSWORD")
        if not uri or not password:
            raise RuntimeError(
                "NEO4J_URI / NEO4J_PASSWORD 가 없다. "
                "AuraDB 무료 인스턴스를 만들고 접속 정보를 환경변수로 넣을 것 "
                "(README의 'Neo4j AuraDB로 검증하기' 참조)."
            )
        return cls(
            uri=uri,
            user=os.environ.get("NEO4J_USER", "neo4j"),
            password=password,
            relations=relations,
            database=os.environ.get("NEO4J_DATABASE") or None,
        )

    def _session(self):
        if self._database:
            return self._driver.session(database=self._database)
        return self._driver.session()

    def close(self) -> None:
        self._driver.close()

    def load(self, batch: int = 500) -> None:
        """그래프를 적재한다(멱등: 기존 :Entity를 지우고 다시 씀).

        관계마다 쿼리를 한 번씩 보내면 로컬에선 괜찮지만 **AuraDB는 원격이라 왕복 지연이
        곱해진다**(관계 1,051개면 수 분). 관계 타입별로 묶어 `UNWIND`로 한 번에 보낸다.
        rtype은 코퍼스가 정의한 고정 집합이라 문자열 보간이 안전하다(값은 전부 파라미터).
        """
        by_type: dict[str, list[dict[str, str]]] = {}
        for r in self._rels:
            by_type.setdefault(r.rtype, []).append({"h": r.head, "t": r.tail})

        with self._session() as s:
            s.run("MATCH (n:Entity) DETACH DELETE n")
            s.run("CREATE INDEX entity_eid IF NOT EXISTS FOR (n:Entity) ON (n.eid)")
            for rtype, pairs in by_type.items():
                for i in range(0, len(pairs), batch):
                    s.run(
                        "UNWIND $rows AS row "
                        "MERGE (a:Entity {eid: row.h}) "
                        "MERGE (b:Entity {eid: row.t}) "
                        f"MERGE (a)-[:{rtype}]->(b)",
                        rows=pairs[i : i + batch],
                    )

    def neighbors(self, eid: str) -> list[tuple[str, str, str]]:
        with self._session() as s:
            rows = s.run(
                "MATCH (a:Entity {eid:$e})-[r]-(b:Entity) "
                "RETURN startNode(r).eid AS h, type(r) AS t, endNode(r).eid AS tl",
                e=eid,
            )
            return [(rec["h"], rec["t"], rec["tl"]) for rec in rows]

    def expand(self, seeds: list[str], max_hops: int) -> dict[str, int]:
        """가변 길이 경로로 최단 홉수를 구한다(방향 무관).

        주의: Cypher의 가변 길이 상한 `[*0..N]`은 **파라미터로 받을 수 없다**(리터럴만 허용).
        그래서 int로 강제 검증한 뒤 보간한다 — 임의 문자열이 들어갈 여지를 막는다.
        """
        hops = int(max_hops)
        if hops < 0:
            raise ValueError("max_hops는 0 이상이어야 한다")
        with self._session() as s:
            rows = s.run(
                "MATCH (a:Entity) WHERE a.eid IN $seeds "
                f"MATCH p = (a)-[*0..{hops}]-(b:Entity) "
                "RETURN b.eid AS eid, min(length(p)) AS d",
                seeds=list(seeds),
            )
            return {rec["eid"]: rec["d"] for rec in rows}
