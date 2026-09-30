"""RDF/SPARQL 백엔드 — 같은 그래프를 트리플로 옮기고 같은 연산을 SPARQL로 수행한다.

기존 백엔드(in-memory·Kuzu·Neo4j)는 모두 속성 그래프(Cypher)다. RDF는 모델이 다르다:
관계는 (주어, 술어, 목적어) 트리플이고 스키마는 OWL 온톨로지로 따로 선언한다.
같은 인터페이스(`neighbors`·`expand`)를 SPARQL로 구현해 **같은 답이 나오는지**를 먼저 확인하고,
그다음에 RDF에서만 되는 것(OWL 추론, SHACL 검증)을 잰다.

SPARQL 1.1 속성 경로에는 "정확히 k번"(`{k}`)이 없다(초안에 있다가 표준에서 빠졌다).
그래서 k홉은 대안 경로 P = (r1|^r1|r2|^r2|…)를 k번 이은 순차 경로 P/P/…/P로 만들고,
k=1부터 올리며 처음 도달한 k를 최단 홉수로 삼는다.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote, unquote

from rdflib import Graph, Namespace, URIRef

from .corpus import RELATIONS, Relation

RFP = Namespace("https://mjholics.github.io/rfp#")
ENT = Namespace("https://mjholics.github.io/rfp/entity/")
ONTOLOGY = Path(__file__).with_name("rfp_ontology.ttl")


def ent_iri(eid: str) -> URIRef:
    return ENT[quote(eid, safe="")]


def iri_ent(iri) -> str:
    return unquote(str(iri)[len(str(ENT)):])


def to_graph(relations: list[Relation]) -> Graph:
    g = Graph()
    g.bind("rfp", RFP)
    g.bind("ent", ENT)
    for r in relations:
        g.add((ent_iri(r.head), RFP[r.rtype], ent_iri(r.tail)))
    return g


class RdfGraphStore:
    """rdflib 인메모리 트리플 저장소 + SPARQL."""

    def __init__(self, relations: list[Relation] | None = None) -> None:
        self._rels = list(relations if relations is not None else RELATIONS)
        self.graph = to_graph(self._rels)
        self._rtypes = sorted({r.rtype for r in self._rels})
        alt = "|".join(f"rfp:{t}|^rfp:{t}" for t in self._rtypes)
        self._step = f"({alt})"

    def neighbors(self, eid: str) -> list[tuple[str, str, str]]:
        e = ent_iri(eid).n3()
        vals = " ".join(f"rfp:{t}" for t in self._rtypes)
        q = f"""
        SELECT ?h ?p ?t WHERE {{
          VALUES ?p {{ {vals} }}
          {{ BIND({e} AS ?h) ?h ?p ?t }} UNION {{ BIND({e} AS ?t) ?h ?p ?t }}
        }}"""
        out = []
        for h, p, t in self.graph.query(q, initNs={"rfp": RFP}):
            out.append((iri_ent(h), str(p)[len(str(RFP)):], iri_ent(t)))
        return out

    def expand(self, seeds: list[str], max_hops: int) -> dict[str, int]:
        hops = int(max_hops)
        if hops < 0:
            raise ValueError("max_hops는 0 이상이어야 한다")
        dist: dict[str, int] = {s: 0 for s in seeds}
        vals = " ".join(ent_iri(s).n3() for s in seeds)
        for k in range(1, hops + 1):
            path = "/".join([self._step] * k)
            q = f"SELECT DISTINCT ?b WHERE {{ VALUES ?a {{ {vals} }} ?a {path} ?b }}"
            for (b,) in self.graph.query(q, initNs={"rfp": RFP}):
                eid = iri_ent(b)
                if eid not in dist:
                    dist[eid] = k
        return dist
