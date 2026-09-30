"""RDF/SPARQL 백엔드 테스트 — 기존 백엔드와 같은 답인지, 온톨로지가 결함을 어떻게 다루는지."""
from __future__ import annotations

import pytest

pytest.importorskip("rdflib", reason="requirements-rdf.txt 미설치")
owlrl = pytest.importorskip("owlrl")
validate = pytest.importorskip("pyshacl").validate

from rdflib import OWL, Graph

from graph_rag.corpus import RELATIONS, Relation
from graph_rag.graph_store import InMemoryGraphStore
from graph_rag.rdf_store import ONTOLOGY, RFP, RdfGraphStore, ent_iri, iri_ent


def _all_entities(rels):
    return sorted({e for r in rels for e in (r.head, r.tail)})


@pytest.mark.parametrize("hops", [0, 1, 2, 3])
def test_expand_matches_inmemory_on_synthetic_corpus(hops):
    mem, rdf = InMemoryGraphStore(RELATIONS), RdfGraphStore(RELATIONS)
    for e in _all_entities(RELATIONS):
        assert rdf.expand([e], hops) == mem.expand([e], hops), e


def test_neighbors_match_as_sets():
    mem, rdf = InMemoryGraphStore(RELATIONS), RdfGraphStore(RELATIONS)
    for e in _all_entities(RELATIONS):
        assert set(rdf.neighbors(e)) == set(mem.neighbors(e)), e


def test_iri_roundtrip_keeps_korean_and_symbols():
    eid = "agency:(사)벤처기업협회 / 50,000,000원#1"
    assert iri_ent(ent_iri(eid)) == eid


def test_negative_hops_rejected():
    with pytest.raises(ValueError):
        RdfGraphStore(RELATIONS).expand(["x"], -1)


# ---- 온톨로지: 같은 결함을 SHACL과 OWL이 다르게 다룬다 ----

MERGED = [  # 같은 사업명으로 합쳐진 두 문서 — 실문서에서 발견한 모양 그대로
    Relation("project:통합", "ISSUED_BY", "agency:A"),
    Relation("project:통합", "ISSUED_BY", "agency:B"),
    Relation("project:통합", "REQUIRES", "tech:Java"),
]

SHAPES = """
@prefix sh: <http://www.w3.org/ns/shacl#> . @prefix rfp: <https://mjholics.github.io/rfp#> .
rfp:S a sh:NodeShape ; sh:targetSubjectsOf rfp:ISSUED_BY ; sh:property [ sh:path rfp:ISSUED_BY ; sh:maxCount 1 ] .
"""


def _closure(rels):
    g = Graph()
    g += RdfGraphStore(rels).graph
    g.parse(ONTOLOGY)
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)
    return g


def test_shacl_reports_two_agencies_as_violation():
    conforms, _, _ = validate(RdfGraphStore(MERGED).graph,
                              shacl_graph=Graph().parse(data=SHAPES, format="turtle"))
    assert conforms is False


def test_owl_concludes_the_two_agencies_are_the_same():
    """열린 세계·고유 이름 가정 없음 — 기능 속성 위반이 오류가 아니라 sameAs 추론이 된다."""
    g = _closure(MERGED)
    assert (ent_iri("agency:A"), OWL.sameAs, ent_iri("agency:B")) in g


def test_property_chain_equals_python_join():
    g = _closure(MERGED)
    got = {(iri_ent(a), iri_ent(b)) for a, b in g.subject_objects(RFP.agencyRequires)}
    assert got == {("agency:A", "tech:Java"), ("agency:B", "tech:Java")}
