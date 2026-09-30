"""실문서 RFP 그래프를 RDF로 옮겨 ① SPARQL이 기존 백엔드와 같은 답을 내는지 ② OWL 추론과
SHACL 검증이 IE 결함을 어떻게 다루는지 잰다.

사용: python run_rdf_ontology.py            → output/rdf_ontology_results.json
"""
from __future__ import annotations

import json
import pickle
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import owlrl
from pyshacl import validate
from rdflib import OWL, RDF, Graph

from graph_rag.graph_store import InMemoryGraphStore
from graph_rag.rdf_store import ONTOLOGY, RFP, RdfGraphStore, ent_iri, iri_ent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
CORPUS = ROOT / "output" / "rfp_corpus.pkl"
OUT = ROOT / "output" / "rdf_ontology_results.json"
PREFIX_CLASS = {"project": "Project", "agency": "Agency", "budget": "Budget",
                "period": "Period", "tech": "Technology"}

# 닫힌 세계 검증 — "사업에는 발주기관·예산·기간이 최대 하나"
SHAPES = """
@prefix sh:  <http://www.w3.org/ns/shacl#> .
@prefix rfp: <https://mjholics.github.io/rfp#> .
rfp:ProjectShape a sh:NodeShape ;
  sh:targetSubjectsOf rfp:ISSUED_BY, rfp:HAS_BUDGET, rfp:HAS_PERIOD, rfp:REQUIRES ;
  sh:property [ sh:path rfp:ISSUED_BY ;  sh:maxCount 1 ; sh:minCount 1 ] ;
  sh:property [ sh:path rfp:HAS_BUDGET ; sh:maxCount 1 ] ;
  sh:property [ sh:path rfp:HAS_PERIOD ; sh:maxCount 1 ] .
"""


def ms(f, reps=1):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = f()
        ts.append((time.perf_counter() - t0) * 1000)
    return r, statistics.median(ts)


def main() -> None:
    rels = pickle.loads(CORPUS.read_bytes()).relations
    ents = sorted({e for r in rels for e in (r.head, r.tail)})
    res: dict = {"relations": len(rels), "entities": len(ents)}

    # ---------------------------------------------------------------- 1. 같은 답인가
    mem = InMemoryGraphStore(rels)
    rdf, t_load = ms(lambda: RdfGraphStore(rels))
    res["triples"] = len(rdf.graph)
    res["duplicate_relations"] = len(rels) - len({(r.head, r.rtype, r.tail) for r in rels})
    res["rdf_load_ms"] = round(t_load, 1)

    eq = {}
    for hops in (1, 2, 3):
        bad, t_mem, t_rdf = 0, [], []
        for e in ents:
            a, tm = ms(lambda: mem.expand([e], hops))
            b, tr = ms(lambda: rdf.expand([e], hops))
            t_mem.append(tm)
            t_rdf.append(tr)
            bad += a != b
        eq[f"expand_h{hops}"] = {"seeds": len(ents), "mismatch": bad,
                                 "inmemory_ms_median": round(statistics.median(t_mem), 3),
                                 "sparql_ms_median": round(statistics.median(t_rdf), 2)}
        print(f"[expand h{hops}] 시드 {len(ents)} 불일치 {bad} · in-memory {eq[f'expand_h{hops}']['inmemory_ms_median']}ms"
              f" · SPARQL {eq[f'expand_h{hops}']['sparql_ms_median']}ms")
    bad_n = sum(set(mem.neighbors(e)) != set(rdf.neighbors(e)) for e in ents)
    eq["neighbors"] = {"seeds": len(ents), "mismatch_as_set": bad_n}
    print(f"[neighbors] 불일치(집합) {bad_n}")
    res["equivalence"] = eq

    # ---------------------------------------------------------------- 2. SHACL(닫힌 세계)
    shapes = Graph().parse(data=SHAPES, format="turtle")
    (conforms, report, _), t_shacl = ms(lambda: validate(rdf.graph, shacl_graph=shapes))
    focus = sorted({iri_ent(o) for o in report.objects(None, _sh("focusNode"))})
    paths = sorted({str(o).split("#")[-1] for o in report.objects(None, _sh("resultPath"))})
    res["shacl"] = {"conforms": conforms, "violating_nodes": focus, "paths": paths,
                    "results": len(list(report.subjects(RDF.type, _sh("ValidationResult")))),
                    "ms": round(t_shacl, 1)}
    print(f"[SHACL] conforms={conforms} · 위반 노드 {focus} · 경로 {paths}")

    # ---------------------------------------------------------------- 3. OWL 2 RL 추론(열린 세계)
    g = Graph()
    g += rdf.graph
    g.parse(ONTOLOGY)
    before = len(g)
    _, t_owl = ms(lambda: owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g))
    res["owl"] = {"triples_before": before, "triples_after": len(g), "ms": round(t_owl, 1),
                  "inconsistency_messages": [str(o) for o in g.objects(None, _err())]}

    # 3a. domain/range로 추론한 타입이 IE 접두사와 맞는가
    wrong, untyped = [], 0
    for e in ents:
        types = {str(t).split("#")[-1] for t in g.objects(ent_iri(e), RDF.type)} & set(PREFIX_CLASS.values())
        want = PREFIX_CLASS[e.split(":")[0]]
        if not types:
            untyped += 1
        elif types != {want}:
            wrong.append((e, sorted(types)))
    res["owl"]["type_inference"] = {"untyped": untyped, "mismatch": wrong}

    # 3b. 기능 속성이 만든 owl:sameAs — 서로 다른 엔티티를 "같다"고 결론 낸 쌍
    same = sorted({tuple(sorted((iri_ent(a), iri_ent(b)))) for a, b in g.subject_objects(OWL.sameAs)
                   if a != b and str(a).startswith(str(ent_iri("")))})
    res["owl"]["sameAs_between_distinct"] = same
    print(f"[OWL-RL] 트리플 {before}→{len(g)} ({t_owl:.0f}ms) · 타입 불일치 {len(wrong)} · 미추론 {untyped}"
          f" · sameAs {same}")

    # 3c. 속성 체인이 Python 조인과 같은가, sameAs가 얼마나 번지는가
    def py_chain(rt):
        by_proj = defaultdict(set)
        for r in rels:
            if r.rtype == rt:
                by_proj[r.head].add(r.tail)
        return {(r.tail, x) for r in rels if r.rtype == "ISSUED_BY" for x in by_proj[r.head]}

    for prop, rt in (("agencyRequires", "REQUIRES"), ("agencyBudget", "HAS_BUDGET")):
        owl_pairs = {(iri_ent(a), iri_ent(b)) for a, b in g.subject_objects(RFP[prop])
                     if str(a).startswith(str(ent_iri(""))) and str(b).startswith(str(ent_iri("")))}
        py = py_chain(rt)
        res["owl"][prop] = {"owl": len(owl_pairs), "python_join": len(py),
                            "owl_only": sorted(owl_pairs - py)[:10], "owl_only_n": len(owl_pairs - py),
                            "python_only_n": len(py - owl_pairs)}
        print(f"[chain {prop}] OWL {len(owl_pairs)} · 조인 {len(py)} · OWL에만 {len(owl_pairs - py)} · 조인에만 {len(py - owl_pairs)}")

    # ---------------------------------------------------------------- 4. 고유 이름 가정을 넣으면
    g2 = Graph()
    g2 += rdf.graph
    g2.parse(ONTOLOGY)
    agencies = [ent_iri(e) for e in ents if e.startswith("agency:")]
    from rdflib import BNode
    from rdflib.collection import Collection
    node, lst = BNode(), BNode()
    Collection(g2, lst, agencies)
    g2.add((node, RDF.type, OWL.AllDifferent))
    g2.add((node, OWL.members, lst))
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g2)
    errs = [str(o) for o in g2.objects(None, _err())]
    res["owl_with_AllDifferent"] = {"inconsistency_messages": errs[:5], "n": len(errs)}
    print(f"[OWL-RL + AllDifferent] 불일치 보고 {len(errs)}건")

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"→ {OUT}")


def _sh(name):
    from rdflib import Namespace
    return Namespace("http://www.w3.org/ns/shacl#")[name]


def _err():
    # owlrl은 불일치를 발견하면 그래프에 '오류' 노드를 남긴다(owlrl.Namespaces.ERRNS)
    from owlrl.Namespaces import ERRNS
    return ERRNS["error"]


if __name__ == "__main__":
    main()
