"""pgvector HNSW의 ef_search를 올렸을 때 recall과 지연이 어떻게 변하는가.

run_vectordb_comparison.py 에서 pgvector 의 recall@20 이 기본값(ef_search=40)에서 94.2% 로
Chroma(97.2%)·Qdrant(99.9%) 보다 낮게 나왔다. 검색 시점 파라미터 ef_search 하나만 바꿔
같은 인덱스·같은 질의로 recall/지연 곡선을 그린다(인덱스 재구축 없음).
    python run_pgvector_ef_sweep.py
"""
from __future__ import annotations
import json, sys, time
import numpy as np
import psycopg2

sys.stdout.reconfigure(encoding="utf-8")
from run_vectordb_comparison import (CORPUS, EMB_CACHE, OUT, PG_DSN, TOP_K, _vec_literal,
                                     brute_force_topk, embed_queries, load_corpus)
from knowledgeops.eval_real import build_real_queries

rc = load_corpus()
ids = [c.chunk_id for c in rc.chunks]
doc_emb = np.load(EMB_CACHE, allow_pickle=True)["emb"]
queries = build_real_queries(rc)
q_emb = embed_queries([q.question for q in queries])
conn = psycopg2.connect(PG_DSN); conn.autocommit = True; cur = conn.cursor()
id_map = {i: c for i, c in enumerate(ids)}

rows = []
for ef in (10, 20, 40, 100, 200, 400):
    cur.execute(f"SET hnsw.ef_search = {ef}")
    rec, lat = [], []
    for i in range(len(queries)):
        bf = set(brute_force_topk(doc_emb, ids, q_emb[i], TOP_K))
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            cur.execute("SELECT id FROM items ORDER BY emb <=> %s::vector LIMIT %s",
                        (_vec_literal(q_emb[i]), TOP_K))
            got = [id_map[r[0]] for r in cur.fetchall()]
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort(); lat.append(ts[2]); rec.append(len(bf & set(got)) / TOP_K)
    row = {"ef_search": ef, "recall_at_20": round(float(np.mean(rec)), 4),
           "latency_ms_median": round(float(np.median(lat)), 3),
           "returned_lt_20": None}
    rows.append(row); print(row)

res = OUT.with_name("pgvector_ef_sweep.json")
res.write_text(json.dumps({"n_queries": len(queries), "top_k": TOP_K, "rows": rows},
                          ensure_ascii=False, indent=2), encoding="utf-8")
print("저장:", res)
