"""두 번째 벡터DB(Qdrant) — ChromaDB와 같은 데이터·같은 질의로 검색 결과·지연을 실측 비교.

## 왜 하는가

RAG 프로젝트 전부(이 저장소·rfp-rag)가 벡터DB로 **ChromaDB 하나만** 쓰고 있었다
(2026-09-13 확인). Neo4j/Kuzu는 그래프DB라 별개다. 브레인크루(DeepConnect) 등
최근 공고가 "VectorDB(pgvector, Qdrant, Chroma 등)를 2개 이상 깊게 다뤄본 경험"을
명시 요구한다 — Graph DB 갭(E10, 세 백엔드 결과 일치 검증)과 같은 패턴으로 닫는다.

## 무엇을 비교하는가

같은 corpus(실 RFP 96건, 7,863청크)를 같은 BGE-M3 임베딩으로 세 방식에 태운다:

  1. **브루트포스**(numpy 내적, 정확) — ANN이 아닌 정답 기준
  2. **ChromaDB**(HNSW 근사, PersistentClient) — 기존 파이프라인이 실제 쓰는 백엔드
  3. **Qdrant**(HNSW 근사, 로컬 임베디드 모드 — 서버·Docker·클라우드 계정 불필요)
  4. **PGVector**(PostgreSQL 16 + pgvector HNSW, Docker 서버 — 2026-09-21 추가. 애자일소다 등
     "Pinecone·Milvus·Qdrant·PGVector 활용 경험" 명시 공고 대응. **앞 둘과 달리 클라이언트-서버
     구조(localhost TCP)라 지연에 네트워크 왕복이 섞인다 — 결과 해석 시 반드시 구분**)

측정: recall@20(브루트포스 대비 순위 집합 일치율) · top-1 일치율(McNemar 페어드 검정) ·
검색 지연(임베딩 제외, 인덱스 조회만) · 인덱스 구축(적재) 1회 비용.

## 실행

    docker run -d --name pgvector-bench -e POSTGRES_PASSWORD=bench -e POSTGRES_DB=vdb         -p 55432:5432 pgvector/pgvector:pg16
    python run_vectordb_comparison.py
"""
from __future__ import annotations

import json
import pickle
import shutil
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from graph_rag.stats import mcnemar, wilson_interval
from knowledgeops.eval_real import build_real_queries

ROOT = Path(__file__).parent
CORPUS = ROOT / "output" / "rfp_corpus.pkl"
CHROMA_DIR = ROOT / "output" / "_vdb_compare_chroma"
QDRANT_DIR = ROOT / "output" / "_vdb_compare_qdrant"
EMB_CACHE = ROOT / "output" / "_vdb_compare_embeddings.npz"
OUT = ROOT / "output" / "vectordb_comparison.json"

PG_DSN = "host=localhost port=55432 dbname=vdb user=postgres password=bench"
TOP_K = 20  # run_rerank_upper_bound.py 와 동일한 상한 관측용 k
N_LATENCY_REPEATS = 5  # 지연은 콜드/캐시 흔들림을 줄이려 5회 반복 중앙값


def load_corpus():
    rc = pickle.load(open(CORPUS, "rb"))
    return rc


def embed_corpus(chunks) -> np.ndarray:
    """청크 텍스트를 BGE-M3로 인코딩. 캐시가 있으면 재사용(7,863건 재인코딩은 수 분 소요)."""
    if EMB_CACHE.exists():
        data = np.load(EMB_CACHE, allow_pickle=True)
        cached_ids = list(data["ids"])
        if cached_ids == [c.chunk_id for c in chunks]:
            print(f"[embed] 캐시 재사용: {EMB_CACHE}")
            return data["emb"]
        print("[embed] 캐시가 현재 코퍼스와 안 맞음 — 재인코딩")

    from sentence_transformers import SentenceTransformer

    print(f"[embed] BGE-M3로 {len(chunks)}건 인코딩 중 (최초 1회, 캐시 저장)...")
    t0 = time.time()
    model = SentenceTransformer("BAAI/bge-m3")
    texts = [c.text for c in chunks]
    emb = model.encode(
        texts, normalize_embeddings=True, batch_size=64, show_progress_bar=True
    )
    emb = np.asarray(emb, dtype=np.float32)
    print(f"[embed] {time.time()-t0:.1f}초")
    np.savez_compressed(EMB_CACHE, emb=emb, ids=np.array([c.chunk_id for c in chunks]))
    return emb


def embed_queries(questions: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-m3")
    return np.asarray(
        model.encode(questions, normalize_embeddings=True, batch_size=32), dtype=np.float32
    )


def brute_force_topk(doc_emb: np.ndarray, ids: list[str], q_emb: np.ndarray, k: int) -> list[str]:
    sims = doc_emb @ q_emb
    idx = np.argsort(-sims)[:k]
    return [ids[i] for i in idx]


def build_chroma(ids, texts, emb) -> "object":
    import chromadb

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    col = client.create_collection("vdb_compare", metadata={"hnsw:space": "cosine"})
    B = 500
    for i in range(0, len(ids), B):
        col.add(
            ids=ids[i : i + B],
            embeddings=emb[i : i + B].tolist(),
            documents=texts[i : i + B],
        )
    return col


def build_qdrant(ids, texts, emb) -> "object":
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct

    if QDRANT_DIR.exists():
        shutil.rmtree(QDRANT_DIR)
    client = QdrantClient(path=str(QDRANT_DIR))
    dim = emb.shape[1]
    client.create_collection(
        collection_name="vdb_compare",
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    B = 500
    for i in range(0, len(ids), B):
        points = [
            PointStruct(id=i + j, vector=emb[i + j].tolist(), payload={"chunk_id": ids[i + j]})
            for j in range(min(B, len(ids) - i))
        ]
        client.upsert(collection_name="vdb_compare", points=points)
    return client, {i: cid for i, cid in enumerate(ids)}


def _vec_literal(v: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in v.tolist()) + "]"


def build_pgvector(ids, emb):
    """PostgreSQL+pgvector. 적재와 HNSW 인덱스 생성 시간을 따로 잰다(m=16, ef_construction=64 기본값)."""
    import psycopg2
    from psycopg2.extras import execute_values

    conn = psycopg2.connect(PG_DSN)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("DROP TABLE IF EXISTS items")
    dim = emb.shape[1]
    cur.execute(f"CREATE TABLE items (id int PRIMARY KEY, emb vector({dim}))")
    t0 = time.time()
    B = 500
    for i in range(0, len(ids), B):
        rows = [(i + j, _vec_literal(emb[i + j])) for j in range(min(B, len(ids) - i))]
        execute_values(cur, "INSERT INTO items (id, emb) VALUES %s", rows, template="(%s, %s::vector)")
    load_s = time.time() - t0
    t0 = time.time()
    cur.execute("CREATE INDEX ON items USING hnsw (emb vector_cosine_ops)")
    cur.execute("ANALYZE items")
    index_s = time.time() - t0
    return conn, {i: cid for i, cid in enumerate(ids)}, load_s, index_s


def query_pgvector(conn, id_map, q_emb: np.ndarray, k: int) -> list[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM items ORDER BY emb <=> %s::vector LIMIT %s", (_vec_literal(q_emb), k)
    )
    return [id_map[r[0]] for r in cur.fetchall()]


def query_chroma(col, q_emb: np.ndarray, k: int) -> list[str]:
    r = col.query(query_embeddings=[q_emb.tolist()], n_results=k, include=[])
    return r["ids"][0]


def query_qdrant(client, id_map, q_emb: np.ndarray, k: int) -> list[str]:
    r = client.query_points(
        collection_name="vdb_compare", query=q_emb.tolist(), limit=k
    ).points
    return [id_map[p.id] for p in r]


def timed_repeat(fn, n: int) -> tuple[list, float]:
    """fn()을 n회 반복, (마지막 결과, 지연 중앙값 ms)를 돌려준다."""
    times = []
    result = None
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return result, times[len(times) // 2]


def main() -> None:
    print("[1/5] 코퍼스·질의 로드")
    rc = load_corpus()
    chunks = rc.chunks
    ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    queries = build_real_queries(rc)
    print(f"  청크 {len(chunks)}건 · 질의 {len(queries)}건")

    print("[2/5] 코퍼스 임베딩")
    doc_emb = embed_corpus(chunks)

    print("[3/5] 백엔드 구축 (ChromaDB · Qdrant · PGVector)")
    t0 = time.time()
    chroma_col = build_chroma(ids, texts, doc_emb)
    chroma_build_s = time.time() - t0

    t0 = time.time()
    qdrant_client, qdrant_id_map = build_qdrant(ids, texts, doc_emb)
    qdrant_build_s = time.time() - t0
    pg_conn, pg_id_map, pg_load_s, pg_index_s = build_pgvector(ids, doc_emb)
    print(
        f"  ChromaDB 적재 {chroma_build_s:.1f}s · Qdrant 적재 {qdrant_build_s:.1f}s · "
        f"PGVector 적재 {pg_load_s:.1f}s + HNSW 인덱스 {pg_index_s:.1f}s"
    )

    print("[4/5] 질의 임베딩 + 4방식 검색")
    q_texts = [q.question for q in queries]
    q_emb = embed_queries(q_texts)

    recalls_chroma, recalls_qdrant, recalls_pg = [], [], []
    top1_chroma_correct, top1_qdrant_correct, top1_pg_correct = [], [], []
    lat_chroma, lat_qdrant, lat_pg, lat_brute = [], [], [], []

    for i, q in enumerate(queries):
        qe = q_emb[i]

        bf_result, bf_ms = timed_repeat(
            lambda: brute_force_topk(doc_emb, ids, qe, TOP_K), N_LATENCY_REPEATS
        )
        chroma_result, chroma_ms = timed_repeat(
            lambda: query_chroma(chroma_col, qe, TOP_K), N_LATENCY_REPEATS
        )
        qdrant_result, qdrant_ms = timed_repeat(
            lambda: query_qdrant(qdrant_client, qdrant_id_map, qe, TOP_K), N_LATENCY_REPEATS
        )

        pg_result, pg_ms = timed_repeat(
            lambda: query_pgvector(pg_conn, pg_id_map, qe, TOP_K), N_LATENCY_REPEATS
        )

        bf_set = set(bf_result)
        recalls_pg.append(len(bf_set & set(pg_result)) / TOP_K)
        top1_pg_correct.append(pg_result[0] == bf_result[0] if pg_result else False)
        lat_pg.append(pg_ms)
        recalls_chroma.append(len(bf_set & set(chroma_result)) / TOP_K)
        recalls_qdrant.append(len(bf_set & set(qdrant_result)) / TOP_K)
        top1_chroma_correct.append(chroma_result[0] == bf_result[0] if chroma_result else False)
        top1_qdrant_correct.append(qdrant_result[0] == bf_result[0] if qdrant_result else False)
        lat_brute.append(bf_ms)
        lat_chroma.append(chroma_ms)
        lat_qdrant.append(qdrant_ms)

        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(queries)} 질의 처리")

    print("[5/5] 집계")

    def summarize(xs):
        xs = sorted(xs)
        n = len(xs)
        return {
            "mean": sum(xs) / n,
            "median": xs[n // 2],
            "p95": xs[int(n * 0.95)] if n > 1 else xs[0],
        }

    mc = mcnemar(top1_chroma_correct, top1_qdrant_correct)
    mc_chroma_pg = mcnemar(top1_chroma_correct, top1_pg_correct)
    mc_qdrant_pg = mcnemar(top1_qdrant_correct, top1_pg_correct)
    n = len(queries)
    chroma_top1_rate = sum(top1_chroma_correct) / n
    qdrant_top1_rate = sum(top1_qdrant_correct) / n

    report = {
        "corpus_chunks": len(chunks),
        "n_queries": n,
        "top_k": TOP_K,
        "pgvector_note": "client-server localhost TCP; chroma/qdrant embedded in-process (latency not like-for-like)",
        "build_time_s": {"chroma": chroma_build_s, "qdrant": qdrant_build_s,
                         "pgvector_load": pg_load_s, "pgvector_hnsw_index": pg_index_s},
        "recall_at_k_vs_bruteforce": {
            "chroma_mean": sum(recalls_chroma) / n,
            "qdrant_mean": sum(recalls_qdrant) / n,
            "pgvector_mean": sum(recalls_pg) / n,
        },
        "top1_match_vs_bruteforce": {
            "chroma_rate": chroma_top1_rate,
            "chroma_wilson95": wilson_interval(sum(top1_chroma_correct), n),
            "qdrant_rate": qdrant_top1_rate,
            "qdrant_wilson95": wilson_interval(sum(top1_qdrant_correct), n),
            "mcnemar": mc,
            "pgvector_rate": sum(top1_pg_correct) / n,
            "pgvector_wilson95": wilson_interval(sum(top1_pg_correct), n),
            "mcnemar_chroma_vs_pgvector": mc_chroma_pg,
            "mcnemar_qdrant_vs_pgvector": mc_qdrant_pg,
        },
        "latency_ms_search_only": {
            "bruteforce_numpy": summarize(lat_brute),
            "chroma": summarize(lat_chroma),
            "qdrant": summarize(lat_qdrant),
            "pgvector_client_server": summarize(lat_pg),
        },
    }

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
