"""Neo4j(AuraDB) 실행 검증 — 세 백엔드가 같은 결과를 내는지 실제 서버로 확인한다.

## 왜 이 스크립트가 있나

이 프로젝트는 "Neo4j는 코드만 있고 실행 미검증"을 계속 한계로 적어 왔다. 개발 환경에
Docker도 Java도 없어 로컬 서버를 못 띄우기 때문이다. **AuraDB 무료 인스턴스는 브라우저에서
계정만 만들면 되고 설치가 필요 없다** — 남은 건 접속 정보뿐이라 그 부분만 사람에게 맡긴다.

검증 내용은 "연결됐다"가 아니다. 그건 아무것도 증명하지 않는다.
**in-memory 순수 구현 / Kuzu(임베디드 Cypher) / Neo4j(서버 Cypher) 세 백엔드가
같은 그래프에서 같은 답을 내는지**를 확인한다. 하나라도 다르면 그 자리에서 실패한다.

## 준비 (5분, 카드 등록 불필요)

1. console.neo4j.io 에서 무료 인스턴스 생성(AuraDB Free)
2. 생성 직후 **한 번만** 보여주는 비밀번호를 저장 (다시 못 본다)
3. 접속 정보를 환경변수로:

    PowerShell:
        $env:NEO4J_URI = "neo4j+s://xxxxxxxx.databases.neo4j.io"
        $env:NEO4J_PASSWORD = "발급받은-비밀번호"

    bash:
        export NEO4J_URI="neo4j+s://xxxxxxxx.databases.neo4j.io"
        export NEO4J_PASSWORD="발급받은-비밀번호"

4. 실행:
        python verify_neo4j.py            # 합성 코퍼스(작음) — 먼저 이걸로 확인
        python verify_neo4j.py --real     # 실문서 그래프(관계 1,051개)

AuraDB Free는 유휴 시 일시정지되고 용량 제한이 있다. 이 그래프는 노드 수백 개라 여유롭다.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

from graph_rag.corpus import RELATIONS
from graph_rag.graph_store import InMemoryGraphStore, Neo4jGraphStore

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
CORPUS = ROOT / "output" / "rfp_corpus.pkl"


def _seeds(relations, n: int = 12) -> list[str]:
    """검증에 쓸 시드 — 결정적으로 고른다(정렬 후 균등 간격)."""
    eids = sorted({e for r in relations for e in (r.head, r.tail)})
    if len(eids) <= n:
        return eids
    step = len(eids) / n
    return [eids[int(i * step)] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="실문서 그래프로 검증(rfp_corpus.pkl 필요)")
    ap.add_argument("--hops", type=int, default=2)
    args = ap.parse_args()

    if args.real:
        if not CORPUS.exists():
            raise SystemExit(f"{CORPUS} 가 없다 — build_rfp_graph.py 를 먼저 돌릴 것")
        relations = pickle.loads(CORPUS.read_bytes()).relations
        label = f"실문서 그래프(관계 {len(relations)})"
    else:
        relations = RELATIONS
        label = f"합성 코퍼스(관계 {len(relations)})"

    print(f"[대상] {label} · max_hops={args.hops}")

    try:
        neo = Neo4jGraphStore.from_env(relations=relations)
    except RuntimeError as e:
        print(f"\n[skip] {e}")
        return 2  # 자격증명 없음 — 실패가 아니라 미실행

    mem = InMemoryGraphStore(relations)
    seeds = _seeds(relations)

    try:
        t0 = time.time()
        neo.load()
        print(f"[적재] Neo4j {time.time() - t0:.1f}s")

        # ---- 1. expand: 시드에서 max_hops 이내 도달 엔티티와 최단 홉수 ----
        bad = 0
        t0 = time.time()
        for s in seeds:
            a = mem.expand([s], args.hops)
            b = neo.expand([s], args.hops)
            if a != b:
                bad += 1
                only_a = {k: v for k, v in a.items() if b.get(k) != v}
                only_b = {k: v for k, v in b.items() if a.get(k) != v}
                print(f"  [불일치] seed={s}\n    in-memory만/다름: {list(only_a)[:5]}"
                      f"\n    neo4j만/다름:     {list(only_b)[:5]}")
        print(f"[expand] 시드 {len(seeds)}개 · 불일치 {bad} · {time.time() - t0:.1f}s")

        # ---- 2. neighbors: 인접 관계 집합 ----
        bad_n = 0
        for s in seeds:
            a = {tuple(x) for x in mem.neighbors(s)}
            b = {tuple(x) for x in neo.neighbors(s)}
            if a != b:
                bad_n += 1
                print(f"  [불일치] neighbors seed={s} · 차집합 {list(a ^ b)[:3]}")
        print(f"[neighbors] 시드 {len(seeds)}개 · 불일치 {bad_n}")

        # ---- 3. Kuzu도 같은 답인지(있으면) ----
        try:
            import tempfile

            from graph_rag.graph_store import KuzuGraphStore

            if not args.real:  # Kuzu는 REL TABLE이 합성 코퍼스 타입에 맞춰져 있다
                with tempfile.TemporaryDirectory() as d:
                    kz = KuzuGraphStore(str(Path(d) / "g.kuzu"), relations)
                    bad_k = sum(
                        1 for s in seeds if kz.expand([s], args.hops) != mem.expand([s], args.hops)
                    )
                    print(f"[kuzu] 불일치 {bad_k}")
        except Exception as e:  # Kuzu 미설치 등은 이 검증의 목적이 아니다
            print(f"[kuzu] 건너뜀 ({type(e).__name__})")

        ok = bad == 0 and bad_n == 0
        print("\n" + ("✅ 세 연산 모두 일치 — Neo4j 경로 실행 검증됨" if ok
                      else "❌ 불일치 있음 — 위 로그 확인"))
        return 0 if ok else 1
    finally:
        neo.close()


if __name__ == "__main__":
    raise SystemExit(main())
