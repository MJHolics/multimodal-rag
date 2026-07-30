"""실문서(HWP) → 지식그래프 인제스천 파이프라인.

## 왜 스크립트로 남기나

처음엔 이 과정을 대화형 명령으로 돌렸다. 그러면 **나만 재현할 수 있다.**
저장소를 클론한 사람(그리고 몇 달 뒤의 나)이 같은 결과를 얻으려면 파이프라인이
파일로 있어야 한다. 특히 이 프로젝트는 "수치는 재실행하면 같아야 한다"를 원칙으로 삼아 왔다.

## 입력이 저장소에 없는 이유

대상 문서는 공공조달 RFP 96건(약 164MB)이고 이 저장소에 넣지 않았다. 대신 경로를 인자로 받는다.
**어떤 HWP 묶음이든** `기관명_사업명.hwp` 규약만 지키면 그대로 돌아간다.
문서가 없으면 `knowledgeops_app.py --demo`가 합성 코퍼스로 동작한다.

사용:
    python build_rfp_graph.py --src "경로/to/hwp_dir"
    python build_rfp_graph.py --src ... --audit HAS_BUDGET   # 추출 품질 표본 확인
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

from knowledgeops.chunking import build_real_corpus, corpus_summary
from knowledgeops.eval_real import build_real_queries, summarize_real
from knowledgeops.extract import audit_sample, build_document, graph_summary
from knowledgeops.hwp import extract_text

ROOT = Path(__file__).parent
OUT = ROOT / "output"

MIN_TEXT = 500  # 이보다 짧으면 파싱 실패로 본다


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HWP 파일이 있는 디렉터리")
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--audit", default="", help="추출 품질 표본을 볼 관계 종류")
    ap.add_argument("--audit-n", type=int, default=20)
    args = ap.parse_args()

    src = Path(args.src)
    files = sorted(src.glob("*.hwp"))
    if not files:
        raise SystemExit(f"HWP 파일이 없다: {src}")

    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    docs, failed = [], []
    for i, f in enumerate(files):
        try:
            text = extract_text(f)
        except Exception as e:  # 한 건 실패가 전체를 멈추지 않게
            failed.append((f.name, str(e)[:60]))
            continue
        if len(text) < MIN_TEXT:
            failed.append((f.name, f"텍스트 부족({len(text)}자)"))
            continue
        docs.append(build_document(f"doc{i:03d}", f.name, text))

    print(f"[1/3] 파싱 {len(docs)}/{len(files)} 성공 · 실패 {len(failed)} "
          f"({time.time() - t0:.1f}s)")
    for name, why in failed[:5]:
        print(f"      실패: {name[:50]} — {why}")

    print(f"[2/3] IE {json.dumps(graph_summary(docs), ensure_ascii=False)}")

    rc = build_real_corpus(docs, size=args.chunk_size)
    print(f"[3/3] 코퍼스 {json.dumps(corpus_summary(rc), ensure_ascii=False)}")

    queries = build_real_queries(rc)
    print(f"      평가셋 {json.dumps(summarize_real(queries), ensure_ascii=False)}")

    pickle.dump(docs, open(OUT / "rfp_docs.pkl", "wb"))
    pickle.dump(rc, open(OUT / "rfp_corpus.pkl", "wb"))
    print(f"저장: {OUT / 'rfp_docs.pkl'} · {OUT / 'rfp_corpus.pkl'}")

    if args.audit:
        print(f"\n=== IE 감사 표본 [{args.audit}] ===")
        print("추출을 그대로 정답으로 쓰면 순환이다. 근거는 판정에 실제로 쓰인 문장이다.")
        for r in audit_sample(docs, args.audit, args.audit_n):
            print(f"  {r['triple'][:70]}")
            print(f"     {r['evidence'][:110]}")

    print("\n다음: python run_real_eval.py  (검색 전략 비교·통계 검정)")
    print("      python knowledgeops_app.py (4탭 콘솔)")


if __name__ == "__main__":
    main()
