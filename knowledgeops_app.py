"""KnowledgeOps — 문서를 넣으면 지식그래프가 서고, 검색이 근거와 함께 답하는 운영 콘솔.

## 왜 이 화면이 필요한가

이 저장소에는 흩어진 조각이 많다 — 하이브리드 검색, 지식그래프 검색, 반복 검색,
RAGAS 지표, McNemar 검정, 실문서 IE. 각각은 수치로 증명돼 있지만 **하나의 흐름으로
보이지 않는다.** 리크루터도 실무자도 "그래서 이게 무슨 제품인가"를 30초 안에 알 수 없다.

그래서 InspectOps와 같은 방식을 쓴다: **이미 테스트된 모듈 위에 화면만 얹는다.**
새 로직을 만들지 않으므로 화면이 거짓말을 할 수 없다.

탭 구성 = 실제 운영 순서다.
  1. 문서 → 그래프   : 실제 HWP를 파싱하고 관계를 뽑아 근거와 함께 보여준다
  2. 검색 비교       : 같은 질문에 네 전략이 무엇을 가져오는지 나란히
  3. 평가 리포트     : 어느 전략이 나은지 통계 검정으로 판정
  4. IE 감사         : 추출이 맞는지 사람이 확인할 표본(순환을 피하는 장치)

실행:
    python knowledgeops_app.py            # 캐시된 실문서 코퍼스 사용
    python knowledgeops_app.py --demo     # 합성 코퍼스(HWP 없이 동작 확인)
"""
from __future__ import annotations

import argparse
import html
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).parent
CORPUS_PKL = ROOT / "output" / "rfp_corpus.pkl"
DOCS_PKL = ROOT / "output" / "rfp_docs.pkl"
REAL_EVAL = ROOT / "output" / "real_eval.json"
SYNTH_EVAL = ROOT / "output" / "graph_eval_auto.json"


# ---------------------------------------------------------------------------
# 상태 — 무거운 것(임베딩)은 요청 시에만 만든다
# ---------------------------------------------------------------------------


class Console:
    def __init__(self, use_real: bool = True) -> None:
        self.use_real = use_real and CORPUS_PKL.exists()
        self.docs = []
        if self.use_real:
            self.rc = pickle.load(open(CORPUS_PKL, "rb"))
            if DOCS_PKL.exists():
                self.docs = pickle.load(open(DOCS_PKL, "rb"))
            self.chunks = self.rc.chunks
            self.relations = self.rc.relations
            self.entities = self.rc.entities
        else:
            from graph_rag.corpus import ENTITY_BY_ID, RELATIONS, build_corpus

            self.rc = None
            self.chunks = build_corpus()
            self.relations = RELATIONS
            self.entities = ENTITY_BY_ID
        self._hybrid = None
        self._graph = None
        self._fused = None
        self._agentic = None

    # -- 리트리버는 처음 쓸 때 만든다(BM25만: 즉시, dense: 필요 시) --
    def retrievers(self):
        if self._hybrid is None:
            from graph_rag.graph_store import InMemoryGraphStore
            from graph_rag.retrieval import (
                FusedRetriever,
                GraphRetriever,
                HybridRetriever,
            )

            if self.use_real:
                import graph_rag.retrieval as R

                from run_real_eval import _link_entities_factory

                R.link_entities = _link_entities_factory(self.rc)

            self._hybrid = HybridRetriever(self.chunks, embedder=None)
            store = InMemoryGraphStore(self.relations)
            self._graph = GraphRetriever(self.chunks, store=store, mode="relation")
            w = 0.5
            self._fused = FusedRetriever(self._hybrid, self._graph, graph_weight=w)
            from graph_rag.agentic import IterativeRetriever

            self._agentic = IterativeRetriever(
                self.chunks, graph=self._graph, hybrid=self._hybrid,
                rounds=2, graph_weight=w,
            )
        return self._hybrid, self._graph, self._fused, self._agentic

    def chunk(self, cid: str):
        for c in self.chunks:
            if c.chunk_id == cid:
                return c
        return None


# ---------------------------------------------------------------------------
# 탭 1 — 문서에서 뽑은 그래프
# ---------------------------------------------------------------------------


def tab_graph(console: Console, doc_query: str) -> tuple[str, str]:
    if not console.docs:
        return "실문서 코퍼스가 없다. `--demo`로 합성 코퍼스를 보거나 인제스천을 먼저 돌릴 것.", ""

    q = (doc_query or "").strip()
    hits = [d for d in console.docs if not q or q in d.project or q in d.agency]
    if not hits:
        return f"'{html.escape(q)}'와 맞는 문서가 없다.", ""
    doc = hits[0]

    rows = []
    for t in doc.triples:
        ev = doc.evidence.get(f"{t.head}|{t.rtype}|{t.tail}", "")
        rows.append(
            f"<tr><td><code>{html.escape(t.rtype)}</code></td>"
            f"<td>{html.escape(t.tail)}</td>"
            f"<td class='ev'>{html.escape(ev[:110])}</td></tr>"
        )
    table = (
        "<table class='ko'><thead><tr><th>관계</th><th>값</th>"
        "<th>근거(추출 판정에 쓰인 문장)</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    head = (
        f"<b>{html.escape(doc.project)}</b><br>"
        f"발주기관 {html.escape(doc.agency)} · 본문 {len(doc.text):,}자 · 관계 {len(doc.triples)}개"
        f"<br><span class='muted'>전체 {len(console.docs)}건 중 '{html.escape(q) or '첫 문서'}' 매칭 "
        f"{len(hits)}건</span>"
    )
    return head, table


# ---------------------------------------------------------------------------
# 탭 2 — 같은 질문, 네 전략
# ---------------------------------------------------------------------------


def tab_search(console: Console, question: str, k: int) -> str:
    q = (question or "").strip()
    if not q:
        return "<p class='muted'>질문을 입력하면 네 전략이 무엇을 가져오는지 나란히 보여준다.</p>"

    hybrid, graph, fused, agentic = console.retrievers()
    plans = [
        ("hybrid_flat", "현행 어휘·의미 하이브리드", lambda: hybrid.retrieve(q, k)),
        ("graph_rel", "지식그래프(관계 인식)", lambda: graph.retrieve(q, k)),
        ("fused_rel", "융합", lambda: fused.retrieve(q, k)),
        ("agentic", "반복 검색(2라운드)", lambda: agentic.retrieve(q, k)),
    ]
    blocks = []
    for name, label, fn in plans:
        try:
            got = fn()
        except Exception as e:  # 화면이 죽지 않게
            blocks.append(f"<div class='card'><h4>{name}</h4><p>오류: {html.escape(str(e))}</p></div>")
            continue
        items = []
        for cid in got:
            c = console.chunk(cid)
            snippet = html.escape((c.text[:150] + "…") if c else "")
            tag = "근거 있음" if (c and c.states) else "방해 문서"
            cls = "ok" if (c and c.states) else "warn"
            items.append(
                f"<li><span class='pill {cls}'>{tag}</span> "
                f"<code>{html.escape(cid)}</code><br><span class='snip'>{snippet}</span></li>"
            )
        blocks.append(
            f"<div class='card'><h4>{name} <span class='muted'>{label}</span></h4>"
            f"<ol>{''.join(items) or '<li>결과 없음</li>'}</ol></div>"
        )
    return f"<div class='grid'>{''.join(blocks)}</div>"


# ---------------------------------------------------------------------------
# 탭 3 — 평가 리포트(저장된 측정 결과를 읽어 보여준다)
# ---------------------------------------------------------------------------


def tab_report(which: str) -> str:
    path = REAL_EVAL if which.startswith("실문서") else SYNTH_EVAL
    if not path.exists():
        return f"<p>측정 결과가 없다: <code>{path.name}</code></p>"
    rep = json.loads(path.read_text(encoding="utf-8"))

    res = rep.get("results", {})
    rows = []
    for name, splits in res.items():
        cells = "".join(
            f"<td>{splits.get(s, {}).get('full_hit', 0):.3f}</td>"
            for s in ("all", "single_hop", "multi_hop")
        )
        rows.append(f"<tr><td><code>{html.escape(name)}</code></td>{cells}</tr>")
    table = (
        "<table class='ko'><thead><tr><th>전략</th><th>전체</th><th>단일홉</th>"
        "<th>멀티홉</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )

    sig = rep.get("significance", {})
    if isinstance(sig, dict) and "tests" in sig:
        sig = sig["tests"]
    srows = []
    for key, r in sig.items():
        verdict = r.get("verdict", "")
        cls = "ok" if verdict == "유의" else ("warn" if "판정불가" in verdict else "")
        srows.append(
            f"<tr><td>{html.escape(key)}</td><td>{r.get('delta', 0):+.3f}</td>"
            f"<td>{r.get('a_only', 0)}:{r.get('b_only', 0)}</td>"
            f"<td>{r.get('p_value', 1):.4f}</td>"
            f"<td><span class='pill {cls}'>{html.escape(verdict)}</span></td></tr>"
        )
    stable = (
        "<table class='ko'><thead><tr><th>비교</th><th>Δ</th><th>불일치</th>"
        "<th>p</th><th>판정</th></tr></thead><tbody>" + "".join(srows) + "</tbody></table>"
    )

    cfg = rep.get("config", {})
    meta = (
        f"<p class='muted'>top-{cfg.get('k')} · dense={html.escape(str(cfg.get('dense')))}"
        f" · graph_weight={cfg.get('graph_weight')}</p>"
    )
    note = (
        "<p class='muted'>지표는 <b>full_hit</b> — 답에 필요한 근거가 <b>전부</b> 상위 k에 "
        "들어야 1점이다. 멀티홉은 근거가 다 모여야 답이 되기 때문이다. "
        "'유의하지 않음'과 '판정불가'는 다르다: 후자는 불일치 쌍이 6개 미만이라 "
        "완전히 쏠려도 유의가 나올 수 없는 상태다.</p>"
    )
    return meta + table + "<h4>짝지음 검정(McNemar 정확검정)</h4>" + stable + note


# ---------------------------------------------------------------------------
# 탭 4 — IE 감사
# ---------------------------------------------------------------------------


def tab_audit(console: Console, rtype: str, n: int) -> str:
    if not console.docs:
        return "<p>실문서 코퍼스가 없다.</p>"
    from knowledgeops.extract import audit_sample

    rows = audit_sample(console.docs, rtype, int(n))
    if not rows:
        return f"<p>{html.escape(rtype)} 관계가 없다.</p>"
    body = "".join(
        f"<tr><td><code>{html.escape(r['doc'])}</code></td>"
        f"<td>{html.escape(r['triple'][:70])}</td>"
        f"<td class='ev'>{html.escape(r['evidence'][:130])}</td></tr>"
        for r in rows
    )
    warn = (
        "<p class='muted'>추출 결과를 그대로 평가 정답으로 쓰면 <b>순환</b>이 된다. "
        "그래서 사람이 직접 확인할 표본을 뽑는다. 근거는 <b>판정에 실제로 쓰인 문장</b>이라 "
        "여기서 틀린 게 보이면 그게 곧 규칙의 결함이다.</p>"
    )
    return (
        warn
        + "<table class='ko'><thead><tr><th>문서</th><th>추출된 관계</th>"
        "<th>근거</th></tr></thead><tbody>" + body + "</tbody></table>"
    )


CSS = """
.ko{width:100%;border-collapse:collapse;font-size:13px}
.ko th,.ko td{border-bottom:1px solid #e5e2dc;padding:6px 8px;text-align:left;vertical-align:top}
.ko thead th{color:#7a736a;font-weight:600;font-size:12px}
.ev{color:#5b554d}
.muted{color:#8a837a;font-size:12px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.card{border:1px solid #e5e2dc;border-radius:8px;padding:10px 12px}
.card h4{margin:0 0 8px;font-size:14px}
.card ol{margin:0;padding-left:18px}
.card li{margin-bottom:8px}
.snip{color:#5b554d;font-size:12px}
.pill{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;
  background:#eee;color:#555;margin-right:4px}
.pill.ok{background:#e3f0e3;color:#2c6b2f}
.pill.warn{background:#fbeee0;color:#8a5a1a}
"""


def build_app(console: Console):
    import gradio as gr

    with gr.Blocks(title="KnowledgeOps") as demo:
        gr.HTML(f"<style>{CSS}</style>")
        src = "실문서 96건(공공조달 RFP)" if console.use_real else "합성 코퍼스(데모)"
        gr.Markdown(
            f"# KnowledgeOps\n"
            f"문서를 넣으면 지식그래프가 서고, 검색이 **근거와 함께** 답한다. "
            f"현재 코퍼스: **{src}** · 청크 {len(console.chunks):,}개 · "
            f"관계 {len(console.relations):,}개"
        )

        with gr.Tab("1. 문서 → 그래프"):
            gr.Markdown(
                "실제 HWP를 파싱해 발주기관·예산·기간·요구기술을 뽑는다. "
                "**모든 관계에 근거 문장이 붙는다** — 근거 없는 추출은 만들지 않는다."
            )
            dq = gr.Textbox(label="문서 검색(사업명 또는 기관명 일부)", value="")
            head = gr.HTML()
            tbl = gr.HTML()
            dq.submit(lambda q: tab_graph(console, q), dq, [head, tbl])
            gr.Button("조회").click(lambda q: tab_graph(console, q), dq, [head, tbl])

        with gr.Tab("2. 검색 비교"):
            gr.Markdown(
                "같은 질문에 네 전략이 무엇을 가져오는지 나란히 본다. "
                "**방해 문서**로 표시된 건 관계가 없는 청크다(실문서 코퍼스의 90%가 이것)."
            )
            sq = gr.Textbox(
                label="질문",
                value="BioIN이(가) 발주한 사업의 예산은 얼마인가?"
                if console.use_real
                else "감속기에 연결되는 부품은(는) 어디에서 공급하는가?",
            )
            kk = gr.Slider(1, 8, value=3, step=1, label="top-k")
            out = gr.HTML()
            sq.submit(lambda q, k: tab_search(console, q, int(k)), [sq, kk], out)
            gr.Button("검색").click(lambda q, k: tab_search(console, q, int(k)), [sq, kk], out)

        with gr.Tab("3. 평가 리포트"):
            gr.Markdown("어느 전략이 나은지는 **통계 검정으로 판정**한다. 집계값만으로는 주장할 수 없다.")
            which = gr.Radio(
                ["실문서(RFP 96건)", "합성 코퍼스"], value="실문서(RFP 96건)", label="측정 대상"
            )
            rep = gr.HTML(tab_report("실문서"))
            which.change(tab_report, which, rep)

        with gr.Tab("4. IE 감사"):
            gr.Markdown("추출이 맞는지 **사람이 확인**하는 표본. 순환을 피하는 장치다.")
            rt = gr.Radio(
                ["HAS_BUDGET", "REQUIRES", "HAS_PERIOD", "ISSUED_BY"],
                value="HAS_BUDGET", label="관계 종류",
            )
            nn = gr.Slider(5, 30, value=12, step=1, label="표본 수")
            aud = gr.HTML()
            rt.change(lambda r, n: tab_audit(console, r, n), [rt, nn], aud)
            nn.change(lambda r, n: tab_audit(console, r, n), [rt, nn], aud)
            gr.Button("표본 뽑기").click(lambda r, n: tab_audit(console, r, n), [rt, nn], aud)

    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="합성 코퍼스로 동작 확인")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    console = Console(use_real=not args.demo)
    build_app(console).launch(server_port=args.port)


if __name__ == "__main__":
    main()
