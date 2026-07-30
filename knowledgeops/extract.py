"""실문서에서 엔티티·관계를 뽑아 지식그래프를 만든다 (IE).

## 이 파일이 프로젝트에서 갖는 의미

`graph_rag/`의 모든 실험은 **관계가 이미 주어졌다는 가정** 위에 있었다. 합성 코퍼스가
생성 규칙으로 (head, rtype, tail)을 공짜로 줬기 때문이다. NOTES.md에 그 한계를 계속
적어 뒀다 — "실문서라면 IE 품질이 그대로 상한이 된다."

여기서 그 가정을 걷어낸다. 대상은 **공공조달 RFP 96건**(실제 HWP 원본, `rfp-rag/files/`).

## 어디까지를 신뢰할 것인가 — 신호별로 다르게 다룬다

실문서 IE에서 제일 중요한 판단은 "무엇을 믿을 수 있는가"다. 여기서는 신호를 셋으로 나눴다.

| 신호 | 출처 | 신뢰도 | 근거 |
|---|---|---|---|
| 발주기관·사업명 | **파일명** | 높음 | 조달 문서는 `기관_사업명.hwp` 규약을 지킨다. 본문 파싱 실패와 무관하게 얻어진다 |
| 예산·기간 | 본문 정규식 | 중간 | 표기가 다양하지만("1억 5천만원", "150,000,000원") 패턴이 유한하다 |
| 요구 기술 | 본문 사전 매칭 | 중간 | 사전에 없는 기술은 원리적으로 못 잡는다(재현율 상한) |

파일명을 1급 신호로 쓰는 게 핵심이다. 본문에서 기관명을 찾으려 하면 "발주기관"과
"수행기관", 참조로 언급된 제3의 기관이 섞여 오탐이 크다. **문서 밖에 있는 확실한 메타데이터를
먼저 쓰고, 본문은 그것으로 설명 안 되는 것만 채우는** 순서다.

## 정직한 한계 (측정 전에 미리 적는다)

- 사전 밖 기술은 못 잡는다. 재현율의 상한이 사전 크기다.
- 예산 정규식은 본문에 여러 금액이 있을 때 **첫 번째**를 택한다. 총사업비가 아닐 수 있다.
- 이 IE 결과를 그대로 평가의 gold로 쓰면 **순환**이 된다. 그래서 `audit_sample`로
  사람이 직접 확인할 표본을 따로 뽑도록 해 뒀다(수치는 그 표본으로 보정해 읽어야 한다).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 기술 사전 — 공공조달 IT 사업에서 반복되는 용어만. 사전 크기가 재현율 상한이다.
# 별칭을 표준형으로 접는다(실문서는 표기가 흔들린다: "MS-SQL"/"MSSQL"/"Microsoft SQL").
# ---------------------------------------------------------------------------
TECH_ALIASES: dict[str, tuple[str, ...]] = {
    "Java": ("자바", "Java", "JAVA"),
    "Spring": ("스프링", "Spring", "전자정부 표준프레임워크", "표준프레임워크"),
    "JavaScript": ("자바스크립트", "JavaScript", "Javascript"),
    "Python": ("파이썬", "Python"),
    "Oracle": ("오라클", "Oracle", "ORACLE"),
    "MySQL": ("MySQL", "마이에스큐엘", "MariaDB", "마리아DB"),
    "MS-SQL": ("MS-SQL", "MSSQL", "Microsoft SQL", "SQL Server"),
    "PostgreSQL": ("PostgreSQL", "포스트그레"),
    "Tibero": ("티베로", "Tibero", "TIBERO"),
    "Linux": ("리눅스", "Linux"),
    "Windows Server": ("윈도우 서버", "Windows Server"),
    "Apache": ("아파치", "Apache"),
    "Tomcat": ("톰캣", "Tomcat"),
    "Nginx": ("Nginx", "엔진엑스"),
    "AWS": ("AWS", "아마존웹서비스", "Amazon Web Services"),
    "Cloud": ("클라우드", "Cloud"),
    "Docker": ("도커", "Docker"),
    "Kubernetes": ("쿠버네티스", "Kubernetes", "K8S"),
    "React": ("리액트", "React"),
    "Vue": ("Vue.js", "뷰제이에스"),
    "REST API": ("REST API", "RESTful", "레스트풀"),
    "AI": ("인공지능", "AI 기술", "머신러닝", "딥러닝"),
    "빅데이터": ("빅데이터", "Big Data"),
    "챗봇": ("챗봇", "chatbot", "대화형 서비스"),
    "OCR": ("OCR", "광학문자인식"),
    "블록체인": ("블록체인", "Blockchain"),
    "모바일앱": ("모바일 앱", "안드로이드", "iOS", "네이티브 앱"),
    "반응형웹": ("반응형 웹", "반응형웹", "모바일 웹"),
    "웹접근성": ("웹 접근성", "웹접근성", "WA 인증"),
    "정보보안": ("정보보호", "보안성 검토", "개인정보보호"),
    "ISMS": ("ISMS", "정보보호관리체계"),
    "SSO": ("SSO", "통합인증"),
    "GIS": ("GIS", "지리정보", "공간정보"),
}

# 관계 타입
ISSUED_BY = "ISSUED_BY"      # 사업 → 발주기관
REQUIRES = "REQUIRES"        # 사업 → 요구 기술
HAS_BUDGET = "HAS_BUDGET"    # 사업 → 예산
HAS_PERIOD = "HAS_PERIOD"    # 사업 → 기간


@dataclass(frozen=True)
class Triple:
    head: str
    rtype: str
    tail: str


@dataclass
class Document:
    doc_id: str
    agency: str
    project: str
    text: str
    triples: list[Triple] = field(default_factory=list)
    evidence: dict[str, str] = field(default_factory=dict)  # 관계키 → 근거 문장


# ---------------------------------------------------------------------------
# 파일명 파싱 — 1급 신호
# ---------------------------------------------------------------------------

_FILENAME_SPLIT = re.compile(r"^(?P<agency>[^_]+)_(?P<project>.+)$")


def parse_filename(name: str) -> tuple[str, str]:
    """`기관_사업명.hwp` → (기관, 사업명). 규약을 안 지키면 (\"\", 전체)를 준다.

    실제 파일명에는 전각 괄호(（）)와 반각이 섞여 있어 정규화한다.
    """
    stem = Path(name).stem.strip()
    stem = stem.replace("（", "(").replace("）", ")")
    m = _FILENAME_SPLIT.match(stem)
    if not m:
        return "", stem
    agency = m.group("agency").strip()
    project = re.sub(r"\s+", " ", m.group("project")).strip()
    return agency, project


# ---------------------------------------------------------------------------
# 본문 정규식 — 2급 신호
# ---------------------------------------------------------------------------

_BUDGET_PATTERNS = [
    # "1,500,000,000원", "150백만원", "15억원"
    re.compile(r"(?P<v>[\d,]+)\s*백만\s*원"),
    re.compile(r"(?P<v>[\d,]+(?:\.\d+)?)\s*억\s*원"),
    re.compile(r"(?P<v>[\d]{1,3}(?:,\d{3}){2,})\s*원"),
]

# ---------------------------------------------------------------------------
# 문맥 가드 — 표본 검증에서 드러난 오탐을 닫는다 (2026-07-30)
#
# 처음엔 "문서에서 첫 금액"을 예산으로, "사전 용어가 등장하면" 요구 기술로 봤다.
# 표본 20건을 눈으로 확인했더니 절반 가까이가 틀렸다:
#
#   "사업금액 20억원 **미만** 계약은 대기업 참여 제한"  → 법령 기준액이지 이 사업 예산이 아니다
#   "변경 소요 비용 **예시)** 1,046 백만원"              → 예시 금액
#   "구매요구액 5천5백만원 **미만** ... 위원 수 4명"      → 심사위원 기준표
#   "제안서는 제본 및 **스프링** 사용"                    → 제본용 스프링(동음이의)
#   "반응형 웹 ... 접근 수요는 **적음**"                  → 오히려 불필요하다는 서술
#   "비공개 **공간정보** 등 비공개 정보 유출"             → 보안 조항
#
# 공통점: **숫자·단어 자체는 맞는데 그것이 놓인 문맥이 다르다.** 그래서 두 가지를 건다.
#   ① 예산은 '사업비·예산·사업금액' 같은 **라벨 근처**에서만 인정한다(양성 문맥 요구).
#   ② 부정·조건·예시 문맥이면 버린다(음성 문맥 배제).
# 같은 전략을 clinical-note-pipeline의 부정 처리에서 이미 썼다 — 표면형 매칭에 문맥 판단을
# 얹는 층은 한국어 실문서에서 반복해 필요하다.
# ---------------------------------------------------------------------------

# 예산으로 인정할 라벨(이 라벨 뒤 N자 안에 금액이 있어야 한다)
_BUDGET_LABELS = re.compile(
    r"(사업\s*비|사업\s*예산|배정\s*예산|총\s*사업비|예산\s*액|사업\s*금액|"
    r"용역\s*비|추정\s*가격|사업\s*대가|계약\s*금액|구매\s*예산)"
)
_BUDGET_LABEL_WINDOW = 60

# 이 표현이 금액 주변에 있으면 이 사업의 예산이 아니다
_BUDGET_NEGATIVE = re.compile(r"(미만|이상|초과|이하|예시|기준|참여\s*제한|위원|배점|벌점)")

# 기술 언급 주변에 이 표현이 있으면 요구사항이 아니다
_TECH_NEGATIVE = re.compile(
    r"(적음|없음|제외|불필요|해당\s*없|유출|금지|위반|제본|아니|미포함|불가)"
)

_PERIOD_PATTERNS = [
    re.compile(r"계약일?로?부터\s*(?P<v>\d{1,3})\s*개월"),
    re.compile(r"(?P<v>\d{1,3})\s*개월\s*(?:간|이내|이하)"),
    re.compile(r"착수일?로?부터\s*(?P<v>\d{1,3})\s*개월"),
]


def _first_match(text: str, patterns: list[re.Pattern]) -> tuple[str, str] | None:
    """가장 먼저 걸리는 값과 그 근거 문장을 준다. 없으면 None."""
    for pat in patterns:
        m = pat.search(text)
        if m:
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 40)
            return m.group(0).strip(), text[start:end].replace("\n", " ").strip()
    return None


def _window(text: str, start: int, end: int, pad: int = 40) -> str:
    return text[max(0, start - pad) : min(len(text), end + pad)].replace("\n", " ").strip()


# 문장 경계로 볼 문자 — 한국어 실문서는 줄바꿈과 항목 기호가 실질적 경계다
_SENT_BREAK = "\n.。!?;:·▪◦□■○●◇◆-–—|"


def _sentence(text: str, start: int, end: int, pad: int = 45) -> str:
    """해당 위치가 속한 **문장**을 준다(경계까지만).

    부정 문맥 판정에 고정 폭 윈도우를 쓰면 **인접 문장의 부정어를 끌어온다.**
    단위테스트가 이걸 잡았다 — "제안서는 제본 및 스프링 사용. 뒤쪽에서 전자정부
    표준프레임워크를 적용한다"에서 뒤 문장의 정당한 언급까지 앞 문장의 '제본' 때문에
    걸러졌다. 실문서의 표나 항목 나열에서도 같은 일이 생긴다(한 줄에 여러 항목이 붙는다).
    부정은 **같은 절 안에서만** 그 언급을 무효화한다.
    """
    lo = max(0, start - pad)
    hi = min(len(text), end + pad)
    seg_start = lo
    for i in range(start - 1, lo - 1, -1):
        if text[i] in _SENT_BREAK:
            seg_start = i + 1
            break
    seg_end = hi
    for i in range(end, hi):
        if text[i] in _SENT_BREAK:
            seg_end = i
            break
    return text[seg_start:seg_end].strip()


def extract_budget(text: str) -> tuple[str, str] | None:
    """예산 표기와 근거 문장 — **라벨 근처**의 금액만, **부정 문맥**은 배제.

    문서 전체의 첫 금액을 쓰면 법령 기준액·예시·심사 기준표에 걸린다(위 주석 참조).
    라벨(사업비·배정예산 등) 뒤 60자 안에 있는 금액만 이 사업의 예산으로 인정한다.
    """
    candidates: list[tuple[int, str, str]] = []
    for pat in _BUDGET_PATTERNS:
        for m in pat.finditer(text):
            ctx = _window(text, m.start(), m.end())
            if _BUDGET_NEGATIVE.search(_sentence(text, m.start(), m.end())):
                continue
            before = text[max(0, m.start() - _BUDGET_LABEL_WINDOW) : m.start()]
            if not _BUDGET_LABELS.search(before):
                continue
            candidates.append((m.start(), m.group(0).strip(), ctx))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])  # 문서에서 가장 먼저 나온 것
    return candidates[0][1], candidates[0][2]


def extract_period(text: str) -> tuple[str, str] | None:
    return _first_match(text, _PERIOD_PATTERNS)


def extract_tech(text: str, aliases: dict[str, tuple[str, ...]] | None = None) -> list[str]:
    """사전 매칭으로 요구 기술을 뽑되 **부정 문맥은 배제**한다. 결정적 순서.

    같은 단어라도 "반응형 웹 접근 수요는 적음"·"제본 및 스프링 사용"·"공간정보 유출"은
    요구사항이 아니다. 언급 위치를 **모두** 보고, 부정 문맥이 아닌 언급이 하나라도 있어야
    인정한다(한 번 부정이 나왔다고 버리면 재현율이 무너진다 — 긴 문서에는 두 문맥이 공존한다).
    """
    return sorted(extract_tech_with_evidence(text, aliases))


def extract_tech_with_evidence(
    text: str, aliases: dict[str, tuple[str, ...]] | None = None
) -> dict[str, str]:
    """{표준형: **판정에 실제로 쓰인** 근거 문장}.

    판정과 근거를 한 함수에서 내는 이유(2026-07-30에 잡은 결함): 처음엔 둘을 따로 계산해
    **판정은 '부정 문맥이 아닌 언급'으로 하고, 근거는 '첫 등장'을 보여주고** 있었다.
    그 탓에 감사 표본에 Spring의 근거로 "제안서는 제본 및 스프링 사용"이 찍혔다 —
    실제 판정은 다른 위치(전자정부 표준프레임워크)로 내려졌는데도 오탐처럼 보인 것이다.
    **사람이 검증할 근거는 반드시 판정에 쓰인 바로 그 위치**여야 한다. 아니면 감사가
    잘못된 신호를 주고, 멀쩡한 규칙을 고치거나 틀린 규칙을 놓친다.
    """
    table = TECH_ALIASES if aliases is None else aliases
    out: dict[str, str] = {}
    for canon, forms in table.items():
        for form in forms:
            start, hit = 0, None
            while True:
                i = text.find(form, start)
                if i < 0:
                    break
                sent = _sentence(text, i, i + len(form))
                if not _TECH_NEGATIVE.search(sent):
                    # 근거 = 판정에 쓴 바로 그 문장. 둘을 다르게 주면 감사가 오도된다.
                    hit = sent
                    break
                start = i + len(form)
            if hit is not None:
                out[canon] = hit
                break
    return out


# ---------------------------------------------------------------------------
# 문서 → 그래프
# ---------------------------------------------------------------------------


def build_document(doc_id: str, filename: str, text: str) -> Document:
    """한 문서에서 엔티티·관계를 뽑는다. 순수 함수(파일 I/O 없음)."""
    agency, project = parse_filename(filename)
    doc = Document(doc_id=doc_id, agency=agency, project=project, text=text)

    if agency:
        doc.triples.append(Triple(project, ISSUED_BY, agency))
        doc.evidence[f"{project}|{ISSUED_BY}|{agency}"] = f"[파일명] {filename}"

    b = extract_budget(text)
    if b:
        doc.triples.append(Triple(project, HAS_BUDGET, b[0]))
        doc.evidence[f"{project}|{HAS_BUDGET}|{b[0]}"] = b[1]

    p = extract_period(text)
    if p:
        doc.triples.append(Triple(project, HAS_PERIOD, p[0]))
        doc.evidence[f"{project}|{HAS_PERIOD}|{p[0]}"] = p[1]

    for tech, ev in sorted(extract_tech_with_evidence(text).items()):
        doc.triples.append(Triple(project, REQUIRES, tech))
        doc.evidence[f"{project}|{REQUIRES}|{tech}"] = ev

    return doc


def graph_summary(docs: list[Document]) -> dict:
    """추출 결과 요약 — 커버리지를 먼저 보고 그다음 품질을 본다."""
    n = len(docs)
    if not n:
        return {"documents": 0}
    by_type: dict[str, int] = {}
    for d in docs:
        for t in d.triples:
            by_type[t.rtype] = by_type.get(t.rtype, 0) + 1
    agencies = {d.agency for d in docs if d.agency}
    techs = {t.tail for d in docs for t in d.triples if t.rtype == REQUIRES}
    return {
        "documents": n,
        "triples": sum(len(d.triples) for d in docs),
        "by_type": dict(sorted(by_type.items())),
        "unique_agencies": len(agencies),
        "unique_techs": len(techs),
        "docs_with_budget": sum(
            1 for d in docs if any(t.rtype == HAS_BUDGET for t in d.triples)
        ),
        "docs_with_period": sum(
            1 for d in docs if any(t.rtype == HAS_PERIOD for t in d.triples)
        ),
        "avg_tech_per_doc": round(
            sum(1 for d in docs for t in d.triples if t.rtype == REQUIRES) / n, 2
        ),
    }


def audit_sample(docs: list[Document], rtype: str, k: int = 20) -> list[dict]:
    """사람이 직접 확인할 표본 — IE를 gold로 쓰면 순환이므로 반드시 필요하다.

    결정적으로 고르고(정렬 후 균등 간격) 근거 문장을 함께 준다.
    """
    rows = [
        {
            "doc": d.doc_id,
            "triple": f"({t.head}, {t.rtype}, {t.tail})",
            "evidence": d.evidence.get(f"{t.head}|{t.rtype}|{t.tail}", ""),
        }
        for d in docs
        for t in d.triples
        if t.rtype == rtype
    ]
    rows.sort(key=lambda r: (r["doc"], r["triple"]))
    if len(rows) <= k:
        return rows
    step = len(rows) / k
    return [rows[int(i * step)] for i in range(k)]
