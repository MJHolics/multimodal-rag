"""IE 감사 모듈 테스트.

여기서 고정하려는 것은 수치가 아니라 **판단 규칙**이다:
  - 전수를 셀 수 있으면 구간을 만들지 않는다(전수에 신뢰구간을 붙이는 건 틀렸다).
  - 오탐 관계에서 유도된 문항은 집계에서 빠진다. 2-hop은 다리가 틀려도 빠진다.
  - 정화는 문항을 지우는 게 아니라 **분모까지 같이** 줄인다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledgeops.audit import (
    BAD,
    OK,
    compare_before_after,
    invalid_triples,
    plan_audit,
    precision,
    purify,
    score,
    systematic_sample,
)

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "output" / "audit_labels.json"


# ---------- 감사 설계 ----------

def test_census_when_population_fits_budget():
    p = plan_audit("HAS_BUDGET", population=49, budget=150)
    assert p.is_census and p.k == 49
    assert "전수" in p.reason


def test_sample_when_population_exceeds_budget():
    p = plan_audit("REQUIRES", population=853, budget=150)
    assert not p.is_census and p.k == 150
    assert "Wilson" in p.reason


def test_census_precision_has_no_interval():
    """전수 감사에 신뢰구간을 붙이면 통계적으로 틀린다 — 추정이 아니기 때문."""
    out = precision([OK] * 47 + [BAD] * 2, population=49, mode="census")
    assert out["precision"] == pytest.approx(47 / 49, abs=1e-4)
    assert out["interval"] is None


def test_sample_precision_has_interval_containing_point():
    out = precision([OK] * 40 + [BAD] * 10, population=853, mode="sample")
    lo, hi = out["interval"]
    assert lo < out["precision"] < hi
    assert 0.0 <= lo and hi <= 1.0


def test_systematic_sample_is_deterministic_and_sized():
    rows = [{"i": i} for i in range(100)]
    a = systematic_sample(rows, 10)
    b = systematic_sample(rows, 10)
    assert a == b and len(a) == 10
    assert systematic_sample(rows, 500) == rows  # k가 모집단보다 크면 전수


# ---------- 정화 ----------

def _pq(qid, hops, correct):
    return {"qid": qid, "hops": hops, "correct": {"s": correct}}


def test_purify_drops_queries_derived_from_bad_triples():
    per_q = [_pq("q1", 1, True), _pq("q2", 1, False), _pq("q3", 2, True)]
    srcs = {"q1": ["T_ok"], "q2": ["T_bad"], "q3": ["T_ok", "T_ok2"]}
    out = purify(per_q, srcs, {"T_bad"})
    assert [r["qid"] for r in out["kept"]] == ["q1", "q3"]
    assert [r["qid"] for r in out["dropped"]] == ["q2"]


def test_purify_drops_two_hop_when_bridge_is_wrong():
    """2-hop은 속성 관계가 맞아도 다리(ISSUED_BY)가 틀리면 gold가 무너진다."""
    per_q = [_pq("q1", 2, True)]
    srcs = {"q1": ["BRIDGE_bad", "ATTR_ok"]}
    assert purify(per_q, srcs, {"BRIDGE_bad"})["dropped"][0]["qid"] == "q1"


def test_score_recomputes_without_rerunning_retrieval():
    per_q = [_pq("a", 1, True), _pq("b", 1, False), _pq("c", 2, True)]
    # score는 보고용으로 4자리에서 반올림한다 — 비교 허용오차를 거기 맞춘다
    assert score(per_q, "s")["full_hit"] == pytest.approx(2 / 3, abs=1e-4)
    assert score(per_q, "s", hops=2)["full_hit"] == 1.0
    assert score(per_q, "s", hops=1)["full_hit"] == 0.5


def test_purified_score_shrinks_denominator_not_just_numerator():
    """오탐 문항을 '틀린 것'으로 세면 안 된다 — 물어보지 말았어야 할 문항이다."""
    per_q = [_pq("a", 1, True), _pq("bad", 1, True)]
    srcs = {"a": ["ok"], "bad": ["nope"]}
    kept = purify(per_q, srcs, {"nope"})["kept"]
    assert score(kept, "s") == {"n": 1, "correct": 1, "full_hit": 1.0}


def test_compare_reports_dropped_count():
    before = [_pq("a", 2, True), _pq("b", 2, False)]
    after = [_pq("a", 2, True)]
    c = compare_before_after(before, after, "s", hops=2)
    assert c["before"]["full_hit"] == 0.5 and c["after"]["full_hit"] == 1.0
    assert c["dropped"] == 1 and c["delta"] == pytest.approx(0.5)


def test_invalid_triples_selects_only_bad():
    recs = [{"key": "a", "label": OK}, {"key": "b", "label": BAD}, {"key": "c"}]
    assert invalid_triples(recs) == {"b"}


# ---------- 실제 라벨 fixture ----------

@pytest.mark.skipif(not LABELS.exists(), reason="감사 라벨이 없다(실문서 필요)")
def test_label_fixture_is_complete_and_census():
    """라벨이 평가에 쓰인 속성 관계를 **전수** 덮는지 — 빠지면 정화가 반쪽이 된다."""
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    recs = data["records"]
    by_type: dict[str, list[str]] = {}
    for r in recs:
        by_type.setdefault(r["rtype"], []).append(r["label"])
    assert len(by_type["HAS_BUDGET"]) == 49
    # 71 → 53: 문맥 가드가 "기간처럼 생긴 다른 기간"을 걸러내고 진짜 사업기간을 찾아왔다
    assert len(by_type["HAS_PERIOD"]) == 53
    assert all(r["label"] in (OK, BAD) for r in recs)
    # 오탐이면 사유가 있어야 한다 — 원인을 모르면 고칠 수 없다
    assert all(r["reason"] for r in recs if r["label"] == BAD)
    # 근거 없는 판정은 재검증이 불가능하다
    assert all(r["evidence"] for r in recs)


@pytest.mark.skipif(not LABELS.exists(), reason="감사 라벨이 없다(실문서 필요)")
def test_requires_is_sampled_not_census():
    """REQUIRES는 모집단 853이라 전수가 불가능하다 — 표본 + 구간으로 간다."""
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    labels = [r["label"] for r in data["records"] if r["rtype"] == "REQUIRES"]
    assert len(labels) == 50 < 853
    out = precision(labels, population=853, mode="sample")
    assert out["interval"] is not None
    lo, hi = out["interval"]
    assert lo < out["precision"] < hi


@pytest.mark.skipif(not LABELS.exists(), reason="감사 라벨이 없다(실문서 필요)")
def test_period_guard_closed_the_gap():
    """전수 감사가 드러낸 격차(예산 0.959 vs 기간 0.437)를 문맥 가드로 닫았음을 고정한다.

    이 테스트는 원래 `test_period_precision_is_far_worse_than_budget`이었고 기간이 0.5
    미만임을 **의도적으로** 못박고 있었다 — 감사 대상을 내가 고친 곳으로만 골랐다는 사실이
    기록에서 지워지지 않게 하려고. 가드를 붙였으니 이제 반대 방향으로 잠근다.
    되돌아가면(가드가 무력화되면) 여기서 깨진다.
    """
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    got = {}
    for rtype in ("HAS_BUDGET", "HAS_PERIOD"):
        labels = [r["label"] for r in data["records"] if r["rtype"] == rtype]
        got[rtype] = precision(labels, len(labels), "census")["precision"]
    assert got["HAS_BUDGET"] > 0.9
    assert got["HAS_PERIOD"] > 0.9, "가드 이전 0.437 — 여기로 되돌아가면 안 된다"


@pytest.mark.skipif(not LABELS.exists(), reason="감사 라벨이 없다(실문서 필요)")
def test_period_guard_was_derived_on_dev_only():
    """가드 규칙은 dev(짝수 문서)에서만 도출했다 — test는 홀드아웃이다.

    내가 붙인 라벨에 가드를 맞추면 정밀도는 정의상 올라간다. 그게 아니라는 증거는
    **한 번도 안 본 절반에서도 오른다**는 것뿐이다. test가 완벽하지 않다는 사실
    (오탐 1건 잔존)이 오히려 홀드아웃이 진짜였다는 증거다.
    """
    data = json.loads(LABELS.read_text(encoding="utf-8"))
    rows = [r for r in data["records"] if r["rtype"] == "HAS_PERIOD"]
    assert {r["split"] for r in rows} == {"dev", "test"}
    for split, floor in (("dev", 0.99), ("test", 0.9)):
        sub = [r["label"] for r in rows if r["split"] == split]
        assert precision(sub, len(sub), "census")["precision"] >= floor
    # 홀드아웃에 남은 오탐은 고치지 않았다(고치면 그건 더 이상 홀드아웃이 아니다)
    assert sum(1 for r in rows if r["split"] == "test" and r["label"] == "bad") == 1
