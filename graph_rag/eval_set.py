"""검색 평가셋 — 단일홉 vs 멀티홉을 **분리**해 정의. 순수·결정적.

## 왜 나눠서 재는가

"GraphRAG가 더 낫다"는 뭉뚱그린 주장은 검증이 안 된다. 두 방식은 이기는 지점이 다르다:

  - **단일홉**("배터리팩 공칭 전압은?") — 답이 한 청크에 있다. 임베딩·BM25가 질의어와
    직접 매칭되므로 플랫 하이브리드가 강하다. 그래프는 여기서 이득이 없거나 오히려
    노이즈를 더할 수 있다.
  - **멀티홉**("셀 모듈을 포함하는 팩은 어디서 공급하나?") — 답이 **여러 청크에 흩어져** 있고
    둘을 잇는 건 그래프상 경로다. 질의 표면어와 정답 청크의 어휘가 겹치지 않을 수 있어
    플랫 검색이 구조적으로 불리하다.

그래서 지표를 전체·단일홉·멀티홉 세 갈래로 낸다. 이렇게 해야 "어디서 왜 이겼는지"를 말할 수 있다.

gold는 사람이 라벨링한 게 아니라 **그래프에서 유도**한다(corpus.relation_home).
즉 정답 정의가 주관이 아니라 생성 규칙에서 나온다.
"""
from __future__ import annotations

from dataclasses import dataclass

from .corpus import ENTITY_BY_ID, Chunk, relation_home


@dataclass(frozen=True)
class Query:
    qid: str
    question: str
    hops: int  # 1 = 단일홉, 2 = 멀티홉
    gold_chunks: tuple[str, ...]  # 정답을 구성하는 청크(모두 필요)
    # 질의가 출발하는 엔티티(그래프 검색의 앵커). 평가에서 시드 추출 성능과 분리해
    # *검색 전략 자체*를 비교하기 위해 명시한다.
    seed_entities: tuple[str, ...]


# (질문, hops, 정답을 구성하는 관계들, 시드 엔티티)
# 관계는 (head, rtype, tail) — corpus.RELATIONS와 같은 표기.
_SPEC: list[tuple[str, str, int, list[tuple[str, str, str]], list[str]]] = [
    # ---- 단일홉: 답이 한 청크에 있다 ----
    ("q01", "고전압 배터리팩의 공칭 전압은 얼마인가?", 1,
     [("cmp_pack", "HAS_SPEC", "spec_voltage")], ["cmp_pack"]),
    ("q02", "리튬이온 셀의 전압 사양은?", 1,
     [("cmp_cell", "HAS_SPEC", "spec_cell_v")], ["cmp_cell"]),
    ("q03", "전방 레이더의 탐지 거리는?", 1,
     [("cmp_radar", "HAS_SPEC", "spec_radar_range")], ["cmp_radar"]),
    ("q04", "냉각수 펌프의 유량 사양은?", 1,
     [("cmp_pump", "HAS_SPEC", "spec_coolant")], ["cmp_pump"]),
    ("q05", "메인 컨택터의 최대 방전 전류는?", 1,
     [("cmp_contactor", "HAS_SPEC", "spec_current")], ["cmp_contactor"]),
    ("q06", "전방 카메라의 화각은?", 1,
     [("cmp_camera", "HAS_SPEC", "spec_camera_fov")], ["cmp_camera"]),
    ("q07", "열관리 시스템은 어떤 부품으로 구성되는가?", 1,
     [("sys_thermal", "CONTAINS", "cmp_chiller"),
      ("sys_thermal", "CONTAINS", "cmp_pump")], ["sys_thermal"]),
    ("q08", "배터리 관리 시스템의 구성 부품은?", 1,
     [("sys_bms", "CONTAINS", "cmp_bms_ecu"),
      ("sys_bms", "CONTAINS", "cmp_contactor")], ["sys_bms"]),

    # ---- 멀티홉: 답이 서로 다른 청크에 흩어져 있다 ----
    ("q09", "셀 모듈을 포함하는 상위 부품은 어느 회사에서 공급하는가?", 2,
     [("cmp_pack", "CONTAINS", "cmp_module"),
      ("cmp_pack", "SUPPLIED_BY", "sup_beta")], ["cmp_module"]),
    ("q10", "알파일렉트로닉스가 공급하는 부품은 어떤 시스템에 속하는가?", 2,
     [("sys_bms", "CONTAINS", "cmp_bms_ecu"),
      ("cmp_bms_ecu", "SUPPLIED_BY", "sup_alpha")], ["sup_alpha"]),
    ("q11", "배터리 칠러와 연결된 부품의 용량은 얼마인가?", 2,
     [("cmp_chiller", "CONNECTS_TO", "cmp_pack"),
      ("cmp_pack", "HAS_SPEC", "spec_capacity")], ["cmp_chiller"]),
    ("q12", "ADAS 제어기에 연결된 센서들은 어디서 공급하는가?", 2,
     [("cmp_radar", "CONNECTS_TO", "cmp_adas_ecu"),
      ("cmp_radar", "SUPPLIED_BY", "sup_gamma")], ["cmp_adas_ecu"]),
    ("q13", "리튬이온 셀을 담고 있는 모듈의 상위 부품은 어떤 전압을 갖는가?", 2,
     [("cmp_pack", "CONTAINS", "cmp_module"),
      ("cmp_pack", "HAS_SPEC", "spec_voltage")], ["cmp_cell"]),
    ("q14", "BMS 제어기가 연결된 부품의 충전 권장 온도는?", 2,
     [("cmp_bms_ecu", "CONNECTS_TO", "cmp_pack"),
      ("cmp_pack", "HAS_SPEC", "spec_temp")], ["cmp_bms_ecu"]),
    ("q15", "베타모빌리티가 공급하는 부품 중 열관리 시스템에 속한 것은?", 2,
     [("cmp_chiller", "SUPPLIED_BY", "sup_beta"),
      ("sys_thermal", "CONTAINS", "cmp_chiller")], ["sup_beta"]),
    ("q16", "전방 카메라와 같은 제어기에 연결된 다른 센서의 탐지 거리는?", 2,
     [("cmp_camera", "CONNECTS_TO", "cmp_adas_ecu"),
      ("cmp_radar", "HAS_SPEC", "spec_radar_range")], ["cmp_camera"]),
]


def build_queries(chunks: list[Chunk]) -> list[Query]:
    """평가 질의 생성. gold 청크는 그래프 관계 → 해당 관계를 서술한 청크로 유도한다."""
    home = relation_home(chunks)
    out: list[Query] = []
    for qid, question, hops, rels, seeds in _SPEC:
        gold: list[str] = []
        for r in rels:
            cid = home.get(r)
            if cid is None:
                raise ValueError(f"{qid}: 코퍼스에 없는 관계 {r}")
            if cid not in gold:
                gold.append(cid)
        for s in seeds:
            if s not in ENTITY_BY_ID:
                raise ValueError(f"{qid}: 알 수 없는 시드 엔티티 {s}")
        out.append(
            Query(
                qid=qid,
                question=question,
                hops=hops,
                gold_chunks=tuple(gold),
                seed_entities=tuple(seeds),
            )
        )
    return out
