"""합성 기술문서 코퍼스 + ground-truth 지식그래프 — 순수·결정적, 외부 의존 없음.

## 왜 합성인가

GraphRAG가 기존 하이브리드 검색보다 나은지 재려면 **정답 관련성(gold relevance)이 정확한**
평가셋이 필요하다. 실 PDF에 사람이 라벨을 달면 (1) 주관이 섞이고 (2) 재현이 안 되며
(3) 멀티홉 질의의 "정답 경로"를 특정하기 어렵다.

여기서는 문서를 **알려진 그래프로부터 생성**한다. 그래서:
  - 각 질의의 gold 청크가 유일하게 결정된다(생성 시점에 안다).
  - 멀티홉 질의를 "그래프상 2-hop 경로"로 정확히 정의할 수 있다.
  - 시드 고정으로 완전 재현된다(랜덤 없음 — 전부 결정적 생성).

정직한 범위: 이건 **검색 전략 비교용 통제 실험**이지 실문서 벤치마크가 아니다.
합성 코퍼스는 엔티티 표기가 일관되므로 실제보다 그래프 구축이 쉽다 —
즉 아래 실측은 GraphRAG에 **유리한 쪽으로 치우친 상한**으로 읽어야 한다.
실 PDF 적용 시 엔티티 정규화·표기 흔들림이 추가 난이도로 붙는다.

## 도메인

EV 배터리/ADAS 기술문서를 본떴다(TechDocRAG의 원 타겟 도메인).
그래프는 부품·시스템·사양 노드와 CONTAINS / CONNECTS_TO / HAS_SPEC / SUPPLIED_BY 관계로 구성.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Entity:
    """지식그래프 노드. name은 문서 표면형과 동일하게 쓴다(합성이라 표기 흔들림 없음)."""

    eid: str
    name: str
    etype: str  # system | component | spec | supplier


@dataclass(frozen=True)
class Relation:
    """방향 있는 관계. (head)-[rtype]->(tail)"""

    head: str  # eid
    rtype: str
    tail: str  # eid


@dataclass
class Chunk:
    """검색 단위. 실제 파이프라인의 PageChunk에 대응(페이지 1개 = 청크 1개로 단순화)."""

    chunk_id: str
    doc_name: str
    page_num: int
    text: str
    # 이 청크가 서술하는 관계들(ground truth). 평가에서 gold 판정에 쓴다.
    states: list[Relation] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)  # eid


# ---------------------------------------------------------------------------
# 그래프 정의 — 이 표가 코퍼스와 정답의 단일 진실 소스
# ---------------------------------------------------------------------------

ENTITIES: list[Entity] = [
    # 시스템
    Entity("sys_bms", "배터리 관리 시스템", "system"),
    Entity("sys_adas", "첨단 운전자 보조 시스템", "system"),
    Entity("sys_thermal", "열관리 시스템", "system"),
    # 부품
    Entity("cmp_pack", "고전압 배터리팩", "component"),
    Entity("cmp_module", "셀 모듈", "component"),
    Entity("cmp_cell", "리튬이온 셀", "component"),
    Entity("cmp_bms_ecu", "BMS 제어기", "component"),
    Entity("cmp_contactor", "메인 컨택터", "component"),
    Entity("cmp_chiller", "배터리 칠러", "component"),
    Entity("cmp_pump", "냉각수 펌프", "component"),
    Entity("cmp_radar", "전방 레이더", "component"),
    Entity("cmp_camera", "전방 카메라", "component"),
    Entity("cmp_adas_ecu", "ADAS 제어기", "component"),
    # 사양
    Entity("spec_voltage", "공칭 전압 400V", "spec"),
    Entity("spec_capacity", "용량 77.4kWh", "spec"),
    Entity("spec_cell_v", "셀 전압 3.6V", "spec"),
    Entity("spec_temp", "충전 권장 온도 0~40도", "spec"),
    Entity("spec_current", "최대 방전 전류 500A", "spec"),
    Entity("spec_radar_range", "탐지 거리 200m", "spec"),
    Entity("spec_camera_fov", "화각 120도", "spec"),
    Entity("spec_coolant", "냉각수 유량 12L/min", "spec"),
    # 공급사
    Entity("sup_alpha", "알파일렉트로닉스", "supplier"),
    Entity("sup_beta", "베타모빌리티", "supplier"),
    Entity("sup_gamma", "감마센서", "supplier"),
]

RELATIONS: list[Relation] = [
    # 배터리 계통 구성
    Relation("sys_bms", "CONTAINS", "cmp_bms_ecu"),
    Relation("sys_bms", "CONTAINS", "cmp_contactor"),
    Relation("cmp_pack", "CONTAINS", "cmp_module"),
    Relation("cmp_module", "CONTAINS", "cmp_cell"),
    # 열관리 계통
    Relation("sys_thermal", "CONTAINS", "cmp_chiller"),
    Relation("sys_thermal", "CONTAINS", "cmp_pump"),
    Relation("cmp_chiller", "CONNECTS_TO", "cmp_pack"),
    # ADAS 계통
    Relation("sys_adas", "CONTAINS", "cmp_adas_ecu"),
    Relation("cmp_radar", "CONNECTS_TO", "cmp_adas_ecu"),
    Relation("cmp_camera", "CONNECTS_TO", "cmp_adas_ecu"),
    # 제어 연결
    Relation("cmp_bms_ecu", "CONNECTS_TO", "cmp_contactor"),
    Relation("cmp_bms_ecu", "CONNECTS_TO", "cmp_pack"),
    # 사양
    Relation("cmp_pack", "HAS_SPEC", "spec_voltage"),
    Relation("cmp_pack", "HAS_SPEC", "spec_capacity"),
    Relation("cmp_cell", "HAS_SPEC", "spec_cell_v"),
    Relation("cmp_pack", "HAS_SPEC", "spec_temp"),
    Relation("cmp_contactor", "HAS_SPEC", "spec_current"),
    Relation("cmp_radar", "HAS_SPEC", "spec_radar_range"),
    Relation("cmp_camera", "HAS_SPEC", "spec_camera_fov"),
    Relation("cmp_pump", "HAS_SPEC", "spec_coolant"),
    # 공급사
    Relation("cmp_bms_ecu", "SUPPLIED_BY", "sup_alpha"),
    Relation("cmp_pack", "SUPPLIED_BY", "sup_beta"),
    Relation("cmp_radar", "SUPPLIED_BY", "sup_gamma"),
    Relation("cmp_camera", "SUPPLIED_BY", "sup_gamma"),
    Relation("cmp_chiller", "SUPPLIED_BY", "sup_beta"),
]

# ---------------------------------------------------------------------------
# 확장(2026-07-29) — 코퍼스를 키워 경쟁을 만든다
#
# 1차 실험(청크 15개)은 top-3가 전체의 20%였다. 무작위로 찍어도 꽤 맞는 크기라
# 어떤 전략이든 점수가 높게 나오고, 전략 간 차이도 과장된다. 그래서 두 방향으로 늘린다:
#
#   (a) 도메인 확장 — 충전·구동·인포테인먼트 계통을 추가해 "관련 없는 정답 후보"를 늘린다.
#   (b) distractor  — 아래 _DISTRACTOR_PLAN 참조. 관계는 없지만 어휘가 겹치는 청크.
#
# 기존 엔티티/관계/청크는 **인덱스까지 그대로 둔다**. RELATIONS 인덱스를 _CHUNK_PLAN이
# 참조하고 평가셋 gold도 관계로부터 유도되므로, 뒤에 append만 해야 1차 결과와 비교가 성립한다.
#
# 공급사 주의: 기존 평가 질의(q10 "알파일렉트로닉스가 공급하는 부품은…", q15 "베타모빌리티…")는
# 답이 유일해야 성립한다. 그래서 신규 부품에는 **새 공급사만** 붙인다. alpha/beta/gamma를
# 재사용하면 질문이 중의적이 되어 평가셋 자체가 깨진다.
# 같은 이유로 ADAS·BMS 제어기에는 새 연결을 달지 않는다(q12·q14의 답이 흔들린다).
# ---------------------------------------------------------------------------

ENTITIES += [
    # 충전 계통
    Entity("sys_charge", "충전 시스템", "system"),
    Entity("cmp_obc", "차량 탑재형 충전기", "component"),
    Entity("cmp_ccu", "충전 제어기", "component"),
    Entity("cmp_inlet", "충전 인렛", "component"),
    Entity("spec_ac_power", "완속 충전 출력 11kW", "spec"),
    Entity("spec_dc_power", "급속 충전 출력 350kW", "spec"),
    # 구동 계통
    Entity("sys_drive", "구동 시스템", "system"),
    Entity("cmp_motor", "구동 모터", "component"),
    Entity("cmp_inverter", "인버터", "component"),
    Entity("cmp_reducer", "감속기", "component"),
    Entity("spec_motor_power", "최고 출력 160kW", "spec"),
    Entity("spec_torque", "최대 토크 350Nm", "spec"),
    Entity("spec_reduction", "감속비 10.65", "spec"),
    # 인포테인먼트 계통
    Entity("sys_infotainment", "인포테인먼트 시스템", "system"),
    Entity("cmp_hu", "헤드유닛", "component"),
    Entity("cmp_tcu", "통신 제어기", "component"),
    Entity("spec_display", "디스플레이 12.3인치", "spec"),
    Entity("spec_telematics", "통신 규격 LTE-V2X", "spec"),
    # 신규 공급사 (기존 공급사 재사용 금지 — 위 주의 참조)
    Entity("sup_delta", "델타파워", "supplier"),
    Entity("sup_epsilon", "엡실론드라이브", "supplier"),
    Entity("sup_zeta", "제타텔레콤", "supplier"),
]

RELATIONS += [
    # 충전 계통 (인덱스 25~)
    Relation("sys_charge", "CONTAINS", "cmp_obc"),
    Relation("sys_charge", "CONTAINS", "cmp_ccu"),
    Relation("cmp_inlet", "CONNECTS_TO", "cmp_obc"),
    Relation("cmp_obc", "CONNECTS_TO", "cmp_pack"),
    Relation("cmp_obc", "HAS_SPEC", "spec_ac_power"),
    Relation("cmp_obc", "HAS_SPEC", "spec_dc_power"),
    Relation("cmp_ccu", "SUPPLIED_BY", "sup_delta"),
    Relation("cmp_obc", "SUPPLIED_BY", "sup_delta"),
    # 구동 계통
    Relation("sys_drive", "CONTAINS", "cmp_motor"),
    Relation("sys_drive", "CONTAINS", "cmp_inverter"),
    Relation("cmp_motor", "CONNECTS_TO", "cmp_reducer"),
    Relation("cmp_inverter", "CONNECTS_TO", "cmp_pack"),
    Relation("cmp_motor", "HAS_SPEC", "spec_motor_power"),
    Relation("cmp_motor", "HAS_SPEC", "spec_torque"),
    Relation("cmp_reducer", "HAS_SPEC", "spec_reduction"),
    Relation("cmp_motor", "SUPPLIED_BY", "sup_epsilon"),
    Relation("cmp_inverter", "SUPPLIED_BY", "sup_epsilon"),
    # 인포테인먼트 계통
    Relation("sys_infotainment", "CONTAINS", "cmp_hu"),
    Relation("sys_infotainment", "CONTAINS", "cmp_tcu"),
    Relation("cmp_hu", "HAS_SPEC", "spec_display"),
    Relation("cmp_tcu", "HAS_SPEC", "spec_telematics"),
    Relation("cmp_hu", "SUPPLIED_BY", "sup_zeta"),
    Relation("cmp_tcu", "SUPPLIED_BY", "sup_zeta"),
]

# ---------------------------------------------------------------------------
# 확장 2차(2026-07-30) — **2-hop 경로 수**를 늘리기 위한 확장
#
# 앞선 확장(07-29)의 목적은 난이도(distractor)였다. 이번 목적은 다르다: 평가셋을 그래프에서
# 자동 생성(`eval_gen.py`)해 보니 **멀티홉이 25문항**밖에 안 나왔다. 검정력 하한이
# 불일치 쌍 6개인데(stats.min_detectable_discordant) 25문항으로는 그 6개가 잘 안 모인다.
#
# 문항을 억지로 만들 수는 없다 — 질의는 그래프에서 유도되므로 **그래프가 작으면 질의도 적다.**
# 그래서 계통을 3개 더 넣는다. 배치 원칙 두 가지:
#   (a) 부품당 HAS_SPEC은 **1개**씩 — 뒷마디 유일성이 성립해야 2-hop이 살아난다.
#   (b) 신규 계통을 기존 계통에 **CONNECTS_TO로 잇는다**(회생제동→인버터, 컴프레서→칠러).
#       문서를 가로지르는 멀티홉이 생겨야 "한 페이지만 찾으면 되는" 질의가 줄어든다.
#
# 기존 제약은 그대로 지킨다: 인덱스는 append만, 기존 공급사(alpha/beta/gamma) 재사용 금지,
# ADAS·BMS 제어기에는 새 연결을 달지 않는다(손수 만든 q12·q14의 답이 흔들린다).
# 신규 CONNECTS_TO는 전부 **신규 부품이 head** 쪽이라 기존 부품의 정방향 유일성도 안 깨진다.
# ---------------------------------------------------------------------------

ENTITIES += [
    # 제동 계통
    Entity("sys_brake", "제동 시스템", "system"),
    Entity("cmp_bcu", "제동 제어기", "component"),
    Entity("cmp_regen", "회생제동 모듈", "component"),
    Entity("cmp_booster", "전동 부스터", "component"),
    Entity("spec_brake_press", "제동 압력 180bar", "spec"),
    Entity("spec_regen_power", "회생 출력 70kW", "spec"),
    # 공조 계통
    Entity("sys_hvac", "공조 시스템", "system"),
    Entity("cmp_compressor", "전동 컴프레서", "component"),
    Entity("cmp_ptc", "PTC 히터", "component"),
    Entity("cmp_hvac_ecu", "공조 제어기", "component"),
    Entity("spec_comp_cap", "냉방 용량 6.5kW", "spec"),
    Entity("spec_ptc_power", "히터 출력 5kW", "spec"),
    # 조향 계통
    Entity("sys_steer", "조향 시스템", "system"),
    Entity("cmp_mdps", "전동식 조향장치", "component"),
    Entity("cmp_steer_sensor", "조향각 센서", "component"),
    Entity("spec_steer_torque", "조향 보조 토크 45Nm", "spec"),
    Entity("spec_steer_angle", "조향각 측정 범위 720도", "spec"),
    # 신규 공급사
    Entity("sup_eta", "에타브레이크", "supplier"),
    Entity("sup_theta", "세타써멀", "supplier"),
    Entity("sup_iota", "이오타스티어", "supplier"),
]

RELATIONS += [
    # 제동 계통 (인덱스 48~)
    Relation("sys_brake", "CONTAINS", "cmp_bcu"),
    Relation("sys_brake", "CONTAINS", "cmp_regen"),
    Relation("sys_brake", "CONTAINS", "cmp_booster"),
    Relation("cmp_regen", "CONNECTS_TO", "cmp_inverter"),  # 구동 계통과 교차
    Relation("cmp_booster", "CONNECTS_TO", "cmp_bcu"),
    Relation("cmp_bcu", "HAS_SPEC", "spec_brake_press"),
    Relation("cmp_regen", "HAS_SPEC", "spec_regen_power"),
    Relation("cmp_bcu", "SUPPLIED_BY", "sup_eta"),
    Relation("cmp_booster", "SUPPLIED_BY", "sup_eta"),
    # 공조 계통
    Relation("sys_hvac", "CONTAINS", "cmp_compressor"),
    Relation("sys_hvac", "CONTAINS", "cmp_ptc"),
    Relation("sys_hvac", "CONTAINS", "cmp_hvac_ecu"),
    Relation("cmp_compressor", "CONNECTS_TO", "cmp_chiller"),  # 열관리 계통과 교차
    Relation("cmp_hvac_ecu", "CONNECTS_TO", "cmp_compressor"),
    Relation("cmp_compressor", "HAS_SPEC", "spec_comp_cap"),
    Relation("cmp_ptc", "HAS_SPEC", "spec_ptc_power"),
    Relation("cmp_compressor", "SUPPLIED_BY", "sup_theta"),
    Relation("cmp_ptc", "SUPPLIED_BY", "sup_theta"),
    # 조향 계통
    Relation("sys_steer", "CONTAINS", "cmp_mdps"),
    Relation("sys_steer", "CONTAINS", "cmp_steer_sensor"),
    Relation("cmp_steer_sensor", "CONNECTS_TO", "cmp_mdps"),
    Relation("cmp_mdps", "HAS_SPEC", "spec_steer_torque"),
    Relation("cmp_steer_sensor", "HAS_SPEC", "spec_steer_angle"),
    Relation("cmp_mdps", "SUPPLIED_BY", "sup_iota"),
    Relation("cmp_steer_sensor", "SUPPLIED_BY", "sup_iota"),
]

ENTITY_BY_ID: dict[str, Entity] = {e.eid: e for e in ENTITIES}


# ---------------------------------------------------------------------------
# 문서 생성 — 관계 1개 = 문장 1개, 청크(=페이지) 1개에 관계 여러 개
# ---------------------------------------------------------------------------

_TEMPLATES = {
    "CONTAINS": "{head}은(는) {tail}을(를) 포함한다.",
    "CONNECTS_TO": "{head}은(는) {tail}과(와) 연결된다.",
    "HAS_SPEC": "{head}의 사양은 {tail}이다.",
    "SUPPLIED_BY": "{head}은(는) {tail}에서 공급한다.",
}

# 청크 구성: (문서명, 페이지, 제목, 담을 관계 인덱스들)
# 의도적으로 **관계를 여러 청크에 흩어 놓는다.** 멀티홉 질의의 답이 한 청크에
# 모여 있으면 플랫 검색으로도 풀려서 비교가 무의미해지기 때문이다.
_CHUNK_PLAN: list[tuple[str, int, str, list[int]]] = [
    ("battery_manual.pdf", 1, "배터리 시스템 개요", [2, 3]),
    ("battery_manual.pdf", 2, "BMS 구성", [0, 1, 10, 11]),
    ("battery_manual.pdf", 3, "배터리팩 사양", [12, 13, 15]),
    ("battery_manual.pdf", 4, "셀 사양", [14]),
    ("battery_manual.pdf", 5, "컨택터 사양", [16]),
    ("thermal_manual.pdf", 1, "열관리 개요", [4, 5]),
    ("thermal_manual.pdf", 2, "칠러 연결", [6]),
    ("thermal_manual.pdf", 3, "펌프 사양", [19]),
    ("adas_manual.pdf", 1, "ADAS 개요", [7]),
    ("adas_manual.pdf", 2, "센서 연결", [8, 9]),
    ("adas_manual.pdf", 3, "레이더 사양", [17]),
    ("adas_manual.pdf", 4, "카메라 사양", [18]),
    ("supply_chain.pdf", 1, "공급사 — 전장", [20]),
    ("supply_chain.pdf", 2, "공급사 — 배터리/열관리", [21, 24]),
    ("supply_chain.pdf", 3, "공급사 — 센서", [22, 23]),
    # ---- 확장(2026-07-29): 신규 도메인 ----
    ("charging_manual.pdf", 1, "충전 시스템 개요", [25, 26]),
    ("charging_manual.pdf", 2, "충전 경로 구성", [27, 28]),
    ("charging_manual.pdf", 3, "충전 출력 사양", [29, 30]),
    ("charging_manual.pdf", 4, "충전 계통 공급사", [31, 32]),
    ("drive_manual.pdf", 1, "구동 시스템 개요", [33, 34]),
    ("drive_manual.pdf", 2, "동력 전달 경로", [35, 36]),
    ("drive_manual.pdf", 3, "모터 사양", [37, 38]),
    ("drive_manual.pdf", 4, "감속기 사양", [39]),
    ("drive_manual.pdf", 5, "구동 계통 공급사", [40, 41]),
    ("infotainment_manual.pdf", 1, "인포테인먼트 개요", [42, 43]),
    ("infotainment_manual.pdf", 2, "디스플레이·통신 사양", [44, 45]),
    ("infotainment_manual.pdf", 3, "인포테인먼트 공급사", [46, 47]),
    # ---- 확장 2차(2026-07-30): 2-hop 경로 확보용 계통 3개 ----
    # 개요/연결/사양/공급사를 **다른 페이지로 쪼갠다** — 한 페이지에 몰면 멀티홉 질의의
    # gold 두 개가 같은 청크가 되어 생성기가 그 질의를 버린다(=경로를 늘린 의미가 없다).
    ("brake_manual.pdf", 1, "제동 시스템 개요", [48, 49, 50]),
    ("brake_manual.pdf", 2, "제동 연결 구조", [51, 52]),
    ("brake_manual.pdf", 3, "제동 사양", [53, 54]),
    ("brake_manual.pdf", 4, "제동 계통 공급사", [55, 56]),
    ("hvac_manual.pdf", 1, "공조 시스템 개요", [57, 58, 59]),
    ("hvac_manual.pdf", 2, "공조 연결 구조", [60, 61]),
    ("hvac_manual.pdf", 3, "공조 사양", [62, 63]),
    ("hvac_manual.pdf", 4, "공조 계통 공급사", [64, 65]),
    ("steering_manual.pdf", 1, "조향 시스템 개요", [66, 67]),
    ("steering_manual.pdf", 2, "조향 연결 구조", [68]),
    ("steering_manual.pdf", 3, "조향 사양", [69, 70]),
    ("steering_manual.pdf", 4, "조향 계통 공급사", [71, 72]),
]


# ---------------------------------------------------------------------------
# distractor — "정답이 아닌데 정답처럼 보이는" 청크
#
# ## 왜 필요한가
# 확장 전 코퍼스는 모든 청크가 gold 관계를 하나씩 갖고 있었다. 즉 **오답 후보가 전부
# 다른 주제**여서, 질의어와 조금만 겹쳐도 정답이 위로 올라왔다. 실문서에는 같은 부품을
# 말하면서 답은 아닌 페이지(주의사항·정비절차·구형 사양·용어집)가 훨씬 많다.
#
# ## 설계에서 가장 중요한 결정
# distractor에 **엔티티를 그대로 넣는다**(states만 비운다). 엔티티를 빼면 GraphRetriever가
# 이 청크들을 아예 점수화하지 않아 그래프만 무풍지대에 놓인다 — 그러면 "그래프가 이겼다"가
# 코퍼스 설계의 산물이 되어버린다. 어휘로도 그래프로도 똑같이 헷갈리게 만들어야 공정하다.
#
# 가장 독한 유형은 '구형 모델 사양'이다. q01(공칭 전압)의 정답 청크와 어휘가 거의 같고
# 숫자만 다르다 — 검색기가 값을 구분하지 못하면 정확히 여기에 걸린다.
#
# states=[]라 relation_home에 잡히지 않으므로 어떤 질의의 gold도 될 수 없다.
# ---------------------------------------------------------------------------

# (문서명, 페이지, 제목, 본문, 언급 엔티티)
_DISTRACTOR_PLAN: list[tuple[str, int, str, str, list[str]]] = [
    # --- 구형 모델 사양: 어휘가 거의 같고 숫자만 다르다(최난도) ---
    ("legacy_spec.pdf", 1, "구형 모델 참고 — 배터리",
     "이전 세대 고전압 배터리팩의 공칭 전압은 350V였다. 현행 사양과 혼동하지 않도록 주의한다.",
     ["cmp_pack"]),
    ("legacy_spec.pdf", 2, "구형 모델 참고 — 셀",
     "구형 리튬이온 셀의 전압은 3.2V로 현행 대비 낮다. 교체 시 호환되지 않는다.",
     ["cmp_cell"]),
    ("legacy_spec.pdf", 3, "구형 모델 참고 — 레이더",
     "구형 전방 레이더의 탐지 거리는 150m였다. 현행 부품과 장착 브래킷이 다르다.",
     ["cmp_radar"]),
    ("legacy_spec.pdf", 4, "구형 모델 참고 — 카메라",
     "구형 전방 카메라의 화각은 90도였다. 현행 부품으로 대체 장착할 수 없다.",
     ["cmp_camera"]),
    ("legacy_spec.pdf", 5, "구형 모델 참고 — 컨택터",
     "구형 메인 컨택터의 최대 방전 전류는 400A였다. 정격이 낮아 현행 고전압 배터리팩에 쓰지 않는다.",
     ["cmp_contactor", "cmp_pack"]),
    ("legacy_spec.pdf", 6, "구형 모델 참고 — 냉각",
     "구형 냉각수 펌프의 유량은 9L/min였다. 현행 열관리 시스템에는 부족하다.",
     ["cmp_pump", "sys_thermal"]),
    # --- 안전 주의사항: 부품을 말하지만 사양·구성 정보가 없다 ---
    ("safety_guide.pdf", 1, "고전압 취급 주의",
     "고전압 배터리팩 취급 시 절연 장갑과 보호구를 착용한다. 메인 컨택터가 개방된 것을 "
     "확인한 뒤 작업을 시작한다.", ["cmp_pack", "cmp_contactor"]),
    ("safety_guide.pdf", 2, "셀 취급 주의",
     "리튬이온 셀은 충격과 관통에 취약하다. 셀 모듈을 분해할 때 금속 공구가 단자에 닿지 "
     "않도록 한다.", ["cmp_cell", "cmp_module"]),
    ("safety_guide.pdf", 3, "냉각 계통 주의",
     "배터리 칠러와 냉각수 펌프는 가동 직후 고온이다. 열관리 시스템 압력이 해제된 뒤 "
     "개방한다.", ["cmp_chiller", "cmp_pump", "sys_thermal"]),
    ("safety_guide.pdf", 4, "센서 취급 주의",
     "전방 레이더와 전방 카메라는 정전기에 민감하다. ADAS 제어기 커넥터를 분리한 상태로 "
     "작업한다.", ["cmp_radar", "cmp_camera", "cmp_adas_ecu"]),
    ("safety_guide.pdf", 5, "충전 중 주의",
     "차량 탑재형 충전기 동작 중에는 충전 인렛을 강제로 분리하지 않는다.",
     ["cmp_obc", "cmp_inlet"]),
    # --- 정비/점검 절차: 동작을 서술할 뿐 구성·사양이 없다 ---
    ("service_manual.pdf", 1, "배터리팩 탈거 절차",
     "고전압 배터리팩을 탈거하기 전 배터리 관리 시스템을 서비스 모드로 전환한다. "
     "리프트로 지지한 뒤 체결 볼트를 순서대로 푼다.", ["cmp_pack", "sys_bms"]),
    ("service_manual.pdf", 2, "BMS 제어기 점검",
     "BMS 제어기의 진단 커넥터에 스캐너를 연결해 고장 코드를 읽는다. 코드 삭제 후 "
     "재시동하여 재발 여부를 확인한다.", ["cmp_bms_ecu"]),
    ("service_manual.pdf", 3, "냉각수 보충 절차",
     "냉각수 펌프를 정지한 상태에서 리저버 상한선까지 보충한다. 배터리 칠러 쪽 공기를 "
     "배출한 뒤 재가동한다.", ["cmp_pump", "cmp_chiller"]),
    ("service_manual.pdf", 4, "센서 교정 절차",
     "전방 카메라 교체 후에는 반드시 에이밍 교정을 수행한다. 전방 레이더도 동일하게 "
     "수평 정렬을 확인한다.", ["cmp_camera", "cmp_radar"]),
    ("service_manual.pdf", 5, "구동계 점검",
     "구동 모터와 감속기 사이 이음 발생 시 마운트 체결 상태를 먼저 확인한다.",
     ["cmp_motor", "cmp_reducer"]),
    ("service_manual.pdf", 6, "충전 계통 점검",
     "충전 제어기 통신 이상 시 충전 인렛 접점 오염을 먼저 확인한다.",
     ["cmp_ccu", "cmp_inlet"]),
    # --- 용어집: 이름만 반복해 어휘 유사도를 크게 올린다 ---
    ("glossary.pdf", 1, "약어 — 배터리",
     "BMS는 배터리 관리 시스템을 가리킨다. 팩은 고전압 배터리팩, 모듈은 셀 모듈을 뜻한다.",
     ["sys_bms", "cmp_pack", "cmp_module"]),
    ("glossary.pdf", 2, "약어 — 주행보조",
     "ADAS는 첨단 운전자 보조 시스템의 약어이며, 여기서 제어기는 ADAS 제어기를 가리킨다.",
     ["sys_adas", "cmp_adas_ecu"]),
    ("glossary.pdf", 3, "약어 — 충전·구동",
     "OBC는 차량 탑재형 충전기, 인버터는 직류를 교류로 변환하는 장치를 뜻한다.",
     ["cmp_obc", "cmp_inverter"]),
    ("glossary.pdf", 4, "단위 표기 규칙",
     "전압은 V, 용량은 kWh, 전류는 A, 유량은 L/min으로 표기한다. 사양 표의 값은 상온 기준이다.",
     []),
    # --- 문서 메타: 엔티티 없이 어휘만 겹치는 유형 ---
    ("doc_meta.pdf", 1, "개정 이력",
     "본 매뉴얼은 3차 개정본이다. 배터리·열관리·주행보조 계통의 사양 표가 갱신되었다.",
     []),
    ("doc_meta.pdf", 2, "적용 범위",
     "본 문서는 해당 차종의 전기 계통 정비에 한정해 적용한다. 사양은 공급사 사정에 따라 "
     "변경될 수 있다.", []),
    ("doc_meta.pdf", 3, "공급사 일반",
     "공급사별 부품 번호는 별도 카탈로그를 따른다. 공급 계약 변경 시 부품 번호가 바뀔 수 있다.",
     []),
    # --- 확장 2차(2026-07-30): 신규 계통에도 **같은 비율로** distractor를 붙인다 ---
    # 이걸 빠뜨리면 신규 계통 질의만 방해 문서가 없는 쉬운 환경에서 평가된다. 그러면
    # "평가셋을 늘렸더니 점수가 올랐다"가 실력이 아니라 **코퍼스 편향**이 된다.
    # 기존 계통과 같은 4유형(구형 사양·안전·정비·용어집)을 그대로 맞춰 넣는다.
    ("legacy_spec.pdf", 7, "구형 모델 참고 — 제동",
     "구형 제동 제어기의 제동 압력은 150bar였다. 현행 사양과 배관 규격이 다르다.",
     ["cmp_bcu"]),
    ("legacy_spec.pdf", 8, "구형 모델 참고 — 공조",
     "구형 전동 컴프레서의 냉방 용량은 5.0kW였다. 현행 공조 시스템 대비 낮다.",
     ["cmp_compressor", "sys_hvac"]),
    ("legacy_spec.pdf", 9, "구형 모델 참고 — 조향",
     "구형 전동식 조향장치의 조향 보조 토크는 35Nm였다. 현행 부품과 장착 규격이 다르다.",
     ["cmp_mdps"]),
    ("safety_guide.pdf", 6, "제동 계통 주의",
     "회생제동 모듈 점검 시 고전압 차단을 먼저 확인한다. 전동 부스터 내부 압력이 해제된 "
     "뒤 작업한다.", ["cmp_regen", "cmp_booster"]),
    ("safety_guide.pdf", 7, "공조 계통 주의",
     "전동 컴프레서는 고전압으로 구동된다. PTC 히터 커넥터를 분리한 뒤 점검한다.",
     ["cmp_compressor", "cmp_ptc"]),
    ("service_manual.pdf", 7, "조향 계통 점검",
     "조향각 센서 교체 후 영점 설정을 수행한다. 전동식 조향장치 경고등이 소등되는지 "
     "확인한다.", ["cmp_steer_sensor", "cmp_mdps"]),
    ("service_manual.pdf", 8, "제동 계통 점검",
     "제동 제어기 진단 시 회생제동 모듈의 협조 제어 이력을 함께 확인한다.",
     ["cmp_bcu", "cmp_regen"]),
    ("glossary.pdf", 5, "약어 — 제동·조향",
     "MDPS는 전동식 조향장치를 가리킨다. 회생제동은 감속 에너지를 전기로 회수하는 방식이다.",
     ["cmp_mdps"]),
    ("glossary.pdf", 6, "약어 — 공조",
     "PTC 히터는 저항 발열식 난방 장치이며, 공조 제어기가 냉난방을 통합 제어한다.",
     ["cmp_ptc", "cmp_hvac_ecu"]),
]


def _render(rel: Relation) -> str:
    head = ENTITY_BY_ID[rel.head].name
    tail = ENTITY_BY_ID[rel.tail].name
    return _TEMPLATES[rel.rtype].format(head=head, tail=tail)


def build_corpus(with_distractors: bool = True) -> list[Chunk]:
    """그래프에서 문서 청크를 결정적으로 생성. 같은 입력 → 항상 같은 출력.

    with_distractors=False면 관계를 서술하는 청크만 만든다. 1차 실험(2026-07-27)의
    코퍼스 조건을 재현하거나, distractor가 점수를 얼마나 끌어내리는지 분리 측정할 때 쓴다.
    """
    chunks: list[Chunk] = []
    for doc, page, title, rel_idx in _CHUNK_PLAN:
        rels = [RELATIONS[i] for i in rel_idx]
        body = " ".join(_render(r) for r in rels)
        eids: list[str] = []
        for r in rels:
            for e in (r.head, r.tail):
                if e not in eids:
                    eids.append(e)
        chunks.append(
            Chunk(
                chunk_id=f"{doc}#p{page}",
                doc_name=doc,
                page_num=page,
                text=f"[{title}] {body}",
                states=rels,
                entities=eids,
            )
        )
    if with_distractors:
        for doc, page, title, body, eids in _DISTRACTOR_PLAN:
            for e in eids:
                if e not in ENTITY_BY_ID:
                    raise ValueError(f"distractor {doc}#p{page}: 알 수 없는 엔티티 {e}")
            chunks.append(
                Chunk(
                    chunk_id=f"{doc}#p{page}",
                    doc_name=doc,
                    page_num=page,
                    text=f"[{title}] {body}",
                    states=[],  # gold가 될 수 없다
                    entities=list(eids),
                )
            )
    return chunks


def chunk_index(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {c.chunk_id: c for c in chunks}


def relation_home(chunks: list[Chunk]) -> dict[tuple[str, str, str], str]:
    """관계 → 그 관계를 서술하는 청크 id. 멀티홉 gold 산출에 쓴다.

    한 관계는 정확히 한 청크에만 등장하도록 _CHUNK_PLAN을 짰다(불변식은 테스트로 고정).
    """
    home: dict[tuple[str, str, str], str] = {}
    for c in chunks:
        for r in c.states:
            key = (r.head, r.rtype, r.tail)
            if key in home:
                raise ValueError(f"관계가 여러 청크에 중복 서술됨: {key}")
            home[key] = c.chunk_id
    return home
