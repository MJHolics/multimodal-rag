"""HWP(한글) 문서에서 본문 텍스트를 뽑는다 — 순수 stdlib + olefile.

## 왜 직접 짰나

`rfp-rag`에 이미 4중 폴백(LibreOffice → hwp5 → olefile → 바이너리 패턴) 파서가 있다.
그런데 여기서 필요한 건 **본문 텍스트 하나뿐**이고, 그 파서는 청킹·메타데이터·멀티모달까지
묶여 있어 통째로 끌어오면 의존성이 딸려온다. 그래서 HWP5 레코드 구조를 직접 읽는
최소 경로만 구현했다(외부 프로그램 불필요, 결정적).

## HWP5 구조 (필요한 만큼만)

- OLE 복합문서. `FileHeader` 37번째 바이트의 최하위 비트가 **압축 여부**.
- 본문은 `BodyText/Section*` 스트림들. 압축이면 raw deflate(`zlib.decompress(d, -15)`).
- 스트림 안은 레코드 열: 4바이트 헤더에 tag_id(10bit) · level(10bit) · size(12bit).
  size가 0xFFF이면 뒤따르는 4바이트가 실제 길이다.
- 본문 문자열은 태그 **67(HWPTAG_PARA_TEXT)**, 인코딩은 UTF-16LE.
- 그 안에 제어 문자(0~31)가 섞이는데, 일부는 8바이트를 더 차지하는 **확장 제어 문자**다.
  이걸 건너뛰지 않으면 쓰레기 문자가 본문에 섞인다(실제로 처음에 그랬다).
"""
from __future__ import annotations

import re
import zlib
from pathlib import Path

# 8 UTF-16 단위를 차지하는 제어 문자 — **두 부류 모두** 건너뛰어야 한다.
#
#   확장(extended): 표·그림·각주 같은 개체
#   인라인(inline) : 탭(9), 필드 시작/끝 등
#
# 처음엔 확장만 건너뛰었더니 목차 줄이 "추진개요螨ȃ 3"처럼 깨져 나왔다.
# 탭(9)이 인라인 제어라 뒤따르는 7단위가 본문 문자로 새어 나온 것이다.
# 실문서(공공조달 RFP)는 목차와 표가 많아 이 경로를 반드시 지나간다.
_EXTENDED = {1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23}
_INLINE = {4, 5, 6, 7, 8, 9, 19, 20}
_SKIP8 = _EXTENDED | _INLINE
# 문단/줄 구분으로 볼 제어 문자
_BREAK = {10, 13}

_PARA_TEXT_TAG = 67


def _records(buf: bytes):
    """레코드 열을 (tag_id, level, payload)로 순회."""
    pos, n = 0, len(buf)
    while pos + 4 <= n:
        header = int.from_bytes(buf[pos : pos + 4], "little")
        tag_id = header & 0x3FF
        level = (header >> 10) & 0x3FF
        size = (header >> 20) & 0xFFF
        pos += 4
        if size == 0xFFF:
            if pos + 4 > n:
                break
            size = int.from_bytes(buf[pos : pos + 4], "little")
            pos += 4
        if pos + size > n:
            break
        yield tag_id, level, buf[pos : pos + size]
        pos += size


def _decode_para(payload: bytes) -> str:
    """PARA_TEXT 페이로드 → 문자열. 제어 문자는 건너뛰고 **서로게이트 쌍은 합친다**.

    서로게이트 처리가 왜 필요한가 (2026-07-31에 잡은 버스):
    본문은 UTF-16LE이라 BMP 밖 문자(희귀 한자·이모지)는 상위/하위 서로게이트 **두 단위**로
    나뉘어 저장된다. 처음엔 각 단위를 그대로 `chr()`로 만들어 붙였는데, 그러면 문자열에
    짝 없는 서로게이트가 남는다. 파이썬은 이걸 담고 있을 수는 있어도 **인코딩할 수 없어서**,
    나중에 BGE-M3 토크나이저가 청크 7,865개 중 260개에서 `TextEncodeInput must be ...`로
    죽었다. 파싱 단계의 조용한 손상이 임베딩 단계에서 터진 셈이다.
    → 쌍이면 합쳐 원래 문자로 복원하고, 짝이 없으면 버린다(복원할 정보가 없다).
    """
    out: list[str] = []
    i, n = 0, len(payload) // 2 * 2
    while i < n:
        code = int.from_bytes(payload[i : i + 2], "little")
        if code in _SKIP8:
            i += 16  # 제어 문자 1 + 딸린 데이터 7 = 8 UTF-16 단위
            continue
        if code in _BREAK:
            out.append("\n")
            i += 2
            continue
        if 0xD800 <= code <= 0xDBFF:  # 상위 서로게이트
            if i + 4 <= n:
                low = int.from_bytes(payload[i + 2 : i + 4], "little")
                if 0xDC00 <= low <= 0xDFFF:
                    out.append(chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)))
                    i += 4
                    continue
            i += 2  # 짝 없는 상위 서로게이트는 버린다
            continue
        if 0xDC00 <= code <= 0xDFFF:  # 짝 없는 하위 서로게이트
            i += 2
            continue
        if code >= 32:
            out.append(chr(code))
        i += 2
    return "".join(out)


def extract_text(path: str | Path) -> str:
    """HWP 파일에서 본문 텍스트를 뽑는다. 실패하면 빈 문자열."""
    import olefile

    p = Path(path)
    if not olefile.isOleFile(str(p)):
        return ""
    ole = olefile.OleFileIO(str(p))
    try:
        header = ole.openstream("FileHeader").read()
        compressed = bool(header[36] & 1)
        sections = sorted(
            (s for s in ole.listdir() if s[0] == "BodyText"),
            key=lambda s: s[-1],
        )
        chunks: list[str] = []
        for s in sections:
            data = ole.openstream(s).read()
            if compressed:
                try:
                    data = zlib.decompress(data, -15)
                except zlib.error:
                    continue
            for tag_id, _level, payload in _records(data):
                if tag_id == _PARA_TEXT_TAG:
                    chunks.append(_decode_para(payload))
        return clean("\n".join(chunks))
    finally:
        ole.close()


def clean(text: str) -> str:
    """공백 정리 — 표에서 온 반복 공백과 빈 줄을 접는다. 순수 함수."""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t ]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return "\n".join(ln.strip() for ln in text.split("\n") if ln.strip())
