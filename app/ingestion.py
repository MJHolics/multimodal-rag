"""PDF 인제스천: 이미지 변환 + 텍스트/표 추출 + 청킹"""
import fitz
import pdfplumber
import re
import json
from pathlib import Path
from dataclasses import dataclass

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class PageChunk:
    doc_name: str
    page_num: int
    chunk_id: str
    text: str
    image_path: str
    content_type: str
    has_table: bool
    char_count: int


def pdf_to_images(pdf_path: Path, dpi: int = 200) -> list[Path]:
    doc = fitz.open(str(pdf_path))
    img_dir = OUTPUT_DIR / pdf_path.stem
    img_dir.mkdir(parents=True, exist_ok=True)
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    paths = []
    for i, page in enumerate(doc):
        p = img_dir / f"page_{i:03d}.png"
        page.get_pixmap(matrix=mat, alpha=False).save(str(p))
        paths.append(p)
    doc.close()
    return paths


def _table_to_markdown(table):
    if not table or not table[0]:
        return ""
    rows = [[str(c).strip() if c else "" for c in row] for row in table]
    widths = [max(len(rows[r][c]) for r in range(len(rows))) for c in range(len(rows[0]))]
    lines = []
    for i, row in enumerate(rows):
        lines.append("| " + " | ".join(cell.ljust(widths[c]) for c, cell in enumerate(row)) + " |")
        if i == 0:
            lines.append("| " + " | ".join("-" * widths[c] for c in range(len(row))) + " |")
    return "\n".join(lines)


def ingest_pdf(pdf_path: Path) -> list[PageChunk]:
    image_paths = pdf_to_images(pdf_path)
    doc = fitz.open(str(pdf_path))

    all_blocks = []
    for page_num, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                texts, sizes = [], []
                for span in line.get("spans", []):
                    t = span["text"].strip()
                    if t:
                        texts.append(t)
                        sizes.append(span["size"])
                if texts:
                    all_blocks.append((page_num, " ".join(texts), sum(sizes) / len(sizes)))

    mean_size = sum(b[2] for b in all_blocks) / len(all_blocks) if all_blocks else 11

    page_texts: dict[int, list] = {}
    for page_num, text, size in all_blocks:
        page_texts.setdefault(page_num, []).append(
            f"## {text}" if size > mean_size * 1.2 else text
        )

    page_tables: dict[int, list] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            mds = [_table_to_markdown(t) for t in page.extract_tables() if t]
            if mds:
                page_tables[i] = mds

    doc.close()

    chunks = []
    for page_num in range(len(image_paths)):
        parts = page_texts.get(page_num, [])
        page_text = "\n".join(parts)
        t_mds = page_tables.get(page_num, [])
        if t_mds:
            page_text += "\n\n" + "\n\n".join(t_mds)

        tlen = sum(len(m) for m in t_mds)
        ctype = (
            "table_heavy" if tlen > len(page_text) * 0.5
            else ("text_heavy" if len(page_text) > 200 else "mixed")
        )

        chunks.append(PageChunk(
            doc_name=pdf_path.stem,
            page_num=page_num,
            chunk_id=f"{pdf_path.stem}_p{page_num:03d}",
            text=page_text.strip(),
            image_path=str(image_paths[page_num]),
            content_type=ctype,
            has_table=len(t_mds) > 0,
            char_count=len(page_text)
        ))
    return chunks
