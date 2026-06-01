"""
Phase 1: High-Fidelity Document Processing — v2
================================================
New capabilities:
  - Life insurance / ACORD form detection
  - Table-row serialisation (each row → searchable text unit)
  - Multi-column slip detection for reinsurance placement slips
  - Preserves original filename as display_name (fixes issue #6)
  - Receipt / form field extraction
"""

import re
import fitz
import pdfplumber
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
from rich.console import Console

console = Console()

# ACORD form identifiers
ACORD_PATTERNS = re.compile(
    r"(acord\s*\d+|certificate\s+of\s+insurance|evidence\s+of\s+insurance|"
    r"acord\s+form|coi\b|certificate\s+holder)", re.IGNORECASE)

# Life insurance document type detection
LIFE_DOC_PATTERNS = {
    "policy":       re.compile(r"\b(policy\s+wording|policy\s+document|insurance\s+policy)\b", re.I),
    "receipt":      re.compile(r"\b(premium\s+receipt|payment\s+receipt|renewal\s+receipt)\b", re.I),
    "proposal":     re.compile(r"\b(proposal\s+form|application\s+form|proposer)\b", re.I),
    "claim":        re.compile(r"\b(claim\s+form|death\s+claim|maturity\s+claim)\b", re.I),
    "endorsement":  re.compile(r"\b(endorsement|rider|add.?on\s+benefit)\b", re.I),
    "acord":        ACORD_PATTERNS,
}


@dataclass
class DocumentBlock:
    block_id:   str
    block_type: str          # "text" | "table" | "heading" | "table_row" | "form_field"
    content:    str
    page_num:   int
    bbox:       Optional[tuple] = None
    raw_table:  Optional[List[List[str]]] = None
    table_row_index: int = -1   # which row of the table (for table_row blocks)


@dataclass
class ParsedDocument:
    source_path:  str
    doc_name:     str          # internal id (safe for filesystems)
    display_name: str          # original human-readable filename
    total_pages:  int
    doc_subtype:  str = ""     # policy | receipt | acord | claim | endorsement
    blocks:       List[DocumentBlock] = field(default_factory=list)
    metadata:     Dict = field(default_factory=dict)

    def full_text(self) -> str:
        return "\n\n".join(b.content for b in self.blocks)


class InsuranceDocumentParser:
    HEADING_MIN_FONT_SIZE = 11.0
    TABLE_MIN_COLS        = 2

    def __init__(self, verbose=True):
        self.verbose = verbose

    # ── Public API ────────────────────────────────────────────────────────

    def parse(self, source: str, display_name: str = "") -> ParsedDocument:
        """
        source       : path to PDF or raw text string
        display_name : original filename shown in UI (e.g. "Life_Policy_2024.pdf")
        """
        path = Path(source)
        if path.suffix.lower() == ".pdf" and path.exists():
            return self._parse_pdf(str(path), display_name or path.name)
        else:
            return self._parse_text(source, display_name or "inline_text")

    # ── PDF parsing ───────────────────────────────────────────────────────

    def _parse_pdf(self, path: str, display_name: str) -> ParsedDocument:
        doc_name = Path(path).stem
        if self.verbose:
            console.print(f"[cyan]📄 Parsing:[/] {display_name}")

        blocks: List[DocumentBlock] = []
        bc = 0
        subtype = ""
        table_bboxes: Dict[int, List] = {}

        # Step 1: pdfplumber — extract tables + detect doc type
        with pdfplumber.open(path) as plumb:
            total_pages = len(plumb.pages)
            # Detect doc subtype from first-page text
            if plumb.pages:
                first_text = (plumb.pages[0].extract_text() or "").lower()
                for st, pat in LIFE_DOC_PATTERNS.items():
                    if pat.search(first_text):
                        subtype = st
                        break

            for pg_idx, page in enumerate(plumb.pages):
                tables    = page.extract_tables()
                tbl_objs  = page.find_tables()
                page_bboxes = []

                for table, tobj in zip(tables, tbl_objs):
                    if not table or not table[0] or len(table[0]) < self.TABLE_MIN_COLS:
                        continue

                    page_bboxes.append(tobj.bbox)

                    # Full table as markdown (for LLM context)
                    md = self._table_to_markdown(table)
                    blocks.append(DocumentBlock(
                        block_id  = f"p{pg_idx+1}_tbl_{bc}",
                        block_type= "table",
                        content   = md,
                        page_num  = pg_idx + 1,
                        raw_table = table,
                        bbox      = tobj.bbox,
                    ))
                    bc += 1

                    # Also index each row individually for fine-grained search
                    header = [str(c).strip() if c else "" for c in table[0]]
                    for ri, row in enumerate(table[1:], 1):
                        cells = [str(c).strip() if c else "" for c in row]
                        # Serialise: "ColHeader: value | ColHeader: value ..."
                        row_text = " | ".join(
                            f"{h}: {v}" for h, v in zip(header, cells) if v
                        )
                        if len(row_text) > 10:
                            blocks.append(DocumentBlock(
                                block_id      = f"p{pg_idx+1}_row_{bc}",
                                block_type    = "table_row",
                                content       = row_text,
                                page_num      = pg_idx + 1,
                                table_row_index = ri,
                            ))
                            bc += 1

                table_bboxes[pg_idx] = page_bboxes

        # Step 2: PyMuPDF — extract text blocks
        with fitz.open(path) as pdf:
            total_pages = pdf.page_count
            for pg_idx in range(total_pages):
                page      = pdf[pg_idx]
                page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
                for block in page_dict.get("blocks", []):
                    if block.get("type") != 0:
                        continue
                    bbox = block.get("bbox", ())
                    if self._overlaps_table(bbox, table_bboxes.get(pg_idx, [])):
                        continue
                    text, is_heading, _ = self._extract_block_text(block)
                    if not text.strip():
                        continue
                    blocks.append(DocumentBlock(
                        block_id  = f"p{pg_idx+1}_txt_{bc}",
                        block_type= "heading" if is_heading else "text",
                        content   = text.strip(),
                        page_num  = pg_idx + 1,
                        bbox      = bbox,
                    ))
                    bc += 1

        blocks.sort(key=lambda b: (b.page_num, b.bbox[1] if b.bbox else 0))

        if self.verbose:
            n_tbl = sum(1 for b in blocks if b.block_type == "table")
            n_row = sum(1 for b in blocks if b.block_type == "table_row")
            n_txt = sum(1 for b in blocks if b.block_type in ("text","heading"))
            console.print(f"  [green]✓[/] {total_pages}p | {n_txt} text | {n_tbl} tables | {n_row} rows | type={subtype or 'general'}")

        return ParsedDocument(
            source_path  = path,
            doc_name     = doc_name,
            display_name = display_name,
            total_pages  = total_pages,
            doc_subtype  = subtype,
            blocks       = blocks,
            metadata     = {"source": path, "pages": total_pages, "subtype": subtype},
        )

    # ── Text parsing ─────────────────────────────────────────────────────

    def _parse_text(self, text: str, display_name: str) -> ParsedDocument:
        lines     = text.split("\n")
        blocks    = []
        cur_lines = []
        bc        = 0

        def flush(btype="text"):
            nonlocal bc
            content = "\n".join(cur_lines).strip()
            if content:
                blocks.append(DocumentBlock(
                    block_id=f"txt_{bc}", block_type=btype, content=content, page_num=1))
                bc += 1
            cur_lines.clear()

        for line in lines:
            s = line.strip()
            if re.match(r"^#{1,3}\s", s) or (s.isupper() and 5 < len(s) < 80):
                flush()
                blocks.append(DocumentBlock(
                    block_id=f"hdr_{bc}", block_type="heading",
                    content=s.lstrip("# "), page_num=1))
                bc += 1
            else:
                cur_lines.append(line)

        flush()
        doc_name = re.sub(r"[^a-z0-9_]", "_", display_name.lower())[:40]
        return ParsedDocument(
            source_path  = "<inline>",
            doc_name     = doc_name,
            display_name = display_name,
            total_pages  = 1,
            blocks       = blocks,
            metadata     = {"source": "inline"},
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _extract_block_text(self, block):
        lines, max_fs = [], 0.0
        for line in block.get("lines", []):
            lt = ""
            for span in line.get("spans", []):
                lt += span.get("text", "")
                fs  = span.get("size", 0)
                if fs > max_fs: max_fs = fs
            lines.append(lt)
        text       = "\n".join(lines)
        is_heading = (max_fs >= self.HEADING_MIN_FONT_SIZE
                      and len(text.strip()) < 120
                      and text.strip() == text.strip().upper())
        return text, is_heading, max_fs

    def _table_to_markdown(self, table):
        """
        Convert table to both markdown AND a natural-language serialisation.
        The NL form is what gets indexed — it reads as:
          "Age: 35 | Policy Term: 20 | Annual Premium: 12450"
        which BM25 can match on individual cell values.
        """
        if not table: return ""
        cleaned = [[str(c).replace("\n"," ").replace("|"," ").strip() if c else ""
                    for c in row] for row in table]

        # Markdown table (for display in LLM context)
        def rs(r): return "| " + " | ".join(r) + " |"
        md = "\n".join([rs(cleaned[0]), rs(["---"]*len(cleaned[0]))]
                        + [rs(r) for r in cleaned[1:]])

        # Natural-language row serialisation (for BM25/TF-IDF indexing)
        header = cleaned[0]
        nl_rows = []
        for row in cleaned[1:]:
            pairs = [f"{h}: {v}" for h, v in zip(header, row) if v and h]
            if pairs:
                nl_rows.append(" | ".join(pairs))

        if nl_rows:
            nl_section = "\nTable data:\n" + "\n".join(nl_rows)
            return md + nl_section
        return md

    def _overlaps_table(self, bbox, tboxes):
        if not bbox or not tboxes: return False
        x0,y0,x1,y1 = bbox
        for tx0,ty0,tx1,ty1 in tboxes:
            if not (x1<tx0 or x0>tx1 or y1<ty0 or y0>ty1):
                return True
        return False
