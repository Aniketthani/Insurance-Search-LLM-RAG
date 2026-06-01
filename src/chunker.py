"""
Phase 2: Domain-Aware Chunking — v2
=====================================
Enhancements:
  - Life insurance section patterns (ACORD, riders, benefit tables)
  - Table-row chunks propagate to child chunks directly
  - Improved metadata: doc_subtype, display_name stored in each chunk
  - Receipt / form field chunking strategy
"""

import re, hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from src.parser import ParsedDocument, DocumentBlock

# Expanded section heading patterns (Life + P&C + ACORD)
SECTION_RE = re.compile(
    r"^(section|clause|article|schedule|endorsement|exhibit|appendix|part|rider"
    r"|benefit|annexure|addendum)\s+[\dA-Z]"
    r"|^\d+\.\s+[A-Z]"
    r"|^[A-Z][A-Z\s]{4,50}$"
    r"|^(EXCLUSIONS|DEFINITIONS|CONDITIONS|COVERAGE|PREMIUM|DEDUCTIBLE|LIMIT"
    r"|BENEFITS|RIDERS|NOMINEES|SURRENDER|MATURITY|GRACE|LAPSE|REVIVAL"
    r"|ACORD|CERTIFICATE|PROPOSAL|TERMS\s+AND\s+CONDITIONS)",
    re.IGNORECASE,
)

NUMERIC_RE = re.compile(
    r"(\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|M|B|K|lakh|crore))?"
    r"|INR\s*[\d,]+(?:\.\d+)?"
    r"|₹\s*[\d,]+(?:\.\d+)?"
    r"|[\d,]+(?:\.\d+)?\s*%"
    r"|xs\s*\$[\d,]+"
    r"|\d{1,3}(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|lakh|crore))?)"
)


@dataclass
class Chunk:
    chunk_id:      str
    text:          str           # with metadata header
    raw_text:      str           # clean text
    chunk_type:    str           # "child" | "parent"
    parent_id:     Optional[str]
    doc_name:      str
    display_name:  str           # ← NEW: original filename
    page_num:      int
    section_title: str
    lob:           str
    doc_category:  str
    token_count:   int
    numeric_values:List[str]
    metadata:      Dict = field(default_factory=dict)


class InsuranceChunker:
    CHILD_MAX_TOKENS    = 450
    PARENT_TARGET_TOKENS= 1500
    OVERLAP_TOKENS      = 75
    CHARS_PER_TOKEN     = 4

    def __init__(self, lob="Life Insurance", doc_category="Policy", verbose=True):
        self.lob          = lob
        self.doc_category = doc_category
        self.verbose      = verbose

    def chunk(self, parsed: ParsedDocument) -> List[Chunk]:
        sections  = self._split_into_sections(parsed)
        all_chunks: List[Chunk] = []
        dn = getattr(parsed, "display_name", parsed.doc_name)

        for title, blks in sections.items():
            text     = "\n\n".join(b.content for b in blks)
            page     = blks[0].page_num if blks else 1
            parent   = self._make_parent(title, text, page, parsed.doc_name, dn)
            children = self._make_children(parent, title, text, page, parsed.doc_name, dn)

            # Table-row blocks get their own direct child chunk (fine-grained)
            for blk in blks:
                if blk.block_type == "table_row":
                    tc = self._make_table_row_chunk(blk, title, parent.chunk_id,
                                                    parsed.doc_name, dn)
                    children.append(tc)

            all_chunks.append(parent)
            all_chunks.extend(children)

        if self.verbose:
            parents  = sum(1 for c in all_chunks if c.chunk_type=="parent")
            children = sum(1 for c in all_chunks if c.chunk_type=="child")
            console_print(f"  ✓ {parents} sections | {children} child chunks")

        return all_chunks

    # ── Section splitting ─────────────────────────────────────────────────

    def _split_into_sections(self, doc: ParsedDocument) -> Dict[str, List[DocumentBlock]]:
        secs: Dict[str, List[DocumentBlock]] = {}
        cur = "Preamble"
        secs[cur] = []
        for blk in doc.blocks:
            is_sec = (
                blk.block_type == "heading" or
                (blk.block_type == "text" and
                 SECTION_RE.match(blk.content.strip()) and
                 len(blk.content.strip()) < 120)
            )
            if is_sec:
                title = blk.content.strip()[:80]
                base, i = title, 2
                while title in secs: title = f"{base} ({i})"; i+=1
                cur = title
                secs[cur] = []
            else:
                secs[cur].append(blk)
        return {k: v for k, v in secs.items() if v}

    # ── Parent / child builders ───────────────────────────────────────────

    def _make_parent(self, title, text, page, doc_name, display_name) -> Chunk:
        hdr  = self._header(title)
        full = f"{hdr}\n\n{text}"
        cid  = self._cid(doc_name, title, "parent")
        nums = NUMERIC_RE.findall(text)
        return Chunk(
            chunk_id=cid, text=full, raw_text=text, chunk_type="parent",
            parent_id=None, doc_name=doc_name, display_name=display_name,
            page_num=page, section_title=title, lob=self.lob,
            doc_category=self.doc_category,
            token_count=self._te(full),
            numeric_values=[n for n in nums if n.strip()],
            metadata={"doc_name":doc_name,"section":title,"page":page},
        )

    def _make_children(self, parent, title, text, page, doc_name, display_name) -> List[Chunk]:
        sents    = self._split_sentences(text)
        children = []
        window, wt, idx = [], 0, 0

        for sent in sents:
            st = self._te(sent)
            if wt + st > self.CHILD_MAX_TOKENS and window:
                children.append(self._finalize_child(
                    window, title, page, doc_name, display_name, parent.chunk_id, idx))
                idx += 1
                # overlap
                ov, ot = [], 0
                for s in reversed(window):
                    t = self._te(s)
                    if ot + t > self.OVERLAP_TOKENS: break
                    ov.insert(0, s); ot += t
                window, wt = ov, ot
            window.append(sent); wt += st

        if window:
            children.append(self._finalize_child(
                window, title, page, doc_name, display_name, parent.chunk_id, idx))
        return children

    def _finalize_child(self, sents, title, page, doc_name, display_name, parent_id, idx) -> Chunk:
        raw  = " ".join(sents).strip()
        hdr  = self._header(title)
        full = f"{hdr}\n\n{raw}"
        cid  = self._cid(doc_name, title, f"child_{idx}")
        nums = NUMERIC_RE.findall(raw)
        return Chunk(
            chunk_id=cid, text=full, raw_text=raw, chunk_type="child",
            parent_id=parent_id, doc_name=doc_name, display_name=display_name,
            page_num=page, section_title=title, lob=self.lob,
            doc_category=self.doc_category,
            token_count=self._te(full),
            numeric_values=[n for n in nums if n.strip()],
            metadata={"doc_name":doc_name,"section":title,"page":page,"parent_id":parent_id},
        )

    def _make_table_row_chunk(self, blk: DocumentBlock, section_title: str,
                               parent_id: str, doc_name: str, display_name: str) -> Chunk:
        """Make a fine-grained chunk from a single table row for precise table search."""
        raw  = blk.content
        hdr  = self._header(section_title)
        full = f"{hdr}\n\n{raw}"
        cid  = self._cid(doc_name, section_title, f"trow_{blk.block_id}")
        nums = NUMERIC_RE.findall(raw)
        return Chunk(
            chunk_id=cid, text=full, raw_text=raw, chunk_type="child",
            parent_id=parent_id, doc_name=doc_name, display_name=display_name,
            page_num=blk.page_num, section_title=section_title, lob=self.lob,
            doc_category=self.doc_category,
            token_count=self._te(full),
            numeric_values=[n for n in nums if n.strip()],
            metadata={"doc_name":doc_name,"table_row":True},
        )

    # ── Helpers ──────────────────────────────────────────────────────────

    def _header(self, section):
        return (f"[Line of Business: {self.lob}] "
                f"[Document Type: {self.doc_category}] "
                f"[Section: {section}]")

    def _split_sentences(self, text):
        text   = re.sub(r"\n+", " \n ", text)
        parts  = re.split(r"(?<=[.?!])\s+(?=[A-Z\d\(])", text)
        final  = []
        for p in parts:
            final.extend(re.split(r"\n\s*(?=\d+\.\s|[A-Z]{2})", p))
        return [p.strip() for p in final if p.strip()]

    def _te(self, text): return max(1, len(text) // self.CHARS_PER_TOKEN)
    def _cid(self, doc_name, title, suffix):
        return hashlib.md5(f"{doc_name}::{title}::{suffix}".encode()).hexdigest()[:16]


def console_print(msg):
    try:
        from rich.console import Console
        Console().print(msg)
    except Exception:
        print(msg)
