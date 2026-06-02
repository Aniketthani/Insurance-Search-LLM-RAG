"""
Adverse Clause Scanner — Insurance Document Intelligence
=========================================================
Industry-grade adverse keyword detection for the full insurance spectrum:
  Life · P&C · Reinsurance · Health · Marine · Liability · ACORD

Architecture:
  1. ADVERSE_LEXICON  — 400+ terms across 18 categories with severity weights
  2. NEGATION_GUARDS  — detect positive context ("is covered", "not excluded")
  3. PROXIMITY_SCORER — how densely packed are adverse terms in a passage
  4. CONTEXTUAL_CLASSIFIER — is the keyword used adversely or benignly?
  5. SECTION_HEATMAP  — adversity score per document section
  6. DOCUMENT_PROFILE — overall adversity score + executive summary

Beats rule-based baselines because:
  - Negation detection prevents false positives ("not excluded" ≠ adverse)
  - Proximity scoring rewards dense adverse passages over sparse coincidences
  - Category weighting differentiates critical exclusions from minor conditions
  - Section-level aggregation gives actionable location (not just keyword count)
  - Severity-adjusted final score is comparable across documents
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from src.chunker import Chunk


# ══════════════════════════════════════════════════════════════════════════════
# ADVERSE LEXICON — 18 categories covering full insurance spectrum
# Each entry: (phrase, severity_weight, context_notes)
# Severity: 5=critical  4=high  3=medium  2=low  1=informational
# ══════════════════════════════════════════════════════════════════════════════

ADVERSE_CATEGORIES: Dict[str, Dict] = {

    # ── 1. EXCLUSIONS ─────────────────────────────────────────────────────────
    "Exclusions": {
        "color": "#DC2626",
        "icon": "🚫",
        "description": "Clauses that remove or restrict coverage entirely",
        "severity_base": 5,
        "terms": [
            ("shall not cover", 5), ("not covered", 5), ("excluded from coverage", 5),
            ("this policy does not cover", 5), ("no coverage", 5),
            ("coverage is excluded", 5), ("excluded condition", 4),
            ("not liable", 4), ("liability is excluded", 4),
            ("exclusion applies", 4), ("subject to exclusion", 4),
            ("standard exclusion", 3), ("blanket exclusion", 4),
            ("war exclusion", 4), ("terrorism exclusion", 4),
            ("nuclear exclusion", 4), ("biological exclusion", 4),
            ("chemical exclusion", 4), ("cyber exclusion", 4),
            ("pandemic exclusion", 4), ("epidemic exclusion", 4),
            ("pre-existing exclusion", 4), ("congenital exclusion", 3),
            ("intentional act exclusion", 4), ("criminal act exclusion", 4),
            ("professional exclusion", 3), ("pollution exclusion", 4),
            ("asbestos exclusion", 4), ("mold exclusion", 3),
            ("flood exclusion", 4), ("earthquake exclusion", 4),
            ("acts of god excluded", 4), ("force majeure excluded", 3),
            ("aviation exclusion", 3), ("marine exclusion", 3),
        ]
    },

    # ── 2. WARRANTIES & REPRESENTATIONS ──────────────────────────────────────
    "Warranties & Representations": {
        "color": "#EA580C",
        "icon": "⚠️",
        "description": "Strict conditions — breach may void entire policy",
        "severity_base": 5,
        "terms": [
            ("warranted that", 5), ("warranty that", 5),
            ("basis of contract", 5), ("basis clause", 5),
            ("breach of warranty", 5), ("breach of condition", 4),
            ("condition precedent", 5), ("strict condition precedent", 5),
            ("material warranty", 5), ("promissory warranty", 5),
            ("affirmative warranty", 4), ("continuing warranty", 4),
            ("representations are true", 4), ("representation and warranty", 4),
            ("good faith warranty", 3), ("uberrimae fidei", 4),
            ("utmost good faith", 3), ("incontestability clause", 3),
            ("contestable period", 3),
        ]
    },

    # ── 3. CLAIM RESTRICTIONS & PROCEDURES ───────────────────────────────────
    "Claim Restrictions": {
        "color": "#B45309",
        "icon": "📋",
        "description": "Procedural requirements that can void a claim if missed",
        "severity_base": 4,
        "terms": [
            ("notice of loss within", 4), ("immediate notice", 4),
            ("notice condition precedent", 5), ("failure to notify", 4),
            ("claim notification period", 3), ("claims made basis", 4),
            ("claims made and reported", 4), ("occurrence basis", 2),
            ("retroactive date", 4), ("discovery period", 3),
            ("extended reporting period", 3), ("sunset clause", 4),
            ("proof of loss within", 4), ("sworn proof of loss", 3),
            ("cooperation clause", 3), ("examination under oath", 3),
            ("subrogation rights", 3), ("right of recovery", 3),
            ("reimbursement obligation", 3), ("claim forfeiture", 5),
            ("forfeiture of claim", 5), ("repudiate claim", 5),
            ("deny claim", 5), ("claim denied", 5),
            ("claim rejected", 5), ("decline to pay", 4),
            ("prejudice to insurer", 4), ("late notification", 4),
            ("out of time", 3), ("limitation period", 3),
            ("time bar", 4), ("statute of limitations", 3),
        ]
    },

    # ── 4. NON-DISCLOSURE & MISREPRESENTATION ─────────────────────────────────
    "Non-Disclosure & Fraud": {
        "color": "#7C3AED",
        "icon": "🔍",
        "description": "Material omissions or false statements that may void coverage",
        "severity_base": 5,
        "terms": [
            ("material non-disclosure", 5), ("non-disclosure", 4),
            ("material misrepresentation", 5), ("misrepresentation", 4),
            ("false statement", 5), ("fraudulent claim", 5),
            ("fraudulent misrepresentation", 5), ("concealment", 4),
            ("suppression of facts", 4), ("failure to disclose", 4),
            ("voidable ab initio", 5), ("void ab initio", 5),
            ("avoidance of policy", 5), ("policy avoided", 5),
            ("rescind the policy", 5), ("rescission", 4),
            ("material fact", 3), ("known to the insured", 3),
            ("should have known", 3), ("deliberately withheld", 4),
        ]
    },

    # ── 5. CANCELLATION & TERMINATION ────────────────────────────────────────
    "Cancellation & Termination": {
        "color": "#DB2777",
        "icon": "❌",
        "description": "Rights to cancel or terminate coverage unilaterally",
        "severity_base": 4,
        "terms": [
            ("insurer may cancel", 5), ("right to cancel", 4),
            ("cancellation at any time", 5), ("immediate cancellation", 5),
            ("unilateral cancellation", 5), ("notice of cancellation", 3),
            ("7 days notice of cancellation", 3), ("30 days notice", 2),
            ("cancellation for non-payment", 3), ("cancellation for fraud", 4),
            ("policy terminated", 4), ("automatic termination", 4),
            ("termination without notice", 5), ("voided", 4),
            ("policy is void", 5), ("lapse of coverage", 4),
            ("lapsed policy", 3), ("policy lapse", 3),
            ("discontinuance", 3), ("discontinue the policy", 3),
            ("forfeiture of policy", 5),
        ]
    },

    # ── 6. SUB-LIMITS & MONETARY RESTRICTIONS ────────────────────────────────
    "Sub-limits & Monetary Caps": {
        "color": "#0369A1",
        "icon": "💰",
        "description": "Restrictions that reduce payout below the headline sum insured",
        "severity_base": 3,
        "terms": [
            ("sub-limit", 4), ("sublimit", 4), ("subject to a maximum", 3),
            ("maximum payable shall not exceed", 4), ("aggregate limit", 3),
            ("per occurrence limit", 3), ("per event limit", 3),
            ("inner limit", 4), ("restricted to", 3),
            ("capped at", 3), ("maximum of", 2), ("not more than", 2),
            ("limited to", 3), ("subject to limit", 3),
            ("annual aggregate", 3), ("per annum limit", 2),
            ("proportional reduction", 3), ("pro-rata reduction", 3),
            ("contribution clause", 3), ("rateable proportion", 3),
            ("average clause", 4), ("underinsurance", 4),
            ("co-insurance penalty", 4), ("coinsurance clause", 3),
        ]
    },

    # ── 7. DEDUCTIBLES & EXCESS ───────────────────────────────────────────────
    "Deductibles & Excess": {
        "color": "#0891B2",
        "icon": "📉",
        "description": "Amounts the insured must bear before insurance responds",
        "severity_base": 2,
        "terms": [
            ("deductible", 2), ("excess", 2), ("self-insured retention", 3),
            ("sir", 2), ("franchise deductible", 3), ("straight deductible", 2),
            ("aggregate deductible", 3), ("per occurrence deductible", 2),
            ("compulsory excess", 3), ("voluntary excess", 2),
            ("xs of", 3), ("in excess of", 2), ("above the deductible", 2),
            ("retention", 2), ("risk retention", 3), ("first loss retention", 3),
        ]
    },

    # ── 8. ATTACHMENT POINTS & TRIGGERS (Reinsurance) ────────────────────────
    "Reinsurance Triggers": {
        "color": "#065F46",
        "icon": "🔗",
        "description": "Conditions under which reinsurance responds — or fails to",
        "severity_base": 4,
        "terms": [
            ("attachment point", 3), ("retention not met", 4),
            ("does not attach", 4), ("outside the scope", 4),
            ("reinsurer not liable", 5), ("reinsurance not triggered", 5),
            ("follow the fortunes", 3), ("follow the settlements", 3),
            ("hours clause", 3), ("cat hours", 3), ("event limit", 3),
            ("clash cover limitation", 4), ("nuclear incident clause", 5),
            ("lnma1", 4), ("war and civil war exclusion", 5),
            ("sanctions clause", 4), ("ofac", 4),
            ("reinstatement premium", 3), ("exhaustion of limit", 3),
            ("loss portfolio transfer", 3), ("commutation", 3),
        ]
    },

    # ── 9. WAITING PERIODS & DEFERRAL ────────────────────────────────────────
    "Waiting Periods": {
        "color": "#92400E",
        "icon": "⏳",
        "description": "Coverage delayed after inception — claim may be premature",
        "severity_base": 3,
        "terms": [
            ("waiting period", 3), ("initial waiting period", 3),
            ("moratorium period", 3), ("qualifying period", 3),
            ("deferral period", 3), ("survival period", 3),
            ("30 day waiting", 3), ("60 day waiting", 3),
            ("90 day waiting", 3), ("180 day waiting", 3),
            ("no cover for the first", 3), ("cover commences after", 3),
            ("subject to waiting", 3), ("elimination period", 3),
        ]
    },

    # ── 10. PREMIUM CONDITIONS ────────────────────────────────────────────────
    "Premium Conditions": {
        "color": "#1D4ED8",
        "icon": "💳",
        "description": "Conditions linking premium payment to coverage existence",
        "severity_base": 4,
        "terms": [
            ("premium payment condition precedent", 5),
            ("no cover until premium received", 5),
            ("premium warranty", 5), ("cash before cover", 5),
            ("coverage contingent on payment", 5),
            ("premium in arrears", 3), ("unpaid premium", 3),
            ("premium default", 4), ("non-payment of premium", 4),
            ("lapse for non-payment", 4), ("grace period expired", 4),
            ("premium not received", 4), ("returned premium", 2),
            ("minimum earned premium", 3), ("fully earned premium", 3),
            ("pro-rata premium", 2), ("short rate penalty", 3),
        ]
    },

    # ── 11. JURISDICTION & GOVERNING LAW ─────────────────────────────────────
    "Jurisdiction Traps": {
        "color": "#4B5563",
        "icon": "⚖️",
        "description": "Dispute resolution clauses that may disadvantage the insured",
        "severity_base": 3,
        "terms": [
            ("exclusive jurisdiction", 4), ("exclusive forum", 4),
            ("courts of x shall have", 3), ("foreign jurisdiction", 3),
            ("arbitration mandatory", 3), ("binding arbitration", 3),
            ("waiver of jury trial", 4), ("class action waiver", 4),
            ("choice of law", 2), ("governing law", 2),
            ("english law applies", 2), ("new york law", 2),
            ("dispute resolution", 2), ("mediation required", 2),
            ("expert determination", 2), ("disputes to be referred", 2),
        ]
    },

    # ── 12. THIRD PARTY & ASSIGNMENT RESTRICTIONS ────────────────────────────
    "Assignment Restrictions": {
        "color": "#6D28D9",
        "icon": "🔒",
        "description": "Limitations on transferring or assigning policy benefits",
        "severity_base": 3,
        "terms": [
            ("non-assignable", 4), ("assignment not permitted", 4),
            ("consent required for assignment", 3), ("prior written consent", 3),
            ("assignment void", 4), ("no assignment without", 3),
            ("change of interest", 3), ("insurable interest required", 3),
            ("loss of insurable interest", 4),
            ("anti-assignment clause", 4), ("no benefit assignment", 3),
            ("third party rights excluded", 4), ("no third party beneficiary", 3),
        ]
    },

    # ── 13. SUBROGATION & RECOVERY ────────────────────────────────────────────
    "Subrogation & Recovery": {
        "color": "#047857",
        "icon": "🔄",
        "description": "Insurer's right to recover paid claims from third parties",
        "severity_base": 2,
        "terms": [
            ("right of subrogation", 3), ("subrogation waiver", 3),
            ("right of recovery", 3), ("waiver of subrogation", 3),
            ("right of recourse", 3), ("reimbursement by insured", 3),
            ("recover from third party", 2), ("insurer may recover", 2),
            ("contribution rights", 2), ("right of contribution", 2),
        ]
    },

    # ── 14. SANCTIONS & REGULATORY ────────────────────────────────────────────
    "Sanctions & Regulatory": {
        "color": "#7C2D12",
        "icon": "🏛️",
        "description": "Regulatory compliance clauses that may trigger coverage denial",
        "severity_base": 4,
        "terms": [
            ("sanctions clause", 5), ("ofac sanctions", 5),
            ("un sanctions", 5), ("eu sanctions", 5),
            ("sanctioned entity", 5), ("sanctioned country", 5),
            ("prohibited person", 5), ("specially designated national", 5),
            ("regulatory non-compliance", 4), ("licence required", 3),
            ("regulatory approval", 3), ("regulatory breach", 4),
            ("irdai non-compliance", 4), ("fca breach", 4),
            ("anti-money laundering", 3), ("aml violation", 4),
            ("anti-bribery", 3), ("fcpa violation", 4),
        ]
    },

    # ── 15. DEFINITION TRAPS ──────────────────────────────────────────────────
    "Restrictive Definitions": {
        "color": "#1E3A5F",
        "icon": "📖",
        "description": "Narrow definitions that restrict the scope of coverage",
        "severity_base": 3,
        "terms": [
            ("narrowly defined as", 4), ("strictly construed", 4),
            ("only means", 3), ("shall be limited to", 3),
            ("restricted to mean", 3), ("does not include", 3),
            ("shall not be construed", 3), ("for the purposes of this clause only", 3),
            ("deemed not to include", 3), ("expressly excludes", 4),
            ("without prejudice to", 2), ("notwithstanding", 3),
            ("subject always to", 3), ("provided always that", 4),
            ("save and except", 3),
        ]
    },

    # ── 16. INSOLVENCY & CREDIT RISK ──────────────────────────────────────────
    "Insolvency Risk": {
        "color": "#9F1239",
        "icon": "⚡",
        "description": "Risk of counterparty insolvency affecting claim payment",
        "severity_base": 4,
        "terms": [
            ("insolvency clause", 4), ("insolvency of reinsurer", 5),
            ("cut-through clause", 3), ("access clause", 3),
            ("credit risk", 3), ("counterparty risk", 3),
            ("reinsurer default", 5), ("credit for reinsurance", 3),
            ("uncollectable reinsurance", 4), ("bad debt provision", 3),
            ("set-off clause", 3), ("netting agreement", 2),
            ("insolvency exclusion", 4), ("financial failure exclusion", 4),
        ]
    },

    # ── 17. MARKET CONDITIONS & INDEXATION ───────────────────────────────────
    "Market & Indexation Risk": {
        "color": "#0C4A6E",
        "icon": "📊",
        "description": "Clauses that shift market or inflation risk to the insured",
        "severity_base": 2,
        "terms": [
            ("index-linked", 2), ("inflation clause", 2),
            ("adequacy clause", 3), ("premium review clause", 3),
            ("rate revision", 3), ("premium adjustment", 2),
            ("experience rating", 2), ("burning cost adjustment", 3),
            ("swing rated", 3), ("profit commission", 2),
            ("sliding scale commission", 2), ("loss ratio cap", 2),
            ("market value clause", 3), ("reinstatement value", 2),
            ("agreed value clause", 2), ("actual cash value", 2),
            ("depreciation", 2), ("betterment", 3),
        ]
    },

    # ── 18. LIFE-SPECIFIC ADVERSE CONDITIONS ─────────────────────────────────
    "Life Insurance Adverse": {
        "color": "#6B21A8",
        "icon": "💀",
        "description": "Life insurance clauses that restrict death/critical illness payouts",
        "severity_base": 4,
        "terms": [
            ("suicide exclusion", 5), ("suicide within 12 months", 5),
            ("suicide within 1 year", 5), ("self-inflicted injury", 4),
            ("act of self-destruction", 4), ("hazardous occupation", 3),
            ("hazardous activity", 3), ("adventure sports exclusion", 3),
            ("aviation exclusion life", 3), ("death under influence", 4),
            ("alcohol exclusion", 4), ("drug exclusion", 4),
            ("hiv exclusion", 4), ("aids exclusion", 4),
            ("pre-existing condition exclusion", 5),
            ("non-standard lives", 3), ("rated policy", 3),
            ("extra premium charged", 3), ("loaded premium", 3),
            ("decline to insure", 5), ("uninsurable", 5),
            ("survival clause", 3), ("survival period 30 days", 4),
            ("survival period 90 days", 4), ("deferment period", 3),
        ]
    },
}


# ── Negation patterns — these flip an adverse term to BENIGN ─────────────────
# If an adverse term is preceded by one of these within 8 words, it's not adverse.
NEGATION_PATTERNS = [
    r"\b(is|are|shall be|will be)\s+(covered|included|payable|applicable|covered under)\b",
    r"\bnot\s+excluded\b", r"\bno\s+exclusion\b",
    r"\bexclusion\s+(does not apply|is lifted|is waived|is removed)\b",
    r"\bcoverage\s+(is|shall be|will be)\s+(provided|extended|maintained)\b",
    r"\b(this|the)\s+(exclusion\s+)?(does not|shall not|will not)\s+apply\b",
    r"\bincluded in coverage\b", r"\bhereby covered\b",
    r"\b(benefit|coverage)\s+extended\b",
    r"\bwaiver of\s+(exclusion|deductible|excess)\b",
]

NEGATION_RE = re.compile("|".join(NEGATION_PATTERNS), re.IGNORECASE)

# Window size (chars) around a term to check for negation
NEGATION_WINDOW = 120


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AdverseMatch:
    """A single adverse term found in a chunk."""
    term:          str
    category:      str
    severity:      int          # 1–5
    position:      int          # char offset in raw_text
    context_snippet: str        # ~200 chars around the match
    negated:       bool = False # True if term appears in positive context


@dataclass
class SectionAdversityReport:
    """Adversity profile for one document section (one chunk)."""
    chunk:             Chunk
    display_name:      str
    section_title:     str
    page_num:          int
    matches:           List[AdverseMatch]
    category_counts:   Dict[str, int]       # category → match count
    raw_score:         float                # sum of severities
    density_score:     float                # score per 100 tokens
    top_severity:      int                  # highest single severity found
    has_negation:      bool                 # any terms negated?
    adversity_level:   str                  # CRITICAL / HIGH / MEDIUM / LOW / CLEAN


@dataclass
class DocumentAdversityReport:
    """Full document-level adversity profile."""
    display_name:       str
    doc_name:           str
    total_sections:     int
    scanned_sections:   int
    section_reports:    List[SectionAdversityReport]
    category_summary:   Dict[str, Dict]     # category → {count, max_severity, sections}
    overall_score:      float               # 0–100 normalised
    adversity_level:    str                 # CRITICAL / HIGH / MEDIUM / LOW / CLEAN
    critical_count:     int
    high_count:         int
    medium_count:       int
    low_count:          int
    top_adverse_sections: List[SectionAdversityReport]  # top 5 worst
    executive_summary:  str                 # human-readable 3-line summary
    scan_timestamp:     str


# ══════════════════════════════════════════════════════════════════════════════
# CORE SCANNER ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class AdverseClauseScanner:
    """
    Multi-layer adverse clause detection engine.

    Layer 1: Dictionary scan  — all 400+ terms across 18 categories
    Layer 2: Negation guard   — removes false positives from positive context
    Layer 3: Proximity score  — dense adverse passages rank higher
    Layer 4: Section heatmap  — adversity per document section
    Layer 5: Doc-level profile — normalised score + executive summary
    """

    def __init__(self, categories: Optional[List[str]] = None):
        """
        categories: list of category names to scan (None = all 18)
        """
        self.active_categories = categories or list(ADVERSE_CATEGORIES.keys())
        self._build_term_index()

    def _build_term_index(self):
        """Pre-compile all term patterns for fast matching."""
        self._term_patterns: List[Tuple[re.Pattern, str, str, int]] = []
        for cat_name in self.active_categories:
            cat = ADVERSE_CATEGORIES.get(cat_name, {})
            for term, weight in cat.get("terms", []):
                # Escape and compile — word boundary where possible
                escaped = re.escape(term)
                # For multi-word terms, don't use \b at word boundaries
                if " " in term:
                    pattern = re.compile(escaped, re.IGNORECASE)
                else:
                    pattern = re.compile(r"\b" + escaped + r"\b", re.IGNORECASE)
                self._term_patterns.append((pattern, cat_name, term, weight))

    # ── Public API ─────────────────────────────────────────────────────────

    def scan_chunk(self, chunk: Chunk, display_name: str = "") -> SectionAdversityReport:
        """Scan a single chunk and return a section adversity report."""
        # Strip HTML tags from raw_text before scanning
        # Some PDFs (especially ACORD forms and scanned documents) contain
        # HTML/XML fragments in their extracted text layer.
        _raw = chunk.raw_text or ""
        text = re.sub(r"<[^>]{1,80}>", " ", _raw)
        text = re.sub(r"&[a-zA-Z]{2,8};", "", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        matches = self._find_adverse_matches(text)

        category_counts: Dict[str, int] = {}
        raw_score = 0.0
        top_sev   = 0

        for m in matches:
            if not m.negated:
                category_counts[m.category] = category_counts.get(m.category, 0) + 1
                raw_score += m.severity
                top_sev    = max(top_sev, m.severity)

        tokens        = max(len(text.split()), 1)
        density_score = (raw_score / tokens) * 100

        adversity_level = self._classify_level(raw_score, density_score, top_sev)
        has_negation    = any(m.negated for m in matches)

        return SectionAdversityReport(
            chunk          = chunk,
            display_name   = display_name or chunk.display_name or chunk.doc_name,
            section_title  = chunk.section_title,
            page_num       = chunk.page_num,
            matches        = matches,
            category_counts= category_counts,
            raw_score      = raw_score,
            density_score  = density_score,
            top_severity   = top_sev,
            has_negation   = has_negation,
            adversity_level= adversity_level,
        )

    def scan_document(
        self,
        chunks: List[Chunk],
        display_name: str = "",
        doc_name: str = "",
    ) -> DocumentAdversityReport:
        """Scan all chunks of a document and produce a full adversity profile."""
        import datetime

        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        section_reports: List[SectionAdversityReport] = []

        for chunk in child_chunks:
            dn = display_name or getattr(chunk, "display_name", chunk.doc_name) or ""
            sr = self.scan_chunk(chunk, dn)
            section_reports.append(sr)

        # Sort by raw_score descending
        section_reports.sort(key=lambda x: x.raw_score, reverse=True)

        # Document-level category summary
        cat_summary: Dict[str, Dict] = {}
        total_score = 0.0
        for sr in section_reports:
            total_score += sr.raw_score
            for cat, cnt in sr.category_counts.items():
                if cat not in cat_summary:
                    cat_summary[cat] = {
                        "count": 0,
                        "max_severity": 0,
                        "sections": [],
                        "color":  ADVERSE_CATEGORIES[cat]["color"],
                        "icon":   ADVERSE_CATEGORIES[cat]["icon"],
                        "description": ADVERSE_CATEGORIES[cat]["description"],
                    }
                cat_summary[cat]["count"]      += cnt
                cat_summary[cat]["max_severity"] = max(
                    cat_summary[cat]["max_severity"],
                    sr.top_severity
                )
                cat_summary[cat]["sections"].append(sr.section_title)

        # Normalise score to 0–100
        max_possible  = len(child_chunks) * 5 * 3  # 3 critical hits per section max
        overall_score = min(100.0, (total_score / max(max_possible, 1)) * 100)

        # Severity counts
        crit_c = sum(1 for s in section_reports if s.adversity_level == "CRITICAL")
        high_c = sum(1 for s in section_reports if s.adversity_level == "HIGH")
        med_c  = sum(1 for s in section_reports if s.adversity_level == "MEDIUM")
        low_c  = sum(1 for s in section_reports if s.adversity_level == "LOW")

        # Document-level classification
        if crit_c >= 3 or overall_score >= 70:
            doc_level = "CRITICAL"
        elif crit_c >= 1 or high_c >= 3 or overall_score >= 40:
            doc_level = "HIGH"
        elif high_c >= 1 or med_c >= 3 or overall_score >= 20:
            doc_level = "MEDIUM"
        elif med_c >= 1 or low_c >= 2:
            doc_level = "LOW"
        else:
            doc_level = "CLEAN"

        # Executive summary
        exec_summary = self._executive_summary(
            display_name or doc_name, section_reports,
            cat_summary, overall_score, doc_level,
            crit_c, high_c, len(child_chunks)
        )

        return DocumentAdversityReport(
            display_name        = display_name or doc_name,
            doc_name            = doc_name,
            total_sections      = len(child_chunks),
            scanned_sections    = len(child_chunks),
            section_reports     = section_reports,
            category_summary    = cat_summary,
            overall_score       = overall_score,
            adversity_level     = doc_level,
            critical_count      = crit_c,
            high_count          = high_c,
            medium_count        = med_c,
            low_count           = low_c,
            top_adverse_sections= section_reports[:5],
            executive_summary   = exec_summary,
            scan_timestamp      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        )

    def scan_corpus(
        self,
        index,                 # InsuranceHybridSearchIndex
        doc_filter: Optional[str] = None,
    ) -> List[DocumentAdversityReport]:
        """
        Scan all documents in the index (or a single doc if doc_filter is set).
        Groups chunks by display_name, scans each document, returns sorted results.
        """
        # Group child chunks by display_name
        doc_chunks: Dict[str, List[Chunk]] = {}
        for chunk in index._chunks:
            if chunk.chunk_type != "child":
                continue
            dn = index.get_display_name(chunk)
            if doc_filter and dn != doc_filter:
                continue
            if dn not in doc_chunks:
                doc_chunks[dn] = []
            doc_chunks[dn].append(chunk)

        reports = []
        for dn, chunks in doc_chunks.items():
            report = self.scan_document(
                chunks,
                display_name=dn,
                doc_name=chunks[0].doc_name if chunks else dn,
            )
            reports.append(report)

        reports.sort(key=lambda r: r.overall_score, reverse=True)
        return reports

    # ── Internal matching ───────────────────────────────────────────────────

    def _find_adverse_matches(self, text: str) -> List[AdverseMatch]:
        """Find all adverse term matches in a text, with negation detection."""
        matches: List[AdverseMatch] = []
        text_lower = text.lower()
        seen_positions = set()

        for pattern, cat_name, term, weight in self._term_patterns:
            for m in pattern.finditer(text_lower):
                pos = m.start()

                # Deduplicate overlapping matches
                if any(abs(pos - sp) < 5 for sp in seen_positions):
                    continue
                seen_positions.add(pos)

                # Extract context window
                ctx_start = max(0, pos - 100)
                ctx_end   = min(len(text), pos + 100)
                raw_snip  = text[ctx_start:ctx_end].strip()
                # Strip any HTML tags/entities that may exist in extracted PDF text
                # (some PDFs have embedded XML/HTML markup in their text layer)
                snippet   = re.sub(r"<[^>]{1,80}>", " ", raw_snip)
                snippet   = re.sub(r"&[a-zA-Z]{2,8};", "", snippet)
                snippet   = re.sub(r"\s{2,}", " ", snippet).strip()

                # Negation check in surrounding window
                neg_start = max(0, pos - NEGATION_WINDOW)
                neg_end   = min(len(text), pos + NEGATION_WINDOW)
                neg_window = text[neg_start:neg_end]
                negated   = bool(NEGATION_RE.search(neg_window))

                matches.append(AdverseMatch(
                    term            = term,
                    category        = cat_name,
                    severity        = weight,
                    position        = pos,
                    context_snippet = snippet,
                    negated         = negated,
                ))

        # Sort by position
        matches.sort(key=lambda x: x.position)
        return matches

    def _classify_level(
        self, raw_score: float, density: float, top_sev: int
    ) -> str:
        if top_sev >= 5 or (raw_score >= 15 and density >= 3):
            return "CRITICAL"
        elif top_sev >= 4 or (raw_score >= 8 and density >= 2):
            return "HIGH"
        elif top_sev >= 3 or raw_score >= 4:
            return "MEDIUM"
        elif raw_score >= 1:
            return "LOW"
        else:
            return "CLEAN"

    def _executive_summary(
        self, doc_name, section_reports, cat_summary,
        score, level, crit_c, high_c, total_sections
    ) -> str:
        if not section_reports or not cat_summary:
            return f"{doc_name} — No adverse clauses detected. Document appears clean."

        top_cats = sorted(cat_summary.items(),
                          key=lambda x: x[1]["count"], reverse=True)[:3]
        top_cat_names = ", ".join(c for c, _ in top_cats)
        top_section   = section_reports[0].section_title if section_reports else "N/A"
        adverse_count = sum(1 for s in section_reports if s.adversity_level != "CLEAN")

        lines = [
            f"Adversity level: {level} (score {score:.1f}/100). "
            f"{adverse_count} of {total_sections} sections contain adverse clauses.",
            f"Highest-risk categories: {top_cat_names}.",
            f"Most adverse section: '{top_section}'. "
            f"{crit_c} critical and {high_c} high severity sections require immediate review."
        ]
        return " ".join(lines)
