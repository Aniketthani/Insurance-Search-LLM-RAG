"""
Insurance Document RAG — Streamlit UI v4
Beautiful, professional design:
  - Deep navy + slate sidebar with glowing accents
  - Warm ivory main canvas — easy on eyes, not plain white
  - Gold / emerald / sapphire accent system
  - Glassmorphism result cards with gradient borders
  - Typography-first hierarchy
"""

import os, sys, tempfile, time, re

# ── Must be set before ANY chromadb import ──────────────────────────────────
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY", "False")
os.environ.setdefault("POSTHOG_DISABLED", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st

st.set_page_config(
    page_title="InsureSearch · AI Document Intelligence",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.parser       import InsuranceDocumentParser
from src.chunker      import InsuranceChunker
from src.search_index import (InsuranceHybridSearchIndex, EMBEDDER_OPTIONS, compute_risk_score)
from src.rag_engine   import (InsuranceRAGEngine, SearchEvaluator,
                               GroqLLM, GROQ_MODELS, OpenAILLM, OPENAI_MODELS)
from src.sample_docs  import SAMPLE_DOCS, get_sample_queries
from src.adverse_scanner import (AdverseClauseScanner, ADVERSE_CATEGORIES,
                                  DocumentAdversityReport, SectionAdversityReport)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Rich, Readable, Beautiful
# Palette:
#   Sidebar  : #0D1B2A (deep navy) / #1A2D40 (cards)
#   Canvas   : #F5F0E8 (warm ivory — NOT cold white)
#   Primary  : #C9A84C (burnished gold)
#   Success  : #2ECC8E (emerald)
#   Info     : #4B9EE8 (sapphire)
#   Danger   : #E85C5C (coral red)
#   Text     : #1C2B3A (near-black on light) / #E8EEF5 (near-white on dark)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* ══════════════════════════════════════════
   MAIN CANVAS — warm ivory, not cold white
══════════════════════════════════════════ */
.main .block-container {
    background: #F5F0E8;
    padding: 2rem 2.5rem 3rem;
}
.stApp { background: #F5F0E8; }

/* ══════════════════════════════════════════
   SIDEBAR — deep navy, rich, readable
══════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #122233 60%, #0D1B2A 100%);
    border-right: none;
}
section[data-testid="stSidebar"] .block-container {
    background: transparent;
    padding: 1.5rem 1.25rem;
}
section[data-testid="stSidebar"] * {
    color: #C8D8E8 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] strong {
    color: #EEF4FA !important;
}
section[data-testid="stSidebar"] label {
    color: #94ABBE !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px !important;
}
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1A2D40 !important;
    border: 1px solid #2A4560 !important;
    color: #EEF4FA !important;
    border-radius: 8px !important;
}
section[data-testid="stSidebar"] .stSlider > div > div > div {
    background: #2A4560 !important;
}
section[data-testid="stSidebar"] .stCheckbox label {
    color: #C8D8E8 !important;
    font-size: 13px !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #C8D8E8 !important;
    font-size: 13px !important;
}
/* Primary build button in sidebar */
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #C9A84C, #E2C06A) !important;
    color: #0D1B2A !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    filter: brightness(1.08) !important;
    transform: translateY(-1px) !important;
}
/* Secondary clear button override */
section[data-testid="stSidebar"] .stButton + .stButton > button {
    background: rgba(255,255,255,0.06) !important;
    color: #94ABBE !important;
    border: 1px solid #2A4560 !important;
}

/* ══════════════════════════════════════════
   SIDEBAR SECTION CARDS
══════════════════════════════════════════ */
.sb-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 14px 16px 10px;
    margin: 0 0 12px;
}
.sb-card-title {
    font-size: 10px;
    font-weight: 700;
    color: #C9A84C !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.sb-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(46,204,142,0.15);
    color: #2ECC8E !important;
    border: 1px solid rgba(46,204,142,0.3);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 14px;
}
.sb-badge::before {
    content: '';
    width: 6px; height: 6px;
    background: #2ECC8E;
    border-radius: 50%;
    display: inline-block;
}

/* ══════════════════════════════════════════
   PAGE HEADER
══════════════════════════════════════════ */
.page-hero {
    background: linear-gradient(135deg, #0D1B2A 0%, #1A3550 50%, #1C3D5C 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}
.page-hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(201,168,76,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 26px;
    font-weight: 700;
    color: #F0EAD6 !important;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
    line-height: 1.2;
}
.hero-subtitle {
    font-size: 13px;
    color: #7A9BB5 !important;
    margin: 0;
    line-height: 1.5;
}
.hero-pills {
    display: flex;
    flex-direction: column;
    gap: 6px;
    align-items: flex-end;
}
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 12px;
    font-weight: 500;
    color: #C8D8E8 !important;
    white-space: nowrap;
}
.hero-pill .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    display: inline-block;
}
.dot-gold    { background: #C9A84C; }
.dot-emerald { background: #2ECC8E; }
.dot-sapphire{ background: #4B9EE8; }

/* ══════════════════════════════════════════
   METRIC STRIP
══════════════════════════════════════════ */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 20px 0 24px;
}
.metric-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 18px 20px;
    border: 1px solid rgba(0,0,0,0.06);
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
}
.mc-plain::after  { background: #D1D5DB; }
.mc-sapphire::after { background: linear-gradient(90deg, #4B9EE8, #7BB8F0); }
.mc-emerald::after  { background: linear-gradient(90deg, #2ECC8E, #4FD9A8); }
.mc-gold::after     { background: linear-gradient(90deg, #C9A84C, #E2C06A); }
.mc-danger::after   { background: linear-gradient(90deg, #E85C5C, #F08080); }
.mc-label {
    font-size: 11px;
    font-weight: 600;
    color: #9CA3AF;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 6px;
}
.mc-value {
    font-size: 26px;
    font-weight: 700;
    color: #1C2B3A;
    line-height: 1.1;
}
.mc-value span { font-size: 14px; color: #9CA3AF; font-weight: 400; }

/* ══════════════════════════════════════════
   QUERY INPUT AREA
══════════════════════════════════════════ */
.query-wrap {
    background: #FFFFFF;
    border: 2px solid #E8E0D0;
    border-radius: 14px;
    padding: 4px 6px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}
.query-wrap:focus-within { border-color: #C9A84C; }
.stTextInput > div > div > input {
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 16px !important;
    font-weight: 400 !important;
    color: #1C2B3A !important;
    padding: 12px 16px !important;
    box-shadow: none !important;
}
.stTextInput > div > div > input::placeholder { color: #B8A898 !important; }
.stTextInput > div { border: none !important; box-shadow: none !important; }

/* ══════════════════════════════════════════
   SUGGESTION CHIPS
══════════════════════════════════════════ */
.sugg-title {
    font-size: 11px;
    font-weight: 700;
    color: #9C8E7A;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 16px 0 8px;
}
.stButton > button {
    background: #FFFFFF !important;
    color: #4A5568 !important;
    border: 1.5px solid #E2D9C8 !important;
    border-radius: 8px !important;
    font-size: 12px !important;
    font-weight: 400 !important;
    padding: 7px 12px !important;
    text-align: left !important;
    white-space: normal !important;
    line-height: 1.45 !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #FDF8EE !important;
    border-color: #C9A84C !important;
    color: #8B6914 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 10px rgba(201,168,76,0.15) !important;
}
/* Primary action overrides */
div[data-testid="column"] .stButton > button[kind="primary"],
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1C3D5C, #2A5580) !important;
    color: #FFFFFF !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.3px !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
}
.stButton > button[kind="primary"]:hover {
    filter: brightness(1.1) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(28,61,92,0.3) !important;
}

/* ══════════════════════════════════════════
   RESULT CARDS  — premium glassmorphism
══════════════════════════════════════════ */
.result-card {
    background: #FFFFFF;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 14px;
    border: 1px solid rgba(0,0,0,0.07);
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s, transform 0.15s;
}
.result-card:hover {
    box-shadow: 0 6px 24px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}
.result-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 5px;
    border-radius: 14px 0 0 14px;
}
.rc-gold::before    { background: linear-gradient(180deg, #C9A84C, #E2C06A); }
.rc-sapphire::before{ background: linear-gradient(180deg, #4B9EE8, #74B8F5); }
.rc-emerald::before { background: linear-gradient(180deg, #2ECC8E, #4FD9A8); }
.rc-violet::before  { background: linear-gradient(180deg, #8B5CF6, #A78BFA); }
.rc-coral::before   { background: linear-gradient(180deg, #E85C5C, #F08080); }
.rc-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 14px;
    margin-bottom: 12px;
}
.rc-rank-badge {
    width: 28px; height: 28px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 800;
    flex-shrink: 0;
    margin-top: 2px;
}
.rank-1 { background: #FEF3C7; color: #92400E; }
.rank-2 { background: #DBEAFE; color: #1E40AF; }
.rank-3 { background: #D1FAE5; color: #065F46; }
.rank-4 { background: #F3E8FF; color: #5B21B6; }
.rank-5 { background: #FEE2E2; color: #991B1B; }
.rc-doc-name {
    font-size: 15px;
    font-weight: 600;
    color: #1C2B3A;
    margin-bottom: 3px;
    line-height: 1.3;
}
.rc-section {
    font-size: 12px;
    color: #8C7A6A;
    display: flex;
    align-items: center;
    gap: 4px;
}
.rc-right {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 5px;
    flex-shrink: 0;
}
.score-badge {
    background: linear-gradient(135deg, #1C2B3A, #2A3E54);
    color: #E2C06A !important;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.3px;
    white-space: nowrap;
}
.risk-pill {
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
    white-space: nowrap;
}
.risk-HIGH   { background: #FEF2F2; color: #991B1B; border: 1px solid #FECACA; }
.risk-MEDIUM { background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }
.risk-LOW    { background: #ECFDF5; color: #065F46; border: 1px solid #A7F3D0; }
.rc-divider {
    height: 1px;
    background: #F0EAE0;
    margin: 10px 0;
}
.rc-snippet {
    font-size: 13.5px;
    color: #4A5568;
    line-height: 1.7;
}
.rc-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 12px;
}
.chip {
    font-size: 11px;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 6px;
    letter-spacing: 0.2px;
}
.chip-bm25   { background: #EFF6FF; color: #1D4ED8; }
.chip-cosine { background: #ECFDF5; color: #065F46; }
.chip-rrf    { background: #FFF7ED; color: #9A3412; }
.chip-phrase { background: #F5F3FF; color: #5B21B6; }
.chip-rerank { background: #FEF2F2; color: #991B1B; }
.rc-figures {
    background: linear-gradient(135deg, #FFFBEB, #FEF3C7);
    border: 1px solid #FDE68A;
    border-radius: 8px;
    padding: 7px 12px;
    margin-top: 10px;
    font-size: 12px;
    color: #78350F;
    font-weight: 500;
}
.rc-riskwords {
    font-size: 11px;
    color: #9C8E7A;
    margin-top: 5px;
}

/* ══════════════════════════════════════════
   LLM ANSWER BOX
══════════════════════════════════════════ */
.llm-box {
    background: linear-gradient(135deg, #F0FDF8, #E6FAF3);
    border: 1px solid #6EE7B7;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 0 0 22px;
    position: relative;
    overflow: hidden;
}
.llm-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #2ECC8E, #4B9EE8);
}
.llm-box.warn {
    background: linear-gradient(135deg, #FFFBEB, #FEF9E7);
    border-color: #FCD34D;
}
.llm-box.warn::before { background: linear-gradient(90deg, #F59E0B, #EF4444); }
.llm-label {
    font-size: 10px;
    font-weight: 800;
    color: #059669 !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.llm-label.warn { color: #D97706 !important; }
.llm-text {
    font-size: 14px;
    color: #1C2B3A;
    line-height: 1.8;
}

/* ══════════════════════════════════════════
   GUARDRAIL STRIP
══════════════════════════════════════════ */
.guardrail {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 18px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 500;
    margin-top: 18px;
}
.guardrail.pass { background: #ECFDF5; border: 1px solid #A7F3D0; color: #065F46; }
.guardrail.warn { background: #FFFBEB; border: 1px solid #FDE68A; color: #92400E; }
.guardrail-icon { font-size: 16px; }

/* ══════════════════════════════════════════
   WELCOME CARDS
══════════════════════════════════════════ */
.welcome-card {
    background: #FFFFFF;
    border: 1px solid #E8E0D0;
    border-radius: 14px;
    padding: 22px 24px;
    height: 100%;
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s;
}
.welcome-card:hover { box-shadow: 0 6px 24px rgba(0,0,0,0.08); }
.welcome-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.wc-gold::before    { background: linear-gradient(90deg, #C9A84C, #E2C06A); }
.wc-emerald::before { background: linear-gradient(90deg, #2ECC8E, #4FD9A8); }
.wc-sapphire::before{ background: linear-gradient(90deg, #4B9EE8, #74B8F5); }
.wc-icon  { font-size: 26px; margin-bottom: 12px; }
.wc-title { font-size: 15px; font-weight: 700; color: #1C2B3A; margin-bottom: 8px; }
.wc-body  { font-size: 13px; color: #7A6E62; line-height: 1.65; }
.wc-step  { font-size: 13px; color: #5C6B7A; line-height: 1.9; }
.wc-step strong { color: #1C2B3A; }

/* ══════════════════════════════════════════
   SECTION LABELS
══════════════════════════════════════════ */
.section-label {
    font-size: 11px;
    font-weight: 700;
    color: #9C8E7A;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 22px 0 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #E8E0D0;
}

/* ══════════════════════════════════════════
   DOCUMENT LIST ITEMS
══════════════════════════════════════════ */
.doc-item {
    background: #FFFFFF;
    border: 1px solid #E8E0D0;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: box-shadow 0.15s;
}
.doc-item:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.07); }
.doc-name  { font-size: 14px; font-weight: 600; color: #1C2B3A; }
.doc-meta  { font-size: 12px; color: #9C8E7A; margin-top: 2px; }
.lob-tag {
    font-size: 11px;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 20px;
    display: inline-block;
}
.lob-life  { background: #EFF6FF; color: #1D4ED8; }
.lob-pc    { background: #ECFDF5; color: #065F46; }
.lob-rei   { background: #F5F3FF; color: #5B21B6; }
.lob-comp  { background: #FFF7ED; color: #9A3412; }
.doc-count { font-size: 20px; font-weight: 700; color: #1C2B3A; text-align: right; }
.doc-count-label { font-size: 11px; color: #9C8E7A; text-align: right; }

/* ══════════════════════════════════════════
   EXPANDERS  — clean and consistent
══════════════════════════════════════════ */
.streamlit-expanderHeader {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #4A5568 !important;
    background: #FAF6F0 !important;
    border-radius: 8px !important;
    padding: 10px 14px !important;
}
.streamlit-expanderContent {
    background: #FDFAF6 !important;
    border-radius: 0 0 8px 8px !important;
}

/* ══════════════════════════════════════════
   TABS  — elegant underline style
══════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #E8E0D0 !important;
    gap: 4px;
    margin-bottom: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-size: 13.5px !important;
    font-weight: 500 !important;
    color: #9C8E7A !important;
    padding: 10px 20px !important;
    background: transparent !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: -2px !important;
    border-radius: 0 !important;
    transition: color 0.15s !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #5C4E3A !important; }
.stTabs [aria-selected="true"] {
    color: #1C2B3A !important;
    border-bottom-color: #C9A84C !important;
    font-weight: 600 !important;
}

/* ══════════════════════════════════════════
   METRICS (native Streamlit)
══════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E8E0D0;
    border-radius: 10px;
    padding: 14px 18px;
}
[data-testid="stMetricLabel"] { font-size: 12px !important; color: #9C8E7A !important; }
[data-testid="stMetricValue"] { font-size: 22px !important; color: #1C2B3A !important; font-weight: 700 !important; }

/* ══════════════════════════════════════════
   INFO / SUCCESS / WARNING boxes
══════════════════════════════════════════ */
.stAlert { border-radius: 10px !important; font-size: 13px !important; }

/* ══════════════════════════════════════════
   DATAFRAME
══════════════════════════════════════════ */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ══════════════════════════════════════════
   RISK SCORE PANEL
══════════════════════════════════════════ */
.risk-panel {
    background: #FFFFFF;
    border: 1px solid #E8E0D0;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.risk-panel-title {
    font-size: 11px;
    font-weight: 800;
    color: #9C8E7A;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.risk-panel-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #E8E0D0;
}
.risk-kw-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
}
.risk-kw-tag {
    font-size: 11px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid;
}
.rkw-high   { background: #FEF2F2; color: #991B1B; border-color: #FECACA; }
.rkw-medium { background: #FFFBEB; color: #92400E; border-color: #FDE68A; }
.rkw-low    { background: #ECFDF5; color: #065F46; border-color: #A7F3D0; }
.risk-score-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}
.risk-score-big {
    font-size: 36px;
    font-weight: 800;
    line-height: 1;
}
.risk-score-big.HIGH   { color: #991B1B; }
.risk-score-big.MEDIUM { color: #92400E; }
.risk-score-big.LOW    { color: #065F46; }
.risk-bar-wrap {
    flex: 1;
    background: #F0EAE0;
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
}
.risk-bar-fill {
    height: 10px;
    border-radius: 6px;
    transition: width 0.4s ease;
}
.risk-bar-fill.HIGH   { background: linear-gradient(90deg, #E85C5C, #C0392B); }
.risk-bar-fill.MEDIUM { background: linear-gradient(90deg, #E2C06A, #C9A84C); }
.risk-bar-fill.LOW    { background: linear-gradient(90deg, #4FD9A8, #2ECC8E); }

/* ══════════════════════════════════════════
   GUARDRAIL DETAIL PANEL
══════════════════════════════════════════ */
.grd-panel {
    background: #FFFFFF;
    border: 1px solid #E8E0D0;
    border-radius: 14px;
    padding: 20px 24px;
    margin-top: 16px;
}
.grd-panel.grd-pass { border-top: 3px solid #2ECC8E; }
.grd-panel.grd-warn { border-top: 3px solid #C9A84C; }
.grd-panel.grd-fail { border-top: 3px solid #E85C5C; }
.grd-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #F0EAE0;
    font-size: 13px;
}
.grd-row:last-child { border-bottom: none; }
.grd-label { color: #7A6E62; font-weight: 500; }
.grd-value { font-weight: 600; color: #1C2B3A; }
.grd-value.pass  { color: #065F46; }
.grd-value.warn  { color: #92400E; }
.grd-value.fail  { color: #991B1B; }
.grd-bar-wrap {
    width: 120px;
    background: #F0EAE0;
    border-radius: 4px;
    height: 6px;
    display: inline-block;
    vertical-align: middle;
    margin-left: 8px;
}
.grd-bar-fill {
    height: 6px;
    border-radius: 4px;
}
.grd-bar-high   { background: #2ECC8E; }
.grd-bar-medium { background: #C9A84C; }
.grd-bar-low    { background: #E85C5C; }

/* No reference found */
.no-ref-box {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-left: 4px solid #E85C5C;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 16px;
    font-size: 14px;
    color: #991B1B;
    font-weight: 500;
}
.no-ref-box .nr-icon { font-size: 20px; margin-right: 8px; }

/* ══════════════════════════════════════════
   ADVERSE SCAN TAB
══════════════════════════════════════════ */
.adv-hero {
    background: linear-gradient(135deg, #1C0A00 0%, #3B0D0D 50%, #450A0A 100%);
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}
.adv-hero::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(220,38,38,0.2) 0%, transparent 70%);
}
.adv-hero-title {
    font-size: 20px; font-weight: 700;
    color: #FEF2F2 !important; margin: 0 0 4px;
}
.adv-hero-sub {
    font-size: 12px; color: #FCA5A5 !important; margin: 0;
}
.adv-score-ring {
    width: 90px; height: 90px;
    border-radius: 50%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    border: 4px solid;
    flex-shrink: 0;
}
.ring-CRITICAL { border-color: #EF4444; background: rgba(239,68,68,0.12); }
.ring-HIGH     { border-color: #F97316; background: rgba(249,115,22,0.12); }
.ring-MEDIUM   { border-color: #EAB308; background: rgba(234,179,8,0.12); }
.ring-LOW      { border-color: #22C55E; background: rgba(34,197,94,0.12); }
.ring-CLEAN    { border-color: #6B7280; background: rgba(107,114,128,0.08); }
.ring-score { font-size: 22px; font-weight: 800; line-height: 1; }
.ring-label { font-size: 10px; font-weight: 600; letter-spacing: 0.5px; }
.score-CRITICAL { color: #EF4444; }
.score-HIGH     { color: #F97316; }
.score-MEDIUM   { color: #EAB308; }
.score-LOW      { color: #22C55E; }
.score-CLEAN    { color: #6B7280; }
.adv-stat-row {
    display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0 0;
}
.adv-stat {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 8px;
    padding: 8px 14px; text-align: center;
}
.adv-stat-num {
    font-size: 22px; font-weight: 800; line-height: 1;
}
.adv-stat-lbl { font-size: 10px; font-weight: 600; color: #FCA5A5 !important; letter-spacing: 0.5px; }
.s-CRITICAL { color: #FCA5A5; }
.s-HIGH     { color: #FDBA74; }
.s-MEDIUM   { color: #FDE68A; }
.s-LOW      { color: #86EFAC; }
.s-CLEAN    { color: #D1D5DB; }
.exec-summary {
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    border-left: 4px solid #EA580C;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 13px;
    color: #7C2D12;
    line-height: 1.7;
    margin: 16px 0;
}
.cat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 10px;
    margin: 14px 0;
}
.cat-card {
    background: #fff;
    border: 1px solid #E8E0D0;
    border-radius: 10px;
    padding: 12px 14px;
    border-left: 4px solid;
}
.cat-card-top {
    display: flex; align-items: center;
    justify-content: space-between; margin-bottom: 4px;
}
.cat-name { font-size: 12px; font-weight: 600; color: #1C2B3A; }
.cat-count {
    background: #1C2B3A; color: #E2C06A !important;
    border-radius: 20px; padding: 1px 8px;
    font-size: 11px; font-weight: 700;
}
.cat-desc { font-size: 11px; color: #9C8E7A; }
.section-adv-card {
    background: #fff;
    border: 1px solid #E8E0D0;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    border-left: 5px solid;
}
.sa-header {
    display: flex; justify-content: space-between;
    align-items: flex-start; gap: 12px; margin-bottom: 8px;
}
.sa-title { font-size: 14px; font-weight: 600; color: #1C2B3A; }
.sa-meta  { font-size: 12px; color: #9C8E7A; }
.sa-right { display: flex; flex-direction: column; align-items: flex-end; gap: 4px; }
.adv-level-pill {
    border-radius: 20px; padding: 3px 10px;
    font-size: 11px; font-weight: 700;
}
.pill-CRITICAL { background: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }
.pill-HIGH     { background: #FFF7ED; color: #9A3412; border: 1px solid #FED7AA; }
.pill-MEDIUM   { background: #FFFBEB; color: #92400E; border: 1px solid #FDE68A; }
.pill-LOW      { background: #F0FDF4; color: #065F46; border: 1px solid #A7F3D0; }
.pill-CLEAN    { background: #F9FAFB; color: #6B7280; border: 1px solid #E5E7EB; }
.match-snippet {
    font-size: 12.5px; color: #374151;
    line-height: 1.65; margin-top: 8px;
    background: #FAFAF8;
    border: 1px solid #E8E0D0;
    border-radius: 6px;
    padding: 8px 12px;
}
.match-term-highlight {
    background: #FEF3C7;
    border-bottom: 2px solid #F59E0B;
    border-radius: 2px;
    padding: 0 2px;
    font-weight: 600;
}
.neg-badge {
    background: #F0FDF4; color: #065F46;
    border: 1px solid #A7F3D0;
    border-radius: 4px; padding: 1px 6px;
    font-size: 10px; font-weight: 600;
}
.heatmap-bar {
    height: 8px; border-radius: 4px;
    background: linear-gradient(90deg, #22C55E 0%, #EAB308 40%, #EF4444 100%);
    margin-top: 6px;
}
.heatmap-marker {
    width: 12px; height: 12px;
    border-radius: 50%;
    border: 2px solid #fff;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2);
    position: relative;
    display: inline-block;
}

/* ══════════════════════════════════════════
   SCROLLBAR
══════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F0EAE0; }
::-webkit-scrollbar-thumb { background: #C8B89A; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    for k, v in dict(
        engine=None, index=None, indexed_docs=[],
        query_history=[], last_response=None, last_elapsed_ms=0,
        eval_results=None, embedder_type="tfidf",
        adverse_reports={}, adverse_scan_done=False,
    ).items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ══════════════════════════════════════════════════════════════════════════════
# AUTO-RESTORE FROM CHROMADB ON PAGE RELOAD
# Streamlit session_state is wiped on every browser refresh. ChromaDB is on disk.
# If the index has data we reload it automatically — user never sees the welcome screen.
# ══════════════════════════════════════════════════════════════════════════════
def _try_auto_restore():
    """
    If session has no engine but ChromaDB has chunks on disk, reconstruct
    the index + engine in-memory without re-parsing any documents.
    Returns True if restore succeeded.
    """
    if st.session_state.engine is not None:
        return True   # already loaded

    try:
        # Peek at ChromaDB without loading everything
        from src.search_index import InsuranceHybridSearchIndex, CHROMA_DIR
        import chromadb as _cdb

        client = _cdb.PersistentClient(path=CHROMA_DIR)
        try:
            col = client.get_collection("insurance_chunks")
        except Exception:
            return False   # collection doesn't exist yet

        # Check if there are any child chunks stored
        result = col.get(include=["metadatas"], limit=1)
        if not result or not result["ids"]:
            return False   # empty

        # Full restore — reconstruct index from persisted data
        with st.spinner("Restoring index from saved data…"):
            idx = InsuranceHybridSearchIndex(
                use_reranker=True, verbose=False,
                embedder_type="tfidf")   # always restore with tfidf (fast)

        if not idx._chunks:
            return False   # nothing loaded

        # Reconstruct indexed_docs list from the chunk metadata
        from src.search_index import CHROMA_DIR as _CD
        all_meta = col.get(include=["metadatas"])["metadatas"]
        seen_docs = {}
        for meta in all_meta:
            if meta.get("chunk_type") != "child":
                continue
            dn = meta.get("display_name") or meta.get("doc_name", "unknown")
            if dn not in seen_docs:
                seen_docs[dn] = {
                    "display_name": dn,
                    "doc_name":     meta.get("doc_name", dn),
                    "lob":          meta.get("lob", "Life Insurance"),
                    "category":     meta.get("doc_category", "Policy"),
                    "chunks":       0,
                    "source":       "restored",
                }
            seen_docs[dn]["chunks"] += 1

        from src.rag_engine import InsuranceRAGEngine
        engine = InsuranceRAGEngine(
            search_index=idx,
            top_k=5,
            use_parent_context=True,
            context_only=True,
            llm_fn=None,
            verbose=False,
        )

        st.session_state.engine       = engine
        st.session_state.index        = idx
        st.session_state.indexed_docs = list(seen_docs.values())
        return True

    except Exception as e:
        # Restore failed — log to console, show welcome screen
        import traceback
        print(f"[Auto-restore failed]: {e}")
        traceback.print_exc()
        return False

_try_auto_restore()

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
DOC_TYPE_HINTS = {
    "policy":("Life Insurance","Policy"), "term":("Life Insurance","Policy"),
    "ulip":("Life Insurance","ULIP"),    "health":("Life Insurance","Health"),
    "rider":("Life Insurance","Rider"),  "receipt":("Life Insurance","Receipt"),
    "acord":("P&C","ACORD"),             "coi":("P&C","ACORD"),
    "certificate":("P&C","ACORD"),       "treaty":("Reinsurance","Treaty"),
    "claims":("P&C","Claims"),           "compliance":("Compliance","Regulatory"),
    "proposal":("Life Insurance","Proposal"), "endorsement":("Life Insurance","Endorsement"),
}
def detect_doc_type(name):
    nl = name.lower()
    for kw,(lob,cat) in DOC_TYPE_HINTS.items():
        if kw in nl: return lob, cat
    return "Life Insurance","Policy"

def lob_tag_cls(lob):
    return {"Life Insurance":"lob-life","P&C":"lob-pc",
            "Reinsurance":"lob-rei","Compliance":"lob-comp"}.get(lob,"lob-life")

def rc_color_cls(rank):
    return {1:"rc-gold",2:"rc-sapphire",3:"rc-emerald",4:"rc-violet",5:"rc-coral"}.get(rank,"")

def rank_badge_cls(rank):
    return {1:"rank-1",2:"rank-2",3:"rank-3",4:"rank-4",5:"rank-5"}.get(rank,"rank-1")


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def build_pipeline(use_demo, pdf_files, use_reranker, top_k, embedder_type, llm_fn):
    parser = InsuranceDocumentParser(verbose=False)
    index  = InsuranceHybridSearchIndex(
        use_reranker=use_reranker, verbose=False, embedder_type=embedder_type)
    indexed_docs = []
    if use_demo:
        for dn, dt in SAMPLE_DOCS.items():
            lob, cat = detect_doc_type(dn)
            parsed   = parser.parse(dt, display_name=dn.replace("_"," ").title())
            parsed.doc_name = dn
            chunks   = InsuranceChunker(lob=lob, doc_category=cat, verbose=False).chunk(parsed)
            index.add_chunks(chunks, display_name=parsed.display_name)
            indexed_docs.append({"display_name":parsed.display_name,"doc_name":dn,
                                  "lob":lob,"category":cat,
                                  "chunks":sum(1 for c in chunks if c.chunk_type=="child"),
                                  "source":"demo"})
    for f in (pdf_files or []):
        lob, cat = detect_doc_type(f.name)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(f.read()); tp = tmp.name
        try:
            parsed = parser.parse(tp, display_name=f.name)
            chunks = InsuranceChunker(lob=lob, doc_category=cat, verbose=False).chunk(parsed)
            index.add_chunks(chunks, display_name=f.name)
            indexed_docs.append({"display_name":f.name,"doc_name":parsed.doc_name,
                                  "lob":lob,"category":cat,
                                  "chunks":sum(1 for c in chunks if c.chunk_type=="child"),
                                  "source":"upload"})
        finally:
            os.unlink(tp)
    engine = InsuranceRAGEngine(
        search_index=index, top_k=top_k, use_parent_context=True,
        context_only=(llm_fn is None), llm_fn=llm_fn, verbose=False)
    return engine, index, indexed_docs


# ══════════════════════════════════════════════════════════════════════════════
# RESULT CARD
# ══════════════════════════════════════════════════════════════════════════════
def render_result_card(r, rank: int):
    dn      = r.chunk.display_name if hasattr(r.chunk,"display_name") else r.chunk.doc_name
    snippet = r.chunk.raw_text.replace("\n"," ").strip()
    snippet = (snippet[:440]+"…") if len(snippet)>440 else snippet
    snippet = snippet.replace("<","&lt;").replace(">","&gt;")

    rcc   = rc_color_cls(rank)
    rbc   = rank_badge_cls(rank)
    risk  = r.risk_info or compute_risk_score(r.chunk.raw_text, r.chunk.doc_category)
    rl    = risk.get("level","LOW")

    phrase_chip = (f'<span class="chip chip-phrase">Phrase +{r.phrase_score:.2f}</span>'
                   if r.phrase_score > 0 else "")
    rerank_chip = (f'<span class="chip chip-rerank">Rerank {r.rerank_score:.3f}</span>'
                   if r.rerank_score is not None else "")

    figs_html = ""
    if r.chunk.numeric_values:
        figs_html = (f'<div class="rc-figures">'
                     f'Key figures &nbsp;·&nbsp; '
                     f'{" &nbsp;·&nbsp; ".join(r.chunk.numeric_values[:6])}</div>')

    mkws = risk.get("matched_keywords",[])
    kw_html = (f'<div class="rc-riskwords">Risk signals: {", ".join(mkws[:5])}</div>'
               if mkws else "")

    st.markdown(f"""
<div class="result-card {rcc}">
  <div class="rc-header">
    <div style="display:flex;align-items:flex-start;gap:12px;flex:1;min-width:0">
      <div class="rc-rank-badge {rbc}">#{rank}</div>
      <div style="min-width:0;flex:1">
        <div class="rc-doc-name">{dn}</div>
        <div class="rc-section">
          <span style="opacity:.5">▸</span>
          {r.chunk.section_title} &nbsp;·&nbsp; p.{r.chunk.page_num}
        </div>
      </div>
    </div>
    <div class="rc-right">
      <span class="score-badge">{r.final_score:.4f}</span>
      <span class="risk-pill risk-{rl}">Risk: {rl}</span>
    </div>
  </div>
  <div class="rc-divider"></div>
  <div class="rc-snippet">{snippet}</div>
  <div class="rc-chips">
    <span class="chip chip-bm25">BM25 &nbsp;{r.bm25_score:.3f}</span>
    <span class="chip chip-cosine">Cosine &nbsp;{r.vector_score:.3f}</span>
    <span class="chip chip-rrf">RRF &nbsp;{r.rrf_score:.4f}</span>
    {phrase_chip}{rerank_chip}
  </div>
  {figs_html}{kw_html}
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style="padding:8px 0 18px">
  <div style="font-size:20px;font-weight:800;color:#EEF4FA;letter-spacing:-0.5px">
    🛡 InsureSearch
  </div>
  <div style="font-size:11px;color:#5A7A96;margin-top:3px;font-weight:500;letter-spacing:0.3px">
    AI · DOCUMENT INTELLIGENCE
  </div>
</div>""", unsafe_allow_html=True)

    if st.session_state.engine:
        n_chunks = len(st.session_state.index._chunks) if st.session_state.index else 0
        restored = any(d.get("source")=="restored"
                       for d in st.session_state.get("indexed_docs",[]))
        badge_text = (f"Restored · {n_chunks} chunks" if restored
                      else f"Index active · {n_chunks} chunks")
        st.markdown(f'''<div class="sb-badge">{badge_text}</div>''',
                    unsafe_allow_html=True)

    # ── Documents ──
    st.markdown('<div class="sb-card"><div class="sb-card-title">📂 Documents</div>',
                unsafe_allow_html=True)
    use_demo = st.checkbox("Built-in demo documents", value=True)
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True,
                                help="Term plans, ULIPs, ACORD forms, receipts, riders")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Embedding ──
    st.markdown('<div class="sb-card"><div class="sb-card-title">🧠 Embedding Engine</div>',
                unsafe_allow_html=True)
    embedder_choice = st.selectbox("Embedding engine", list(EMBEDDER_OPTIONS.keys()),
        label_visibility="collapsed",
        format_func=lambda k: {"tfidf":"TF-IDF  (offline)",
                               "bge":"BGE-large  (neural)",
                               "qwen3vl":"Qwen3-VL-8B  (GPU)"}.get(k,k))
    if embedder_choice=="qwen3vl": st.caption("⚠ ~16 GB · GPU required")
    elif embedder_choice=="bge":   st.caption("⚠ HuggingFace download")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Search ──
    st.markdown('<div class="sb-card"><div class="sb-card-title">⚙ Search Settings</div>',
                unsafe_allow_html=True)
    top_k        = st.slider("Results per query", 1, 10, 5)
    use_reranker = st.checkbox("Enable reranker", True)
    bm25_w       = st.slider("BM25 weight α", 0.0, 1.0, 0.45, 0.05)
    vec_w        = round(1.0 - bm25_w, 2)
    st.caption(f"Semantic β = {vec_w}  ·  Phrase bonus = 0.25")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Filter ──
    st.markdown('<div class="sb-card"><div class="sb-card-title">🔎 Filter</div>',
                unsafe_allow_html=True)

    # Build document name list from indexed docs
    _doc_names = ["All documents"] + sorted({
        d["display_name"]
        for d in st.session_state.get("indexed_docs", [])
        if d.get("display_name")
    })
    _doc_sel = st.selectbox(
        "Filter by document",
        options=_doc_names,
        index=0,
        key="doc_filter_select",
    )
    doc_filter_input = None if _doc_sel == "All documents" else _doc_sel

    lob_sel    = st.selectbox("By line of business",
                              ["All","Life Insurance","P&C","Reinsurance","Compliance"])
    lob_filter = None if lob_sel=="All" else lob_sel
    st.markdown('</div>', unsafe_allow_html=True)

    # ── LLM ──
    st.markdown('<div class="sb-card"><div class="sb-card-title">🤖 Language Model</div>',
                unsafe_allow_html=True)
    llm_choice = st.radio("LLM Provider", ["None","Groq","OpenAI"], horizontal=True,
                          label_visibility="collapsed",
                          key="llm_provider_radio")
    groq_key, groq_model     = "", GroqLLM.DEFAULT_MODEL
    openai_key, openai_model = "", OpenAILLM.DEFAULT_MODEL
    groq_enabled = openai_enabled = False
    if llm_choice == "Groq":
        groq_enabled = True
        groq_key   = st.text_input("Groq API key", type="password", placeholder="gsk_...",
                                   key="groq_key_input")
        groq_model = st.selectbox("Groq model", list(GROQ_MODELS.keys()),
                                  format_func=lambda k: GROQ_MODELS[k].split("—")[0].strip(),
                                  key="groq_model_select")
    elif llm_choice == "OpenAI":
        openai_enabled = True
        openai_key   = st.text_input("OpenAI API key", type="password", placeholder="sk-...",
                                     key="openai_key_input")
        openai_model = st.selectbox("OpenAI model", list(OPENAI_MODELS.keys()),
                                    format_func=lambda k: OPENAI_MODELS[k].split("—")[0].strip(),
                                    key="openai_model_select")

    # ── Hot-swap LLM button (no reindex needed) ─────────────────────────
    if st.session_state.engine is not None:
        apply_llm_btn = st.button("⚡ Apply LLM (no reindex)", use_container_width=True,
                                  key="apply_llm_btn")
    else:
        apply_llm_btn = False
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    engine_ready = st.session_state.engine is not None
    build_label  = "➕ Add Documents" if engine_ready else "🚀 Build Index"
    c1, c2 = st.columns(2)
    build_btn = c1.button(build_label, type="primary", use_container_width=True)
    clear_btn = c2.button("Clear Index", use_container_width=True)

    if st.session_state.engine:
        stats = st.session_state.index.stats()
        st.markdown(f"""
<div style="padding:12px 0 0;border-top:1px solid rgba(255,255,255,0.06);margin-top:8px">
  <div style="font-size:12px;color:#5A7A96;line-height:1.9">
    Chunks &nbsp;<span style="color:#C8D8E8;font-weight:600">
      {stats['total_child_chunks']}</span><br>
    Sections &nbsp;<span style="color:#C8D8E8;font-weight:600">
      {stats['total_parent_sections']}</span><br>
    Embedder &nbsp;<span style="color:#C8D8E8;font-weight:600">
      {stats['embedder_type']}</span><br>
    Dim &nbsp;<span style="color:#C8D8E8;font-weight:600">
      {stats['vector_dimensions']:,}</span>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LLM HOT-SWAP  (no reindex — just replace llm_fn on existing engine)
# ══════════════════════════════════════════════════════════════════════════════
if apply_llm_btn and st.session_state.engine is not None:
    _new_llm = None
    if groq_enabled and groq_key.strip():
        try:
            _new_llm = GroqLLM(api_key=groq_key.strip(), model=groq_model)
            st.sidebar.success(f"Groq activated: {groq_model.split('-')[0].upper()}")
        except Exception as e:
            st.sidebar.error(f"Groq error: {e}")
    elif openai_enabled and openai_key.strip():
        try:
            _new_llm = OpenAILLM(api_key=openai_key.strip(), model=openai_model)
            st.sidebar.success(f"OpenAI activated: {openai_model}")
        except Exception as e:
            st.sidebar.error(f"OpenAI error: {e}")
    elif llm_choice == "None":
        st.sidebar.info("LLM disabled — search-only mode active")

    # Swap the llm_fn in place — no reindex, no re-parse
    st.session_state.engine.llm_fn       = _new_llm
    st.session_state.engine.context_only = (_new_llm is None)
    # Store for display in hero pills
    st.session_state["active_llm_label"] = (
        f"Groq · {groq_model}"    if groq_enabled  and _new_llm else
        f"OpenAI · {openai_model}" if openai_enabled and _new_llm else
        "No LLM"
    )

# ══════════════════════════════════════════════════════════════════════════════
# CLEAR / BUILD
# ══════════════════════════════════════════════════════════════════════════════
if clear_btn and st.session_state.index:
    st.session_state.index.clear_index()
    for k in ("engine","indexed_docs","last_response"):
        st.session_state[k] = None if k!="indexed_docs" else []
    st.sidebar.success("Index cleared")
    st.rerun()

if build_btn:
    # ── LLM setup ───────────────────────────────────────────────────────
    llm_fn = None
    if groq_enabled and groq_key.strip():
        try:   llm_fn = GroqLLM(api_key=groq_key.strip(), model=groq_model)
        except Exception as e: st.sidebar.error(f"Groq: {e}")
    elif openai_enabled and openai_key.strip():
        try:   llm_fn = OpenAILLM(api_key=openai_key.strip(), model=openai_model)
        except Exception as e: st.sidebar.error(f"OpenAI: {e}")

    engine_exists = st.session_state.engine is not None

    if engine_exists and not use_demo and not uploaded:
        # Engine already loaded, nothing new to add — just update LLM
        if llm_fn:
            st.session_state.engine.llm_fn       = llm_fn
            st.session_state.engine.context_only  = False
            st.sidebar.success("LLM updated. Start searching.")
        else:
            st.sidebar.info("Index already loaded. Upload new PDFs to add them.")

    elif engine_exists and uploaded:
        # ── INCREMENTAL ADD: only parse + index the new uploaded files ──
        parser = InsuranceDocumentParser(verbose=False)
        idx    = st.session_state.index
        idx.BM25_WEIGHT   = bm25_w
        idx.VECTOR_WEIGHT = vec_w
        new_docs = []
        with st.spinner(f"Adding {len(uploaded)} new document(s)…"):
            t0 = time.time()
            for f in uploaded:
                # Skip if already indexed (same display name)
                existing_names = {d["display_name"]
                                  for d in st.session_state.indexed_docs}
                if f.name in existing_names:
                    st.sidebar.warning(f"'{f.name}' already indexed — skipped")
                    continue
                lob, cat = detect_doc_type(f.name)
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(f.read()); tp = tmp.name
                try:
                    parsed = parser.parse(tp, display_name=f.name)
                    chunks = InsuranceChunker(lob=lob, doc_category=cat,
                                             verbose=False).chunk(parsed)
                    idx.add_chunks(chunks, display_name=f.name)
                    n_child = sum(1 for c in chunks if c.chunk_type=="child")
                    new_docs.append({"display_name":f.name,
                                     "doc_name":parsed.doc_name,
                                     "lob":lob,"category":cat,
                                     "chunks":n_child,"source":"upload"})
                finally:
                    os.unlink(tp)
            elapsed = time.time() - t0

        if new_docs:
            st.session_state.indexed_docs.extend(new_docs)
            # Update engine with new top_k / llm settings
            st.session_state.engine.top_k        = top_k
            st.session_state.engine.llm_fn        = llm_fn
            st.session_state.engine.context_only  = (llm_fn is None)
            st.sidebar.success(
                f"✅  Added {len(new_docs)} doc(s) in {elapsed:.1f}s")
            st.rerun()

    else:
        # ── FULL BUILD: no existing engine, build from scratch ──────────
        if not use_demo and not uploaded:
            st.sidebar.error("Select demo docs or upload PDFs first.")
        else:
            with st.spinner(f"Building index with {embedder_choice}…"):
                t0 = time.time()
                engine, index, indexed_docs = build_pipeline(
                    use_demo, uploaded, use_reranker, top_k, embedder_choice, llm_fn)
                index.BM25_WEIGHT   = bm25_w
                index.VECTOR_WEIGHT = vec_w
                elapsed = time.time() - t0
            st.session_state.update(engine=engine, index=index,
                                    indexed_docs=indexed_docs,
                                    eval_results=None, query_history=[])
            st.sidebar.success(f"✅  Ready — {len(indexed_docs)} docs in {elapsed:.1f}s")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
# Use stored active LLM label (updated by hot-swap), else derive from sidebar
provider_label = st.session_state.get("active_llm_label") or (
    "Groq · " + groq_model.replace("-"," ").title() if groq_enabled and groq_key else
    "OpenAI · " + openai_model if openai_enabled and openai_key else
    "No LLM"
)
st.markdown(f"""
<div class="page-hero">
  <div>
    <div class="hero-title">Insurance Document Intelligence</div>
    <div class="hero-subtitle">
      Semantic search across life insurance policies · ACORD forms · Receipts · Riders
    </div>
  </div>
  <div class="hero-pills">
    <div class="hero-pill">
      <span class="dot dot-gold"></span> BM25 + Cosine + RRF
    </div>
    <div class="hero-pill">
      <span class="dot dot-emerald"></span> {embedder_choice.upper()} embeddings
    </div>
    <div class="hero-pill">
      <span class="dot dot-sapphire"></span> {provider_label}
    </div>
  </div>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# WELCOME STATE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.engine is None:   # show welcome only if no index at all
    c1, c2, c3 = st.columns(3, gap="medium")
    c1.markdown("""<div class="welcome-card wc-gold">
  <div class="wc-icon">⚡</div>
  <div class="wc-title">Get started in 30 seconds</div>
  <div class="wc-step">
    <strong>1.</strong> Tick <em>Built-in demo documents</em> in the sidebar<br>
    <strong>2.</strong> Click <em>Build Index</em><br>
    <strong>3.</strong> Type any insurance question below
  </div>
</div>""", unsafe_allow_html=True)
    c2.markdown("""<div class="welcome-card wc-emerald">
  <div class="wc-icon">📄</div>
  <div class="wc-title">Supported document types</div>
  <div class="wc-body">
    Term insurance plans &amp; ULIPs<br>
    Health benefit riders<br>
    ACORD COI / certificate forms<br>
    Premium payment receipts<br>
    Proposals &amp; endorsements
  </div>
</div>""", unsafe_allow_html=True)
    c3.markdown("""<div class="welcome-card wc-sapphire">
  <div class="wc-icon">💾</div>
  <div class="wc-title">Persistent vector index</div>
  <div class="wc-body">
    ChromaDB stores your index on disk.<br>
    Restart the app — documents stay.<br>
    Connect Groq or OpenAI for full<br>
    natural-language answers.
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-label">Sample queries to try</div>', unsafe_allow_html=True)
    examples = [
        ("Organ donor coverage",  "Is organ donor medical expense covered under the policy?"),
        ("Death benefit",         "What is the death benefit sum assured for the term plan?"),
        ("Premium lapse",         "What happens if I miss the premium payment?"),
        ("ULIP fund options",     "What fund options and NAV values are in the ULIP?"),
        ("Surrender value",       "What is the surrender charge if I exit in Year 2?"),
        ("ACORD limits",          "What is the general aggregate limit in the ACORD COI?"),
    ]
    g1, g2, g3 = st.columns(3, gap="small")
    for i,(lbl,q) in enumerate(examples):
        [g1,g2,g3][i%3].markdown(f"**{lbl}**  \n`{q}`")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_search, tab_hist, tab_adv, tab_docs = st.tabs([
    "  🔍  Search  ", "  🕐  History  ", "  🚨  Adverse Scan  ", "  📁  Documents  "
])
# Evaluation tab commented out — re-enable by adding back "  📊  Evaluation  " and tab_eval


# ─────────────────────────────────────────────────────────
# SEARCH TAB
# ─────────────────────────────────────────────────────────
with tab_search:
    st.markdown('<div class="section-label">Quick queries</div>', unsafe_allow_html=True)
    suggs = [
        "Is organ donor medical expense covered under the policy?",
        "What is the sum assured for the term plan?",
        "What happens if I miss the premium payment?",
        "What fund options are available in the ULIP?",
        "What is the surrender charge if I exit in Year 2?",
        "What is the general aggregate limit in the ACORD certificate?",
    ]
    sc1, sc2, sc3 = st.columns(3, gap="small")
    for i, sq in enumerate(suggs):
        if [sc1,sc2,sc3][i%3].button(sq, key=f"sug_{i}", use_container_width=True):
            st.session_state["prefill_query"] = sq

    st.markdown("")
    prefill = st.session_state.pop("prefill_query", "")
    query   = st.text_input("Insurance query", value=prefill,
                             placeholder="Ask anything about your insurance documents…",
                             label_visibility="collapsed")

    qa, qb = st.columns([8, 1], gap="small")
    do_search = qa.button("Search", type="primary", use_container_width=True)
    if qb.button("✕", use_container_width=True):
        st.session_state.last_response = None; st.rerun()

    if (do_search or query) and query.strip():
        # ── Auto-apply LLM if key is present but engine still in context_only mode ──
        # This means user entered a key without clicking "Apply LLM" — activate it now.
        _engine = st.session_state.engine
        if _engine.context_only or _engine.llm_fn is None:
            _auto_llm = None
            if groq_enabled and groq_key.strip():
                try:
                    _auto_llm = GroqLLM(api_key=groq_key.strip(), model=groq_model)
                except Exception:
                    pass
            elif openai_enabled and openai_key.strip():
                try:
                    _auto_llm = OpenAILLM(api_key=openai_key.strip(), model=openai_model)
                except Exception:
                    pass
            if _auto_llm:
                _engine.llm_fn       = _auto_llm
                _engine.context_only = False
                st.session_state["active_llm_label"] = (
                    f"Groq · {groq_model}"     if groq_enabled else
                    f"OpenAI · {openai_model}" if openai_enabled else "No LLM"
                )

        df_in = doc_filter_input if doc_filter_input and doc_filter_input != "All documents" else None
        with st.spinner("Searching your documents…"):
            t0   = time.time()
            resp = st.session_state.engine.query(
                query, doc_filter=df_in, lob_filter=lob_filter)
            elapsed_ms = (time.time()-t0)*1000
        st.session_state.last_response   = resp
        st.session_state.last_elapsed_ms = elapsed_ms
        st.session_state.query_history.insert(0,{"query":query,"response":resp,"ms":elapsed_ms})

    resp = st.session_state.last_response
    if resp:
        n          = len(resp.source_chunks)
        elapsed_ms = st.session_state.get("last_elapsed_ms",0)
        gs         = resp.groundedness_score

        # No reference found banner
        # Only show when: (a) no_reference_found is True AND
        #                 (b) there is no real LLM answer (answer is empty or context-only format)
        _REFUSAL_STARTS = [
            "⚠ no reference found", "no reference found",
            "not present in the indexed", "⚠️ no reference",
        ]
        _has_real_answer = (
            bool(resp.answer) and
            not resp.answer.startswith("Query:") and
            len(resp.answer.strip()) > 60 and
            not any(resp.answer.strip().lower().startswith(p) for p in _REFUSAL_STARTS)
        )
        if resp.no_reference_found and not _has_real_answer:
            st.markdown("""<div class="no-ref-box">
  <span class="nr-icon">⚠</span>
  <strong>No Reference Found</strong> — The LLM could not find relevant information
  in the indexed documents for this query. Try uploading documents that contain
  the relevant content, or rephrase your query.
</div>""", unsafe_allow_html=True)

        # Metric strip
        grd_clr   = "mc-emerald" if resp.guardrail_passed else "mc-danger"
        grd_val   = "Passed" if resp.guardrail_passed else "Warning"
        gs_clr    = "mc-emerald" if gs>=0.7 else "mc-gold" if gs>=0.4 else "mc-danger"

        st.markdown(f"""
<div class="metric-row">
  <div class="metric-card mc-plain">
    <div class="mc-label">Results</div>
    <div class="mc-value">{n}</div>
  </div>
  <div class="metric-card mc-sapphire">
    <div class="mc-label">Latency</div>
    <div class="mc-value">{elapsed_ms:.0f}<span> ms</span></div>
  </div>
  <div class="metric-card {gs_clr}">
    <div class="mc-label">Groundedness</div>
    <div class="mc-value">{gs:.0%}</div>
  </div>
  <div class="metric-card {grd_clr}">
    <div class="mc-label">Guardrail</div>
    <div class="mc-value" style="font-size:18px">{grd_val}</div>
  </div>
</div>""", unsafe_allow_html=True)

        # LLM answer
        if resp.answer and not resp.answer.startswith("Query:"):
            # Split main answer from "Note:" qualifier if present
            _full_answer = resp.answer.strip()
            _note_phrases = [
                "Note: Complete information",
                "Note: complete information",
                "complete information on this topic was not found",
            ]
            _note_text = ""
            _main_answer = _full_answer
            for _np in _note_phrases:
                if _np.lower() in _full_answer.lower():
                    _parts = _full_answer.split(_np, 1) if _np in _full_answer else                              _full_answer.lower().split(_np.lower(), 1)
                    # Reconstruct using original case position
                    _split_idx = _full_answer.lower().find(_np.lower())
                    if _split_idx > 0:
                        _main_answer = _full_answer[:_split_idx].strip().rstrip(".")
                        _note_text   = _full_answer[_split_idx:].strip()
                    break

            box_cls = "llm-box" if resp.guardrail_passed else "llm-box warn"
            lbl_cls = "llm-label" if resp.guardrail_passed else "llm-label warn"
            prov    = ("Groq" if groq_enabled else "OpenAI" if openai_enabled else "LLM")
            main_html = _main_answer.replace("<","&lt;").replace(">","&gt;").replace("\n","<br>")
            note_html = ""
            if _note_text:
                nt = _note_text.replace("<","&lt;").replace(">","&gt;")
                note_html = (f'''<div style="margin-top:10px;padding:8px 12px;'''
                             f'''background:#FFFBEB;border-left:3px solid #F59E0B;'''
                             f'''border-radius:0 6px 6px 0;font-size:12px;color:#92400E">'''
                             f'''📋 {nt}</div>''')
            st.markdown(f"""
<div class="{box_cls}">
  <div class="{lbl_cls}">◆ {prov} Answer</div>
  <div class="llm-text">{main_html}</div>
  {note_html}
</div>""", unsafe_allow_html=True)
            for w in resp.guardrail_warnings: st.warning(w)

        # Score breakdown
        with st.expander("Score breakdown — how results were ranked", expanded=False):
            import pandas as pd
            rows = [{
                "Rank":i+1,
                "Document": st.session_state.index.get_display_name(r.chunk),
                "Section": r.chunk.section_title[:42],
                "BM25":   round(r.bm25_score,4),
                "Cosine": round(r.vector_score,4),
                "Phrase": round(r.phrase_score,3),
                "RRF":    round(r.rrf_score,5),
                "Rerank": round(r.rerank_score,3) if r.rerank_score else "—",
                "Final":  round(r.final_score,4),
                "Risk":   r.risk_info["level"] if r.risk_info else "—",
            } for i,r in enumerate(resp.source_chunks)]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Result cards
        st.markdown(f'<div class="section-label">Top {n} results</div>', unsafe_allow_html=True)
        for i, r in enumerate(resp.source_chunks):
            render_result_card(r, i+1)

        # Full section context
        if resp.parent_contexts:
            with st.expander(f"Full section context · {len(resp.parent_contexts)} sections loaded"):
                for pc in resp.parent_contexts:
                    dn = getattr(pc,"display_name",pc.doc_name)
                    st.markdown(f"**{dn}  ·  {pc.section_title}**")
                    st.text_area(f"Section content", value=pc.raw_text[:1000]+
                                 ("…" if len(pc.raw_text)>1000 else ""),
                                 height=130, disabled=True,
                                 label_visibility="collapsed",
                                 key=f"pc_{pc.chunk_id}")

        st.markdown("")
        left_col, right_col = st.columns(2, gap="large")

        # ── RISK SCORE PANEL (left) ─────────────────────────────────────
        with left_col:
            from src.search_index import RISK_KEYWORDS
            # Aggregate risk across all result chunks
            all_risk_kws: dict = {}
            max_score = 0
            for r in resp.source_chunks:
                ri = r.risk_info or {}
                for kw in ri.get("matched_keywords", []):
                    weight = RISK_KEYWORDS.get(kw, 1)
                    all_risk_kws[kw] = max(all_risk_kws.get(kw, 0), weight)
                max_score = max(max_score, ri.get("score", 0))

            # Overall aggregate level
            agg_level = ("HIGH" if max_score >= 8 else
                         "MEDIUM" if max_score >= 4 else "LOW")
            bar_pct   = min(100, int(max_score / 20 * 100))

            # Build keyword tags html
            kw_html = ""
            for kw, weight in sorted(all_risk_kws.items(),
                                     key=lambda x: -x[1]):
                cls = ("rkw-high" if weight >= 3 else
                       "rkw-medium" if weight == 2 else "rkw-low")
                kw_html += f'<span class="risk-kw-tag {cls}">{kw}</span>'

            st.markdown(f"""
<div class="risk-panel">
  <div class="risk-panel-title">P&amp;C Risk Signals</div>
  <div class="risk-score-row">
    <div>
      <div style="font-size:11px;color:#9C8E7A;font-weight:600;text-transform:uppercase;letter-spacing:0.5px">
        Aggregate risk
      </div>
      <div class="risk-score-big {agg_level}">{max_score}</div>
    </div>
    <div style="flex:1">
      <div style="font-size:11px;color:#9C8E7A;margin-bottom:4px">
        Level: <strong style="color:#1C2B3A">{agg_level}</strong>
        &nbsp;·&nbsp; {len(all_risk_kws)} signal(s) detected
      </div>
      <div class="risk-bar-wrap">
        <div class="risk-bar-fill {agg_level}" style="width:{bar_pct}%"></div>
      </div>
    </div>
  </div>
  <div style="font-size:11px;color:#9C8E7A;margin-bottom:6px;font-weight:600">
    KEYWORD SIGNALS DETECTED
  </div>
  <div class="risk-kw-grid">
    {kw_html if kw_html else '<span style="color:#B8A898;font-size:12px">No risk signals detected</span>'}
  </div>
</div>""", unsafe_allow_html=True)

        # ── GUARDRAIL DETAIL PANEL (right) ─────────────────────────────
        with right_col:
            gd = resp.guardrail_detail or {}
            grd_cls   = ("grd-pass" if resp.guardrail_passed else
                         "grd-warn" if gd.get("warnings_count",0) <= 1 else "grd-fail")
            gs_pct    = int(gs * 100)
            gs_bar    = ("grd-bar-high" if gs >= 0.7 else
                         "grd-bar-medium" if gs >= 0.4 else "grd-bar-low")
            gs_cls    = ("pass" if gs >= 0.7 else "warn" if gs >= 0.4 else "fail")
            nref      = "⚠ YES" if gd.get("no_reference") else "✓ Not triggered"
            nref_cls  = "fail" if gd.get("no_reference") else "pass"
            hal_cls   = "fail" if gd.get("hallucination_flag") else "pass"
            hal_val   = "⚠ Detected" if gd.get("hallucination_flag") else "✓ Clear"
            unverified= gd.get("num_unverified", 0)
            unver_cls = "fail" if unverified > 0 else "pass"
            unver_val = (f"⚠ {unverified} figure(s)" if unverified > 0
                         else "✓ All verified")
            verified  = gd.get("verified_figures", [])
            ver_html  = (", ".join(f"<code>{v}</code>" for v in verified[:4])
                         if verified else "—")
            warn_html = ""
            for w in resp.guardrail_warnings:
                warn_html += f'<div style="font-size:12px;color:#92400E;margin-top:6px;padding:6px 10px;background:#FFFBEB;border-radius:6px">• {w}</div>'

            st.markdown(f"""
<div class="grd-panel {grd_cls}">
  <div class="risk-panel-title">Guardrail &amp; Quality Scores</div>
  <div class="grd-row">
    <span class="grd-label">Groundedness score</span>
    <span>
      <span class="grd-value {gs_cls}">{gs_pct}%</span>
      <span class="grd-bar-wrap">
        <span class="grd-bar-fill {gs_bar}" style="display:block;width:{gs_pct}%"></span>
      </span>
    </span>
  </div>
  <div class="grd-row">
    <span class="grd-label">Unverified figures</span>
    <span class="grd-value {unver_cls}">{unver_val}</span>
  </div>
  <div class="grd-row">
    <span class="grd-label">Verified figures</span>
    <span class="grd-value" style="font-size:12px">{ver_html}</span>
  </div>
  <div class="grd-row">
    <span class="grd-label">No reference found</span>
    <span class="grd-value {nref_cls}">{nref}</span>
  </div>
  <div class="grd-row">
    <span class="grd-label">Hallucination flag</span>
    <span class="grd-value {hal_cls}">{hal_val}</span>
  </div>
  <div class="grd-row">
    <span class="grd-label">Total warnings</span>
    <span class="grd-value {'fail' if resp.guardrail_warnings else 'pass'}">
      {len(resp.guardrail_warnings)} warning(s)
    </span>
  </div>
  {warn_html}
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# HISTORY TAB
# ─────────────────────────────────────────────────────────
with tab_hist:
    st.markdown('<div class="section-label">Query history</div>', unsafe_allow_html=True)
    hist = st.session_state.query_history
    if not hist:
        st.info("No queries yet — run a search first.")
    else:
        if st.button("Clear all history"):
            st.session_state.query_history = []; st.rerun()
        for i, item in enumerate(hist):
            r  = item["response"]
            n  = len(r.source_chunks)
            td = (st.session_state.index.get_display_name(r.source_chunks[0].chunk)
                  if n else "—")
            with st.expander(
                    f"Q{len(hist)-i} · {item['query'][:62]} · {item['ms']:.0f} ms"):
                ca, cb, cc = st.columns(3)
                ca.metric("Results", n)
                cb.metric("Groundedness", f"{r.groundedness_score:.0%}")
                cc.metric("Guardrail", "Passed" if r.guardrail_passed else "Warning")
                st.markdown(f"**Top result:** {td}")
                if n:
                    st.markdown(
                        f"> {r.source_chunks[0].chunk.raw_text[:280].replace(chr(10),' ')}…")
                if st.button("Re-run", key=f"rerun_{i}"):
                    st.session_state["prefill_query"] = item["query"]; st.rerun()


# ── EVALUATION TAB COMMENTED OUT ─────────────────────────────────────────
# Uncomment the eval tab in the st.tabs() call above to re-enable


# ─────────────────────────────────────────────────────────
# ADVERSE SCAN TAB
# ─────────────────────────────────────────────────────────
with tab_adv:

    def _level_color(level):
        return {"CRITICAL":"#EF4444","HIGH":"#F97316",
                "MEDIUM":"#EAB308","LOW":"#22C55E","CLEAN":"#6B7280"}.get(level,"#6B7280")

    def _highlight_terms(text, matches):
        """Highlight matched adverse terms in snippet text."""
        result = text
        # Sort matches by length descending to avoid partial replacements
        sorted_terms = sorted({m.term for m in matches if not m.negated},
                               key=len, reverse=True)
        for term in sorted_terms[:8]:  # limit highlights
            escaped = re.escape(term)
            result = re.sub(
                escaped,
                f'<span class="match-term-highlight">{term}</span>',
                result, flags=re.IGNORECASE, count=2)
        return result

    # Hero header
    st.markdown("""
<div class="adv-hero">
  <div style="display:flex;align-items:center;justify-content:space-between;gap:20px">
    <div>
      <div class="adv-hero-title">🚨 Adverse Clause Scanner</div>
      <div class="adv-hero-sub">
        18 risk categories · 400+ adverse terms · Negation detection ·
        Section heatmap · Executive summary
      </div>
    </div>
    <div style="font-size:40px;opacity:0.3">⚖️</div>
  </div>
</div>""", unsafe_allow_html=True)

    if not st.session_state.engine:
        st.info("Build the index first using the sidebar, then return here to scan.")
        st.stop()

    # ── Controls ─────────────────────────────────────────────────────────────
    adv_c1, adv_c2, adv_c3 = st.columns([3, 2, 2], gap="medium")

    with adv_c1:
        _doc_list = ["All indexed documents"] + sorted({
            d["display_name"] for d in st.session_state.get("indexed_docs", [])
            if d.get("display_name")
        })
        adv_doc_sel = st.selectbox(
            "Document to scan",
            _doc_list,
            key="adv_doc_sel"
        )

    with adv_c2:
        _all_cats   = list(ADVERSE_CATEGORIES.keys())
        adv_cat_sel = st.multiselect(
            "Risk categories",
            _all_cats,
            default=_all_cats,
            key="adv_cat_sel"
        )

    with adv_c3:
        adv_min_level = st.selectbox(
            "Minimum severity",
            ["ALL", "LOW", "MEDIUM", "HIGH", "CRITICAL"],
            index=0,
            key="adv_min_level"
        )

    _level_order = {"CLEAN":0,"LOW":1,"MEDIUM":2,"HIGH":3,"CRITICAL":4,"ALL":-1}

    scan_btn = st.button(
        "🔍 Run Adverse Scan",
        type="primary",
        use_container_width=False,
        key="adv_scan_btn"
    )

    if scan_btn:
        _filter = None if adv_doc_sel == "All indexed documents" else adv_doc_sel
        scanner = AdverseClauseScanner(categories=adv_cat_sel or None)

        with st.spinner("Scanning for adverse clauses…"):
            reports = scanner.scan_corpus(st.session_state.index, doc_filter=_filter)

        st.session_state.adverse_reports    = {r.display_name: r for r in reports}
        st.session_state.adverse_scan_done  = True

    # ── Results ───────────────────────────────────────────────────────────────
    if st.session_state.get("adverse_scan_done") and st.session_state.adverse_reports:
        reports_list = list(st.session_state.adverse_reports.values())

        # ── Summary strip across all scanned docs ────────────────────────────
        total_crit = sum(r.critical_count for r in reports_list)
        total_high = sum(r.high_count     for r in reports_list)
        total_med  = sum(r.medium_count   for r in reports_list)
        total_secs = sum(r.total_sections for r in reports_list)
        avg_score  = sum(r.overall_score  for r in reports_list) / max(len(reports_list),1)

        st.markdown('<div class="section-label">Portfolio overview</div>',
                    unsafe_allow_html=True)
        ms1,ms2,ms3,ms4,ms5 = st.columns(5, gap="small")
        for col, lbl, val, cls in [
            (ms1,"Documents scanned", len(reports_list), ""),
            (ms2,"Critical sections",  total_crit, "s-CRITICAL"),
            (ms3,"High sections",       total_high, "s-HIGH"),
            (ms4,"Medium sections",     total_med,  "s-MEDIUM"),
            (ms5,"Avg adversity score", f"{avg_score:.1f}", ""),
        ]:
            col.markdown(f"""<div class="metric-card mc-plain" style="padding:12px 14px">
  <div class="mc-label">{lbl}</div>
  <div class="mc-value {cls}" style="font-size:22px">{val}</div>
</div>""", unsafe_allow_html=True)

        st.markdown("")

        # ── Per-document result ───────────────────────────────────────────────
        for rpt in sorted(reports_list, key=lambda r: r.overall_score, reverse=True):
            lc = _level_color(rpt.adversity_level)
            sc = f"{rpt.overall_score:.1f}"

            with st.expander(
                f"{'🔴' if rpt.adversity_level=='CRITICAL' else '🟠' if rpt.adversity_level=='HIGH' else '🟡' if rpt.adversity_level=='MEDIUM' else '🟢'}  "
                f"{rpt.display_name}  ·  {rpt.adversity_level}  ·  Score {sc}/100",
                expanded=(rpt.adversity_level in ("CRITICAL","HIGH"))
            ):
                # ── Score ring + stats ────────────────────────────────────────
                ring_col, stat_col = st.columns([1, 4], gap="medium")
                with ring_col:
                    st.markdown(f"""
<div style="display:flex;flex-direction:column;align-items:center;gap:8px;padding:10px 0">
  <div class="adv-score-ring ring-{rpt.adversity_level}">
    <span class="ring-score score-{rpt.adversity_level}">{sc}</span>
    <span class="ring-label score-{rpt.adversity_level}">/100</span>
  </div>
  <span class="adv-level-pill pill-{rpt.adversity_level}">{rpt.adversity_level}</span>
</div>""", unsafe_allow_html=True)

                with stat_col:
                    st.markdown(f"""
<div class="adv-stat-row">
  <div class="adv-stat">
    <div class="adv-stat-num s-CRITICAL">{rpt.critical_count}</div>
    <div class="adv-stat-lbl">CRITICAL</div>
  </div>
  <div class="adv-stat">
    <div class="adv-stat-num s-HIGH">{rpt.high_count}</div>
    <div class="adv-stat-lbl">HIGH</div>
  </div>
  <div class="adv-stat">
    <div class="adv-stat-num s-MEDIUM">{rpt.medium_count}</div>
    <div class="adv-stat-lbl">MEDIUM</div>
  </div>
  <div class="adv-stat">
    <div class="adv-stat-num s-LOW">{rpt.low_count}</div>
    <div class="adv-stat-lbl">LOW</div>
  </div>
  <div class="adv-stat">
    <div class="adv-stat-num" style="color:#C8D8E8">{rpt.total_sections}</div>
    <div class="adv-stat-lbl">SECTIONS</div>
  </div>
  <div class="adv-stat">
    <div class="adv-stat-num" style="color:#C8D8E8">{len(rpt.category_summary)}</div>
    <div class="adv-stat-lbl">CATEGORIES</div>
  </div>
</div>""", unsafe_allow_html=True)

                # ── Executive summary ─────────────────────────────────────────
                st.markdown(
                    f'<div class="exec-summary">📋 {rpt.executive_summary}</div>',
                    unsafe_allow_html=True)

                # ── Category heatmap grid ─────────────────────────────────────
                if rpt.category_summary:
                    st.markdown('<div class="section-label">Category breakdown</div>',
                                unsafe_allow_html=True)
                    cat_html = '<div class="cat-grid">'
                    for cat_name, info in sorted(
                            rpt.category_summary.items(),
                            key=lambda x: x[1]["count"], reverse=True):
                        cat_html += f"""
<div class="cat-card" style="border-left-color:{info['color']}">
  <div class="cat-card-top">
    <span class="cat-name">{info['icon']} {cat_name}</span>
    <span class="cat-count">{info['count']}</span>
  </div>
  <div class="cat-desc">{info['description'][:55]}…</div>
</div>"""
                    cat_html += "</div>"
                    st.markdown(cat_html, unsafe_allow_html=True)

                # ── Section-level drill-down ──────────────────────────────────
                min_lvl_val = _level_order.get(adv_min_level, -1)
                filtered_sections = [
                    s for s in rpt.section_reports
                    if _level_order.get(s.adversity_level, 0) >= max(min_lvl_val, 1)
                ]

                if filtered_sections:
                    st.markdown(
                        f'<div class="section-label">Flagged sections '
                        f'({len(filtered_sections)} shown)</div>',
                        unsafe_allow_html=True)

                    for sec in filtered_sections[:20]:  # cap at 20 per doc
                        lc_sec = _level_color(sec.adversity_level)
                        active_matches = [m for m in sec.matches if not m.negated]
                        neg_matches    = [m for m in sec.matches if m.negated]

                        # Category tags for this section
                        cat_tags = " ".join(
                            f'<span class="chip" style="background:{ADVERSE_CATEGORIES[c]["color"]}18;'
                            f'color:{ADVERSE_CATEGORIES[c]["color"]};border:1px solid {ADVERSE_CATEGORIES[c]["color"]}40">'
                            f'{ADVERSE_CATEGORIES[c]["icon"]} {c}</span>'
                            for c in sec.category_counts.keys()
                        )

                        st.markdown(f"""
<div class="section-adv-card" style="border-left-color:{lc_sec}">
  <div class="sa-header">
    <div>
      <div class="sa-title">{sec.section_title}</div>
      <div class="sa-meta">Page {sec.page_num} &nbsp;·&nbsp;
        {len(active_matches)} adverse match(es)
        {f'&nbsp;·&nbsp; <span class="neg-badge">✓ {len(neg_matches)} negated</span>' if neg_matches else ''}
      </div>
    </div>
    <div class="sa-right">
      <span class="adv-level-pill pill-{sec.adversity_level}">{sec.adversity_level}</span>
      <span style="font-size:11px;color:#9C8E7A">Score {sec.raw_score:.1f}</span>
    </div>
  </div>
  <div style="margin:6px 0 8px">{cat_tags}</div>
</div>""", unsafe_allow_html=True)

                        # Show top 3 match snippets
                        shown = sorted(active_matches, key=lambda m: m.severity, reverse=True)[:3]
                        for match in shown:
                            highlighted = _highlight_terms(match.context_snippet, [match])
                            sev_color   = _level_color(
                                "CRITICAL" if match.severity>=5 else
                                "HIGH" if match.severity>=4 else
                                "MEDIUM" if match.severity>=3 else "LOW")
                            st.markdown(
                                f'<div class="match-snippet">'
                                f'<span style="font-size:10px;font-weight:700;color:{sev_color};'
                                f'text-transform:uppercase;letter-spacing:0.5px">'
                                f'{match.category} · Severity {match.severity}/5</span><br>'
                                f'{highlighted}</div>',
                                unsafe_allow_html=True)

        # ── Category deep-dive selector ───────────────────────────────────────
        st.markdown('<div class="section-label">Deep-dive by category</div>',
                    unsafe_allow_html=True)
        all_cats_found = sorted({
            cat for r in reports_list
            for cat in r.category_summary.keys()
        })
        if all_cats_found:
            dd_cat = st.selectbox(
                "Select a category to see all matches across documents",
                ["— pick a category —"] + all_cats_found,
                key="adv_deepdive_cat"
            )
            if dd_cat and dd_cat != "— pick a category —":
                cat_info = ADVERSE_CATEGORIES[dd_cat]
                st.markdown(
                    f'<div class="exec-summary" style="border-left-color:{cat_info["color"]}">'
                    f'{cat_info["icon"]} <strong>{dd_cat}</strong> — '
                    f'{cat_info["description"]}</div>',
                    unsafe_allow_html=True)
                for rpt in reports_list:
                    if dd_cat not in rpt.category_summary:
                        continue
                    st.markdown(f"**{rpt.display_name}**")
                    for sec in rpt.section_reports:
                        cat_matches = [m for m in sec.matches
                                       if m.category == dd_cat and not m.negated]
                        if not cat_matches:
                            continue
                        for m in cat_matches[:3]:
                            highlighted = _highlight_terms(m.context_snippet, [m])
                            st.markdown(
                                f'<div class="match-snippet">'
                                f'<small style="color:#9C8E7A">{sec.section_title} · p.{sec.page_num}</small><br>'
                                f'{highlighted}</div>',
                                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# DOCUMENTS TAB
# ─────────────────────────────────────────────────────────
with tab_docs:
    docs = st.session_state.indexed_docs
    st.markdown('<div class="section-label">Indexed corpus</div>', unsafe_allow_html=True)
    if not docs:
        st.info("No documents indexed yet.")
    else:
        tc = sum(d["chunks"] for d in docs)
        da, db, dc = st.columns(3, gap="medium")
        da.metric("Documents", len(docs))
        db.metric("Search chunks", tc)
        dc.metric("Sections", st.session_state.index.stats()["total_parent_sections"])
        st.markdown("")
        for doc in docs:
            src = "Uploaded" if doc["source"]=="upload" else "Demo"
            lc  = lob_tag_cls(doc["lob"])
            st.markdown(f"""
<div class="doc-item">
  <div>
    <div class="doc-name">📄 &nbsp;{doc['display_name']}</div>
    <div class="doc-meta">
      <span class="lob-tag {lc}">{doc['lob']}</span>
      &nbsp; {doc['category']} &nbsp;·&nbsp; {src}
    </div>
  </div>
  <div>
    <div class="doc-count">{doc['chunks']}</div>
    <div class="doc-count-label">chunks</div>
  </div>
</div>""", unsafe_allow_html=True)
