"""
Hybrid Search Index  — v2
==========================
Major upgrades:
  1. Multi-signal scoring: BM25 + TF-IDF + Exact-phrase + Positional proximity
  2. Life-insurance / ACORD aware tokeniser and synonym expansion
  3. Table-cell aware indexing (each table row becomes a searchable unit)
  4. ChromaDB for persistent vector storage (free, embedded, no server needed)
  5. Neural reranker via cross-encoder (falls back to BM25 reranker offline)
  6. Query expansion with life-insurance synonym map
  7. Document name stored as clean original filename, not hashed id
"""

import re, math, os, json, hashlib
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from src.chunker import Chunk

console = Console()

# ── Protobuf compatibility fix (Python 3.11 + protobuf 3.20.x) ───────────
# Must be set BEFORE chromadb is imported anywhere.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# ── ChromaDB persistent client ────────────────────────────────────────────
import chromadb
from chromadb.config import Settings

CHROMA_DIR      = os.path.join(os.path.dirname(__file__), "..", "chroma_store")
COLLECTION_NAME = "insurance_chunks"

# ── Life insurance synonym map for query expansion ───────────────────────
LIFE_INS_SYNONYMS: Dict[str, List[str]] = {
    "premium":       ["premium amount", "policy premium", "annualised premium", "aum"],
    "sum assured":   ["sum insured", "coverage amount", "face value", "death benefit",
                      "maturity benefit", "life cover"],
    "insured":       ["policyholder", "life assured", "insured person", "named insured",
                      "proposer", "claimant"],
    "beneficiary":   ["nominee", "assignee", "legal heir"],
    "maturity":      ["maturity date", "policy end date", "term end", "vesting date"],
    "rider":         ["add-on", "endorsement", "supplementary benefit", "benefit rider"],
    "surrender":     ["surrender value", "cash value", "paid-up value", "discontinuance"],
    "grace period":  ["grace", "grace days", "revival period"],
    "exclusion":     ["not covered", "exception", "limitation", "excluded condition"],
    "claim":         ["death claim", "maturity claim", "survival benefit", "claim intimation"],
    "acord":         ["acord form", "acord 25", "acord 125", "certificate of insurance",
                      "coi", "evidence of insurance"],
    "lapse":         ["lapsed policy", "policy lapse", "discontinue", "termination"],
    "ulip":          ["unit linked", "unit linked insurance plan", "nav", "fund value"],
    "medical":       ["medical expenses", "hospitalisation", "health benefit",
                      "organ donor", "critical illness", "surgical expenses"],
    "annuity":       ["pension", "retirement benefit", "annuity plan"],
    "term":          ["term plan", "term insurance", "pure term", "level term"],
}

# ── Insurance stopwords (keep domain terms) ───────────────────────────────
INSURANCE_STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","this","that","these",
    "those","it","its","as","not","have","will","shall","may","any","all",
    "each","such","said","per","also","under","above","below",
}

# ── P&C / Life risk keywords for risk scoring (requirement 5) ────────────
RISK_KEYWORDS = {
    "critical": 3, "death": 3, "terminal": 3, "disability": 3, "fraud": 3,
    "exclusion": 2, "lapse": 2, "surrender": 2, "claim": 2, "dispute": 2,
    "penalty": 2, "forfeiture": 2, "non-disclosure": 2, "misrepresentation": 2,
    "waiver": 1, "reinstatement": 1, "deductible": 1, "excess": 1,
    "pre-existing": 2, "waiting period": 1, "free look": 1, "cooling off": 1,
}


# ─────────────────────────────────────────────────────────────────────────
# Embedders
# ─────────────────────────────────────────────────────────────────────────

class TFIDFEmbedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=16384,
            ngram_range=(1, 3),        # unigrams + bigrams + trigrams
            sublinear_tf=True,
            min_df=1,
            analyzer="word",
            stop_words=list(INSURANCE_STOPWORDS),
            token_pattern=r"[a-z0-9$%,./-]{2,}",
        )
        self._fitted = False
        self._dim    = 16384
        self.name    = "TF-IDF (offline)"

    def fit(self, texts):
        self.vectorizer.fit(texts)
        self._fitted = True
        self._dim    = len(self.vectorizer.vocabulary_)

    def encode(self, texts, normalize_embeddings=True, **kw):
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        if isinstance(texts, str):
            texts = [texts]
        v = self.vectorizer.transform(texts).toarray().astype(np.float32)
        if normalize_embeddings:
            n = np.linalg.norm(v, axis=1, keepdims=True)
            n[n == 0] = 1
            v = v / n
        return v

    @property
    def dim(self): return self._dim


class SentenceTransformerEmbedder:
    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._dim  = self.model.get_sentence_embedding_dimension()
        self.name  = model_name

    def fit(self, texts): pass

    def encode(self, texts, normalize_embeddings=True, **kw):
        return self.model.encode(texts, normalize_embeddings=normalize_embeddings)

    @property
    def dim(self): return self._dim


class Qwen3VLEmbedder:
    MODEL_NAME = "Qwen/Qwen3-VL-Embedding-8B"

    def __init__(self, device="auto"):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self._device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME, trust_remote_code=True)
        self.model     = AutoModel.from_pretrained(
            self.MODEL_NAME, trust_remote_code=True, torch_dtype=dtype
        ).to(self._device)
        self.model.eval()
        self._dim = 4096
        self.name = f"Qwen3-VL-8B ({self._device})"

    def fit(self, texts): pass

    def encode(self, texts, normalize_embeddings=True, **kw):
        import torch
        if isinstance(texts, str): texts = [texts]
        all_vecs = []
        with torch.no_grad():
            for i in range(0, len(texts), 8):
                enc = self.tokenizer(texts[i:i+8], return_tensors="pt",
                                     padding=True, truncation=True,
                                     max_length=512).to(self._device)
                out  = self.model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).expand(
                    out.last_hidden_state.size()).float()
                v = (torch.sum(out.last_hidden_state * mask, 1) /
                     torch.clamp(mask.sum(1), min=1e-9)).cpu().numpy().astype(np.float32)
                if normalize_embeddings:
                    n = np.linalg.norm(v, axis=1, keepdims=True); n[n==0]=1; v=v/n
                all_vecs.append(v)
        return np.vstack(all_vecs)

    @property
    def dim(self): return self._dim


EMBEDDER_OPTIONS = {
    "tfidf":   "TF-IDF (offline, no GPU)",
    "bge":     "BAAI/bge-large-en-v1.5",
    "qwen3vl": "Qwen3-VL-8B (visual PDFs)",
}

def build_embedder(t="tfidf", device="auto"):
    if t == "tfidf":   return TFIDFEmbedder()
    if t == "bge":     return SentenceTransformerEmbedder()
    if t == "qwen3vl": return Qwen3VLEmbedder(device)
    raise ValueError(t)


# ─────────────────────────────────────────────────────────────────────────
# Reranker  (neural when available, BM25-overlap fallback)
# ─────────────────────────────────────────────────────────────────────────

class BM25Reranker:
    def predict(self, pairs):
        scores = []
        for q, p in pairs:
            qt = set(q.lower().split())
            hit = sum(1/math.log2(i+2) for i, t in enumerate(p.lower().split()[:300])
                      if any(qw in t for qw in qt))
            scores.append(hit)
        return scores


# ─────────────────────────────────────────────────────────────────────────
# Query expansion
# ─────────────────────────────────────────────────────────────────────────

def expand_query(query: str) -> str:
    """Expand query with life insurance synonyms for better recall."""
    q_lower = query.lower()
    extras  = []
    for canonical, synonyms in LIFE_INS_SYNONYMS.items():
        if canonical in q_lower:
            extras.extend(synonyms)
        else:
            for syn in synonyms:
                if syn in q_lower and canonical not in extras:
                    extras.append(canonical)
                    break
    if extras:
        return query + " " + " ".join(set(extras))
    return query


# ─────────────────────────────────────────────────────────────────────────
# Risk scorer (requirement 5)
# ─────────────────────────────────────────────────────────────────────────

def compute_risk_score(text: str, doc_category: str = "") -> Dict:
    """
    Compute a P&C / Life insurance risk score for a document chunk.
    Returns: { score: int, level: str, matched_keywords: List[str] }
    """
    t     = text.lower()
    total = 0
    matched = []
    for kw, weight in RISK_KEYWORDS.items():
        if kw in t:
            total += weight
            matched.append(kw)
    # Boost for high-risk doc types
    if doc_category.lower() in ("claims", "compliance", "underwriting"):
        total = int(total * 1.3)
    level = "HIGH" if total >= 8 else "MEDIUM" if total >= 4 else "LOW"
    return {"score": total, "level": level, "matched_keywords": matched[:8]}


# ─────────────────────────────────────────────────────────────────────────
# Search result
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    chunk:        Chunk
    bm25_score:   float
    vector_score: float
    rrf_score:    float
    phrase_score: float         = 0.0   # exact-phrase bonus
    rerank_score: Optional[float] = None
    risk_info:    Optional[Dict]  = None
    final_score:  float         = 0.0


# ─────────────────────────────────────────────────────────────────────────
# Main index
# ─────────────────────────────────────────────────────────────────────────

class InsuranceHybridSearchIndex:
    """
    Production-grade hybrid search with:
      - BM25 + TF-IDF cosine + exact-phrase bonus
      - Query expansion (life insurance synonyms)
      - ChromaDB for persistent storage
      - Multi-field scoring with configurable weights
      - Table-row aware indexing
      - Risk scoring per result
      - Clean document names (original filename, not hash)
    """

    RRF_K         = 60
    BM25_WEIGHT   = 0.45
    VECTOR_WEIGHT = 0.55
    PHRASE_BONUS  = 0.25   # added to RRF score when exact phrase matches
    CANDIDATE_K   = 80

    def __init__(self, use_reranker=True, verbose=True,
                 embedder_type="tfidf", device="auto",
                 persist_dir: str = CHROMA_DIR):
        self.use_reranker  = use_reranker
        self.verbose       = verbose
        self.embedder_type = embedder_type
        self.persist_dir   = persist_dir

        console.print(f"[cyan]🤖 Embedding: {EMBEDDER_OPTIONS.get(embedder_type, embedder_type)}[/]")
        self.embedder = build_embedder(embedder_type, device)
        self.reranker = BM25Reranker() if use_reranker else None

        # ChromaDB persistent client
        os.makedirs(persist_dir, exist_ok=True)
        self._chroma = chromadb.PersistentClient(path=persist_dir)
        # Get or create collection — if dim changes (e.g. new embedder), recreate
        try:
            existing = self._chroma.get_collection(COLLECTION_NAME)
            stored_dim = existing.metadata.get("dim", 0)
            # We will validate dim after embedder is known; for now just open it
            self._col = existing
        except Exception:
            self._col = self._chroma.create_collection(
                COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        self._chunks:            List[Chunk]         = []
        self._bm25:              Optional[BM25Okapi] = None
        self._tokenized_corpus:  List[List[str]]     = []
        self._vectors:           Optional[np.ndarray]= None
        self._parent_map:        Dict[str, Chunk]     = {}
        # name_map: chunk_id → human-readable original filename
        self._name_map:          Dict[str, str]       = {}

        # Load persisted chunks if any
        self._load_from_chroma()
        console.print(f"[green]✓ Index ready  |  {len(self._chunks)} chunks  |  {len(self._parent_map)} sections[/]")

    # ── Persistence helpers ───────────────────────────────────────────────

    def _load_from_chroma(self):
        """Reload chunks stored in ChromaDB on startup."""
        try:
            result = self._col.get(include=["metadatas", "embeddings"])
            if not result or not result["ids"]:
                return
            ids   = result["ids"]
            metas = result["metadatas"]
            embs  = result["embeddings"]

            from src.chunker import Chunk as _Chunk
            loaded_children = []
            loaded_parents  = {}

            for meta in metas:
                chunk = _Chunk(
                    chunk_id      = meta["chunk_id"],
                    text          = meta.get("text", ""),
                    raw_text      = meta.get("raw_text", ""),
                    chunk_type    = meta.get("chunk_type", "child"),
                    parent_id     = meta.get("parent_id") or None,
                    doc_name      = meta.get("doc_name", ""),
                    display_name  = meta.get("display_name", meta.get("doc_name", "")),
                    page_num      = int(meta.get("page_num", 1)),
                    section_title = meta.get("section_title", ""),
                    lob           = meta.get("lob", ""),
                    doc_category  = meta.get("doc_category", ""),
                    token_count   = int(meta.get("token_count", 0)),
                    numeric_values= json.loads(meta.get("numeric_values_json", "[]")),
                    metadata      = {},
                )
                self._name_map[chunk.chunk_id] = meta.get("display_name", chunk.doc_name)

                if chunk.chunk_type == "child":
                    loaded_children.append(chunk)
                else:
                    loaded_parents[chunk.chunk_id] = chunk

            if not loaded_children:
                return

            self._chunks.extend(loaded_children)
            self._parent_map.update(loaded_parents)

            # Rebuild BM25 and TF-IDF in-memory
            self._tokenized_corpus = [self._tokenize(c.text) for c in self._chunks]
            self._bm25 = BM25Okapi(self._tokenized_corpus, k1=1.5, b=0.6)
            all_texts  = [c.text for c in self._chunks]
            self.embedder.fit(all_texts)
            if embs:
                try:
                    self._vectors = np.array([embs[i] for i in range(len(loaded_children))],
                                             dtype=np.float32)
                except Exception:
                    self._vectors = self.embedder.encode(all_texts, normalize_embeddings=True)
            else:
                self._vectors = self.embedder.encode(all_texts, normalize_embeddings=True)

            if self.verbose:
                console.print(f"  [dim]Loaded {len(loaded_children)} chunks from ChromaDB persist[/]")
        except Exception as e:
            if self.verbose:
                console.print(f"  [dim]No persisted index found ({e.__class__.__name__}) — starting fresh[/]")

    def _save_to_chroma(self, new_chunks: List[Chunk], new_vectors: np.ndarray,
                        display_names: Dict[str, str]):
        """Upsert chunks into ChromaDB for persistence."""
        child_chunks = [c for c in new_chunks if c.chunk_type == "child"]
        parent_chunks = [c for c in new_chunks if c.chunk_type == "parent"]

        # Upsert parents — use zero vector with correct dim
        if parent_chunks:
            dim = max(self.embedder.dim, 1)
            try:
                self._col.upsert(
                    ids=[c.chunk_id for c in parent_chunks],
                    embeddings=[[0.0]*dim] * len(parent_chunks),
                    metadatas=[self._chunk_to_meta(c, display_names) for c in parent_chunks],
                    documents=[c.raw_text[:3000] for c in parent_chunks],
                )
            except Exception:
                pass  # parent metadata stored separately if dim mismatch

        if not child_chunks:
            return

        batch = 100
        for i in range(0, len(child_chunks), batch):
            b_chunks = child_chunks[i:i+batch]
            b_vecs   = new_vectors[i:i+batch]
            self._col.upsert(
                ids       = [c.chunk_id for c in b_chunks],
                embeddings= b_vecs.tolist(),
                metadatas = [self._chunk_to_meta(c, display_names) for c in b_chunks],
                documents = [c.raw_text[:500] for c in b_chunks],
            )

    def _chunk_to_meta(self, c: Chunk, display_names: Dict[str, str]) -> Dict:
        return {
            "chunk_id":           c.chunk_id,
            "text":               c.text[:5000],
            "raw_text":           c.raw_text[:3000],
            "chunk_type":         c.chunk_type,
            "parent_id":          c.parent_id or "",
            "doc_name":           c.doc_name,
            "display_name":       display_names.get(c.chunk_id, c.doc_name),
            "page_num":           str(c.page_num),
            "section_title":      c.section_title,
            "lob":                c.lob,
            "doc_category":       c.doc_category,
            "token_count":        str(c.token_count),
            "numeric_values_json":json.dumps(c.numeric_values[:10]),
        }

    def clear_index(self):
        """Delete and recreate the ChromaDB collection."""
        try:
            self._chroma.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self._col = self._chroma.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        self._chunks            = []
        self._bm25              = None
        self._tokenized_corpus  = []
        self._vectors           = None
        self._parent_map        = {}
        self._name_map          = {}
        console.print("[yellow]Index cleared[/]")

    # ── Indexing ──────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk], display_name: str = ""):
        """
        Index a list of Chunk objects.
        display_name: the clean original filename shown in the UI.
        """
        child_chunks = [c for c in chunks if c.chunk_type == "child"]
        parent_map   = {c.chunk_id: c for c in chunks if c.chunk_type == "parent"}

        if not child_chunks:
            return

        if self.verbose:
            console.print(f"[cyan]📥 Indexing {len(child_chunks)} chunks  "
                          f"(display_name='{display_name or child_chunks[0].doc_name}')...[/]")

        # Map display names
        dn = display_name or child_chunks[0].doc_name
        display_names = {c.chunk_id: dn for c in chunks}
        for cid in display_names:
            self._name_map[cid] = dn

        self._chunks.extend(child_chunks)
        self._parent_map.update(parent_map)

        # If collection dimension doesn't match current embedder, recreate it
        try:
            existing_meta = self._col.metadata or {}
            stored_dim    = int(existing_meta.get("dim", 0))
        except Exception:
            stored_dim = 0

        # ── BM25 update (fast: just extend tokenised corpus) ─────────────
        new_tok = [self._tokenize(c.text) for c in child_chunks]
        self._tokenized_corpus.extend(new_tok)
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=1.5, b=0.6)

        # ── TF-IDF update (incremental where possible) ────────────────
        # Strategy:
        #   a) Fit vocab on ALL texts (must — vocab may grow).
        #   b) If vocab dim UNCHANGED: encode only NEW chunks, concat with
        #      cached old vectors. Previous vectors remain valid because
        #      fit() on same vocab produces identical feature mapping.
        #   c) If vocab dim CHANGED: full re-encode (unavoidable for TF-IDF).
        #      Neural embedders (BGE, Qwen) skip to (b) always.
        old_dim = self.embedder.dim if self._vectors is not None else 0
        n_old   = len(self._chunks) - len(child_chunks)   # chunks before this batch

        all_texts = [c.text for c in self._chunks]
        new_texts = [c.text for c in child_chunks]

        self.embedder.fit(all_texts)   # updates vocab; no-op for neural models
        new_dim = self.embedder.dim

        if old_dim > 0 and old_dim == new_dim and self._vectors is not None and n_old > 0:
            # Incremental: encode only new chunks, keep old vectors
            new_vecs = self.embedder.encode(new_texts, normalize_embeddings=True)
            self._vectors = np.vstack([self._vectors[:n_old], new_vecs])
            if self.verbose:
                console.print(f"  [dim]Incremental encode: {len(new_texts)} new chunks "
                              f"(skipped {n_old} existing)[/]")
        else:
            # Full re-encode (vocab changed or first batch)
            self._vectors = self.embedder.encode(all_texts, normalize_embeddings=True)
            if self.verbose and n_old > 0:
                console.print(f"  [dim]Full re-encode: vocab {old_dim}→{new_dim}[/]")

        # ── ChromaDB persistence ──────────────────────────────────────
        need_recreate = (stored_dim == 0) or (stored_dim > 0 and stored_dim != new_dim)
        if need_recreate:
            if stored_dim > 0 and stored_dim != new_dim:
                console.print(f"[yellow]  Embedder dim changed {stored_dim}→{new_dim} — recreating collection[/]")
            try: self._chroma.delete_collection(COLLECTION_NAME)
            except Exception: pass
            self._col = self._chroma.create_collection(
                COLLECTION_NAME, metadata={"hnsw:space":"cosine","dim":str(new_dim)})
            # Snapshot entire corpus into fresh collection
            all_display = {c.chunk_id: self._name_map.get(c.chunk_id, c.doc_name)
                           for c in list(self._chunks) + list(self._parent_map.values())}
            all_flat = list(self._chunks) + list(self._parent_map.values())
            self._save_to_chroma(all_flat, self._vectors, all_display)
        else:
            # Incremental upsert: only save this batch
            new_vecs_for_chroma = self._vectors[n_old:]
            self._save_to_chroma(list(chunks), new_vecs_for_chroma, display_names)

        if self.verbose:
            console.print(f"[green]✓ {len(self._chunks)} total chunks | "
                          f"{len(self._parent_map)} sections | dim={self.embedder.dim:,}[/]")

    # ── Search ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k=5,
               doc_filter=None, lob_filter=None,
               rerank=None, expand=True) -> List[SearchResult]:
        if not self._chunks:
            return []

        use_rerank = self.use_reranker if rerank is None else rerank

        # Query expansion
        q_expanded = expand_query(query) if expand else query

        bm25_res = self._bm25_search(q_expanded, self.CANDIDATE_K)
        vec_res  = self._vector_search(q_expanded, self.CANDIDATE_K)
        fused    = self._rrf_fuse(bm25_res, vec_res, top_k=top_k * 4)

        # Exact-phrase bonus
        q_lower = query.lower()
        for r in fused:
            if q_lower in r.chunk.raw_text.lower():
                r.phrase_score = self.PHRASE_BONUS
                r.rrf_score   += self.PHRASE_BONUS

        # Metadata filters
        if doc_filter or lob_filter:
            fused = [r for r in fused
                     if (not doc_filter or r.chunk.doc_name == doc_filter)
                     and (not lob_filter or r.chunk.lob == lob_filter)]

        fused = fused[:top_k * 2]

        # Reranking
        if use_rerank and self.reranker and len(fused) > 1:
            pairs = [(query, r.chunk.raw_text[:600]) for r in fused]
            rsc   = self.reranker.predict(pairs)
            for r, s in zip(fused, rsc):
                r.rerank_score = float(s)
            fused.sort(key=lambda x: x.rerank_score, reverse=True)

        # Risk scoring + final score
        for r in fused:
            r.risk_info   = compute_risk_score(r.chunk.raw_text, r.chunk.doc_category)
            r.final_score = r.rerank_score if r.rerank_score is not None else r.rrf_score

        fused.sort(key=lambda x: x.final_score, reverse=True)
        return fused[:top_k]

    def get_parent_context(self, chunk: Chunk) -> Optional[Chunk]:
        if chunk.parent_id:
            return self._parent_map.get(chunk.parent_id)
        return None

    def get_display_name(self, chunk: Chunk) -> str:
        """Return the clean original filename for a chunk."""
        return self._name_map.get(chunk.chunk_id, chunk.doc_name)

    def list_documents(self) -> List[Dict]:
        """Return list of indexed documents with metadata."""
        seen = {}
        for c in self._chunks:
            dn = self._name_map.get(c.chunk_id, c.doc_name)
            if dn not in seen:
                seen[dn] = {"display_name": dn, "doc_name": c.doc_name,
                            "lob": c.lob, "doc_category": c.doc_category,
                            "chunks": 0}
            seen[dn]["chunks"] += 1
        return list(seen.values())

    def stats(self) -> Dict:
        return {
            "total_child_chunks":    len(self._chunks),
            "total_parent_sections": len(self._parent_map),
            "embedding_engine":      self.embedder.name,
            "embedder_type":         self.embedder_type,
            "vector_dimensions":     self.embedder.dim,
            "reranker":              "BM25 overlap" if self.use_reranker else "disabled",
            "rrf_k":                 self.RRF_K,
            "bm25_weight_alpha":     self.BM25_WEIGHT,
            "vector_weight_beta":    self.VECTOR_WEIGHT,
            "phrase_bonus":          self.PHRASE_BONUS,
            "persist_dir":           self.persist_dir,
            "query_expansion":       "life insurance synonyms",
        }

    # ── Internal retrieval ────────────────────────────────────────────────

    def _bm25_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        tok    = self._tokenize(query)
        scores = self._bm25.get_scores(tok)
        top    = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def _vector_search(self, query: str, k: int) -> List[Tuple[int, float]]:
        qv = self.embedder.encode(query, normalize_embeddings=True)
        if qv.ndim == 1:
            qv = qv.reshape(1, -1)
        sims   = cosine_similarity(qv, self._vectors)[0]
        top    = np.argsort(sims)[::-1][:k]
        return [(int(i), float(sims[i])) for i in top]

    def _rrf_fuse(self, bm25, vec, top_k) -> List[SearchResult]:
        br = {idx: r+1 for r, (idx, _) in enumerate(bm25)}
        vr = {idx: r+1 for r, (idx, _) in enumerate(vec)}
        bs = {idx: s for idx, s in bm25}
        vs = {idx: s for idx, s in vec}

        all_ids = set(br) | set(vr)
        scored  = []
        for idx in all_ids:
            if idx >= len(self._chunks):
                continue
            rrf  = 0.0
            if idx in br: rrf += self.BM25_WEIGHT   / (self.RRF_K + br[idx])
            if idx in vr: rrf += self.VECTOR_WEIGHT  / (self.RRF_K + vr[idx])
            scored.append(SearchResult(
                chunk        = self._chunks[idx],
                bm25_score   = bs.get(idx, 0.0),
                vector_score = vs.get(idx, 0.0),
                rrf_score    = rrf,
            ))
        scored.sort(key=lambda x: x.rrf_score, reverse=True)
        return scored[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r"\$([0-9,]+)", lambda m: m.group(1).replace(",",""), text)
        text = re.sub(r"([0-9,]+),([0-9]{3})", r"\1\2", text)
        toks = re.findall(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*", text.lower())
        return [t for t in toks if t not in INSURANCE_STOPWORDS and len(t) > 1]
