"""
Phase 4: Context Injection + LLM Engine
Phase 5: Guardrails & Evaluation
=========================================
- Parent-context expansion (child chunk → parent section for LLM)
- Strict system prompt (underwriter + claims auditor persona)
- Numerical audit guardrail: cross-checks answer numbers against source
- Context groundedness check: answer must cite source chunks
- Automated evaluation: nDCG, precision@k, groundedness score
- Works with any OpenAI-compatible API OR prints context-only mode
"""

import re
import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from src.search_index import InsuranceHybridSearchIndex, SearchResult
from src.chunker import Chunk

console = Console()

# ── Monetary / numerical pattern (for guardrail audit) ──────────────────
NUMERIC_RE = re.compile(
    r"\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|M|B|K))?|"
    r"\d+(?:,\d{3})*(?:\.\d+)?\s*%|"
    r"\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion))?",
    re.IGNORECASE
)

SYSTEM_PROMPT = """You are an expert insurance analyst. Answer ONLY from the CONTEXT below.

CONTEXT:
---
{context}
---
Key figures: {attachment_info}

RULES (follow strictly):
1. Use ONLY the CONTEXT above. No outside knowledge.
2. If the topic is NOT mentioned anywhere in the context, respond EXACTLY:
   "⚠ NO REFERENCE FOUND: The query topic is not present in the indexed documents."
3. If partially mentioned but incomplete, say what IS found, then add:
   "Note: Complete information on this topic was not found in the indexed documents."
4. Cite the exact section or table row (e.g. "Section 3 — Exclusions").
5. Quote exact figures (premium amounts, percentages, dates).
6. Be concise — 3 to 5 sentences maximum.

USER QUERY: {query}

ANSWER:"""


@dataclass
class RAGResponse:
    query: str
    answer: str
    source_chunks: List[SearchResult]
    parent_contexts: List[Chunk]
    guardrail_passed: bool
    guardrail_warnings: List[str]
    groundedness_score: float
    numerical_audit: Dict
    no_reference_found: bool = False
    guardrail_detail: Dict = None


class InsuranceRAGEngine:
    """
    RAG engine for P&C / Reinsurance document Q&A.
    
    Modes:
        context_only=True  → prints retrieved context (no LLM needed — great for PoC demo)
        llm_fn             → pass any callable(prompt: str) -> str to use a real LLM
    """

    def __init__(
        self,
        search_index: InsuranceHybridSearchIndex,
        top_k: int = 5,
        use_parent_context: bool = True,
        context_only: bool = True,
        llm_fn=None,
        verbose: bool = True,
    ):
        self.index = search_index
        self.top_k = top_k
        self.use_parent_context = use_parent_context
        self.context_only = context_only
        self.llm_fn = llm_fn
        self.verbose = verbose

    # ------------------------------------------------------------------ #
    #  Main query method                                                   #
    # ------------------------------------------------------------------ #

    def query(
        self,
        query: str,
        doc_filter: Optional[str] = None,
        lob_filter: Optional[str] = None,
    ) -> RAGResponse:
        # 1. Hybrid search
        results = self.index.search(
            query,
            top_k=self.top_k,
            doc_filter=doc_filter,
            lob_filter=lob_filter,
        )

        if not results:
            return RAGResponse(
                query=query, answer="No relevant documents found.",
                source_chunks=[], parent_contexts=[],
                guardrail_passed=True, guardrail_warnings=[],
                groundedness_score=0.0, numerical_audit={},
            )

        # 2. Parent context expansion
        parent_contexts = []
        if self.use_parent_context:
            seen_parents = set()
            for r in results:
                parent = self.index.get_parent_context(r.chunk)
                if parent and parent.chunk_id not in seen_parents:
                    parent_contexts.append(parent)
                    seen_parents.add(parent.chunk_id)

        # 3. Build context string — pass llm budget so it is pre-capped at source level
        llm_context_budget = getattr(self.llm_fn, "context_budget", 0)
        context_str, attachment_info = self._build_context(
            results, parent_contexts,
            max_context_chars=llm_context_budget,
        )

        # 4. Generate answer
        if self.context_only or self.llm_fn is None:
            answer = self._context_only_answer(query, results)
        else:
            prompt = SYSTEM_PROMPT.format(
                context=context_str,
                attachment_info=attachment_info,
                query=query,
            )
            answer = self.llm_fn(prompt)

        # 5. Guardrails
        # Guardrails are most meaningful for LLM-generated answers
        if self.context_only or self.llm_fn is None:
            warnings, num_audit = [], {'answer_numbers': [], 'context_numbers': [], 'unverified': []}
        else:
            warnings, num_audit = self._run_guardrails(answer, results, context_str)
        groundedness = self._groundedness_score(answer, results)
        passed = len(warnings) == 0

        # Detect "no reference found" responses
        no_ref_phrases = [
            "no reference found",
            "not present in the indexed",
            "not found in the indexed",
            "cannot be confirmed from the provided",
            "no relevant documents found",
        ]
        no_reference = any(p in answer.lower() for p in no_ref_phrases)

        # Guardrail detail dict for UI display
        guardrail_detail = {
            "groundedness_score":  groundedness,
            "groundedness_level":  ("high" if groundedness >= 0.70 else
                                    "medium" if groundedness >= 0.40 else "low"),
            "num_unverified":      len(num_audit.get("unverified", [])),
            "unverified_figures":  num_audit.get("unverified", []),
            "verified_figures":    [n for n in num_audit.get("answer_numbers", [])
                                    if n not in num_audit.get("unverified", [])],
            "hallucination_flag":  any("hallucination" in w for w in warnings),
            "coverage_opinion_flag": any("opinion" in w for w in warnings),
            "no_reference":        no_reference,
            "warnings_count":      len(warnings),
        }

        return RAGResponse(
            query=query,
            answer=answer,
            source_chunks=results,
            parent_contexts=parent_contexts,
            guardrail_passed=passed,
            guardrail_warnings=warnings,
            groundedness_score=groundedness,
            numerical_audit=num_audit,
            no_reference_found=no_reference,
            guardrail_detail=guardrail_detail,
        )

    # ------------------------------------------------------------------ #
    #  Context builder                                                     #
    # ------------------------------------------------------------------ #

    def _build_context(
        self,
        results: List[SearchResult],
        parents: List[Chunk],
        max_context_chars: int = 0,
    ) -> Tuple[str, str]:
        """
        Assemble context string, hard-capping total size before returning.

        max_context_chars:
          0  → use child chunk text only (compact, always safe)
          >0 → use parent text where available, but cap each section and total

        For LLM calls we always pass the llm_fn's context_budget so the
        context is already the right size before SYSTEM_PROMPT.format().
        """
        parts = []
        attachment_values = []

        parent_by_id = {p.chunk_id: p for p in parents}
        used_parents  = set()

        # Per-source char budget — share evenly across top results
        n_results = len(results)
        if max_context_chars > 0 and n_results > 0:
            # Reserve 20% for labels + separators overhead
            usable         = int(max_context_chars * 0.80)
            # Top result gets 40%, rest share equally
            top_share      = int(usable * 0.40)
            rest_share     = (usable - top_share) // max(n_results - 1, 1)
            per_source_cap = [top_share] + [rest_share] * (n_results - 1)
        else:
            # No cap — use child chunk text only (compact)
            per_source_cap = [0] * n_results

        for i, r in enumerate(results):
            source = (
                f"[Source {i+1}: {r.chunk.doc_name}, "
                f"Section: {r.chunk.section_title}, Page {r.chunk.page_num}]"
            )

            # Choose text: parent (full section) if budget allows, else child chunk
            parent = None
            if r.chunk.parent_id and parent_by_id:
                parent = parent_by_id.get(r.chunk.parent_id)

            if parent and parent.chunk_id not in used_parents and max_context_chars > 0:
                text = parent.raw_text
                used_parents.add(parent.chunk_id)
            else:
                text = r.chunk.raw_text  # always safe — child chunks are small

            # Apply per-source hard cap
            cap = per_source_cap[i]
            if cap > 0 and len(text) > cap:
                text = text[:cap] + "\n[...section truncated for token budget...]"

            parts.append(f"{source}\n{text}")

            nums = NUMERIC_RE.findall(text)
            attachment_values.extend(nums[:3])

        context = "\n\n---\n\n".join(parts)
        att_str = ", ".join(attachment_values[:5]) if attachment_values else "See context above"
        return context, att_str

    def _context_only_answer(self, query: str, results: List[SearchResult]) -> str:
        """Produce a structured answer without an LLM by surfacing source passages."""
        lines = [f"Query: {query}\n"]
        lines.append("Most relevant passages from your insurance documents:\n")
        for i, r in enumerate(results):
            lines.append(
                f"[{i+1}] {r.chunk.doc_name} | Section: {r.chunk.section_title} | "
                f"Page {r.chunk.page_num} | Score: {r.final_score:.4f}"
            )
            lines.append(f"    {r.chunk.raw_text[:400]}...")
            if r.chunk.numeric_values:
                lines.append(f"    💰 Key figures: {', '.join(r.chunk.numeric_values[:5])}")
            lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Guardrails (Phase 5)                                                #
    # ------------------------------------------------------------------ #

    def _run_guardrails(
        self,
        answer: str,
        results: List[SearchResult],
        context: str,
    ) -> Tuple[List[str], Dict]:
        """
        Guardrail checks:
        1. Numerical audit — every number in the answer must appear in source context
        2. Hallucination signal — if answer is very long but context is short
        3. Disclaimer check — LLM shouldn't add unauthorised coverage opinions
        """
        warnings = []
        audit = {"answer_numbers": [], "context_numbers": [], "unverified": []}

        # ── Numerical audit ────────────────────────────────────────────
        answer_nums  = set(NUMERIC_RE.findall(answer))
        context_nums = set(NUMERIC_RE.findall(context))
        audit["answer_numbers"]  = list(answer_nums)
        audit["context_numbers"] = list(context_nums)

        # Normalise for comparison (strip $ commas)
        def norm(n): return re.sub(r"[$,\s]", "", n.lower())
        norm_ctx = {norm(n) for n in context_nums}

        unverified = []
        for num in answer_nums:
            if norm(num) not in norm_ctx:
                unverified.append(num)

        audit["unverified"] = unverified
        if unverified:
            warnings.append(
                f"⚠️ Numerical audit: {len(unverified)} figure(s) in answer not found in source: "
                + ", ".join(unverified[:5])
            )

        # ── Hallucination length heuristic ────────────────────────────
        if len(answer) > 2 * len(context) and len(context) < 500:
            warnings.append("⚠️ Answer is significantly longer than source context — possible hallucination.")

        # ── Unauthorised coverage opinion ─────────────────────────────
        opinion_phrases = [
            "this policy covers", "you are covered", "the claim will be paid",
            "coverage is guaranteed", "this is covered"
        ]
        for phrase in opinion_phrases:
            if phrase in answer.lower() and "context" not in answer.lower():
                warnings.append(f"⚠️ Possible unauthorised coverage opinion: '{phrase}'")

        return warnings, audit

    def _groundedness_score(self, answer: str, results: List[SearchResult]) -> float:
        """
        Simple lexical groundedness: fraction of answer content words
        that appear in retrieved chunks.
        """
        if not answer or not results:
            return 0.0

        answer_words = set(re.findall(r"\b[a-z]{4,}\b", answer.lower()))
        if not answer_words:
            return 0.0

        source_words = set()
        for r in results:
            source_words.update(re.findall(r"\b[a-z]{4,}\b", r.chunk.text.lower()))

        overlap = answer_words & source_words
        return round(len(overlap) / len(answer_words), 3)


# ─────────────────────────────────────────────────────────────────────────
# GROQ LLM INTEGRATION
# ─────────────────────────────────────────────────────────────────────────

# Available Groq models (as of mid-2025)
GROQ_MODELS = {
    "llama-3.3-70b-versatile":  "Llama 3.3 70B — best quality, recommended",
    "llama-3.1-8b-instant":     "Llama 3.1 8B — fastest, low latency",
    "mixtral-8x7b-32768":       "Mixtral 8x7B — 32k context window",
    "gemma2-9b-it":             "Gemma 2 9B — Google, instruction-tuned",
    "llama3-70b-8192":          "Llama 3 70B — 8k context",
}

# Max TOKENS the context portion of the prompt may use (answer budget excluded).
# Free-tier Groq TPM limits (tokens-per-minute), minus 800 reserved for answer + instructions.
# We store raw token counts and convert to chars via CHARS_PER_TOKEN = 3.8 (conservative).
MODEL_MAX_CONTEXT_TOKENS = {
    "llama-3.1-8b-instant":     3_200,   # 6k TPM free tier  → 3200 tokens safe context
    "gemma2-9b-it":             3_200,   # 6k TPM free tier
    "llama3-70b-8192":          6_000,   # 8k context window
    "llama-3.3-70b-versatile": 24_000,   # 30k TPM free tier
    "mixtral-8x7b-32768":      28_000,   # 32k context window
}
DEFAULT_MAX_CONTEXT_TOKENS = 3_200
CHARS_PER_TOKEN = 3.8  # conservative estimate; actual GPT tokeniser ≈ 4

def _tokens_to_chars(tokens: int) -> int:
    return int(tokens * CHARS_PER_TOKEN)

MODEL_CONTEXT_BUDGET = {
    k: _tokens_to_chars(v) for k, v in MODEL_MAX_CONTEXT_TOKENS.items()
}
DEFAULT_CONTEXT_BUDGET = _tokens_to_chars(DEFAULT_MAX_CONTEXT_TOKENS)


class GroqLLM:
    """
    ChatGroq integration for the Insurance RAG engine.

    Provides a callable compatible with InsuranceRAGEngine.llm_fn.
    Uses the Groq Python SDK for ultra-low latency inference.

    Setup:
        1. Get a free API key at https://console.groq.com
        2. Set environment variable: GROQ_API_KEY=gsk_...
           OR pass api_key= directly to the constructor.

    Usage:
        llm = GroqLLM(api_key="gsk_...", model="llama-3.3-70b-versatile")
        engine = InsuranceRAGEngine(..., context_only=False, llm_fn=llm)
        response = engine.query("What is the flood deductible?")
        print(response.answer)   # Full grounded answer from Groq
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(
        self,
        api_key:     Optional[str] = None,
        model:       str = DEFAULT_MODEL,
        temperature: float = 0.0,    # 0 = deterministic — required for insurance accuracy
        max_tokens:  int   = 512,   # 512 → faster, concise answers
    ):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError(
                "groq package not installed. Run: pip install groq"
            )

        resolved_key = api_key or os.environ.get("GROQ_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No Groq API key provided. Either pass api_key= or set "
                "the GROQ_API_KEY environment variable."
            )

        try:
            self.client = Groq(api_key=resolved_key)
        except TypeError as e:
            if "proxies" in str(e):
                # Older groq SDK passes 'proxies' to httpx which removed it.
                # Patch: instantiate with only the api_key kwarg via httpx directly.
                import httpx
                http_client = httpx.Client()
                self.client = Groq(api_key=resolved_key, http_client=http_client)
            else:
                raise

        self.model          = model
        self.temperature    = temperature
        self.max_tokens     = max_tokens
        self.name           = f"Groq / {model}"
        self.context_budget = MODEL_CONTEXT_BUDGET.get(model, DEFAULT_CONTEXT_BUDGET)

        console.print(
            f"[green]✓ Groq LLM ready: {self.name}  "
            f"| context budget: {self.context_budget:,} chars[/]"
        )

    # ── Token-budget-aware context truncation ─────────────────────────────

    def _truncate_prompt(self, prompt: str) -> tuple:
        """
        Fit the prompt inside the model context budget.
        Priority: keep system instructions + query intact,
        truncate source bodies proportionally, drop lowest sources if needed.
        Returns (truncated_prompt, was_truncated, chars_removed).
        """
        if len(prompt) <= self.context_budget:
            return prompt, False, 0

        import re as _re

        sources_start = prompt.find("[Source 1:")
        tail_match    = _re.search(r"\nUSER QUERY:", prompt)
        tail_start    = tail_match.start() if tail_match else len(prompt)

        if sources_start == -1 or sources_start >= tail_start:
            budget = self.context_budget - 200
            trunc_note = "\n[...context truncated to fit token limit...]\n"
            return prompt[:budget] + trunc_note + prompt[tail_start:], True, len(prompt) - self.context_budget

        preamble      = prompt[:sources_start]
        tail          = prompt[tail_start:]
        sources_block = prompt[sources_start:tail_start]

        source_sections = _re.split(r"\n\n---\n\n", sources_block)
        available       = self.context_budget - len(preamble) - len(tail) - 100

        if available <= 0:
            omit_note = "[Context omitted — query too large for token budget]\n"
            return preamble + omit_note + tail, True, len(sources_block)

        n = len(source_sections)
        if n == 1:
            budgets = [available]
        else:
            first_share = int(available * 0.40)
            rest_share  = (available - first_share) // max(n - 1, 1)
            budgets = [first_share] + [rest_share] * (n - 1)

        truncated_sections = []
        chars_removed = 0
        for sec, bud in zip(source_sections, budgets):
            if len(sec) <= bud:
                truncated_sections.append(sec)
            else:
                label_end   = sec.find("\n")
                label       = sec[:label_end + 1] if label_end != -1 else ""
                body        = sec[label_end + 1:] if label_end != -1 else sec
                body_budget = bud - len(label) - 60
                if body_budget > 100:
                    truncated_sections.append(label + body[:body_budget] + "\n[...truncated...]")
                else:
                    truncated_sections.append(label + "[...truncated to fit budget...]")
                chars_removed += len(sec) - bud

        rebuilt = "\n\n---\n\n".join(truncated_sections)
        return preamble + rebuilt + tail, True, chars_removed

    def __call__(self, prompt: str) -> str:
        """
        Truncates context to fit the model token budget, then calls Groq.
        Returns the model answer string.
        """
        prompt, was_truncated, chars_removed = self._truncate_prompt(prompt)
        if was_truncated:
            console.print(
                f"[yellow]  Context truncated: {chars_removed:,} chars removed "
                f"(budget={self.context_budget:,} chars for {self.model})[/]"
            )
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "413" in err or "rate_limit_exceeded" in err or "tokens" in err.lower():
                console.print(f"[red]Groq token limit hit: {e}[/]")
                return (
                    "The context is still too large for this model. "
                    "Switch to Mixtral 8x7B (32k context) or Llama 3.3 70B "
                    "in the sidebar and rebuild the index."
                )
            console.print(f"[red]Groq API error: {e}[/]")
            return f"[Groq API error: {e}]"

    def test_connection(self) -> bool:
        """Quick connectivity check — returns True if the API key is valid."""
        try:
            self("Say OK in one word")
            return True
        except Exception:
            return False


def build_groq_llm(
    api_key: Optional[str] = None,
    model:   str = GroqLLM.DEFAULT_MODEL,
) -> GroqLLM:
    """
    Convenience factory used by the Streamlit UI.
    Reads GROQ_API_KEY from environment if api_key not passed.
    """
    return GroqLLM(api_key=api_key, model=model)


# ─────────────────────────────────────────────────────────────────────────
# OPENAI LLM INTEGRATION
# ─────────────────────────────────────────────────────────────────────────

OPENAI_MODELS = {
    "gpt-4o":              "GPT-4o — best quality, multimodal",
    "gpt-4o-mini":         "GPT-4o Mini — fast & cost-effective",
    "gpt-4-turbo":         "GPT-4 Turbo — 128k context",
    "gpt-3.5-turbo":       "GPT-3.5 Turbo — fastest, lowest cost",
}

# OpenAI context budgets (tokens → chars, conservative)
OPENAI_CONTEXT_BUDGET = {
    "gpt-4o":        int(100_000 * 3.8),
    "gpt-4o-mini":   int(100_000 * 3.8),
    "gpt-4-turbo":   int(120_000 * 3.8),
    "gpt-3.5-turbo": int(14_000  * 3.8),
}


class OpenAILLM:
    """
    OpenAI LLM integration for the Insurance RAG engine.
    Drop-in compatible with GroqLLM — same __call__ interface.

    Setup:
        1. Get an API key at https://platform.openai.com/api-keys
        2. Set env var: OPENAI_API_KEY=sk-...
           OR pass api_key= directly.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(self, api_key=None, model=DEFAULT_MODEL,
                 temperature=0.0, max_tokens=1024):
        try:
            from openai import OpenAI as _OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No OpenAI API key provided. Pass api_key= or set OPENAI_API_KEY.")

        self.client         = _OpenAI(api_key=resolved_key)
        self.model          = model
        self.temperature    = temperature
        self.max_tokens     = max_tokens
        self.name           = f"OpenAI / {model}"
        self.context_budget = OPENAI_CONTEXT_BUDGET.get(model, int(14_000 * 3.8))

        console.print(f"[green]✓ OpenAI LLM ready: {self.name}  "
                      f"| context budget: {self.context_budget:,} chars[/]")

    def _truncate_prompt(self, prompt: str) -> tuple:
        """Same truncation logic as GroqLLM."""
        if len(prompt) <= self.context_budget:
            return prompt, False, 0
        import re as _re
        sources_start = prompt.find("[Source 1:")
        tail_match    = _re.search(r"\nUSER QUERY:", prompt)
        tail_start    = tail_match.start() if tail_match else len(prompt)
        if sources_start == -1 or sources_start >= tail_start:
            budget = self.context_budget - 200
            return (prompt[:budget] + "\n[...truncated...]\n" + prompt[tail_start:],
                    True, len(prompt) - self.context_budget)
        preamble      = prompt[:sources_start]
        tail          = prompt[tail_start:]
        sources_block = prompt[sources_start:tail_start]
        sections      = _re.split(r"\n\n---\n\n", sources_block)
        available     = self.context_budget - len(preamble) - len(tail) - 100
        if available <= 0:
            return preamble + "[Context omitted]\n" + tail, True, len(sources_block)
        n = len(sections)
        budgets = ([available] if n==1 else
                   [int(available*0.40)] + [(available - int(available*0.40))//max(n-1,1)]*(n-1))
        trunc = []
        removed = 0
        for sec, bud in zip(sections, budgets):
            if len(sec) <= bud:
                trunc.append(sec)
            else:
                le = sec.find("\n")
                lbl = sec[:le+1] if le!=-1 else ""
                body = sec[le+1:] if le!=-1 else sec
                bb = bud - len(lbl) - 60
                trunc.append(lbl + (body[:bb]+"\n[...truncated...]" if bb>100
                                    else "[...truncated...]"))
                removed += len(sec) - bud
        return preamble + "\n\n---\n\n".join(trunc) + tail, True, removed

    def __call__(self, prompt: str) -> str:
        prompt, was_trunc, removed = self._truncate_prompt(prompt)
        if was_trunc:
            console.print(f"[yellow]  Prompt truncated: {removed:,} chars (OpenAI {self.model})[/]")
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            console.print(f"[red]OpenAI API error: {e}[/]")
            return f"[OpenAI API error: {e}]"

    def test_connection(self) -> bool:
        try:
            self("Say OK")
            return True
        except Exception:
            return False


def build_openai_llm(api_key=None, model=OpenAILLM.DEFAULT_MODEL) -> OpenAILLM:
    return OpenAILLM(api_key=api_key, model=model)



# ------------------------------------------------------------------ #
#  Evaluator                                                           #
# ------------------------------------------------------------------ #

class SearchEvaluator:
    """
    Offline evaluation of search quality.
    Metrics: nDCG@k, MRR, Precision@k, Recall@k
    """

    @staticmethod
    def evaluate(
        engine: InsuranceRAGEngine,
        eval_set: List[Dict],
    ) -> Dict:
        """
        eval_set: list of {"query": str, "relevant_sections": [str, ...]}
        """
        ndcg_scores, mrr_scores, prec_scores, recall_scores = [], [], [], []
        results_log = []

        for item in eval_set:
            query    = item["query"]
            relevant = set(item.get("relevant_sections", []))
            response = engine.query(query)
            retrieved_sections = [r.chunk.section_title for r in response.source_chunks]

            # Precision@k
            prec = sum(1 for s in retrieved_sections if s in relevant) / max(len(retrieved_sections), 1)
            prec_scores.append(prec)

            # Recall@k
            rec  = sum(1 for s in retrieved_sections if s in relevant) / max(len(relevant), 1)
            recall_scores.append(rec)

            # MRR
            mrr = 0.0
            for rank, sec in enumerate(retrieved_sections):
                if sec in relevant:
                    mrr = 1.0 / (rank + 1)
                    break
            mrr_scores.append(mrr)

            # nDCG@k
            k = len(retrieved_sections)
            dcg  = sum(
                (1.0 / math.log2(rank + 2)) for rank, sec in enumerate(retrieved_sections) if sec in relevant
            )
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

            results_log.append({
                "query": query,
                "precision": round(prec, 3),
                "recall": round(rec, 3),
                "mrr": round(mrr, 3),
                "ndcg": round(ndcg_scores[-1], 3),
            })

        def avg(lst): return round(sum(lst) / len(lst), 4) if lst else 0.0

        return {
            "num_queries": len(eval_set),
            "precision_at_k": avg(prec_scores),
            "recall_at_k": avg(recall_scores),
            "mrr": avg(mrr_scores),
            "ndcg_at_k": avg(ndcg_scores),
            "per_query": results_log,
        }