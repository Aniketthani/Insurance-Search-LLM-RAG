#!/usr/bin/env python3
"""
Insurance Document RAG — Main Entry Point
==========================================
P&C / Reinsurance Hybrid Search PoC
100% Open-Source

Usage:
    python main.py --demo                         # Built-in sample docs
    python main.py --pdf path/policy.pdf          # Single PDF
    python main.py --pdf p1.pdf p2.pdf p3.pdf     # Multiple PDFs
    python main.py --interactive                  # Interactive Q&A loop
    python main.py --eval                         # Run evaluation suite
    python main.py --demo --no-reranker           # Skip reranker (faster)
    python main.py --demo --query "flood limit"   # Single query
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from rich.console import Console
from rich.rule import Rule
from rich.prompt import Prompt

from src.parser     import InsuranceDocumentParser
from src.chunker    import InsuranceChunker
from src.search_index import InsuranceHybridSearchIndex
from src.rag_engine import InsuranceRAGEngine, SearchEvaluator
from src.sample_docs import SAMPLE_DOCS, get_sample_queries
from src.display    import (
    print_banner, print_index_stats, print_search_results, print_eval_results
)

console = Console()

# ── Document type hints for better LOB / category tagging ────────────────
DOC_TYPE_HINTS = {
    "policy":    ("Property & Casualty", "Policy"),
    "treaty":    ("Reinsurance", "Treaty"),
    "claims":    ("Property & Casualty", "Claims"),
    "sop":       ("Property & Casualty", "SOP"),
    "endorsement": ("Property & Casualty", "Endorsement"),
    "compliance":  ("Compliance", "Regulatory"),
}


def detect_doc_type(doc_name: str):
    name_lower = doc_name.lower()
    for keyword, (lob, category) in DOC_TYPE_HINTS.items():
        if keyword in name_lower:
            return lob, category
    return "Property & Casualty", "Policy"


def build_pipeline(pdf_paths=None, use_demo=False, use_reranker=True):
    """Parse, chunk, and index documents. Return the RAG engine."""

    parser  = InsuranceDocumentParser(verbose=True)
    index   = InsuranceHybridSearchIndex(use_reranker=use_reranker, verbose=True)

    all_chunks = []

    # ── Load documents ─────────────────────────────────────────────────
    if use_demo:
        console.print(Rule("[bold cyan]Loading sample insurance documents", style="cyan"))
        for doc_name, doc_text in SAMPLE_DOCS.items():
            lob, category = detect_doc_type(doc_name)
            console.print(f"\n[cyan]▶ {doc_name}[/]  [dim](LOB: {lob}, Type: {category})[/]")

            parsed  = parser.parse(doc_text)
            parsed.doc_name = doc_name

            chunker = InsuranceChunker(lob=lob, doc_category=category, verbose=True)
            chunks  = chunker.chunk(parsed)
            all_chunks.extend(chunks)

    if pdf_paths:
        console.print(Rule("[bold cyan]Loading PDF documents", style="cyan"))
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                console.print(f"[red]✗ File not found: {pdf_path}[/]")
                continue

            doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
            lob, category = detect_doc_type(doc_name)
            console.print(f"\n[cyan]▶ {doc_name}[/]  [dim](LOB: {lob}, Type: {category})[/]")

            parsed  = parser.parse(pdf_path)
            chunker = InsuranceChunker(lob=lob, doc_category=category, verbose=True)
            chunks  = chunker.chunk(parsed)
            all_chunks.extend(chunks)

    if not all_chunks:
        console.print("[red]No documents loaded. Use --demo or --pdf[/]")
        sys.exit(1)

    # ── Index all chunks ──────────────────────────────────────────────
    console.print(Rule("[bold cyan]Building Hybrid Index", style="cyan"))
    index.add_chunks(all_chunks)

    # ── Build RAG engine ──────────────────────────────────────────────
    engine = InsuranceRAGEngine(
        search_index=index,
        top_k=5,
        use_parent_context=True,
        context_only=True,   # No LLM required for PoC
        verbose=True,
    )

    return engine


def run_demo_queries(engine):
    """Run a representative set of insurance queries."""
    demo_queries = [
        "What is the deductible for flood damage?",
        "Does this policy cover cyber-induced business interruption?",
        "What is the attachment point for Layer 1 reinsurance?",
        "How long does the insured have to notify a claim?",
        "What is the maximum reinsurance liability per occurrence?",
        "Who handles claims above $2 million?",
    ]

    console.print(Rule("[bold yellow]Demo Queries", style="yellow"))
    for q in demo_queries:
        response = engine.query(q)
        print_search_results(response)


def run_evaluation(engine):
    """Run the labelled evaluation suite and print metrics."""
    eval_set = get_sample_queries()
    console.print(Rule("[bold magenta]Running Evaluation Suite", style="magenta"))
    console.print(f"[dim]{len(eval_set)} labelled queries...[/]\n")
    results = SearchEvaluator.evaluate(engine, eval_set)
    print_eval_results(results)
    return results


def interactive_mode(engine):
    """Interactive Q&A loop."""
    console.print(Rule("[bold green]Interactive Mode — type 'quit' to exit", style="green"))
    console.print("[dim]Tip: prefix with 'doc:<name>' to filter by document[/]\n")

    while True:
        try:
            query = Prompt.ask("[bold cyan]Query")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/]")
            break

        if query.lower() in ("quit", "exit", "q"):
            break
        if not query.strip():
            continue

        # Parse optional doc filter: "doc:property_policy flood deductible"
        doc_filter = None
        if query.startswith("doc:"):
            parts = query.split(" ", 1)
            doc_filter = parts[0][4:]
            query = parts[1] if len(parts) > 1 else ""

        response = engine.query(query, doc_filter=doc_filter)
        print_search_results(response)


# ── Entry point ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Insurance Document RAG — Hybrid Search PoC"
    )
    ap.add_argument("--demo",        action="store_true", help="Use built-in sample documents")
    ap.add_argument("--pdf",         nargs="+",           help="Path(s) to PDF files")
    ap.add_argument("--query",       type=str,            help="Single query to run")
    ap.add_argument("--interactive", action="store_true", help="Interactive Q&A mode")
    ap.add_argument("--eval",        action="store_true", help="Run evaluation suite")
    ap.add_argument("--no-reranker", action="store_true", help="Disable cross-encoder reranker")
    ap.add_argument("--show-stats",  action="store_true", help="Print index statistics")
    args = ap.parse_args()

    print_banner()

    # Default to demo if nothing specified
    if not args.demo and not args.pdf:
        args.demo = True

    use_reranker = not args.no_reranker

    # Build pipeline
    engine = build_pipeline(
        pdf_paths=args.pdf,
        use_demo=args.demo,
        use_reranker=use_reranker,
    )

    if args.show_stats:
        print_index_stats(engine.index.stats())

    # Run modes
    if args.query:
        console.print(Rule("[bold yellow]Single Query", style="yellow"))
        response = engine.query(args.query)
        print_search_results(response)

    if args.eval:
        run_evaluation(engine)

    if args.interactive:
        interactive_mode(engine)

    if not args.query and not args.eval and not args.interactive:
        # Default: run demo queries
        run_demo_queries(engine)

    console.print()
    console.print(Rule("[dim]Done", style="dim"))


if __name__ == "__main__":
    main()
