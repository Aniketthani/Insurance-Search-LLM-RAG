"""
Display utilities — rich terminal output for the insurance RAG PoC.
"""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.rule import Rule
from rich import box
from rich.progress import Progress, SpinnerColumn, TextColumn
from src.rag_engine import RAGResponse

console = Console()


def print_banner():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]🏛  Insurance Document RAG — Hybrid Search PoC[/]\n"
        "[dim]BM25 + Dense Vector + RRF Fusion + Cross-Encoder Reranker[/]\n"
        "[dim]100% Open-Source · P&C / Reinsurance Domain[/]",
        border_style="cyan",
        padding=(1, 4),
    ))
    console.print()


def print_index_stats(stats: dict):
    t = Table(box=box.ROUNDED, show_header=False, border_style="dim")
    t.add_column("Key",   style="cyan")
    t.add_column("Value", style="white")
    for k, v in stats.items():
        t.add_row(str(k).replace("_", " ").title(), str(v))
    console.print(Panel(t, title="[bold]Index Stats", border_style="cyan"))


def print_search_results(response: RAGResponse, show_scores: bool = True):
    console.print()
    console.print(Rule(f"[bold yellow]Query:[/] {response.query}", style="yellow"))
    console.print()

    for i, r in enumerate(response.source_chunks):
        score_str = ""
        if show_scores:
            score_str = (
                f"[dim]BM25={r.bm25_score:.3f}  "
                f"Vec={r.vector_score:.3f}  "
                f"RRF={r.rrf_score:.4f}"
            )
            if r.rerank_score is not None:
                score_str += f"  [bold green]Rerank={r.rerank_score:.3f}[/]"
            score_str += "[/]"

        nums = ""
        if r.chunk.numeric_values:
            nums = f"\n[bold yellow]💰 Key figures:[/] {', '.join(r.chunk.numeric_values[:6])}"

        snippet = r.chunk.raw_text[:380].replace("\n", " ")
        if len(r.chunk.raw_text) > 380:
            snippet += "…"

        console.print(Panel(
            f"[bold]{r.chunk.doc_name}[/] · Section: [cyan]{r.chunk.section_title}[/] · Page {r.chunk.page_num}\n"
            f"{score_str}\n\n"
            f"{snippet}"
            f"{nums}",
            title=f"[bold green]#{i+1}[/]  Final score: [bold]{r.final_score:.4f}[/]",
            border_style="green" if i == 0 else "dim",
            expand=False,
        ))

    # Guardrails panel
    grd_color = "green" if response.guardrail_passed else "red"
    grd_icon  = "✅" if response.guardrail_passed else "⚠️"
    grd_lines = [
        f"{grd_icon} Guardrail: {'PASSED' if response.guardrail_passed else 'WARNINGS'}",
        f"📊 Groundedness score: {response.groundedness_score:.1%}",
    ]
    if response.guardrail_warnings:
        for w in response.guardrail_warnings:
            grd_lines.append(w)
    if response.numerical_audit.get("unverified"):
        grd_lines.append(f"🔢 Unverified numbers: {', '.join(response.numerical_audit['unverified'])}")

    console.print(Panel(
        "\n".join(grd_lines),
        title="[bold]Guardrails & Audit",
        border_style=grd_color,
        expand=False,
    ))


def print_eval_results(eval_results: dict):
    console.print()
    console.print(Rule("[bold magenta]Evaluation Results", style="magenta"))
    console.print()

    # Summary metrics
    t = Table(box=box.ROUNDED, border_style="magenta")
    t.add_column("Metric",    style="cyan", justify="left")
    t.add_column("Score",     style="bold white", justify="center")
    t.add_column("Target",    style="dim", justify="center")
    t.add_column("Status",    justify="center")

    targets = {
        "precision_at_k": 0.85,
        "recall_at_k": 0.80,
        "mrr": 0.88,
        "ndcg_at_k": 0.82,
    }
    labels = {
        "precision_at_k": "Precision@k",
        "recall_at_k": "Recall@k",
        "mrr": "MRR",
        "ndcg_at_k": "nDCG@k",
    }

    for key, label in labels.items():
        score  = eval_results.get(key, 0.0)
        target = targets[key]
        status = "✅" if score >= target else "⚠️"
        t.add_row(label, f"{score:.4f}", f">{target}", status)

    console.print(t)
    console.print()

    # Per-query table
    pq = eval_results.get("per_query", [])
    if pq:
        t2 = Table(title="Per-Query Breakdown", box=box.SIMPLE, border_style="dim")
        t2.add_column("Query", style="cyan", max_width=45, no_wrap=False)
        t2.add_column("P@k",  justify="center")
        t2.add_column("R@k",  justify="center")
        t2.add_column("MRR",  justify="center")
        t2.add_column("nDCG", justify="center")
        for row in pq:
            t2.add_row(
                row["query"][:45],
                str(row["precision"]),
                str(row["recall"]),
                str(row["mrr"]),
                str(row["ndcg"]),
            )
        console.print(t2)
