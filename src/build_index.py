"""Index building script for the RAG chatbot.

This script orchestrates the full pipeline:
1. Load and chunk documents (PDFs and MP4 transcripts)
2. Generate embeddings
3. Store in ChromaDB vector database

Usage:
    python -m src.build_index              # Build/update index
    python -m src.build_index --rebuild    # Clear and rebuild from scratch
    python -m src.build_index --stats      # Show index statistics only
"""

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import DATA_DIR, CHROMA_DB_DIR, print_config
from src.data_loader import load_and_chunk_documents
from src.retriever import index_documents, get_collection_stats, clear_index


app = typer.Typer(help="RAG Chatbot Index Builder")
console = Console()


@app.command()
def build(
    rebuild: bool = typer.Option(
        False,
        "--rebuild",
        "-r",
        help="Clear existing index and rebuild from scratch"
    ),
    stats_only: bool = typer.Option(
        False,
        "--stats",
        "-s",
        help="Show index statistics only (no building)"
    ),
    data_dir: Path = typer.Option(
        None,
        "--data-dir",
        "-d",
        help="Override data directory path"
    ),
) -> None:
    """Build or update the document index.

    This command loads documents from the data directory, generates embeddings,
    and stores them in ChromaDB for semantic search.
    """
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]RAG Chatbot - Index Builder[/bold cyan]",
        border_style="cyan"
    ))

    # Show configuration
    if not stats_only:
        console.print("\n[bold]Configuration:[/bold]")
        print_config()

    # Show stats if requested
    if stats_only:
        try:
            stats = get_collection_stats()
            _display_stats(stats)
        except Exception as e:
            console.print(f"[red]Error getting stats: {e}[/red]")
            sys.exit(1)
        return

    # Validate data directory exists
    data_path = data_dir if data_dir else DATA_DIR
    if not data_path.exists():
        console.print(f"\n[red]Error: Data directory not found: {data_path}[/red]")
        console.print("[yellow]Please create the directory and add PDF/MP4 files.[/yellow]")
        sys.exit(1)

    # Check for files
    pdf_files = list(data_path.glob("*.pdf"))
    mp4_files = list(data_path.glob("*.mp4")) + list(data_path.glob("*.MP4"))

    if not pdf_files and not mp4_files:
        console.print(f"\n[red]Error: No PDF or MP4 files found in {data_path}[/red]")
        console.print("[yellow]Please add source documents to the data directory.[/yellow]")
        sys.exit(1)

    console.print(f"\n[green]Found {len(pdf_files)} PDF(s) and {len(mp4_files)} MP4(s)[/green]")

    # Clear index if rebuild requested
    if rebuild:
        console.print("\n[yellow]Clearing existing index...[/yellow]")
        try:
            clear_index()
        except Exception as e:
            console.print(f"[red]Error clearing index: {e}[/red]")
            sys.exit(1)

    # Step 1: Load and chunk documents
    console.print("\n[bold cyan]Step 1: Loading and chunking documents[/bold cyan]")
    try:
        documents = load_and_chunk_documents(data_path)
    except Exception as e:
        console.print(f"\n[red]Error loading documents: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if not documents:
        console.print("[red]No documents were loaded. Exiting.[/red]")
        sys.exit(1)

    # Step 2: Generate embeddings and index
    console.print("\n[bold cyan]Step 2: Generating embeddings and indexing[/bold cyan]")
    try:
        index_documents(documents)
    except Exception as e:
        console.print(f"\n[red]Error indexing documents: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 3: Show final statistics
    console.print("\n[bold cyan]Step 3: Index Statistics[/bold cyan]")
    try:
        stats = get_collection_stats()
        _display_stats(stats)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not fetch stats: {e}[/yellow]")

    # Success message
    console.print("\n")
    console.print(Panel.fit(
        "[bold green]✓ Index built successfully![/bold green]\n\n"
        "You can now query the chatbot:\n"
        "  [cyan]python -m src.chatbot query \"What is RAG?\"[/cyan]",
        border_style="green"
    ))


def _display_stats(stats: dict) -> None:
    """Display index statistics in a formatted table."""
    table = Table(title="Index Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Collection Name", stats["collection_name"])
    table.add_row("Document Count", str(stats["count"]))
    table.add_row("Storage Path", stats["storage_path"])

    console.print(table)


@app.command()
def clear() -> None:
    """Clear the entire index (delete all documents).

    Warning: This operation is irreversible!
    """
    # Confirm with user
    console.print("[yellow]Warning: This will delete all indexed documents![/yellow]")
    confirm = typer.confirm("Are you sure you want to continue?")

    if not confirm:
        console.print("[cyan]Operation cancelled.[/cyan]")
        return

    try:
        clear_index()
        console.print("[green]✓ Index cleared successfully[/green]")
    except Exception as e:
        console.print(f"[red]Error clearing index: {e}[/red]")
        sys.exit(1)


@app.command()
def stats() -> None:
    """Show current index statistics."""
    try:
        stats = get_collection_stats()
        console.print("\n")
        _display_stats(stats)
        console.print("\n")
    except Exception as e:
        console.print(f"[red]Error getting stats: {e}[/red]")
        sys.exit(1)


def main() -> None:
    """Entry point for the index builder."""
    app()


if __name__ == "__main__":
    main()
