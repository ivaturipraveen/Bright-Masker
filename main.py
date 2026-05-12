from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from config import AppConfig, Config
from pipeline.orchestrator import PiiMaskingPipeline
from utils.logger import configure_logging, get_logger

app = typer.Typer(name="pii-masker", help="Production-grade PII masking pipeline.")
console = Console()
log = get_logger(__name__)


def _build_app_config(config_path: Path) -> AppConfig:
    settings = Config(entities_config_path=config_path)
    configure_logging(settings.log_level)
    return AppConfig(settings=settings)


@app.command("mask-text")
def mask_text(
    text: str = typer.Option(..., "--text", help="Input text as string"),
    config: Path = typer.Option(
        Path("entities_config.yaml"), "--config", help="Path to entities_config.yaml"
    ),
    output: str = typer.Option("table", "--output", help="Output format: table | json"),
    show_stats: bool = typer.Option(False, "--show-stats", help="Show per-layer timing stats"),
) -> None:
    app_config = _build_app_config(config)
    pipeline = PiiMaskingPipeline(app_config)
    result = pipeline.process_sync(text)

    if output == "json":
        console.print_json(result.model_dump_json())
        return

    table = Table(title="PII Detection Results")
    table.add_column("Entity Type", style="cyan")
    table.add_column("Original", style="red")
    table.add_column("Masked", style="green")
    table.add_column("Confidence", justify="right")
    table.add_column("Source", style="dim")
    table.add_column("Strategy", style="dim")

    for span in result.detected_spans:
        masked = next(
            (m.masked for m in result.masked_spans if m.entity_id == span.entity_id and m.original == span.text),
            "N/A",
        )
        entity_cfg = app_config.entities_by_id.get(span.entity_id)
        strategy = entity_cfg.masking.strategy.value if entity_cfg else "unknown"
        table.add_row(
            span.display_name,
            span.text,
            masked,
            f"{span.confidence:.2f}",
            span.source,
            strategy,
        )

    console.print(table)
    s = result.stats
    llm_part = f" | llm {s.llm_ms:.0f}ms" if s.llm_ms is not None else ""
    console.print(
        f"[bold]{s.spans_total} entities detected[/bold], "
        f"{s.total_ms:.0f}ms total "
        f"(pattern {s.pattern_ms:.0f}ms | ner {s.ner_ms:.0f}ms{llm_part})"
    )

    if show_stats:
        console.print(result.stats.model_dump_json(indent=2))

    console.rule()
    console.print("[bold]Masked text:[/bold]")
    console.print(result.masked_text)


@app.command("mask-file")
def mask_file(
    input: Path = typer.Option(..., "--input", help="Input file (txt, json, csv)"),
    output: Path = typer.Option(..., "--output", help="Output file path"),
    format: str = typer.Option("text", "--format", help="Output format: json | text"),
    config: Path = typer.Option(Path("entities_config.yaml"), "--config"),
) -> None:
    app_config = _build_app_config(config)
    pipeline = PiiMaskingPipeline(app_config)

    text = input.read_text(encoding="utf-8")
    result = pipeline.process_sync(text)

    if format == "json":
        output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    else:
        output.write_text(result.masked_text, encoding="utf-8")

    console.print(
        f"[green]✓[/green] Masked {result.stats.spans_total} entities "
        f"in {result.stats.total_ms:.0f}ms → {output}"
    )


@app.command("mask-batch")
def mask_batch(
    input_dir: Path = typer.Option(..., "--input-dir", help="Directory of input files"),
    output_dir: Path = typer.Option(..., "--output-dir", help="Directory for masked output"),
    workers: int = typer.Option(4, "--workers", help="Concurrency"),
    config: Path = typer.Option(Path("entities_config.yaml"), "--config"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    app_config = _build_app_config(config)
    pipeline = PiiMaskingPipeline(app_config)

    files = list(input_dir.glob("*.txt")) + list(input_dir.glob("*.json"))
    if not files:
        console.print("[yellow]No .txt or .json files found.[/yellow]")
        return

    texts = [f.read_text(encoding="utf-8") for f in files]

    async def _run() -> None:
        results = await pipeline.process_batch(texts, max_concurrency=workers)
        for file, result in zip(files, results):
            out_path = output_dir / file.name
            out_path.write_text(result.masked_text, encoding="utf-8")

    asyncio.run(_run())
    console.print(f"[green]✓[/green] Processed {len(files)} files → {output_dir}")


@app.command("list-entities")
def list_entities(
    config: Path = typer.Option(Path("entities_config.yaml"), "--config"),
) -> None:
    app_config = _build_app_config(config)

    table = Table(title="Configured PII Entities")
    table.add_column("ID", style="cyan")
    table.add_column("Display Name")
    table.add_column("Enabled", justify="center")
    table.add_column("Strategy")
    table.add_column("Format")
    table.add_column("Layers")

    all_entities_raw = _load_all_entities_raw(config)

    for entity in all_entities_raw:
        eid = entity.get("id", "")
        layers = []
        if entity.get("presidio_type"):
            layers.append("pattern")
        layers.append("ner")
        layers.append("llm")

        masking = entity.get("masking", {})
        table.add_row(
            eid,
            entity.get("display_name", ""),
            "[green]✓[/green]" if entity.get("enabled", True) else "[red]✗[/red]",
            masking.get("strategy", ""),
            masking.get("format", ""),
            "/".join(layers),
        )

    console.print(table)


def _load_all_entities_raw(config_path: Path) -> list[dict]:
    import yaml
    with open(config_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data.get("entities", [])


@app.command("benchmark")
def benchmark(
    config: Path = typer.Option(Path("entities_config.yaml"), "--config"),
) -> None:
    from benchmarks.benchmark import run_benchmark
    asyncio.run(run_benchmark(config))


if __name__ == "__main__":
    app()
