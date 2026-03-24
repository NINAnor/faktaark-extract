#!/usr/bin/env python3

"""Fetch naturtype faktaark data from the Miljødirektoratet ArcGIS REST API."""

import logging
import pathlib
from typing import Annotated, Generator

import dlt
import pyarrow
import pyarrow.csv
import pyarrow.feather
import pyarrow.json
import pyarrow.orc
import pyarrow.parquet
import typer
from dlt.sources.rest_api import rest_api_source

logger = logging.getLogger(__name__)

ARCGIS_BASE_URL = (
    "https://arcgis06.miljodirektoratet.no"
    "/arcgis/rest/services/faktaark/naturtyper/MapServer/0/"
)

app = typer.Typer(
    help="Fetch naturtype faktaark from Miljødirektoratet ArcGIS REST API and write to parquet."
)


def read_table(path: pathlib.Path) -> pyarrow.Table:
    """Read a dataframe file into a PyArrow Table, auto-detecting format by extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pyarrow.csv.read_csv(path)
    if suffix == ".parquet":
        return pyarrow.parquet.read_table(path)
    if suffix in {".json", ".ndjson"}:
        return pyarrow.json.read_json(path)
    if suffix in {".arrow", ".feather"}:
        return pyarrow.feather.read_table(path)
    if suffix == ".orc":
        return pyarrow.orc.ORCFile(path).read()
    raise typer.BadParameter(
        f"Unsupported file format '{suffix}'. "
        "Supported: .csv, .parquet, .json, .ndjson, .arrow, .feather, .orc"
    )


@dlt.resource(name="naturtype_batches", selected=False)
def naturtype_batches(
    ids: list[str], batch_size: int
) -> Generator[list[dict], None, None]:
    """Seed resource that yields batches of naturtypeId WHERE clauses."""
    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]
        quoted = ", ".join(f"'{v}'" for v in batch)
        where_clause = f"naturtypeId IN ({quoted})"
        logger.debug("Batch %d–%d: %s", i, i + len(batch), where_clause)
        yield [{"where_clause": where_clause}]


def build_pipeline_source(ids: list[str], batch_size: int) -> object:
    """Build a dlt REST API source that fetches naturtype records for the given IDs."""
    config = {
        "client": {
            "base_url": ARCGIS_BASE_URL,
        },
        "resources": [
            naturtype_batches(ids, batch_size),
            {
                "name": "naturtyper",
                "endpoint": {
                    "path": "query",
                    "params": {
                        "where": "{resources.naturtype_batches.where_clause}",
                        "outFields": "*",
                        "returnGeometry": "false",
                        "f": "json",
                    },
                    "data_selector": "features",
                    "paginator": {"type": "single_page"},
                },
                "processing_steps": [
                    {"map": lambda record: record["attributes"]},
                ],
            },
        ],
    }
    return rest_api_source(config)  # type: ignore[arg-type]


@app.command()
def fetch(
    input_file: Annotated[
        pathlib.Path,
        typer.Argument(
            exists=True,
            readable=True,
            help="Path to the input dataframe file (.csv, .parquet, .json, .ndjson, .arrow, .feather, .orc).",
        ),
    ],
    column: Annotated[
        str,
        typer.Argument(help="Column name whose values are used as naturtypeId values."),
    ],
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Number of naturtypeIds to include per API request.",
            min=1,
        ),
    ] = 100,
    output: Annotated[
        pathlib.Path,
        typer.Option(
            "--output",
            help="Output directory where parquet files will be written.",
        ),
    ] = pathlib.Path("output"),
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Read INPUT_FILE, extract COLUMN values and fetch their faktaark from the ArcGIS REST API."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # ── 1. Read input dataframe ──────────────────────────────────────────────
    typer.echo(f"Reading {input_file} …")
    try:
        table = read_table(input_file)
    except Exception as exc:
        typer.echo(f"Error reading file: {exc}", err=True)
        raise typer.Exit(1) from exc

    if column not in table.schema.names:
        available = ", ".join(table.schema.names)
        typer.echo(
            f"Column '{column}' not found. Available columns: {available}", err=True
        )
        raise typer.Exit(1)

    ids: list[str] = [str(v) for v in table.column(column).to_pylist() if v is not None]
    typer.echo(f"Found {len(ids)} IDs in column '{column}'.")

    if not ids:
        typer.echo("No IDs to fetch. Exiting.")
        raise typer.Exit(0)

    # ── 2. Build dlt pipeline ────────────────────────────────────────────────
    bucket_url = output.resolve().as_posix()
    typer.echo(
        f"Fetching in batches of {batch_size} → writing parquet to {bucket_url} …"
    )

    pipeline = dlt.pipeline(
        pipeline_name="naturtyper",
        destination=dlt.destinations.filesystem(
            bucket_url=bucket_url,
            kwargs={"auto_mkdir": True},
        ),
        dataset_name="naturtyper",
    )

    source = build_pipeline_source(ids, batch_size)
    load_info = pipeline.run(source, loader_file_format="parquet")

    # ── 3. Report ────────────────────────────────────────────────────────────
    typer.echo(load_info)


def start() -> None:
    """Entry point for the installed package script."""
    app()


if __name__ == "__main__":
    app()
