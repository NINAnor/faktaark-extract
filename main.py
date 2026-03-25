#!/usr/bin/env python3

"""Fetch naturtype faktaark data from the Miljødirektoratet ArcGIS REST API."""

import asyncio
import logging
import pathlib
import tempfile
from enum import Enum
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
from pydantic import BaseModel, Field
from pydantic_ai import Agent

logger = logging.getLogger(__name__)

ARCGIS_BASE_URL = (
    "https://arcgis06.miljodirektoratet.no"
    "/arcgis/rest/services/faktaark/naturtyper/MapServer/0/"
)

app = typer.Typer(
    help="Fetch naturtype faktaark from Miljødirektoratet ArcGIS REST API and write to parquet."
)

# ── Extraction models ────────────────────────────────────────────────────────

_TEXT_FIELDS: dict[str, str] = {
    "innledning": "Innledning",
    "beliggenhet_og_naturgrunnlag": "Beliggenhet og naturgrunnlag",
    "naturtyper_og_utforminger": "Naturtyper og utforminger",
    "artsmangfold": "Artsmangfold",
    "paavirkning": "Påvirkning",
    "fremmede_arter": "Fremmede arter",
    "raad_om_skjoetsel_og_hensyn": "Råd om skjøtsel og hensyn",
    "landskap": "Landskap",
}

_HEVDSTATUS_LABELS: dict[int, str] = {
    1: "god hevd",
    2: "svak hevd",
    3: "ingen hevd / restaureringstilstand",
    4: "gjengrodd",
    5: "ikke vurdert",
}

_SYSTEM_PROMPT = """\
Du er ekspert på norsk naturtypeforvaltning og biologisk mangfold.
Du analyserer beskrivelsestekster fra Miljødirektoratets naturtypekartlegging \
og trekker ut strukturert informasjon.

Den viktigste oppgaven er å vurdere om området er utsatt for eller i risiko \
for gjengroing (tilgroing / reforestation / overgrowth).

Regler:
- Svar kun på grunnlag av det som faktisk er beskrevet i teksten.
- Spekuler ikke utover det som er nevnt.
- Hvis et felt ikke er relevant eller ikke omtalt, bruk standardverdier \
(tom liste, null, eller "ukjent").
- Svar med norske verdier der det er naturlig.\
"""


class GjengroingGrad(str, Enum):
    ingen = "ingen"
    lav = "lav"
    moderat = "moderat"
    høy = "høy"


class Tilstand(str, Enum):
    god = "god"
    moderat = "moderat"
    dårlig = "dårlig"
    ukjent = "ukjent"


class NaturtypeAnalyse(BaseModel):
    """Structured extraction of a naturtype locality description."""

    gjengroing_risiko: bool = Field(
        description=(
            "Er området utsatt for eller i risiko for gjengroing "
            "(tilgroing / reforestation / overgrowth)?"
        )
    )
    gjengroing_grad: GjengroingGrad = Field(
        description="Nåværende grad av gjengroing observert i eller rundt lokaliteten."
    )
    gjengroing_beskrivelse: str | None = Field(
        default=None,
        description=(
            "Kort beskrivelse av gjengroingssituasjonen, eller null hvis ikke omtalt."
        ),
    )
    skjoetsel_anbefalt: bool = Field(
        description="Er aktiv skjøtsel eller tiltak anbefalt for å bevare naturverdiene?"
    )
    skjoetsel_tiltak: list[str] = Field(
        default_factory=list,
        description="Konkrete skjøtselstiltak som anbefales i teksten.",
    )
    fremmede_arter_liste: list[str] = Field(
        default_factory=list,
        description="Fremmede eller invaderende arter nevnt i teksten.",
    )
    paavirkningsfaktorer: list[str] = Field(
        default_factory=list,
        description="Påvirknings- og trusselfaktorer som er identifisert.",
    )
    tilstand: Tilstand = Field(
        description="Samlet vurdering av den økologiske tilstanden basert på teksten."
    )
    noekkelarter: list[str] = Field(
        default_factory=list,
        description="Nøkkelarter eller viktige naturverdier nevnt i teksten.",
    )


# ── Extraction helpers ───────────────────────────────────────────────────────


def build_prompt(row: dict) -> str:
    """Build a text prompt for a single naturtype row."""
    hevd = row.get("hevdstatus")
    hevd_label = (
        _HEVDSTATUS_LABELS.get(hevd, str(hevd)) if hevd is not None else "ikke oppgitt"
    )

    parts = [
        f"Naturtype: {row.get('naturtype', '?')} | "
        f"Verdi: {row.get('verdi', '?')} | "
        f"Hevdstatus: {hevd_label}",
        f"Områdenavn: {row.get('omraadenavn', '?')}",
        "",
    ]

    for field, label in _TEXT_FIELDS.items():
        val = (row.get(field) or "").strip()
        if val:
            parts.append(f"--- {label} ---")
            parts.append(val)
            parts.append("")

    return "\n".join(parts)


async def _extract_one(
    semaphore: asyncio.Semaphore,
    agent: "Agent[None, NaturtypeAnalyse]",
    naturtype_id: str,
    prompt: str,
) -> tuple[str, NaturtypeAnalyse | None]:
    async with semaphore:
        try:
            result = await agent.run(prompt)
            return naturtype_id, result.output
        except Exception as exc:
            logger.error("Failed to extract %s: %s", naturtype_id, exc)
            return naturtype_id, None


async def _run_all_extractions(
    rows: list[dict],
    model_str: str,
    concurrency: int,
) -> list[tuple[str, NaturtypeAnalyse | None]]:
    agent: Agent[None, NaturtypeAnalyse] = Agent(
        model_str,
        output_type=NaturtypeAnalyse,
        system_prompt=_SYSTEM_PROMPT,
    )
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [
        _extract_one(semaphore, agent, row["naturtype_id"], build_prompt(row))
        for row in rows
    ]
    return list(await asyncio.gather(*tasks))


# ── Fetch helpers ────────────────────────────────────────────────────────────


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
                    "method": "POST",
                    "data": {
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


# ── Commands ─────────────────────────────────────────────────────────────────


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

    # ── 2. Run dlt pipeline into a temp dir, then consolidate ────────────────
    output.mkdir(parents=True, exist_ok=True)
    out_file = output / "naturtyper.parquet"

    typer.echo(f"Fetching in batches of {batch_size} …")

    with tempfile.TemporaryDirectory() as tmp:
        pipeline = dlt.pipeline(
            pipeline_name="naturtyper",
            destination=dlt.destinations.filesystem(
                bucket_url=tmp,
                kwargs={"auto_mkdir": True},
            ),
            dataset_name="naturtyper",
        )

        source = build_pipeline_source(ids, batch_size)
        pipeline.run(source, loader_file_format="parquet")

        # Collect all parquet files written for the naturtyper table
        parquet_files = sorted(
            pathlib.Path(tmp).glob("naturtyper/naturtyper/*.parquet")
        )
        if not parquet_files:
            typer.echo("No data returned by the API.", err=True)
            raise typer.Exit(1)

        tables = [pyarrow.parquet.read_table(f) for f in parquet_files]

    # Merge, strip dlt metadata columns, write single file
    merged = pyarrow.concat_tables(tables)
    dlt_cols = {c for c in merged.schema.names if c.startswith("_dlt_")}
    if dlt_cols:
        merged = merged.select([c for c in merged.schema.names if c not in dlt_cols])

    pyarrow.parquet.write_table(merged, out_file)

    # ── 3. Report ────────────────────────────────────────────────────────────
    typer.echo(
        f"Wrote {merged.num_rows} rows × {merged.num_columns} columns" f" → {out_file}"
    )


@app.command()
def extract(
    input_file: Annotated[
        pathlib.Path,
        typer.Option(
            "--input",
            exists=True,
            readable=True,
            help="Input parquet file produced by the fetch command.",
        ),
    ] = pathlib.Path("output/naturtyper.parquet"),
    output: Annotated[
        pathlib.Path,
        typer.Option(
            "--output",
            help="Output parquet file with AI-extracted fields.",
        ),
    ] = pathlib.Path("output/naturtyper_analyse.parquet"),
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help=(
                "PydanticAI model string, e.g. 'google-gla:gemini-2.0-flash' "
                "or 'openai:gpt-4o-mini'. The provider's API key must be set "
                "in the appropriate environment variable beforehand."
            ),
        ),
    ] = "google-gla:gemini-3-flash-preview",
    concurrency: Annotated[
        int,
        typer.Option(
            "--concurrency",
            min=1,
            help="Maximum number of parallel API calls.",
        ),
    ] = 10,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Process only the first N rows (useful for testing).",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Extract structured information from text descriptions using an AI agent.

    Reads the parquet produced by the fetch command, sends each row's text
    descriptions to the configured AI model and writes the structured results
    to a separate parquet file joinable on naturtype_id.

    The most important extracted field is gjengroing_risiko, which indicates
    whether the area can be subject to reforestation (gjengroing).
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # ── 1. Read input ─────────────────────────────────────────────────────────
    typer.echo(f"Reading {input_file} …")
    try:
        table = pyarrow.parquet.read_table(input_file)
    except Exception as exc:
        typer.echo(f"Error reading file: {exc}", err=True)
        raise typer.Exit(1) from exc

    col_dict = table.to_pydict()
    n = len(table)
    row_list = [{col: col_dict[col][i] for col in col_dict} for i in range(n)]

    if limit is not None:
        row_list = row_list[:limit]

    typer.echo(f"Extracting from {len(row_list)} rows using {model} …")

    # ── 2. Run async extraction ───────────────────────────────────────────────
    results: list[tuple[str, NaturtypeAnalyse | None]] = asyncio.run(
        _run_all_extractions(row_list, model, concurrency)
    )

    # ── 3. Build output parquet ───────────────────────────────────────────────
    naturtype_ids = [r[0] for r in results]
    analyses = [r[1] for r in results]

    failed = sum(1 for a in analyses if a is None)
    if failed:
        typer.echo(
            f"Warning: {failed} rows failed extraction and will have null values.",
            err=True,
        )

    out_data: dict[str, list] = {"naturtype_id": naturtype_ids}
    for field_name in NaturtypeAnalyse.model_fields:
        col = []
        for a in analyses:
            if a is None:
                col.append(None)
            else:
                val = getattr(a, field_name)
                col.append(val.value if isinstance(val, Enum) else val)
        out_data[field_name] = col

    out_table = pyarrow.Table.from_pydict(out_data)

    output.parent.mkdir(parents=True, exist_ok=True)
    pyarrow.parquet.write_table(out_table, output)

    # ── 4. Report ─────────────────────────────────────────────────────────────
    typer.echo(
        f"Wrote {out_table.num_rows} rows × {out_table.num_columns} columns → {output}"
    )


def start() -> None:
    """Entry point for the installed package script."""
    app()


if __name__ == "__main__":
    app()
