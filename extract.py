#!/usr/bin/env python3

"""AI-powered structured extraction of naturtype locality descriptions."""

import asyncio
import logging
import pathlib
import random
import re
import time
from enum import Enum
from typing import Annotated

from dotenv import load_dotenv

load_dotenv()

import pyarrow
import pyarrow.parquet
import typer
from pydantic import BaseModel, Field
from pydantic_ai import Agent

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Extract structured ecological assessments from naturtype text descriptions."
)

# ── Models ───────────────────────────────────────────────────────────────────

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


# ── Helpers ──────────────────────────────────────────────────────────────────


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


class _RateLimiter:
    """Token-bucket rate limiter: allows at most `rate` calls per 60 seconds."""

    def __init__(self, rate: float) -> None:
        self._interval = 60.0 / rate  # seconds between tokens
        self._lock = asyncio.Lock()
        self._next_allowed = time.monotonic()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next_allowed = (
                max(self._next_allowed, time.monotonic()) + self._interval
            )


async def _extract_one(
    semaphore: asyncio.Semaphore,
    rate_limiter: _RateLimiter,
    agent: "Agent[None, NaturtypeAnalyse]",
    naturtype_id: str,
    prompt: str,
    max_retries: int,
) -> tuple[str, NaturtypeAnalyse | None]:
    for attempt in range(max_retries + 1):
        await rate_limiter.acquire()
        async with semaphore:
            try:
                result = await agent.run(prompt)
                return naturtype_id, result.output
            except Exception as exc:
                exc_str = str(exc)
                is_rate_limit = "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str
                if is_rate_limit and attempt < max_retries:
                    match = re.search(r"retryDelay.*?(\d+)s", exc_str)
                    suggested = int(match.group(1)) if match else 30
                    jitter = random.uniform(1.0, 5.0)
                    wait = suggested + jitter
                    logger.warning(
                        "Rate limited on %s, retrying in %.0fs (attempt %d/%d)",
                        naturtype_id,
                        wait,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(wait)
                    continue
                logger.error("Failed to extract %s: %s", naturtype_id, exc)
                return naturtype_id, None
    logger.error("Gave up on %s after %d retries", naturtype_id, max_retries)
    return naturtype_id, None


async def _run_all_extractions(
    rows: list[dict],
    model_str: str,
    concurrency: int,
    rpm: float,
    max_retries: int,
) -> list[tuple[str, NaturtypeAnalyse | None]]:
    from tqdm.asyncio import tqdm

    agent: Agent[None, NaturtypeAnalyse] = Agent(
        model_str,
        output_type=NaturtypeAnalyse,
        system_prompt=_SYSTEM_PROMPT,
    )
    semaphore = asyncio.Semaphore(concurrency)
    rate_limiter = _RateLimiter(rpm)

    tasks = [
        _extract_one(
            semaphore,
            rate_limiter,
            agent,
            row["naturtype_id"],
            build_prompt(row),
            max_retries,
        )
        for row in rows
    ]
    return list(await tqdm.gather(*tasks, desc="Extracting", unit="row"))


# ── Command ──────────────────────────────────────────────────────────────────


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
    rpm: Annotated[
        float,
        typer.Option(
            "--rpm",
            min=0.1,
            help="Maximum requests per minute sent to the API (rate limit).",
        ),
    ] = 5.0,
    max_retries: Annotated[
        int,
        typer.Option(
            "--max-retries",
            min=0,
            help="Maximum number of retries per row on rate-limit (429) errors.",
        ),
    ] = 12,
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

    typer.echo(f"Extracting from {len(row_list)} rows using {model} (≤{rpm} RPM) …")

    # ── 2. Run async extraction ───────────────────────────────────────────────
    results: list[tuple[str, NaturtypeAnalyse | None]] = asyncio.run(
        _run_all_extractions(row_list, model, concurrency, rpm, max_retries)
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
