# Industrial policy subsidy → markup pipeline

End-to-end, reproducible Python pipeline for the research project **“Industrial policy subsidy → markup pipeline (incidence as rents vs capacity)”**.

## Requirements

- Windows + VS Code friendly
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for environments and dependency management

## Setup (Windows)

```powershell
# From repo root
uv venv
uv pip install -e .
```

For development (tests + linting):

```powershell
uv venv
uv pip install -e ".[dev]"
```

## Configure

1. Copy the environment template:

```powershell
copy .env.example .env
```

2. Update `SEC_USER_AGENT` in `.env` or directly in `config/config.yaml` (environment variables override YAML):

```yaml
sec:
  user_agent: "YOUR NAME your.email@domain.com"
```

This is required to comply with SEC fair-access rules.

## Run the full pipeline

```powershell
uv run industrial-policy all --config config/config.yaml
```

Or run steps independently:

```powershell
uv run industrial-policy ingest usaspending --config config/config.yaml
uv run industrial-policy ingest sec --config config/config.yaml
uv run industrial-policy match recipients --config config/config.yaml
uv run industrial-policy build panel --config config/config.yaml
uv run industrial-policy match controls --config config/config.yaml
uv run industrial-policy estimate --config config/config.yaml
```

A convenience script is also provided:

```powershell
scripts/run_pipeline.ps1
```

## Outputs

All outputs are written to `outputs/`:

- `outputs/tables/` for CSV + LaTeX event-study tables
- `outputs/figures/` for event-study plots
- `outputs/logs/` for pipeline logs

## Data handling

- All downloads are cached in `data/cache/` and `data/raw/`.
- DuckDB + Parquet are used as intermediate storage for large SEC FSDS files.
- No proprietary data is required (public USAspending + SEC FSDS).

## Development

Run tests:

```powershell
uv run pytest
```

Run linting:

```powershell
uv run ruff check .
```
