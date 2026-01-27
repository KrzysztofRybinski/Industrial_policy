# Industrial policy subsidy -> markup pipeline

End-to-end, reproducible Python pipeline for the research project **"Industrial policy subsidy -> markup pipeline (incidence as rents vs capacity)"**.

## Requirements

- Windows + VS Code friendly
- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for environments and dependency management

## Setup (Windows)

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
uv run industrial-policy all --config-path config/config.yaml
```

The CLI also accepts `--config` and `-c` as aliases for `--config-path`.

Or run steps independently:

```powershell
uv run industrial-policy doctor --config-path config/config.yaml
uv run industrial-policy ingest usaspending-download --config-path config/config.yaml
uv run industrial-policy ingest sec --config-path config/config.yaml
uv run industrial-policy match recipients --config-path config/config.yaml
uv run industrial-policy build panel --config-path config/config.yaml
uv run industrial-policy match controls --config-path config/config.yaml
uv run industrial-policy estimate --config-path config/config.yaml
uv run industrial-policy incidence --config-path config/config.yaml
uv run industrial-policy robustness --config-path config/config.yaml
```

The legacy, slower USAspending crawler is still available as `industrial-policy ingest usaspending`.

## Smoke run (fast)

For quick end-to-end validation on a tiny USAspending window:

```powershell
uv run industrial-policy all --config-path config/config.smoke.yaml
```

This writes to `data_smoke/` and `outputs_smoke/` (both ignored by git).

To force re-downloads (otherwise cached chunks/ZIPs are reused):

```powershell
uv run industrial-policy ingest usaspending-download --config-path config/config.yaml --force
uv run industrial-policy ingest sec --config-path config/config.yaml --force
```

A convenience script is also provided:

```powershell
scripts/run_pipeline.ps1
```

## Outputs

All outputs are written to `outputs/`:

- `outputs/tables/` for CSV event-study tables and diagnostics
- `outputs/figures/` for event-study plots
- `outputs/logs/` for pipeline logs

## Data handling

- `data/` and `outputs/` are ignored by git; the pipeline creates all required subfolders at runtime.
- All downloads are cached in `data/raw/` and `data/derived/` (re-runs reuse cached chunks/ZIPs unless `--force` is used).
- DuckDB + Parquet are used as intermediate storage for large SEC FSDS files.
- No proprietary data is required (public USAspending + SEC FSDS).

## Optional HHI input

If you have a Census concentration file, set the path in `config/config.yaml` under `hhi:`. When provided, the pipeline will generate HHI-based heterogeneity outputs; otherwise it logs a skip.

## Development

Run tests:

```powershell
uv run --extra dev pytest
```

Run linting:

```powershell
uv run --extra dev ruff check .
```
