"""CLI entrypoints for the industrial policy pipeline."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import typer

from industrial_policy.analysis.did import run_event_studies
from industrial_policy.analysis.incidence import run_incidence_analysis
from industrial_policy.analysis.outputs import save_event_study_outputs, save_event_study_summary
from industrial_policy.analysis.robustness import run_robustness_suite
from industrial_policy.analysis.sample_report import write_sample_construction_report
from industrial_policy.config import load_config
from industrial_policy.entity.name_matching import match_recipients
from industrial_policy.entity.sec_company_lookup import fetch_company_lookup
from industrial_policy.features.awards import prepare_awards
from industrial_policy.features.financials import compute_financial_features
from industrial_policy.features.panel import build_event_panel
from industrial_policy.ingest.eu_state_aid import load_eu_state_aid
from industrial_policy.ingest.census_concentration import ingest_census_concentration
from industrial_policy.ingest.sec_fsds import fetch_sec_fsds
from industrial_policy.ingest.usaspending import fetch_usaspending_awards
from industrial_policy.log import setup_logging
from industrial_policy.match.diagnostics import write_matching_diagnostics
from industrial_policy.match.nearest_neighbor import nearest_neighbor_match
from industrial_policy.match.propensity import build_propensity_scores
from industrial_policy.utils.paths import ensure_dirs
from industrial_policy.viz.event_study_plots import plot_event_study

load_dotenv()

app = typer.Typer(add_completion=False)
ingest_app = typer.Typer()
match_app = typer.Typer()
build_app = typer.Typer()

app.add_typer(ingest_app, name="ingest")
app.add_typer(match_app, name="match")
app.add_typer(build_app, name="build")


def _configure(config_path: Path) -> dict:
    config = load_config(config_path)
    ensure_dirs(config)
    outputs_dir = Path(config["project"]["outputs_dir"])
    setup_logging(outputs_dir / "logs")
    return config


config_option = typer.Option(
    "config/config.yaml",
    "--config-path",
    "--config",
    "-c",
    envvar="INDPOLICY_CONFIG",
    help="Path to config YAML",
)


def _require_file(path: Path, command: str) -> None:
    if not path.exists():
        raise typer.BadParameter(f"Missing required file: {path}. Run: {command}")


@ingest_app.command("usaspending")
def ingest_usaspending(
    config_path: Path = config_option,
    force: bool = typer.Option(False, "--force", help="Re-download USAspending data."),
) -> None:
    """Fetch USAspending awards."""
    config = _configure(config_path)
    fetch_usaspending_awards(config, force=force)


@ingest_app.command("sec")
def ingest_sec(
    config_path: Path = config_option,
    force: bool = typer.Option(False, "--force", help="Re-download SEC FSDS data."),
) -> None:
    """Fetch SEC FSDS data."""
    config = _configure(config_path)
    fetch_sec_fsds(config, force=force)


@ingest_app.command("eu")
def ingest_eu(config_path: Path = config_option) -> None:
    """Load optional EU State Aid data."""
    config = _configure(config_path)
    load_eu_state_aid(config["project"]["data_dir"])


@match_app.command("recipients")
def match_recipients_cmd(config_path: Path = config_option) -> None:
    """Match USAspending recipients to SEC CIKs."""
    config = _configure(config_path)
    data_dir = Path(config["project"]["data_dir"]) / "derived"
    awards_path = data_dir / "usaspending_awards.parquet"
    _require_file(
        awards_path,
        f"uv run industrial-policy ingest usaspending --config-path {config_path}",
    )
    awards = pd.read_parquet(awards_path)
    awards = prepare_awards(awards, config)
    lookup = fetch_company_lookup(config["project"]["data_dir"], config["sec"]["user_agent"])
    match_recipients(awards, lookup, config)


@build_app.command("panel")
def build_panel_cmd(config_path: Path = config_option) -> None:
    """Build event-time panel."""
    config = _configure(config_path)
    data_dir = Path(config["project"]["data_dir"]) / "derived"
    awards_path = data_dir / "awards_with_cik.parquet"
    sec_path = data_dir / "sec_firm_period_base.parquet"
    _require_file(
        awards_path,
        f"uv run industrial-policy match recipients --config-path {config_path}",
    )
    _require_file(
        sec_path,
        f"uv run industrial-policy ingest sec --config-path {config_path}",
    )
    awards = pd.read_parquet(awards_path)
    sec_panel = pd.read_parquet(sec_path)
    sec_features = compute_financial_features(sec_panel)
    sec_features.to_parquet(data_dir / "sec_firm_period_features.parquet", index=False)
    ingest_census_concentration(config)
    build_event_panel(awards, sec_features, config)


@match_app.command("controls")
def match_controls_cmd(config_path: Path = config_option) -> None:
    """Match treated firms with control firms."""
    config = _configure(config_path)
    data_dir = Path(config["project"]["data_dir"]) / "derived"
    treated_path = data_dir / "event_panel_treated.parquet"
    control_path = data_dir / "panel_pool_controls.parquet"
    _require_file(
        treated_path,
        f"uv run industrial-policy build panel --config-path {config_path}",
    )
    _require_file(
        control_path,
        f"uv run industrial-policy build panel --config-path {config_path}",
    )
    treated_panel = pd.read_parquet(treated_path)
    control_pool = pd.read_parquet(control_path)
    propensity = build_propensity_scores(treated_panel, control_pool, config)
    propensity.to_parquet(data_dir / "propensity_scores.parquet", index=False)
    nearest_neighbor_match(propensity, config)


@app.command("estimate")
def estimate_cmd(config_path: Path = config_option) -> None:
    """Estimate event-study DID and write outputs."""
    config = _configure(config_path)
    data_dir = Path(config["project"]["data_dir"]) / "derived"
    treated_path = data_dir / "event_panel_treated.parquet"
    control_path = data_dir / "panel_pool_controls.parquet"
    matches_path = data_dir / "matches.parquet"
    _require_file(
        treated_path,
        f"uv run industrial-policy build panel --config-path {config_path}",
    )
    _require_file(
        control_path,
        f"uv run industrial-policy build panel --config-path {config_path}",
    )
    _require_file(
        matches_path,
        f"uv run industrial-policy match controls --config-path {config_path}",
    )
    treated_panel = pd.read_parquet(treated_path)
    control_pool = pd.read_parquet(control_path)
    matches = pd.read_parquet(matches_path)

    results = run_event_studies(treated_panel, control_pool, matches, config)
    save_event_study_outputs(results, config["project"]["outputs_dir"])
    save_event_study_summary(results, config["project"]["outputs_dir"])
    for outcome, payload in results.items():
        coef_df = payload["coefficients"]
        if not coef_df.empty:
            plot_event_study(coef_df, outcome, config["project"]["outputs_dir"])
    write_sample_construction_report(config, config["project"]["outputs_dir"])
    write_matching_diagnostics(config, config["project"]["outputs_dir"])


@app.command("all")
def run_all(
    config_path: Path = config_option,
    force: bool = typer.Option(False, "--force", help="Force re-downloads for ingest steps."),
) -> None:
    """Run the full pipeline."""
    ingest_usaspending(config_path=config_path, force=force)
    ingest_sec(config_path=config_path, force=force)
    match_recipients_cmd(config_path=config_path)
    build_panel_cmd(config_path=config_path)
    match_controls_cmd(config_path=config_path)
    estimate_cmd(config_path=config_path)
    incidence_cmd(config_path=config_path)
    robustness_cmd(config_path=config_path)


@app.command("incidence")
def incidence_cmd(config_path: Path = config_option) -> None:
    """Compute incidence-in-dollars outputs."""
    config = _configure(config_path)
    run_incidence_analysis(config)


@app.command("robustness")
def robustness_cmd(config_path: Path = config_option) -> None:
    """Run robustness suite."""
    config = _configure(config_path)
    run_robustness_suite(config)


@app.command("doctor")
def doctor_cmd(config_path: Path = config_option) -> None:
    """Run configuration and data checks."""
    config = _configure(config_path)
    data_dir = Path(config["project"]["data_dir"])
    outputs_dir = Path(config["project"]["outputs_dir"])
    typer.echo(f"Config path: {Path(config_path).resolve()}")
    typer.echo(f"Data dir: {data_dir.resolve()}")
    typer.echo(f"Outputs dir: {outputs_dir.resolve()}")

    sec_user_agent = config.get("sec", {}).get("user_agent")
    if sec_user_agent:
        typer.echo("SEC_USER_AGENT: configured")
    else:
        typer.echo("SEC_USER_AGENT: missing (set in .env or config/config.yaml)")

    def _dataset_rows(path: Path) -> int | None:
        if not path.exists():
            return None
        if path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq

                return pq.ParquetFile(path).metadata.num_rows
            except Exception:
                return None
        if path.suffix == ".csv":
            try:
                with path.open("r", encoding="utf-8", errors="ignore") as handle:
                    return max(sum(1 for _ in handle) - 1, 0)
            except Exception:
                return None
        return None

    checks = [
        (
            data_dir / "derived" / "usaspending_awards.parquet",
            f"uv run industrial-policy ingest usaspending --config-path {config_path}",
        ),
        (
            data_dir / "derived" / "sec_firm_period_base.parquet",
            f"uv run industrial-policy ingest sec --config-path {config_path}",
        ),
        (
            data_dir / "derived" / "awards_with_cik.parquet",
            f"uv run industrial-policy match recipients --config-path {config_path}",
        ),
        (
            data_dir / "derived" / "event_panel_treated.parquet",
            f"uv run industrial-policy build panel --config-path {config_path}",
        ),
        (
            data_dir / "derived" / "matches.parquet",
            f"uv run industrial-policy match controls --config-path {config_path}",
        ),
        (
            data_dir / "derived" / "event_panel_stacked.parquet",
            f"uv run industrial-policy build panel --config-path {config_path}",
        ),
        (
            outputs_dir / "tables" / "event_study_summary.csv",
            f"uv run industrial-policy estimate --config-path {config_path}",
        ),
        (
            outputs_dir / "tables" / "incidence_ratios_summary.csv",
            f"uv run industrial-policy incidence --config-path {config_path}",
        ),
    ]
    typer.echo("\nPipeline checkpoints:")
    for path, command in checks:
        if not path.exists():
            status = "MISSING"
            rows = None
        else:
            rows = _dataset_rows(path)
            status = "EMPTY" if rows == 0 else "OK"

        desc = f" ({rows} rows)" if rows is not None else ""
        typer.echo(f"- {status}: {path}{desc}")
        if not path.exists():
            typer.echo(f"  -> Run: {command}")
        elif status == "EMPTY":
            typer.echo(f"  -> Rebuild: {command}")

if __name__ == "__main__":
    app()
