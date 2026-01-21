"""CLI entrypoints for the industrial policy pipeline."""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
import typer

from industrial_policy.analysis.did import run_event_studies
from industrial_policy.analysis.outputs import save_event_study_outputs
from industrial_policy.analysis.sample_report import write_sample_construction_report
from industrial_policy.config import load_config
from industrial_policy.entity.name_matching import match_recipients
from industrial_policy.entity.sec_company_lookup import fetch_company_lookup
from industrial_policy.features.awards import prepare_awards
from industrial_policy.features.financials import compute_financial_features
from industrial_policy.features.panel import build_event_panel
from industrial_policy.ingest.eu_state_aid import load_eu_state_aid
from industrial_policy.ingest.sec_fsds import fetch_sec_fsds
from industrial_policy.ingest.usaspending import fetch_usaspending_awards
from industrial_policy.log import setup_logging
from industrial_policy.match.diagnostics import write_matching_diagnostics
from industrial_policy.match.nearest_neighbor import nearest_neighbor_match
from industrial_policy.match.propensity import build_propensity_scores
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
    outputs_dir = Path(config["project"]["outputs_dir"])
    setup_logging(outputs_dir / "logs")
    return config


ingest_config_option = typer.Option("config/config.yaml", help="Path to config YAML")


@ingest_app.command("usaspending")
def ingest_usaspending(config_path: Path = ingest_config_option) -> None:
    """Fetch USAspending awards."""
    config = _configure(config_path)
    fetch_usaspending_awards(config)


@ingest_app.command("sec")
def ingest_sec(config_path: Path = ingest_config_option) -> None:
    """Fetch SEC FSDS data."""
    config = _configure(config_path)
    fetch_sec_fsds(config)


@ingest_app.command("eu")
def ingest_eu(config_path: Path = ingest_config_option) -> None:
    """Load optional EU State Aid data."""
    config = _configure(config_path)
    load_eu_state_aid(config["project"]["data_dir"])


@match_app.command("recipients")
def match_recipients_cmd(config_path: Path = ingest_config_option) -> None:
    """Match USAspending recipients to SEC CIKs."""
    config = _configure(config_path)
    data_dir = Path(config["project"]["data_dir"]) / "derived"
    awards_path = data_dir / "usaspending_awards.parquet"
    awards = pd.read_parquet(awards_path)
    awards = prepare_awards(awards, config)
    lookup = fetch_company_lookup(config["project"]["data_dir"], config["sec"]["user_agent"])
    match_recipients(awards, lookup, config)


@build_app.command("panel")
def build_panel_cmd(config_path: Path = ingest_config_option) -> None:
    """Build event-time panel."""
    config = _configure(config_path)
    data_dir = Path(config["project"]["data_dir"]) / "derived"
    awards = pd.read_parquet(data_dir / "awards_with_cik.parquet")
    sec_panel = pd.read_parquet(data_dir / "sec_firm_period_base.parquet")
    sec_features = compute_financial_features(sec_panel)
    sec_features.to_parquet(data_dir / "sec_firm_period_features.parquet", index=False)
    build_event_panel(awards, sec_features, config)


@match_app.command("controls")
def match_controls_cmd(config_path: Path = ingest_config_option) -> None:
    """Match treated firms with control firms."""
    config = _configure(config_path)
    data_dir = Path(config["project"]["data_dir"]) / "derived"
    treated_panel = pd.read_parquet(data_dir / "event_panel_treated.parquet")
    control_pool = pd.read_parquet(data_dir / "panel_pool_controls.parquet")
    propensity = build_propensity_scores(treated_panel, control_pool, config)
    propensity.to_parquet(data_dir / "propensity_scores.parquet", index=False)
    nearest_neighbor_match(propensity, config)


@app.command("estimate")
def estimate_cmd(config_path: Path = ingest_config_option) -> None:
    """Estimate event-study DID and write outputs."""
    config = _configure(config_path)
    data_dir = Path(config["project"]["data_dir"]) / "derived"
    treated_panel = pd.read_parquet(data_dir / "event_panel_treated.parquet")
    control_pool = pd.read_parquet(data_dir / "panel_pool_controls.parquet")
    matches = pd.read_parquet(data_dir / "matches.parquet")

    results = run_event_studies(treated_panel, control_pool, matches, config)
    save_event_study_outputs(results, config["project"]["outputs_dir"])
    for outcome, payload in results.items():
        coef_df = payload["coefficients"]
        if not coef_df.empty:
            plot_event_study(coef_df, outcome, config["project"]["outputs_dir"])
    write_sample_construction_report(config, config["project"]["outputs_dir"])
    write_matching_diagnostics(config, config["project"]["outputs_dir"])


@app.command("all")
def run_all(config_path: Path = ingest_config_option) -> None:
    """Run the full pipeline."""
    ingest_usaspending(config_path)
    ingest_sec(config_path)
    match_recipients_cmd(config_path)
    build_panel_cmd(config_path)
    match_controls_cmd(config_path)
    estimate_cmd(config_path)


if __name__ == "__main__":
    app()
