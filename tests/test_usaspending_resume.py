from __future__ import annotations

import pandas as pd

import industrial_policy.ingest.usaspending as usaspending


def test_usaspending_resume_skips_existing_chunks(monkeypatch, tmp_path):
    def fake_fetch_interval(*args, **kwargs):
        raise AssertionError("Downloader should not be called when chunks exist.")

    monkeypatch.setattr(usaspending, "_fetch_interval", fake_fetch_interval)

    config = {
        "project": {"data_dir": str(tmp_path), "outputs_dir": str(tmp_path / "outputs")},
        "usaspending": {
            "base_url": "https://api.usaspending.gov",
            "endpoint": "/api/v2/search/spending_by_award/",
            "filters": {
                "award_type_codes": ["A"],
                "time_period": [{"start_date": "2020-01-01", "end_date": "2020-01-02"}],
            },
            "fields": ["Award ID"],
            "page_size": 100,
            "max_pages_per_query": 1,
            "request_timeout_seconds": 1,
            "retry": {"max_attempts": 1, "backoff_seconds": 0.1},
            "chunking": {"initial_granularity": "quarter", "min_granularity_days": 1},
        },
    }

    chunk_dir = tmp_path / "derived" / "usaspending" / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / "usaspending_2020-01-01_2020-01-02.parquet"
    pd.DataFrame({"Award ID": ["A1"]}).to_parquet(chunk_path, index=False)

    df = usaspending.fetch_usaspending_awards(config, force=False)
    assert not df.empty
