from __future__ import annotations

from datetime import datetime

import pandas as pd

import industrial_policy.ingest.usaspending as usaspending


def test_interval_split_writes_chunks(monkeypatch, tmp_path):
    def fake_fetch_interval(
        session,
        url,
        payload,
        request_timeout,
        max_pages,
        adaptive_split_on_422,
        max_attempts,
        backoff_seconds,
    ):
        start = datetime.fromisoformat(payload["filters"]["time_period"][0]["start_date"])
        end = datetime.fromisoformat(payload["filters"]["time_period"][0]["end_date"])
        days = (end - start).days + 1
        if days > 2:
            raise usaspending.PageLimitError("too many pages")
        df = pd.DataFrame({"Award ID": [f"{start.date()}_{end.date()}"]})
        return df, {"pages_downloaded": 1, "total_pages": 1, "stop_reason": "total_pages"}

    monkeypatch.setattr(usaspending, "_fetch_interval", fake_fetch_interval)

    config = {
        "project": {"data_dir": str(tmp_path), "outputs_dir": str(tmp_path / "outputs")},
        "usaspending": {
            "base_url": "https://api.usaspending.gov",
            "endpoint": "/api/v2/search/spending_by_award/",
            "filters": {
                "award_type_codes": ["A"],
                "time_period": [{"start_date": "2020-01-01", "end_date": "2020-01-06"}],
            },
            "fields": ["Award ID"],
            "page_size": 100,
            "max_pages_per_query": 1,
            "request_timeout_seconds": 1,
            "retry": {"max_attempts": 1, "backoff_seconds": 0.1},
            "chunking": {"initial_granularity": "quarter", "min_granularity_days": 1},
        },
    }
    df = usaspending.fetch_usaspending_awards(config, force=True)
    chunk_dir = tmp_path / "derived" / "usaspending" / "chunks"
    chunks = list(chunk_dir.glob("*.parquet"))
    assert len(chunks) > 1
    assert not df.empty
