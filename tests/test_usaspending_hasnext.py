from __future__ import annotations

import requests

import industrial_policy.ingest.usaspending as usaspending


class DummyResponse:
    def __init__(self, payload: dict):
        self.status_code = 200
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_fetch_interval_stops_on_hasnext_false_at_max_pages(monkeypatch):
    def fake_post_with_retry(*args, **kwargs):
        page = kwargs.get("payload", args[2]).get("page")
        has_next = page < 3
        return DummyResponse(
            {
                "results": [{"Award ID": f"A{page}"}],
                "page_metadata": {"page": page, "hasNext": has_next},
            }
        )

    monkeypatch.setattr(usaspending, "_post_with_retry", fake_post_with_retry)

    df, meta = usaspending._fetch_interval(
        session=requests.Session(),
        url="https://example.test",
        payload={
            "filters": {"time_period": [{"start_date": "2020-01-01", "end_date": "2020-01-01"}]},
            "fields": ["Award ID"],
            "limit": 1,
            "subawards": False,
        },
        request_timeout=1,
        max_pages=3,
        adaptive_split_on_422=True,
        max_attempts=1,
        backoff_seconds=0.1,
        sleep_seconds=0.0,
    )

    assert len(df) == 3
    assert meta["pages_downloaded"] == 3
    assert meta["stop_reason"] == "no_next_page"
    assert meta["has_next"] is False


def test_fetch_interval_raises_when_hasnext_true_at_max_pages(monkeypatch):
    def fake_post_with_retry(*args, **kwargs):
        page = kwargs.get("payload", args[2]).get("page")
        return DummyResponse(
            {
                "results": [{"Award ID": f"A{page}"}],
                "page_metadata": {"page": page, "hasNext": True},
            }
        )

    monkeypatch.setattr(usaspending, "_post_with_retry", fake_post_with_retry)

    try:
        usaspending._fetch_interval(
            session=requests.Session(),
            url="https://example.test",
            payload={
                "filters": {"time_period": [{"start_date": "2020-01-01", "end_date": "2020-01-01"}]},
                "fields": ["Award ID"],
                "limit": 1,
                "subawards": False,
            },
            request_timeout=1,
            max_pages=2,
            adaptive_split_on_422=True,
            max_attempts=1,
            backoff_seconds=0.1,
            sleep_seconds=0.0,
        )
    except usaspending.PageLimitError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Expected PageLimitError when hasNext is True at max_pages.")

