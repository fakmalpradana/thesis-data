"""HTTP client for the DSDA DKI Jakarta flood warning portal.

No VIEWSTATE needed - discovery (docs/DISCOVERY.md §1) found a plain GET
endpoint that returns the chart data directly. One session, one request at a
time, randomized delay between requests (the server serves public flood
warnings - overloading it is not just rude, it can degrade a safety function).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import date

import requests


class FetchError(Exception):
    """Raised when a chunk could not be fetched after all retries."""


@dataclass
class Settings:
    base_url: str
    user_agent: str
    chunk_days: int
    delay_min_s: float
    delay_max_s: float
    max_retries: int
    retry_backoff_base_s: float
    request_timeout_s: float


class TmaClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = requests.Session()
        self.session.headers["User-Agent"] = settings.user_agent
        self._last_request_at: float | None = None

    def _throttle(self) -> None:
        if self._last_request_at is not None:
            elapsed = time.monotonic() - self._last_request_at
            wait = random.uniform(self.settings.delay_min_s, self.settings.delay_max_s)
            remaining = wait - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def fetch_raw(self, station_id: int, start: date, end: date) -> str:
        """GET one chunk. Retries 5xx/network errors with exponential backoff;
        never retries 4xx (those won't fix themselves)."""
        params = {
            "IdPintuAir": station_id,
            "StartDate": start.strftime("%d/%m/%Y"),
            "EndDate": end.strftime("%d/%m/%Y"),
        }
        last_exc: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(
                    self.settings.base_url,
                    params=params,
                    timeout=self.settings.request_timeout_s,
                )
            except requests.RequestException as exc:
                last_exc = exc
            else:
                if resp.status_code < 400:
                    return resp.text
                if resp.status_code < 500:
                    raise FetchError(
                        f"HTTP {resp.status_code} for station={station_id} "
                        f"{start}..{end} (not retrying, client error)"
                    )
                last_exc = FetchError(f"HTTP {resp.status_code}")
            if attempt < self.settings.max_retries:
                backoff = self.settings.retry_backoff_base_s * (2**attempt)
                time.sleep(backoff)
        raise FetchError(
            f"failed after {self.settings.max_retries + 1} attempts: "
            f"station={station_id} {start}..{end}: {last_exc}"
        )
