"""Uniform interface every source module implements (brief §4)."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import NamedTuple, Protocol

import xarray as xr


class BBox(NamedTuple):
    lon_min: float
    lat_min: float
    lon_max: float
    lat_max: float


class Source(Protocol):
    def available_range(self) -> tuple[date, date]: ...
    def fetch(self, start: date, end: date, bbox: BBox) -> Path: ...
    def parse(self, path: Path) -> xr.Dataset: ...
