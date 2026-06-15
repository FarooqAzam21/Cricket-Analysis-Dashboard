"""Smoke tests for year-wise analytics data loading."""

from pathlib import Path

import pandas as pd

from src.data_loader import _get_csv_cache_key, load_all_data


BASE_DIR = Path(__file__).resolve().parent


def test_yearwise_csv_loads_from_workspace():
    df = pd.read_csv(BASE_DIR / "yearwise_data.csv")

    assert not df.empty
    assert "player" in df.columns
    assert df["player"].nunique() > 0


def test_data_loader_returns_yearwise_data():
    _, _, _, _, year_wise, _, _, _ = load_all_data(_csv_cache_key=_get_csv_cache_key())

    assert year_wise is not None
    assert not year_wise.empty
    assert "player" in year_wise.columns
