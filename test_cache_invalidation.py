"""Smoke test for analytics CSV cache invalidation."""

from pathlib import Path

from src.data_loader import _get_csv_cache_key


CSV_FILES = [
    "odi_batsman.csv",
    "odi_bowler.csv",
    "odi_all_rounders.csv",
    "yearwise_data.csv",
]


def test_csv_cache_key_available():
    cache_key = _get_csv_cache_key()
    assert cache_key
    assert len(cache_key) == 32


def test_analytics_csv_files_exist():
    base_dir = Path(__file__).resolve().parent
    missing = [csv_file for csv_file in CSV_FILES if not (base_dir / csv_file).exists()]
    assert not missing, f"Missing analytics CSV files: {missing}"
