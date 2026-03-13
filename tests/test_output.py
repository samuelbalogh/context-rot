import json
import csv
from pathlib import Path

from src.output.json_writer import write_json
from src.output.csv_writer import write_csv


def test_write_json(tmp_path):
    results = [{"model": "m1", "correct": "correct"}]
    path = tmp_path / "out" / "results.json"
    write_json(results, path)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded == results


def test_write_csv(tmp_path):
    results = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    path = tmp_path / "results.csv"
    write_csv(results, path)
    assert path.exists()
    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["a"] == "1"
    assert rows[1]["a"] == "3"


def test_write_csv_empty(tmp_path):
    path = tmp_path / "empty.csv"
    write_csv([], path)
    assert not path.exists()
