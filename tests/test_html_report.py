from pathlib import Path
from src.output.html_report import write_html_report


def test_write_html_report(tmp_path):
    results = [
        {"model": "m1", "variant": "pg", "context_length": 1000, "position": 0, "correct": "correct"},
        {"model": "m1", "variant": "pg", "context_length": 1000, "position": 0.5, "correct": "incorrect"},
    ]
    path = tmp_path / "report.html"
    write_html_report(results, path)
    assert path.exists()
    html = path.read_text()
    assert "Context Rot" in html
    assert "m1" in html
    assert "Chart.js" in html or "chart" in html.lower()
