import html
import json
from pathlib import Path

from src.config import load_config
from src.evaluation.metrics import aggregate_metrics, accuracy


def _pair_lookup(config: dict) -> dict:
    out = {}
    for variant, cfg in config.get("needles", {}).items():
        for p in cfg.get("pairs", []):
            pid = p.get("id", "")
            out[(variant, pid)] = (p.get("question", ""), p.get("answer", ""))
        if "question" in cfg and "answer" in cfg:
            out[(variant, "0")] = (cfg.get("question", ""), cfg.get("answer", ""))
    return out


def _cell(text: str, max_len: int = 200) -> str:
    escaped = html.escape(str(text or ""))
    if len(escaped) > max_len:
        title = escaped.replace('"', "&quot;")
        return f'<span title="{title}">{escaped[:max_len]}...</span>'
    return escaped


def write_html_report(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = path.parent / "meta.json"
    trials = 1
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            trials = meta.get("trials", 1)
        except Exception:
            pass
    metrics = aggregate_metrics(results)
    models = sorted(set(r.get("model", "") for r in results))
    lengths = sorted(set(r.get("context_length", 0) for r in results))
    positions = sorted(set(r.get("position", 0) for r in results))
    pairs = sorted(set(r.get("needle_pair", "") for r in results))
    model_acc = {m: accuracy(metrics["by_model"].get(m, {})) for m in models}
    length_acc = {l: accuracy(metrics["by_length"].get(l, {})) for l in lengths}
    pos_acc = {p: accuracy(metrics["by_position"].get(p, {})) for p in positions}
    pair_acc = {p: accuracy(metrics["by_needle_pair"].get(p, {})) for p in pairs}
    pair_lookup = _pair_lookup(load_config())
    html = _template(models, lengths, positions, pairs, model_acc, length_acc, pos_acc, pair_acc, results, pair_lookup, trials)
    path.write_text(html)


def _template(models, lengths, positions, pairs, model_acc, length_acc, pos_acc, pair_acc, results, pair_lookup, trials=1):
    length_labels = [str(l) for l in lengths]
    pos_labels = [str(p) for p in positions]
    model_vals = [round(model_acc.get(m, 0) * 100, 1) for m in models]
    length_vals = [round(length_acc.get(l, 0) * 100, 1) for l in lengths]
    length_chart_data = [{"x": int(l), "y": round(length_acc.get(l, 0) * 100, 1)} for l in lengths]
    pos_vals = [round(pos_acc.get(p, 0) * 100, 1) for p in positions]
    pair_vals = [round(pair_acc.get(p, 0) * 100, 1) for p in pairs]

    def row(r):
        q = r.get("question") or pair_lookup.get((r.get("variant"), r.get("needle_pair")), ("", ""))[0]
        exp = r.get("expected_answer") or pair_lookup.get((r.get("variant"), r.get("needle_pair")), ("", ""))[1]
        out = r.get("model_output", "")
        return f'<tr><td>{r.get("model","")}</td><td>{r.get("variant","")}</td><td>{r.get("needle_pair","")}</td><td>{r.get("context_length","")}</td><td>{r.get("position","")}</td><td class="correct-cell">{r.get("correct","")}</td><td>{_cell(q, 80)}</td><td>{_cell(exp, 80)}</td><td>{_cell(out)}</td></tr>'

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Context Rot NIAH Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ font-family: system-ui; max-width: 1400px; margin: 2rem auto; padding: 0 1rem; }}
h1 {{ margin-bottom: 1.5rem; }}
.chart-container {{ position: relative; height: 280px; margin-bottom: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #f5f5f5; }}
td {{ max-width: 300px; word-break: break-word; }}
td.correct-cell {{ white-space: nowrap; }}
</style>
</head>
<body>
<h1>Context Rot Needle-in-Haystack Report</h1>
{f'<p><em>Aggregated over {trials} trials per condition.</em></p>' if trials > 1 else ''}
<div class="chart-container">
<canvas id="modelChart"></canvas>
</div>
<div class="chart-container">
<canvas id="lengthChart"></canvas>
</div>
<div class="chart-container">
<canvas id="positionChart"></canvas>
</div>
<div class="chart-container">
<canvas id="pairChart"></canvas>
</div>
<h2>Results</h2>
<table>
<thead><tr><th>Model</th><th>Variant</th><th>Needle Pair</th><th>Context Length</th><th>Position</th><th>Correct</th><th>Question</th><th>Expected Answer</th><th>Model Output</th></tr></thead>
<tbody>
{"".join(row(r) for r in results)}
</tbody>
</table>
<script>
new Chart(document.getElementById("modelChart"), {{
  type: "bar",
  data: {{
    labels: {repr([m.replace("-", " ").replace(":", " ") for m in models])},
    datasets: [{{ label: "Accuracy %", data: {model_vals}, backgroundColor: "rgba(54,162,235,0.6)" }}]
  }},
  options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ min: 0, max: 100 }} }} }}
}});
new Chart(document.getElementById("lengthChart"), {{
  type: "line",
  data: {{
    datasets: [{{
      label: "Accuracy %",
      data: {repr(length_chart_data)},
      borderColor: "rgb(75,192,192)",
      fill: false
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    scales: {{
      x: {{ type: "logarithmic", title: {{ display: true, text: "Context length (tokens)" }} }},
      y: {{ min: 0, max: 100, title: {{ display: true, text: "Accuracy %" }} }}
    }}
  }}
}});
new Chart(document.getElementById("positionChart"), {{
  type: "bar",
  data: {{
    labels: {repr(pos_labels)},
    datasets: [{{ label: "Accuracy %", data: {pos_vals}, backgroundColor: "rgba(255,99,132,0.6)" }}]
  }},
  options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ min: 0, max: 100 }} }} }}
}});
new Chart(document.getElementById("pairChart"), {{
  type: "bar",
  data: {{
    labels: {repr(pairs)},
    datasets: [{{ label: "Accuracy %", data: {pair_vals}, backgroundColor: "rgba(153,102,255,0.6)" }}]
  }},
  options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ min: 0, max: 100 }} }} }}
}});
</script>
</body>
</html>"""
