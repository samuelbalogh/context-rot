from collections import defaultdict


def aggregate_metrics(results: list[dict]) -> dict:
    by_model = defaultdict(lambda: {"correct": 0, "total": 0})
    by_position = defaultdict(lambda: {"correct": 0, "total": 0})
    by_length = defaultdict(lambda: {"correct": 0, "total": 0})
    by_needle_pair = defaultdict(lambda: {"correct": 0, "total": 0})
    by_model_position = defaultdict(lambda: {"correct": 0, "total": 0})
    by_model_length = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        correct = 1 if r.get("correct") == "correct" else 0
        model = r.get("model", "")
        pos = r.get("position", 0)
        length = r.get("context_length", 0)
        pair = r.get("needle_pair", "")
        by_model[model]["correct"] += correct
        by_model[model]["total"] += 1
        by_position[pos]["correct"] += correct
        by_position[pos]["total"] += 1
        by_length[length]["correct"] += correct
        by_length[length]["total"] += 1
        by_needle_pair[pair]["correct"] += correct
        by_needle_pair[pair]["total"] += 1
        by_model_position[f"{model}_{pos}"]["correct"] += correct
        by_model_position[f"{model}_{pos}"]["total"] += 1
        by_model_length[f"{model}_{length}"]["correct"] += correct
        by_model_length[f"{model}_{length}"]["total"] += 1
    return {
        "by_model": dict(by_model),
        "by_position": dict(by_position),
        "by_length": dict(by_length),
        "by_needle_pair": dict(by_needle_pair),
        "by_model_position": dict(by_model_position),
        "by_model_length": dict(by_model_length),
    }


def accuracy(m: dict) -> float:
    total = m.get("total", 0)
    if total == 0:
        return 0.0
    return m.get("correct", 0) / total
