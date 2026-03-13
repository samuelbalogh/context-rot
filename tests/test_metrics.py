from src.evaluation.metrics import aggregate_metrics, accuracy


def test_accuracy_full():
    assert accuracy({"correct": 3, "total": 3}) == 1.0


def test_accuracy_zero():
    assert accuracy({"correct": 0, "total": 3}) == 0.0


def test_accuracy_empty():
    assert accuracy({}) == 0.0


def test_aggregate_metrics():
    results = [
        {"model": "m1", "position": 0, "context_length": 1000, "correct": "correct"},
        {"model": "m1", "position": 0.5, "context_length": 1000, "correct": "incorrect"},
        {"model": "m2", "position": 0, "context_length": 1000, "correct": "correct"},
    ]
    m = aggregate_metrics(results)
    assert m["by_model"]["m1"]["correct"] == 1
    assert m["by_model"]["m1"]["total"] == 2
    assert m["by_model"]["m2"]["correct"] == 1
    assert m["by_model"]["m2"]["total"] == 1
    assert m["by_position"][0]["correct"] == 2
    assert m["by_position"][0.5]["correct"] == 0
