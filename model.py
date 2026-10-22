"""Synthetic fleet telemetry, leakage-aware features, and fault classification."""
import argparse
import json
import numpy as np


def generate(units=200, hours=72, seed=7):
    if units < 1 or hours < 30:
        raise ValueError("need positive units and at least 30 hours")
    rng = np.random.default_rng(seed)
    faulty = rng.random(units) < 0.35
    start = rng.integers(hours // 2, hours - 5, units)
    data = []
    for hour in range(hours):
        degradation = np.where(faulty, np.clip((hour - start) / 12, 0, 2), 0)
        temperature = 65 + degradation * 12 + rng.normal(0, 2, units)
        current = 10 + degradation * 3 + rng.normal(0, 0.7, units)
        vibration = 1 + degradation * 0.8 + rng.normal(0, 0.12, units)
        label = (faulty & (degradation >= 0.65)).astype(int)
        data.append(np.stack([temperature, current, vibration, label], axis=1))
    return np.stack(data)


def features(data):
    hours, units, _ = data.shape
    vectors, labels, times = [], [], []
    last_alarm = np.full(units, -1, dtype=int)
    for hour in range(24, hours):
        signal = data[:, :, :3]
        columns = []
        for window in (1, 6, 24):
            trailing = signal[hour - window + 1:hour + 1]
            columns.extend([trailing.mean(axis=0), trailing.std(axis=0)])
        columns.append(signal[hour] - signal[hour - 1])
        columns.append((hour - last_alarm)[:, None])
        vectors.append(np.concatenate(columns, axis=1))
        labels.append(data[hour, :, 3].astype(int))
        times.extend([hour] * units)
        last_alarm = np.where(signal[hour, :, 2] > 1.7, hour, last_alarm)
    return np.concatenate(vectors), np.concatenate(labels), np.asarray(times)


def train_logistic(x, y, iterations=350, learning_rate=0.1):
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    z = np.clip((x - mean) / scale, -10, 10)
    weights = np.zeros(z.shape[1] + 1)
    for _ in range(iterations):
        probability = 1 / (1 + np.exp(-np.clip(z @ weights[:-1] + weights[-1], -30, 30)))
        weights[:-1] -= learning_rate * (z.T @ (probability - y) / len(y) + 0.001 * weights[:-1])
        weights[-1] -= learning_rate * np.mean(probability - y)
    return weights, mean, scale


def predict(x, model):
    weights, mean, scale = model
    z = np.clip((x - mean) / scale, -10, 10)
    return 1 / (1 + np.exp(-np.clip(z @ weights[:-1] + weights[-1], -30, 30)))


def evaluate(y, probability, cutoff=0.5):
    predicted = probability >= cutoff
    tp = int(np.sum(predicted & (y == 1)))
    fp = int(np.sum(predicted & (y == 0)))
    fn = int(np.sum(~predicted & (y == 1)))
    return {"precision": round(tp / (tp + fp), 4) if tp + fp else 0,
            "recall": round(tp / (tp + fn), 4) if tp + fn else 0,
            "true_positives": tp, "false_alarms": fp, "missed_faults": fn}


def experiment(units=200, hours=72, seed=7):
    data = generate(units, hours, seed)
    x, y, times = features(data)
    cutoff = 24 + int((hours - 24) * 0.7)
    train, test = times < cutoff, times >= cutoff
    model = train_logistic(x[train], y[train])
    probability = predict(x[test], model)
    baseline = (x[test, 2] >= 1.7).astype(float)
    return {"train_rows": int(train.sum()), "test_rows": int(test.sum()), "split_hour": cutoff,
            "logistic": evaluate(y[test], probability), "threshold": evaluate(y[test], baseline),
            "precision_recall": [{"threshold": t, **evaluate(y[test], probability, t)} for t in (0.2, 0.4, 0.6, 0.8)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--units", type=int, default=200)
    p.add_argument("--hours", type=int, default=72)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--json")
    a = p.parse_args()
    report = experiment(a.units, a.hours, a.seed)
    print(json.dumps(report, indent=2))
    if a.json:
        from pathlib import Path
        Path(a.json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

