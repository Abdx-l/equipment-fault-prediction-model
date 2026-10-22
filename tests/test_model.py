import unittest
import numpy as np
from model import evaluate, experiment, features, generate


class ModelTests(unittest.TestCase):
    def test_deterministic_and_leakage_window(self):
        first = generate(10, 36, 3)
        np.testing.assert_array_equal(first, generate(10, 36, 3))
        x, y, times = features(first)
        self.assertEqual(len(x), 120)
        self.assertEqual(times.min(), 24)
        self.assertEqual(x.shape[1], 22)

    def test_training_report_and_counts(self):
        report = experiment(30, 48, 4)
        self.assertEqual(report["train_rows"] + report["test_rows"], 30 * (48 - 24))
        metric = evaluate(np.array([1, 0, 1]), np.array([0.9, 0.7, 0.1]))
        self.assertEqual(metric["false_alarms"], 1)
        self.assertEqual(metric["missed_faults"], 1)

