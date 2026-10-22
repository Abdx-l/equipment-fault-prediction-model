# Equipment Fault Prediction Model

Generate transformer telemetry with gradual degradation, build rolling temperature, current, and vibration features, then compare a trained logistic model with a vibration threshold baseline. A time-based split keeps later readings out of training. The report includes precision, recall, false alarms, and missed faults.

```sh
python model.py --units 200 --hours 72 --seed 7 --json report.json
python -m unittest discover -s tests -v
```

Features use trailing 1-, 6-, and 24-hour windows, rate of change, and time since the last alarm. The NumPy implementation trains by gradient descent. Data are synthetic; the model is a reproducible experiment rather than an operational maintenance decision system.

