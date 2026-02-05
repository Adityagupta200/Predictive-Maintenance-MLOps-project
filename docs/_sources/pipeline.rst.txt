Pipeline
========

The training pipeline is defined in ``dvc.yaml`` and produces:

- ``artifacts/processed/train.csv``
- ``artifacts/processed/val.csv``
- ``artifacts/metrics/metrics.json``
- ``models/best_model.joblib``

Run:

- ``make pipeline`` (or ``dvc repro``)
