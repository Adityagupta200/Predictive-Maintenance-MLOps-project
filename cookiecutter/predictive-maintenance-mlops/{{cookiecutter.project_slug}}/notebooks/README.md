# Notebooks (CMAPSS + MLflow)

These notebooks provide recruiter-grade proof of:
- MLflow experiment tracking (params/metrics/model artifacts)
- Run comparison and model selection (leaderboard + selection audit)
- Reproducible MLflow UI demo via a portable tracking URI

The dataset is NASA C-MAPSS (FD00x) style: time-series per engine (`unit`) over operating cycles (`cycle`), with operational settings + sensor channels, and RUL labels derived from run-to-failure trajectories.  

## Quickstart

### 1) Install notebook dependencies
From repo root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-api.txt
pip install -r requirements-notebooks.txt
