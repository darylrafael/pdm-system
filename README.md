# PdM System — Predictive Maintenance for Heavy Equipment

> **Portfolio Project** | Target: Astra Group (United Tractors / PAMAPERSADA)
> Built as part of Binus University Enrichment Program preparation.

---

## Business Context

PAMAPERSADA Nusantara (Astra Group) operates hundreds of heavy equipment units
(excavators, bulldozers, dump trucks) across mining sites in Kalimantan and Sumatera.

**The Problem:** Unplanned equipment downtime costs **Rp 500M–1B per day**.
Current maintenance is schedule-based (every X hours), ignoring actual machine condition.

**This System:** Replaces schedule-based with **condition-based maintenance** using
real-time sensor data to predict Remaining Useful Life (RUL) and alert before failure.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        PdM System v1.0                       │
├──────────────┬──────────────┬───────────────┬───────────────┤
│  Data Layer  │ Model Layer  │   API Layer   │  Dashboard    │
│              │              │               │               │
│ NASA CMAPSS  │  XGBoost     │  FastAPI      │  Streamlit    │
│ FD001-FD004  │  Regression  │  /predict     │               │
│              │              │  /health      │  Machine      │
│  Feature     │  MLflow      │  /monitoring  │  Status Table │
│  Engineering │  Tracking    │               │               │
│              │              │  API Key Auth │  Sensor       │
│  SQLite      │  Evidently   │  Pydantic v2  │  Trend Charts │
│  Persistence │  Drift Det.  │  Async        │               │
└──────────────┴──────────────┴───────────────┴───────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| API Framework | FastAPI + Uvicorn (async) |
| ML | XGBoost + scikit-learn |
| Experiment Tracking | MLflow |
| Drift Detection | Evidently |
| Validation | Pydantic v2 |
| Config | pydantic-settings + dotenv |
| Persistence | SQLite + SQLAlchemy (async) |
| Dashboard | Streamlit + Plotly |
| Testing | pytest + pytest-asyncio |
| Linting | ruff |
| Type Checking | mypy (strict) |
| Security | bandit |
| Containerization | Docker + Docker Compose |
| CI/CD | GitHub Actions |
| Dependency Mgmt | uv + pyproject.toml |

---

## Quick Start

### Prerequisites
- Python 3.12+
- Docker & Docker Compose (for containerized run)

### Local Development

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/pdm-system.git
cd pdm-system

# 2. First-time setup (installs uv + dependencies + creates .env)
make setup

# 3. Edit .env with your configuration
nano .env

# 4. Download NASA CMAPSS dataset
# Place train_FD001.txt, test_FD001.txt, RUL_FD001.txt into data/raw/

# 5. Run quality checks
make ci

# 6. Start API server
make run-api

# 7. Start dashboard (new terminal)
make run-dashboard

# 8. View MLflow experiments (new terminal)
make run-mlflow
```

### Docker (Full Stack)

```bash
make docker-build
make docker-up
# API       → http://localhost:8000
# Dashboard → http://localhost:8501
# MLflow    → http://localhost:5000
```

---

## API Reference

Auto-generated docs available at `http://localhost:8000/docs` when running.

### Authentication
All endpoints (except `/health`) require:
```
X-API-Key: your-api-key
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check — no auth required |
| POST | `/predict` | Single unit RUL prediction |
| GET | `/monitoring/drift` | Current drift detection status |

---

## Key Metrics (KPIs)

| Metric | Target | Achieved |
|---|---|---|
| RMSE | < 20 cycles | TBD |
| MAE | < 15 cycles | TBD |
| Drift Detection | Live alert | TBD |
| API Latency | < 100ms | TBD |
| Test Coverage | > 70% | TBD |

---

## Project Structure

```
pdm-system/
├── src/
│   ├── config.py              # Centralized config via pydantic-settings
│   ├── data/
│   │   ├── loader.py          # NASA CMAPSS data ingestion
│   │   ├── preprocessor.py    # Normalization, null handling
│   │   └── feature_engineer.py # Rolling stats, rate-of-change features
│   ├── models/
│   │   ├── trainer.py         # Training loop + MLflow logging
│   │   ├── evaluator.py       # Metrics (RMSE, MAE, AUC-ROC)
│   │   └── schemas.py         # Pydantic I/O contracts
│   ├── monitoring/
│   │   └── drift_detector.py  # Evidently drift detection (live)
│   └── api/
│       ├── main.py            # FastAPI app factory
│       ├── dependencies.py    # Auth, shared dependencies
│       ├── routers/           # predict, health, monitoring
│       └── middleware/        # Security headers
├── dashboard/app.py           # Streamlit UI
├── tests/                     # Unit + integration tests
├── docs/adr/                  # Architecture Decision Records
├── docker/                    # Dockerfiles per service
└── .github/workflows/ci.yml   # GitHub Actions pipeline
```

---

## Development Workflow

```
Linear (Issue) → Claude (Design) → Antigravity (Implement)
→ SENTINEL Review → GitHub PR → CI Pipeline → Merge
```

See `docs/` for Architecture Decision Records (ADRs).

---

## License

MIT — Built for educational/portfolio purposes.
