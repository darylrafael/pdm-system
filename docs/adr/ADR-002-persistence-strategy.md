# ADR-002: Persistence Strategy — SQLite for Prediction History

**Date:** 2025-07-17
**Status:** ACCEPTED
**Deciders:** Engineering Team
**Related:** ADR-001 (model choice), Issue #09

---

## Context

The PdM system needs to persist two categories of data:

1. **Prediction history** — every `/predict` API call result (unit_id,
   cycle, predicted_rul, risk_level, timestamp) for dashboard display
   and trend analysis
2. **Drift reference window** — the last N inference inputs used by
   Evidently to compute data drift scores in real-time

Without persistence, the Streamlit dashboard can only show the last
prediction (stateless), and drift detection has no reference distribution
to compare against. Neither is acceptable for a production-realistic
portfolio system.

Three options were evaluated:

| Option | Description |
|---|---|
| A | **Stateless** — no database, every request independent |
| B | **SQLite** — embedded file-based relational database |
| C | **PostgreSQL** — full client-server relational database |

---

## Decision

**Option B — SQLite via SQLAlchemy async (`aiosqlite`).**

A single `pdm.db` file stores prediction history and drift reference
data. Accessed through SQLAlchemy's async engine to keep FastAPI's event
loop unblocked.

---

## Rationale

### Why not stateless (Option A)?

Stateless inference means the dashboard can only display the most recent
prediction per request. It cannot show:
- Degradation curves over time per unit
- Historical alert counts
- Drift monitoring over a rolling window

These are exactly the features that differentiate this system from a
bare model endpoint. Stateless was rejected unconditionally.

### Why SQLite over PostgreSQL (Option C)?

| Criteria | SQLite | PostgreSQL |
|---|---|---|
| **Setup complexity** | Zero — file on disk | Server process, auth, network |
| **Portfolio deployment** | Single `docker-compose.yml` | Requires separate DB container + volume + healthcheck |
| **Throughput** | ~10K writes/sec | ~100K writes/sec |
| **Expected load** | Low (one prediction at a time) | Enterprise scale |
| **Migration path** | SQLAlchemy makes swap trivial | — |

PAMAPERSADA operates hundreds of heavy equipment units. Even at one
prediction per unit per minute, that is ~14,400 writes/day — well within
SQLite's practical throughput ceiling. SQLite's write-ahead logging (WAL
mode) handles the single concurrent writer constraint comfortably at this
scale.

The SQLAlchemy ORM layer means switching to PostgreSQL for a real
deployment is a one-line config change (`DATABASE_URL` in `.env`) with
zero application code changes.

### Why async SQLAlchemy (`aiosqlite`)?

FastAPI is built on an async event loop (uvicorn + asyncio). Synchronous
database calls from within an `async def` endpoint block the event loop,
preventing the API from handling other requests during the I/O wait.
`aiosqlite` + SQLAlchemy async engine keeps the loop unblocked throughout
the database write.

---

## Schema (Planned — implemented in Issue #16)

```sql
-- Prediction history
CREATE TABLE predictions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id          INTEGER NOT NULL,
    cycle            INTEGER NOT NULL,
    predicted_rul    REAL    NOT NULL,
    risk_level       TEXT    NOT NULL,  -- HIGH | MEDIUM | NORMAL
    model_version    TEXT    NOT NULL,
    created_at       TEXT    NOT NULL   -- ISO 8601 UTC timestamp
);

-- Drift reference window (rolling, capped at DRIFT_REFERENCE_WINDOW rows)
CREATE TABLE drift_reference (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id          INTEGER NOT NULL,
    cycle            INTEGER NOT NULL,
    feature_snapshot TEXT    NOT NULL,  -- JSON: {feature_name: value}
    created_at       TEXT    NOT NULL
);

CREATE INDEX idx_predictions_unit    ON predictions (unit_id);
CREATE INDEX idx_predictions_created ON predictions (created_at);
```

---

## Consequences

**Positive:**
- Zero infrastructure overhead — no separate container or service
- Full historical trend data available for dashboard degradation curves
- Drift reference window persists across API restarts
- SQLAlchemy ORM enables trivial migration to PostgreSQL if needed

**Negative / Constraints:**
- Single writer constraint — concurrent write load must stay low
  (acceptable for portfolio demo scope)
- `pdm.db` file must be excluded from version control (`.gitignore` ✅)
- Docker volume must be configured to persist `data/pdm.db` across
  container restarts (addressed in docker-compose.yml, Issue #29)

---

## Alternatives Rejected

**Redis:** Rejected — adds another service dependency. Appropriate for
high-throughput caching or pub/sub, not for durable prediction history
that must survive restarts.

**InfluxDB / TimescaleDB:** Rejected — time-series databases optimized
for sensor streams, not for structured prediction records with relational
joins. Over-engineered for this scope.

**In-memory dict / list:** Rejected — equivalent to stateless; data lost
on every API restart. Ruled out alongside Option A.