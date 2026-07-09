# ADR-001: Model Architecture — Regression vs Classification

**Date:** 2025-07-09
**Status:** ACCEPTED
**Deciders:** Engineering Team

---

## Context

The PdM system must predict equipment failure risk from NASA CMAPSS sensor data.
Two viable approaches were evaluated:

**Option A — Regression:** Predict exact Remaining Useful Life (RUL) value in cycles.
**Option B — Classification:** Predict binary flag — will this unit fail within N cycles?

---

## Decision

We implement **both**, in sequence:

1. **Primary: XGBoost Regression** → Predicts exact RUL (float)
2. **Derived: Risk Classification** → Derived from regression output at inference time

```python
# At inference time — no separate model needed
predicted_rul = regression_model.predict(features)
risk_level = "HIGH" if predicted_rul < 30 else "MEDIUM" if predicted_rul < 60 else "NORMAL"
```

---

## Rationale

| Criteria | Regression | Classification |
|---|---|---|
| **Information richness** | ✅ Exact RUL value | ⚠️ Binary only |
| **Actionability** | ⚠️ Requires interpretation | ✅ Immediate |
| **Dashboard display** | ✅ Numeric + trend chart | ✅ Color badge |
| **Retraining frequency** | Same | Same |

By deriving classification from regression output at inference time, we get
the benefits of both approaches with a single trained model.

---

## Consequences

- API response includes both `predicted_rul` (float) and `risk_level` (enum)
- Dashboard displays both the number and a color-coded badge
- RMSE and MAE are primary training metrics; classification metrics derived post-hoc
- Threshold values (30, 60 cycles) are configurable via environment variable

---

## Alternatives Rejected

**Pure Classification (binary):** Rejected — loses granularity. "Risk in 29 cycles" and
"Risk in 1 cycle" would produce the same output, which is unacceptable for scheduling.

**LSTM/Transformer:** Rejected for Sprint 1 — adds complexity without proven benefit
over XGBoost on CMAPSS tabular data. Revisit in Sprint 4 if XGBoost RMSE > 20.
