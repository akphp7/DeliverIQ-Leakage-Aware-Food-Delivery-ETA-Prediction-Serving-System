# Architecture

Diagrams for every part of DeliverIQ. Each one reflects code that exists in this repository;
the only design-only diagram is marked as such.

1. [Module map](#1-module-map)
2. [Phase pipeline and artifacts](#2-phase-pipeline-and-artifacts)
3. [Data cleaning and repair](#3-data-cleaning-and-repair)
4. [Order-time feature construction](#4-order-time-feature-construction)
5. [Out-of-fold history encoding](#5-out-of-fold-history-encoding)
6. [Evaluation splits and time-series CV](#6-evaluation-splits-and-time-series-cv)
7. [Model selection flow](#7-model-selection-flow)
8. [Conformalized quantile regression](#8-conformalized-quantile-regression)
9. [SQL feature lineage](#9-sql-feature-lineage)
10. [Prediction service internals](#10-prediction-service-internals)
11. [Model bundle contents](#11-model-bundle-contents)
12. [Test coverage map](#12-test-coverage-map)
13. [Production design (design only)](#13-production-design-design-only)

---

## 1. Module map

```mermaid
flowchart TD
    CFG[config.py]
    subgraph data
        P[parsing.py] --> CL[cleaning.py]
    end
    subgraph features
        PH[prep_history.py] --> B[build.py]
    end
    subgraph evaluation
        MT[metrics.py]
        SP[splits.py]
        SG[significance.py]
    end
    subgraph models
        F[factory.py] --> EN[ensemble.py]
        F --> IV[intervals.py]
        F --> EX[explain.py]
    end
    subgraph api
        SV[service.py] --> AP[app.py]
    end
    CL --> B --> F
    CL --> SV
    B --> SV
    IV --> SV
    EX --> SV
    CFG -.-> CL
```

`stats/tests.py` (hypothesis tests) and `sqlfeatures/duck.py` (DuckDB) are used by the
Phase 3 and Phase 4 pipelines only.

## 2. Phase pipeline and artifacts

```mermaid
flowchart LR
    R[(data/raw)] --> P0[p0 data fixes]
    P0 --> P1[p1 evaluation]
    P1 -->|drop history feature| P2[p2 models]
    P2 --> MB[[final_bundle.joblib]]
    R --> P3[p3 statistics]
    R --> P4[p4 SQL features]
    MB --> P4
    MB --> P5[p5 production]
    MB --> P6[p6 paper benchmark]
    MB --> API[FastAPI service]
    P5 --> FB[[fallback.json]]
    FB --> API
```

| Phase | Main outputs (`outputs/v2/phaseN/`) |
|---|---|
| 0 | `data_audit.csv`, `model_comparison.csv`, `v1_vs_v2.csv`, `leakage_demo.csv`, `error_by_*.csv` |
| 1 | `random_vs_temporal.csv`, `cv5_*.csv`, `time_cv_*.csv`, `significance.csv`, `ablation.csv`, `final_feature_check.csv` |
| 2 | `xgb_search.csv`, `final_model_comparison.csv`, `interval_*.csv/png`, `late_vs_early_tradeoff.csv`, `shap_*` |
| 3 | `tests_*.csv`, `summary_by_*.csv`, `ols_*.csv`, `vif_*.csv`, three charts |
| 4 | `01–06_*.csv`, `sql_features_*.csv`, `sql_leakage_check.csv` |
| 5 | `service_smoke_test.csv`, `latency.csv`, `ab_test_sample_size.csv`, `drift_psi.csv`, UI screenshot |
| 6 | `paper_benchmark.csv` |

## 3. Data cleaning and repair

`deliveriq/data/cleaning.py` never drops an order and never uses the target. Every repair is
counted in an audit.

```mermaid
flowchart TD
    RAW[Raw row] --> T{Order time<br/>parseable?}
    T -->|HH:MM or day fraction| OK1[minutes of day]
    T -->|blank| IMP[pickup − 10 min<br/>flag order_time_imputed]
    RAW --> G{Restaurant inside<br/>India bounds?}
    G -->|yes| D1[Haversine distance]
    G -->|no, 0,0| D2[city centre + offset<br/>flag coords_missing]
    RAW --> RP{Rider profile<br/>valid?}
    RP -->|age 50 + rating 6<br/>or age 15 + rating 1| NA[set to missing<br/>flag profile_invalid]
    RAW --> CAT[traffic / weather<br/>blank → Missing + flag]
    RAW --> ID[rider_id → city_code,<br/>restaurant_id]
```

| Repair | Rows |
|---|---|
| Order time in day-fraction format, recovered | 4,068 |
| Order time blank, filled from pickup time | 1,731 |
| Restaurant coordinates invalid, distance recovered | 3,640 |
| Placeholder rider profiles blanked | 91 |
| Rows dropped | 0 |

## 4. Order-time feature construction

```mermaid
flowchart LR
    C[Cleaned order] --> N[Numeric<br/>age, rating, vehicle,<br/>multiple deliveries]
    C --> DI[distance_km]
    C --> TM[hour, sin/cos hour,<br/>day of week, weekend, peak]
    C --> TR[traffic_level 0-3]
    DI --> X[distance × traffic]
    TR --> X
    C --> CT[one-hot: weather, traffic,<br/>order type, vehicle, festival,<br/>city type, city code, time of day]
    C --> FL[missing-value flags]
    N & DI & TM & TR & X & CT & FL --> M[Model matrix]
```

The order's own pickup time and the target are never features (`tests/test_leakage.py`).

## 5. Out-of-fold history encoding

Restaurant prep-time history is built from pickup delays, which are only known after an order is
placed. If a training row's average includes its own value, the model sees information in
training that it will never have at serving time.

```mermaid
flowchart TD
    TR[Training rows] --> K[Split into 5 folds]
    K --> F1[Fold k held out]
    K --> O[Other 4 folds]
    O --> ST[Per-restaurant sum and count]
    ST --> SM[Smoothed mean<br/>sum + k·global / count + k]
    SM --> F1
    TR --> ALL[All training rows]
    ALL --> FULL[Full-train table]
    FULL --> TE[Test and inference rows]
    FULL --> NEW[Unseen restaurant<br/>→ global mean]
```

Correlation of the feature with each row's own value: 0.109 with in-sample averaging, 0.001
out-of-fold. The feature was later removed from the final model because it carried no signal
(CV MAE 3.199 → 3.109 without it).

## 6. Evaluation splits and time-series CV

**Chronological split** (Phase 2 onward):

```mermaid
gantt
    title Train core, validation and test
    dateFormat YYYY-MM-DD
    axisFormat %d %b
    tickInterval 1week
    section Train core
    Feb block        :a1, 2022-02-11, 2022-02-19
    Mar block        :a2, 2022-03-01, 2022-03-25
    section Validation
    25-28 Mar        :crit, a3, 2022-03-25, 2022-03-29
    section Test
    29 Mar - 6 Apr   :active, a4, 2022-03-29, 2022-04-07
```

**Rolling-origin CV** (Phase 1 and 4): each fold trains on all earlier days and tests on the
next 4 days with orders.

```mermaid
gantt
    title Expanding-window folds
    dateFormat YYYY-MM-DD
    axisFormat %d %b
    tickInterval 1week
    section Fold 0
    train :f0a, 2022-03-01, 2022-03-21
    test  :crit, f0b, 2022-03-21, 2022-03-26
    section Fold 1
    train :f1a, 2022-03-01, 2022-03-26
    test  :crit, f1b, 2022-03-26, 2022-03-30
    section Fold 2
    train :f2a, 2022-03-01, 2022-03-30
    test  :crit, f2b, 2022-03-30, 2022-04-03
    section Fold 3
    train :f3a, 2022-03-01, 2022-04-03
    test  :crit, f3b, 2022-04-03, 2022-04-07
```

Training in every fold also includes 11–18 Feb. 22 Mar has no orders, so fold 0's four test
days are 21, 23, 24 and 25 Mar.

## 7. Model selection flow

```mermaid
flowchart TD
    A[LR, RF, XGBoost<br/>same features] --> B{Tree models<br/>beat LR?}
    B -->|yes, by ~1.6 min| C[Ablation: drop<br/>prep history]
    C --> D{Better in all<br/>5 folds?}
    D -->|yes| E[Final feature set]
    E --> F[Tune XGBoost<br/>20 configs on validation]
    F --> G[Blend RF + XGB<br/>weights on validation]
    G --> H{Blend gain<br/>worth it?}
    H -->|no, 0.008 min| I[Serve tuned XGBoost]
    E --> J[SQL history features]
    J --> K{Improve?}
    K -->|no, +0.036 min| E
```

## 8. Conformalized quantile regression

```mermaid
flowchart LR
    A[Train core] --> Q1[XGBoost q=0.10]
    A --> Q2[XGBoost q=0.90]
    V[Validation block] --> S["Scores on validation:<br/>max(lo − y, y − hi)"]
    Q1 --> S
    Q2 --> S
    S --> QH["Conformal quantile q<br/>rank ⌈(n+1)·0.8⌉"]
    Q1 --> OUT["80% range =<br/>[lo − q, hi + q]"]
    Q2 --> OUT
    QH --> OUT
```

Split conformal (the comparison method) uses a single `|y − ŷ|` quantile, giving every order
the same ±4.9 min width.

## 9. SQL feature lineage

```mermaid
flowchart LR
    O[(orders)] --> RD[rider_day<br/>GROUP BY rider, day]
    RD --> RC[rider_cum<br/>SUM OVER ... 1 PRECEDING<br/>LAG day]
    RC --> J1[LEFT JOIN on rider, day]
    O --> J1
    O --> ED[rest_day<br/>GROUP BY restaurant, day]
    ED --> RR[rest_roll<br/>RANGE 7 DAY .. 1 DAY PRECEDING]
    RR --> J2[LEFT JOIN on restaurant, day]
    O --> J2
    J1 --> H[History features<br/>previous days only]
    J2 --> H
    H --> CK[pandas cross-check<br/>300 of 300 match]
```

## 10. Prediction service internals

```mermaid
flowchart TD
    REQ[POST /v2/predict] --> VAL{Pydantic<br/>validation}
    VAL -->|invalid| E422[HTTP 422]
    VAL -->|valid| RAW[OrderInput → raw row<br/>pickup time = None]
    RAW --> CLN[clean_orders<br/>+ saved city centroids]
    CLN --> FT[base_features]
    FT --> TRY{Model OK?}
    TRY -->|yes| PT[XGBoost ETA]
    PT --> RG[CQR 80% range]
    RG --> SH[Grouped SHAP<br/>top 3 reasons]
    TRY -->|error| FBK[Fallback: training<br/>median by traffic]
    SH --> WR[Warnings: unknown traffic,<br/>invalid rider, city centre used]
    FBK --> WR
    WR --> RESP[JSON response]
```

## 11. Model bundle contents

```mermaid
flowchart LR
    B[[final_bundle.joblib]] --> P[point_model<br/>preprocessor + tuned XGBoost]
    B --> Q[cqr80<br/>two quantile models + q]
    B --> SC[split_conformal_q80]
    B --> CR[city_reference<br/>city centroids]
    B --> FS[numeric / categorical<br/>feature lists]
    B --> MD[xgb_params, train_period,<br/>test_metrics]
```

## 12. Test coverage map

```mermaid
flowchart LR
    T1[test_data.py<br/>14 tests] --> A1[clock parsing, audit counts,<br/>no dropped rows, distances]
    T2[test_leakage.py<br/>7 tests] --> A2[out-of-fold encoder, split order,<br/>no pickup/target features, SQL windows]
    T3[test_models.py<br/>5 tests] --> A3[business metrics, conformal rank,<br/>coverage, paired test, Holm]
    T4[test_api.py<br/>7 tests] --> A4[health, range contains ETA,<br/>Jam slower than Low, validation, batch]
```

## 13. Production design (design only)

Not implemented; see [PRODUCTION_DESIGN.md](PRODUCTION_DESIGN.md) for details.

```mermaid
stateDiagram-v2
    [*] --> Checkout: order placed
    Checkout --> RiderAssigned: re-predict with rider location
    RiderAssigned --> FoodReady: re-predict with actual prep
    FoodReady --> PickedUp: re-predict with GPS
    PickedUp --> NearCustomer: last-mile correction
    NearCustomer --> [*]: delivered
    note right of Checkout: DeliverIQ covers this stage
```
