# 🏡 Immo Eliza — Real-Estate Price Prediction (Regression)

Machine-learning pipeline that predicts **sale prices** and **monthly rents** of
Belgian residential properties for the fictional real-estate company *Immo
Eliza*. The project takes scraped-and-cleaned listings, builds a **reusable
preprocessing + modelling pipeline**, trains and tunes four families of
regression models for each market, evaluates them on held-out data, and uses
them to price brand-new (dummy) properties.

> Two separate markets are modelled — **sale** and **rent** — because their
> price scales are completely different (hundreds of thousands of euros vs a few
> hundred euros per month). Both use the **same feature set and the same
> preprocessing**, so the only thing that differs is the data each model is fit
> on.

---

## 📁 Project structure

```
immo-eliza-ml/
├── data/
│   ├── in/                      # raw input (provided): cleaned_sale/rent_properties.csv
│   ├── training/                # cleaned TRAIN split  (sale_train.csv, rent_train.csv)
│   ├── test/                    # cleaned TEST  split  (sale_test.csv,  rent_test.csv)
│   └── dummy/                   # 10 sale + 10 rent unseen listings (NO price) for prediction
├── models/
│   ├── 0_empty/                 # untrained pipelines        (pricing_<algo>_<market>.joblib)
│   ├── 1_trained/               # trained on default params
│   ├── 2_tuned/                 # best params from cross-validated search
│   └── evaluation_results.csv   # full metrics table written by evaluate.py
├── src/
│   ├── preprocessing.py         # clean → define feature set + reusable transformer → split → save
│   ├── create_models.py         # build the untrained pipelines           → models/0_empty
│   ├── train_models.py          # fit each pipeline on the training data   → models/1_trained
│   ├── evaluate.py              # score trained & tuned models on the test set (R²/MAE/RMSE)
│   ├── tune_models.py           # cross-validated hyper-parameter search   → models/2_tuned
│   └── predict.py               # price the dummy properties with every model
├── other/exercise_README.md     # original assignment brief
├── requirements.txt
└── README.md
```

---

## 🔧 Setup

```bash
python -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt
# macOS only — XGBoost needs the OpenMP runtime:
brew install libomp
```

## ▶️ How to run (full pipeline, in order)

```bash
python src/preprocessing.py    # 1. clean + split + save data/training & data/test
python src/create_models.py    # 2. build 8 untrained pipelines      -> models/0_empty
python src/train_models.py     # 3. train them                       -> models/1_trained
python src/evaluate.py         # 4. accuracy of the trained models
python src/tune_models.py      # 5. hyper-parameter tuning           -> models/2_tuned
python src/evaluate.py         # 6. accuracy of trained vs tuned (same script, now richer)
python src/predict.py          # 7. price the 20 dummy properties
```

---

## 🧠 Approach, step by step

### 1. Cleaning & preprocessing — `preprocessing.py`
This module is the **single source of truth** for what the models see. It is
designed as a *reusable pipeline* so the **exact same** transformations are
applied to the training data and to any new property at prediction time.

**Cleaning (`clean_data`)**
- Drop exact duplicates and duplicate `property_id`s.
- **Target hygiene:** drop rows with a missing price or one outside sane,
  *hard-coded* per-market domain bounds (e.g. a €0 house or a €1 rent). These
  constant bounds are leakage-free, so they apply to both splits (~99.6% of rows
  survive). The extreme **1st/99th-percentile price tails are trimmed on the
  training split only** (after the train/test split), so the held-out test set is
  never filtered using statistics it should not have seen — keeping the
  evaluation honest.
- **Drop leakage:** `price_per_sqm` is `price ÷ surface` — it would leak the
  target — so it is never used. `cadastral_income` (100% missing for rent) and
  high-cardinality identifiers (`property_id`, `street`, `locality`, …) are
  excluded too.
- Null-out impossible values (0 m² surfaces, negative/absurd energy scores,
  build years outside 1750–2031, room counts > 15) so the imputer can replace
  them.
- Coerce amenity columns (terrace, garden, pool, …) to clean **0/1** flags —
  a missing amenity is treated as *absent*.

**Feature set (identical for every model & both markets — 29 features)**
| Type | Handling | Features |
|------|----------|----------|
| **Numeric** (12) | median impute → standardise | livable_surface, bedrooms, bathrooms, toilets, build_year, facades, number_of_floors, primary_energy_consumption, land_surface, latitude, longitude, nearest_city_distance_km |
| **Categorical** (7) | "missing" category → one-hot | property_type, province, region, epc, building_state, kitchen_equipment, heating_type |
| **Binary** (10) | impute 0 (passthrough) | new_construction, furnished, terrace, garden, swimming_pool, elevator, cellar, solar_panels, air_conditioning, has_parking |

The transformer (`build_preprocessor`) bundles imputation + one-hot encoding
(`handle_unknown='ignore'`) + standardisation into a single scikit-learn
`ColumnTransformer`. It is **embedded inside every model pipeline** and fitted on
the training fold only — no test/prediction leakage.

The cleaned data is split **80/20** into `data/training` and `data/test`.

### 2. Build the models — `create_models.py`
Each model is a `Pipeline(preprocessor → estimator)` so it consumes raw cleaned
features end-to-end. Four algorithm families span the bias/variance spectrum:

| Algorithm | Why it's here |
|-----------|---------------|
| **Linear Regression** | simple, interpretable **baseline** |
| **Decision Tree** | single non-linear tree (shows over-fitting clearly) |
| **Random Forest** | **bagging** ensemble — strong, low-variance |
| **XGBoost** | **gradient boosting** — usually the top performer |

4 algorithms × 2 markets = **8 models**, saved as
`pricing_<algorithm>_<market>.joblib`.

### 3. Train — `train_models.py`
Loads each untrained pipeline, fits it on the matching training split, and saves
it to `models/1_trained`. (Models are saved with `compress=3` so the random
forests stay well under GitHub's 100 MB limit.)

### 4. Evaluate — `evaluate.py`
Scores every model on the **held-out test set**. Because this is regression, the
head-line "accuracy" is **R²** (share of price variance explained); **MAE** and
**RMSE** (in euros) are reported alongside. The script prints the trained and
tuned models side by side and also shows **train R²** so over-fitting is obvious.

### 5. Tune — `tune_models.py`
Cross-validated (`cv=4`, scoring R²) hyper-parameter search per model
(`GridSearchCV` for the linear/Ridge penalty, `RandomizedSearchCV` for the
trees/boosting). Each search space **includes the default parameters**, so
tuning never regresses on its cross-validation objective (on the held-out test
set the already-strong ensembles then move only within noise). Plain OLS has
nothing to tune, so its tuned version is a regularised **Ridge** regression.
Best models are saved to `models/2_tuned`.

### 6. Predict — `predict.py`
Loads the dummy listings (raw features only, **no price**), runs every untuned
and tuned model, and prints the estimates next to each property description,
plus a head-line estimate from the best overall model (tuned XGBoost).

---

## 📊 Results

Head-line metric = **R²** on the **held-out test set** (higher is better; 1.0 =
perfect). MAE/RMSE are in euros. "Default" = trained model; "Tuned" = after the
cross-validated hyper-parameter search.

### 🏠 SALE — sale price (€)
| Model | Test R² (default) | Test R² (tuned) | MAE (tuned) | RMSE (tuned) |
|---|:---:|:---:|---:|---:|
| **XGBoost** ⭐ | 0.808 | **0.812** | €81,840 | €184,419 |
| Random Forest | 0.774 | 0.772 | €90,111 | €203,203 |
| Decision Tree | 0.633 | 0.707 | €116,293 | €230,361 |
| Linear Regression *(baseline)* | 0.642 | 0.642 | €139,201 | €254,549 |

### 🔑 RENT — monthly rent (€)
| Model | Test R² (default) | Test R² (tuned) | MAE (tuned) | RMSE (tuned) |
|---|:---:|:---:|---:|---:|
| **XGBoost** ⭐ | 0.624 | **0.627** | €254 | €659 |
| Random Forest | 0.591 | 0.586 | €269 | €694 |
| Decision Tree | 0.493 | 0.518 | €334 | €749 |
| Linear Regression *(baseline)* | 0.532 | 0.532 | €353 | €738 |

> These are **honest, leakage-free** numbers: the test set is evaluated on the
> full in-domain price range (the percentile outlier trim is applied to the
> *training* split only). See the luxury-tail note below for why the rent R² in
> particular looks modest.

### 🔎 What the metrics say
- **Best model = tuned XGBoost:** explains **~81%** of sale-price variance and
  **~63%** of rent variance overall, with a typical error (MAE) of **~€82k** on
  sale prices and **~€254/month** on rents.
- **Strong on the typical market, weak on the luxury tail:** R² is dragged down
  by a handful of ultra-expensive outliers the model has too few examples of and
  systematically under-prices. Excluding the top ~1.5% (sales > €2M, rents >
  €5,000/mo), tuned XGBoost reaches **R² ≈ 0.86 (sale)** and **R² ≈ 0.83 (rent)**
  — i.e. it prices the *bulk* of the market very well. This tail effect is far
  larger for rent (few high-end rentals in the data), which is why its head-line
  R² is the lowest.
- **Ensembles ≫ baseline:** Random Forest and XGBoost beat the linear baseline by
  **~0.10–0.17 R²** — price depends on the data in clearly *non-linear*,
  interacting ways (location × surface × condition) that a straight line cannot
  capture.
- **Over-fitting is real and visible:** the default Decision Tree gets
  **train R² ≈ 1.00 but test R² = 0.63** (sale) — textbook memorisation. Tuning
  (depth/leaf limits) closes much of the gap (test R² **0.63 → 0.71**). The
  ensembles also show a train–test gap but generalise far better thanks to
  bagging/boosting.
- **Tuning helped, especially the weak models:** big win on the Decision Tree
  (**+0.07** sale, **+0.03** rent) by curbing its over-fitting; the
  already-strong ensembles moved only within noise on the held-out set
  (XGBoost +0.003–0.004; Random Forest a hair either way). The Ridge-regularised
  "linear" model matched OLS. Each search space *included the defaults*, so
  tuning never regresses **in cross-validation** (the objective it optimises).

### 🏷️ Example predictions (tuned XGBoost, on unseen dummy properties)
| For-sale property | Est. price | | Rental property | Est. rent |
|---|---:|---|---|---:|
| Charleroi budget 2-bed flat (70 m², EPC E) | €101,400 | | Liège budget 1-bed (55 m², EPC E) | €665 /mo |
| Liège terraced house to renovate (130 m², EPC F) | €159,900 | | Antwerp 2-bed apartment (90 m²) | €981 /mo |
| Brussels 2-bed apartment (85 m², EPC C) | €340,800 | | Hasselt suburban house (140 m²) | €1,228 /mo |
| Antwerp townhouse to renovate (165 m², EPC D) | €376,900 | | Walloon Brabant family house (180 m²) | €2,078 /mo |
| Knokke coastal penthouse (140 m², EPC B) | €978,600 | | Knokke furnished coastal flat (80 m²) | €1,746 /mo |
| Walloon Brabant villa + pool (280 m², EPC B) | €1,087,400 | | Brussels luxury furnished penthouse (150 m²) | €4,146 /mo |

These line up well with current Belgian market levels (see *Market context*
below). **One honest limitation:** the model **over-prices very small studios**
(e.g. a 30 m² Ghent studio → ~€1,600/month) and **under-prices the ultra-luxury
tail**, because both extremes are sparse in the training data — a good reminder
that models extrapolate poorly where data is thin.

### 🇧🇪 Market context (2025–2026 — used to ground the dummy data & sanity-check predictions)
- National median sale prices in 2025: **~€280k** semi/terraced houses,
  **~€385–390k** detached houses, **~€249–255k** apartments (houses +2–7% YoY).
  ([statbel.fgov.be](https://statbel.fgov.be/en/news/house-prices-first-semester-2025))
- Strong **regional gradient:** detached-house median ~€330k in Wallonia,
  ~€425–430k in Flanders, **>€1,000,000 in Brussels**; cheapest markets are rural
  Wallonia / Hainaut, priciest are Flemish & Walloon Brabant.
  ([statbel.fgov.be](https://statbel.fgov.be/en/news/house-prices-first-semester-2025))
- Flemish cities 2025: **Leuven** dearest (house €450k), then **Ghent** (€380k),
  **Antwerp** (€365k), **Bruges** (€350k); **Charleroi** is the cheap end
  (~€1,400–1,500/m²). ([pandwijzer.be](https://pandwijzer.be/en/vastgoedprijzen/gent))
- **Rents 2025–2026:** Flanders ~€948/mo on average (apartments ~€893, houses
  ~€1,014–1,255), **Brussels ~€1,346/mo**, Wallonia cheapest at ~€634–879/mo.
  ([cib.be](https://community.cib.be/actua/news/3fd5cdef-8a5a-4e20-bd59-43f26e05b396))
- **EPC label moves price:** in Flanders an A-rated house sells **~+13%** and an
  F-rated **~−10%** vs an identical D-rated home (~23pp spread); the stock is old
  and energy-inefficient (~38% pre-1945).
  ([nbb.be](https://www.nbb.be/doc/ts/publications/economicreview/2025/ecorevi2025_h01.pdf))
- Typical sizes: existing Flemish house **~100 m²**, apartment **~64 m²**,
  new-build houses **~170 m²**; 2-bedroom homes are now the most common type.
  ([oximo.be](https://www.oximo.be/nl/nieuws/zo-groot-zijn-vlaamse-woningen))

---

## 📦 The dummy data (`data/dummy/`)
20 realistic, never-seen Belgian listings (10 for sale, 10 to rent) with **no
price**, containing exactly the 29 model features plus a human-readable
`description` label (used only for display, not by the models). Values are
grounded in the real dataset (province coordinates, typical surfaces per
property type, valid category values) and in current Belgian-market figures
(see *Market context* above) — covering a deliberate spread of regions, property
types, sizes, energy labels and conditions.

## ✅ Key takeaways / interview notes (STAR-ready)
- **Messy data:** ~25 of 58 raw columns were >40% missing; the fix was a
  principled split into *median-imputed numerics*, *one-hot categoricals with a
  "missing" level*, and *amenity flags where missing = absent* — all inside one
  reusable transformer.
- **Leakage (feature):** spotting and dropping `price_per_sqm` (a function of the
  target) was essential — keeping it would have produced a deceptively "perfect"
  model.
- **Leakage (evaluation):** a self-review caught a *subtle* leak — trimming price
  outliers by percentile **before** the train/test split let the test
  distribution influence cleaning. Moving the percentile trim to the **training
  split only** dropped the head-line rent R² from an inflated **0.80** to an
  honest **0.63** — a reminder that catching leakage often makes your numbers
  *worse* but *truer*.
- **Reusable pipeline:** preprocessing lives inside every model, so pricing a new
  house is a single `pipeline.predict(raw_features)` call.
- **Over-fitting:** the lone Decision Tree memorised the training set
  (train R² ≈ 1.0, far above its test R²); the ensembles (Random Forest, XGBoost)
  generalise much better — exactly why bagging/boosting win here.
