# Fraud Detection Project — End-to-End Review Document

> **Purpose:** A complete reference for explaining this project in interviews — architecture, ML decisions, engineering gotchas, and the reasoning behind every major choice.
>
> **Live app:** Deployed on Hugging Face Spaces (Docker runtime)
> **Repo:** https://github.com/yashaur/fraud-detection-project
> **Team:** Shaurya Singru & Paul Babu (2-person team)
> **My ownership:** Modelling pipeline (benchmarking, model selection, hyperparameter tuning) and app architecture (session-state/caching design, Predict page, Threshold Slider page). Paul owned the Dashboard page and initial SHAP work.

---

## 1. Elevator Pitch (30-second version)

An end-to-end machine learning system that detects fraudulent financial transactions. I took a 6.3-million-row synthetic mobile-money dataset from Kaggle, performed EDA, benchmarked 7 classifiers under proper cross-validation, selected and tuned a LightGBM model (96.4% precision / 83.6% recall on a 2.7M-row held-out test set), and shipped it as an interactive multi-page Streamlit app — with live single-transaction prediction, SHAP explainability, and a user-adjustable decision threshold — containerised with Docker and deployed on Hugging Face Spaces.

**Key differentiators to emphasise:**
- Not just a notebook — a deployed, containerised product with an explainability layer.
- Rigorous model selection: 7 models × stratified 5-fold CV, ranked on PR-AUC (the right metric for 0.13% class imbalance), with training/scoring *cost* as an explicit selection criterion.
- The decision threshold is exposed as a product feature, turning the precision/recall trade-off into something a business user can control.

---

## 2. Problem Statement & Dataset

### The problem
Binary classification: given a financial transaction (type, amount, account balances before/after, time of day), predict whether it is fraudulent. The core challenge is **extreme class imbalance**: only ~0.13% of transactions are fraud (~1 in 770).

### The dataset
- **Source:** Kaggle — `amanalisiddiqui/fraud-detection-dataset` (a PaySim-style synthetic dataset simulating mobile money transactions).
- **Size:** ~6.36 million rows.
- **Columns:**

| Column | Description | Kept? |
|---|---|---|
| `step` | Time step, 1 unit = 1 hour (744 steps ≈ 31 days) | ❌ Dropped (see below) — but used to derive `hour_of_day` |
| `type` | Transaction type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER | ✅ |
| `amount` | Transaction amount | ✅ |
| `nameOrig` / `nameDest` | Origin/destination account IDs | ❌ Dropped |
| `oldbalanceOrg` / `newbalanceOrig` | Origin account balance before/after | ✅ |
| `oldbalanceDest` / `newbalanceDest` | Destination account balance before/after | ✅ |
| `isFlaggedFraud` | Flag from the dataset's naive rule-based system | ❌ Dropped |
| `isFraud` | **Target variable** | ✅ |

### Memory optimisation at load time
The CSV was read with explicit dtypes (`float32` for amounts, `int16` for step, `category` for type) instead of pandas defaults (`float64`/`object`). On 6.3M rows this roughly halves memory usage — necessary to work with the full dataset in Colab.

---

## 3. EDA — Key Findings & Decisions

1. **Class imbalance:** `isFraud` is ~0.13% positive. This drove everything downstream: choice of metrics (PR-AUC over accuracy/ROC-AUC), class weighting, stratified splits.

2. **Fraud concentrates in specific transaction types.** The fraud *rate* is by far highest for **TRANSFER** transactions (with CASH_OUT the other significant type). PAYMENT/CASH_IN/DEBIT fraud is essentially nonexistent. This made `type` an obviously powerful categorical feature.

3. **Fraud amounts are larger.** A boxplot of `amount` vs `isFraud` (filtered to < 100k for readability) showed fraudulent transactions have a higher median (~40k) and wider IQR than legitimate ones. Amount distribution overall is heavily right-skewed, so it was visualised with `log1p`.

4. **The `step` (time) variable is broken — a synthetic-data artifact.** Plotting fraud rate by day-of-month showed a flat ~2–3% rate for days 1–30, then a spike to ~100% on day 31 (legitimate transactions simply stop being generated at the end of the simulation). Because this pattern is an artifact and not real signal, we **dropped `step` and all derived calendar features** (`day_of_month`, `weekday`, `timestamp`) to avoid the model learning nonsense. This is a good interview story: *we deliberately threw away a "predictive" feature because its predictive power was an artifact of the data-generating process and wouldn't generalise.*

5. **But hour-of-day is meaningful.** We derived `hour_of_day = step % 24`, and fraud rate does vary by hour (fraud is relatively more common at night when legitimate volume drops). This was the one time-derived feature we kept.

6. **Dropped `nameOrig`/`nameDest`:** account IDs are nearly all unique in this dataset, so they carry no generalisable signal (they would just memorise accounts). Dropped `isFlaggedFraud`: it's the output of an existing naive rule (flags transfers > 200k) — keeping it would be leaking another model's answer.

7. **Correlation matrix** across the numeric features + target showed no single dominant linear correlation with fraud — supporting the choice of non-linear tree-based models.

8. **Explored-then-abandoned approaches (good to mention):** In early EDA we prototyped a Logistic Regression baseline with StandardScaler + OneHotEncoder + **SMOTE** oversampling, and also experimented with **Weight of Evidence (WoE)** encoding for `type` (a credit-risk technique: log-odds of good vs bad per category, with Laplace smoothing). For the final pipeline we moved away from SMOTE in favour of **class weighting** (`class_weight='balanced'` / `scale_pos_weight`) because: (a) SMOTE on ~4.4M training rows is computationally heavy; (b) synthetic interpolation of fraud rows risks creating unrealistic samples; (c) class weights achieve the same rebalancing effect inside the loss function without touching the data; and (d) SMOTE must be applied inside each CV fold to avoid leakage, which complicates the pipeline.

---

## 4. Feature Engineering

Final feature set (9 features fed to the model):

```
type (categorical), amount, oldbalanceOrg, newbalanceOrig,
oldbalanceDest, newbalanceDest, hour_of_day, sin_hour, cos_hour
```

### Cyclical encoding of hour (the flagship feature-engineering decision)
`hour_of_day` is cyclical — hour 23 and hour 0 are adjacent, but numerically they're maximally far apart. We encoded it as:

```python
sin_hour = sin(hour * 2π / 24)
cos_hour = cos(hour * 2π / 24)
```

This maps each hour onto a point on the unit circle, so midnight and 11 PM are close in feature space. You need *both* sin and cos: either one alone is ambiguous (sin(2)=sin(10) o'clock positions collide); together they uniquely identify the hour while preserving circular distance.

- For **LightGBM/XGBoost** we used `sin_hour`/`cos_hour` (plus raw `hour_of_day` as a categorical).
- For **RandomForest** (no native categorical support in sklearn) we one-hot encoded `hour_of_day` and `type` instead.

### Native categorical handling
LightGBM handles `category`-dtype columns natively (no one-hot needed) — it finds optimal category splits directly. This is both faster and often more accurate than one-hot for high-cardinality categoricals, and is one of the practical reasons LightGBM was pleasant to deploy.

---

## 5. Benchmarking & Model Selection (my core contribution)

### Data splitting strategy
```
6.36M rows
├── 15%   → benchmark set (stratified) — used for 5-fold CV model comparison
└── 85%
    ├── 50% → validation set (~42.5% of total, ~2.7M rows) — used for hyperparameter tuning
    └── 50% → test set (~42.5% of total, ~2.7M rows) — held out for final evaluation
```

Rationale: with 6.3M rows, 15% (~950k rows) is plenty for reliable CV comparison of 7 models while keeping compute tractable, and it leaves two *very large* untouched sets — one to tune on and one that is only touched once, at the very end. The benchmark split was **stratified** on the target so each fold preserves the 0.13% fraud rate.

### The benchmarking harness (engineering I'm proud of)
I built a reusable benchmarking framework in the notebook rather than ad-hoc cells:

- **`get_preprocessor()`** — builds a `ColumnTransformer`: median imputation (+ optional `StandardScaler`) for numerics; most-frequent imputation + `OneHotEncoder(handle_unknown='ignore', drop='first')` for categoricals. Scaling is *conditional* because tree models don't need it (they split on thresholds, invariant to monotone transforms) while LogReg/SVC/KNN do.
- **`build_model_constructors()`** — one factory per model, with class imbalance handled per-model: `class_weight='balanced'` for sklearn models, `scale_pos_weight = n_negative/n_positive` for the boosting models. Solver choices were deliberate (e.g. `solver='sag'` for LogReg because of large-n, low-dimensional, dense data).
- **`build_pipelines()`** — pairs each model with the correct preprocessor (scaled vs unscaled) into a sklearn `Pipeline`, so preprocessing is fitted *inside* each CV fold — **no leakage of test-fold statistics into training**.
- **`benchmark_models()`** — runs `cross_validate` with `StratifiedKFold(n_splits=5, shuffle=True)` and four scorers: precision, recall, ROC-AUC, **PR-AUC (average precision)**. Every run auto-saves results, confusion matrices (CSV + heatmap PNG), a JSON metadata file, and appends to a persistent execution log on Google Drive — so experiments were reproducible and comparable across sessions. Each execution got a numbered, timestamped folder.

### Results (5-fold CV, ranked by PR-AUC)

| Model | PR-AUC | ROC-AUC | Recall | Precision | F1 | Total time |
|---|---|---|---|---|---|---|
| **LightGBM** | **0.929** | 0.999 | 0.945 | 0.524 | 0.674 | **2.5 min** |
| XGBoost | 0.921 | 0.999 | **0.985** | 0.247 | 0.395 | 2.8 min |
| RandomForest | 0.874 | 0.975 | 0.761 | **0.980** | **0.857** | 16.8 min |
| DecisionTree | 0.658 | 0.984 | 0.970 | 0.090 | 0.165 | 0.7 min |
| LogisticRegression | 0.641 | 0.993 | 0.948 | 0.032 | 0.062 | 65.5 min |
| KNN | 0.542 | 0.856 | 0.407 | 0.876 | 0.556 | 85.1 min |
| GaussianNB | 0.041 | 0.918 | 0.583 | 0.025 | 0.047 | fast |

(SVC was excluded from the run — kernel SVMs are O(n²–n³) in samples and infeasible at ~1M rows.)

### How the winner was chosen (tell this story in interviews)
- **Why PR-AUC as the primary metric:** with 0.13% positives, ROC-AUC is misleadingly optimistic (note LogReg's 0.993 ROC-AUC alongside 3% precision — it flags ~30 false alarms for every real fraud). PR-AUC focuses on performance on the positive class across all thresholds, which is what matters when positives are rare. Accuracy is worthless here — predicting "never fraud" is 99.87% accurate.
- **Threshold-independent ranking:** precision/recall at the default 0.5 threshold are just one operating point; PR-AUC summarises the whole trade-off curve. This is also why the *app* exposes a threshold slider — the "best" threshold is a business decision, not an ML one.
- **Cost as a first-class criterion:** LogReg took >1 hour to fit (even with SAG); KNN took 84 minutes just to *score* (it's lazy — all compute at inference, disqualifying for a real-time fraud system). LightGBM matched or beat everything at 2.5 minutes total.
- **Shortlist:** LightGBM (best PR-AUC, balanced, fastest), RandomForest (best precision & F1, moderate cost), XGBoost (best recall — in fraud, missing a fraud usually costs more than a false alarm). KNN was cut despite decent precision because of its inference cost.

---

## 6. Hyperparameter Tuning

- **Method:** `RandomizedSearchCV` (100 candidates × 3 folds = 300 fits for LightGBM), run on GPU (Colab Tesla T4 / Kaggle). Optuna was set up as a planned next step (Bayesian/TPE search) but randomized search already converged to strong parameters.
- **Why randomized over grid search:** with 5+ continuous hyperparameters a grid explodes combinatorially, and randomized search explores the space more efficiently for the same budget (Bergstra & Bengio) — important parameters get many distinct values tried instead of a few grid points.
- **Search space design (LightGBM):** `learning_rate ~ loguniform(0.01, 0.2)` (log-scale because learning-rate effects are multiplicative), `n_estimators ~ randint(100, 2000)`, `num_leaves ~ randint(31, 256)`, `reg_alpha`/`reg_lambda ~ loguniform(1e-5, 10)`.
- **Winning configuration (the deployed model, verified from the pickle):**

```python
LGBMClassifier(
    objective='binary', boosting_type='gbdt',
    learning_rate=0.01099,   # low LR...
    n_estimators=1321,       # ...compensated by many trees (classic slow-and-steady GBM recipe)
    num_leaves=31, random_state=42, n_jobs=-1
)
# 9 features: type, amount, oldbalanceOrg, newbalanceOrig,
#             oldbalanceDest, newbalanceDest, hour_of_day, sin_hour, cos_hour
```

- The final model was retrained on proper training data after tuning, then serialised with `joblib` to `model/lgbm.pkl` (~4.7 MB).

### Final held-out test performance (~2.7M rows, 3,473 actual frauds)

| Metric | Value |
|---|---|
| **Precision** | **96.44%** |
| **Recall** | **83.56%** |
| **F1** | **89.54%** |
| Balanced accuracy | 91.78% |
| Average precision (PR-AUC) | 0.806 |

Confusion matrix:

```
                 Predicted Neg   Predicted Pos
Actual Neg         2,700,534           107      ← only 107 false alarms in 2.7M legit transactions
Actual Pos               571         2,902      ← caught 2,902 of 3,473 frauds
```

**How to talk about this:** out of 2.7 million legitimate transactions, the model raised only 107 false alarms, while catching ~84% of all fraud. In production, the remaining 16% miss rate is exactly why the threshold slider exists — an operations team can lower the threshold to trade some precision for higher recall.

---

## 7. Application Architecture (my design)

### Repository structure

```
fraud-detection-project/
├── app.py                     # Entry point: session init + st.navigation router
├── pages/
│   ├── dashboard.py           # 📊 Fleet-level analytics + global SHAP (Paul's page)
│   ├── predict.py             # 🚨 Single-transaction prediction + local SHAP explanation
│   └── threshold_slider.py    # 🎛️ Interactive precision/recall trade-off explorer
├── utils/
│   ├── init.py                # init_session_vars() — idempotent app bootstrap
│   ├── data.py                # Loading, preprocessing, dashboard aggregations
│   ├── model.py               # load_model() / predict() with caching
│   ├── precision_recall.py    # Hand-rolled vectorised precision/recall + PR curve
│   ├── shap.py                # TreeExplainer wrapper (global + local SHAP)
│   ├── shap_init.py           # Lazy SHAP-values init (deferred to dashboard visit)
│   └── charts.py              # All Plotly figure factories
├── model/lgbm.pkl             # Tuned LightGBM (joblib, ~4.7MB)
├── data/
│   ├── X_sample.csv           # 10k rows sampled from held-out test set (app data)
│   ├── y_sample.csv           # Matching labels
│   ├── prediction_samples.json# 32 curated transactions for the demo button
│   └── sample.py              # Script that generated the samples
├── notebooks/                 # 1 EDA → 2 Benchmarking → 3 Model Selection → 4 Tuning
├── Dockerfile
├── requirements.txt
├── .streamlit/config.toml     # Theme, viewer-mode toolbar, runOnSave
└── README.md                  # HF Spaces frontmatter (title, sdk, app_file)
```

Design principle: **pages contain only UI code; all logic lives in `utils/`**. Every utils module has an `if __name__ == '__main__':` block so it can be smoke-tested standalone (`python -m utils.model`) without launching Streamlit.

### The core Streamlit problem this architecture solves
Streamlit **re-executes the entire script top-to-bottom on every widget interaction**. Naively, that means reloading a 10k-row CSV, a model, a SHAP explainer, and recomputing predictions on *every slider tick*. The architecture defends against this with two layers:

**Layer 1 — Caching (survives across users/sessions):**
- `@st.cache_resource` for **singleton, unhashable/unserialisable objects**: the LightGBM model, the SHAP `TreeExplainer`. One shared instance, never copied.
- `@st.cache_data` for **data artifacts**: loaded DataFrames, prediction arrays, PR-curve arrays, aggregations. Returns copies, keyed on input hashes.
- **The underscore convention (a real gotcha I solved):** `st.cache_data` hashes all arguments to form the cache key, but a LightGBM model isn't hashable. Prefixing the parameter with an underscore (`def predict(_model, X)`) tells Streamlit to *skip hashing that argument*. Trade-off: the cache won't invalidate if the model changes — acceptable here because the model is a static artifact loaded once.

**Layer 2 — Session state (per-user, survives page switches):**
- `init_session_vars()` runs at the top of `app.py` *and* every page (pages can be hit directly via URL), but is **idempotent**: an `_app_initialised` flag plus per-key `if 'x' not in st.session_state` checks make repeat calls free.
- It eagerly warms everything the app needs: data, model, predictions on all 10k rows, the full PR array (precision/recall at 101 thresholds), the SHAP explainer, and dashboard aggregations. It also logs exactly which pieces were initialised — an intentional observability aid during development.
- **Exception — SHAP values are lazily initialised** (`shap_init()` is only called from the dashboard, behind a spinner): computing SHAP values over 10k rows takes tens of seconds, and users who never open the dashboard shouldn't pay for it. This eager-vs-lazy split was a deliberate startup-latency decision.

### Cross-page shared state: the threshold
The decision threshold (default 50%) lives in `st.session_state['threshold']` and is read by all three pages — set it on the slider page, and the Predict page's verdicts and the Dashboard's alert list immediately reflect it. **Gotcha solved:** a Streamlit widget's state is tied to its key and can reset when you navigate away. The fix is a two-key pattern — the slider widget uses its own key (`threshold_slider`) with an `on_change` callback that copies into the canonical `threshold` key, and the slider's `value=` reads from the canonical key. The "Reset to 50%" button updates both.

### Page-by-page

**🚨 Predict (`pages/predict.py`)** — my page
- Form for 7 fields (type, amount, hour, 4 balances), built from a central `field_names` dict (single source of truth mapping model column names → human labels, reused across pages and SHAP output).
- "Cycle Random Demo Values" pulls from 32 curated real transactions (`prediction_samples.json`, sampled from the test set) — so demos show realistic mixed outcomes instead of hand-typed inputs.
- Validates all fields are filled, preprocesses (`preprocess_input(..., source='app')`), predicts a fraud probability, and classifies against the *session* threshold.
- **Explainability:** an expander computes SHAP values for that single transaction and reports the strongest risk-increasing feature (for fraud verdicts) or strongest risk-reducing feature (for legit verdicts) in plain English.

**🎛️ Threshold Slider (`pages/threshold_slider.py`)** — my page
- A slider (0–100%) with live **precision & recall doughnut gauges** and a **PR curve with a red marker showing the current operating point** moving along it.
- The PR curve is precomputed once at startup (`precision_recall_array`: 101 thresholds over the cached 10k-row probabilities), so slider interaction only recomputes one (precision, recall) pair — instant feedback.
- I hand-rolled the confusion-matrix math with vectorised NumPy boolean masks (`tp = np.sum(ap_mask * pp_mask)` etc.) rather than calling sklearn per tick — it's transparent, fast, and handles the `thresh=1.0` edge case where zero positive predictions makes precision 0/0 (defined as 1.0, the correct limit).

**📊 Dashboard (`pages/dashboard.py`)** — Paul's page, refactored jointly
- "Top Fraud Alerts": all sample transactions above the current threshold, sorted by fraud probability.
- Fraud *rate* by hour of day (12-hour-labelled buckets like "2PM - 3PM"), fraud per 1,000 transactions by Day/Night segment and by transaction type — **rates, not counts**, so busy hours don't look fraudulent just from volume.
- **Global SHAP importance:** mean absolute SHAP value per feature across the 10k sample, as a horizontal bar chart — shows what drives the model overall (lazily computed, cached in session state afterwards).

### Preprocessing at inference time (`preprocess_input`)
A single function accepts a dict (from the form), Series, or DataFrame and reproduces the *exact* training-time representation:
1. Recomputes `sin_hour`/`cos_hour`.
2. Normalises type strings (uppercase, spaces→underscores: "Cash Out" → `CASH_OUT`).
3. Reorders columns to the exact training order and casts dtypes.
4. **Rebuilds the categorical with the full fixed category list** — `pd.Categorical(x, categories=['CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER'])`. This is critical: LightGBM encodes categories by their *category codes*, so a single-row input whose `category` dtype only knows one category would silently produce wrong predictions. The categories (and their order) must match training exactly.

---

## 8. Deployment

### Docker
```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY <app code> ./
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Deliberate details:
- **`libgomp1` (the best deployment gotcha):** LightGBM's compiled library depends on GNU OpenMP for multithreading. `python:3.11-slim` strips it out, so `import lightgbm` crashes with `libgomp.so.1: cannot open shared object file` *only inside the container* — it works fine locally. Diagnosed from container logs and fixed with a one-line apt install, keeping the slim base (vs the much larger full image). `rm -rf /var/lib/apt/lists/*` keeps the layer small.
- **Layer-cache ordering:** `requirements.txt` is copied and installed *before* the app code, so code-only changes don't re-trigger the multi-minute pip install layer.
- **`--server.address=0.0.0.0`:** Streamlit binds to localhost by default, which is unreachable through Docker's port mapping; binding to all interfaces is required in a container.

### Hugging Face Spaces
The app runs live on HF Spaces using the **Docker runtime** — the README's YAML frontmatter (`title`, `emoji`, `sdk`, `app_file`) is HF Spaces configuration metadata ("Huggingface alignment" commit). Spaces builds the image from the repo and serves it.

### Version pinning (`requirements.txt`)
- `scikit-learn==1.6.1` — **pinned for pickle compatibility**: the model artifact must be loaded by a compatible library version, or joblib deserialisation can break or warn.
- `streamlit==1.55.0` — pinned to match the HF frontmatter and because the app uses newer APIs (`st.navigation`, `st.space`, container `horizontal=True`).

### `.streamlit/config.toml` (production polish)
`toolbarMode = "viewer"` and `hideTopBar = true` hide Streamlit's developer chrome from end users; `showErrorDetails = "none"` prevents raw stack traces leaking to visitors; custom theme colours.

---

## 9. Gotchas & War Stories (interview gold)

1. **`libgomp1` missing in slim Docker images** — see §8. The generalisable lesson: Python wheels with compiled extensions have *system-level* shared-library dependencies that minimal base images don't guarantee.

2. **The hour off-by-one between UI and model.** The model uses `hour_of_day` ∈ 0–23; the UI presents 1–24 (more natural for users). `preprocess_input(..., source='app')` subtracts 1 only for app-sourced input, and the demo button *adds* 1 when populating the form from raw samples. Getting this wrong silently shifts every prediction's time feature by an hour — the kind of quiet train/serve skew that never throws an error. (This was the "Prediction page bug" commit.)

3. **`st.cache_data` vs `st.cache_resource`, and the underscore-argument trick** — see §7. Know this cold: `cache_resource` = shared singletons (models, DB connections); `cache_data` = serialisable data (returns copies); `_arg` = exclude from cache key.

4. **Categorical dtype mismatches at inference** — see §7. Single-row inference must reconstruct the full training category set or LightGBM's category codes shift.

5. **Feature order matters.** LightGBM (fitted on a DataFrame) expects columns in training order; `preprocess_input` explicitly reindexes to `correct_order`.

6. **Widget state loss on page navigation** — the two-key threshold pattern with `on_change` callbacks (§7).

7. **Division-by-zero at threshold extremes.** At threshold 1.0 there are no predicted positives, so precision = 0/0. We define it as 1.0 (the limiting value) so the PR curve renders sensibly end-to-end.

8. **SMOTE abandoned for class weights** — the leakage/compute/realism argument in §3.8.

9. **Dropping a "predictive" feature (`step`)** because its signal was a simulation artifact (§3.4) — demonstrates data scepticism.

10. **Startup latency budget:** eager init for cheap things (data, model, PR array), lazy init + spinner for the expensive one (10k-row SHAP). Also the app runs on a 10k-row *sample* of the test set, not the 2.7M-row test file — enough for meaningful dashboard statistics, but responsive.

11. **Colab/Kaggle operational realities:** benchmarking runs auto-checkpointed everything to Google Drive with an execution log (numbered runs, timestamps, fold counts, data proportions) because Colab sessions die; GPU training (Tesla T4) for the 300-fit random search; a `keep_session_active()` busy-loop hack to stop Colab idling out mid-search. Shows you've felt real experiment-management pain — and can motivate why tools like MLflow/W&B exist.

---

## 10. Concepts You Should Be Ready to Explain

| Concept | Where it appears | One-liner |
|---|---|---|
| Precision vs recall | Everywhere | Of flagged, how many are fraud (precision) vs of fraud, how many we caught (recall) |
| PR-AUC vs ROC-AUC | Model selection | ROC-AUC inflated under heavy imbalance (FPR denominator is huge); PR-AUC tracks positive-class performance |
| Class imbalance strategies | Benchmarking | Class weights (loss reweighting) vs SMOTE (data-level oversampling); we chose weights |
| Stratified K-fold CV | Benchmarking | Preserves the 0.13% fraud rate per fold; plain K-fold could yield folds with almost no fraud |
| Pipeline + ColumnTransformer | Benchmarking | Preprocessing fitted inside each fold → no leakage |
| Gradient boosting (LightGBM) | Final model | Sequential trees each fitting the previous ensemble's errors; LightGBM = histogram binning + leaf-wise growth → speed |
| LightGBM vs XGBoost vs RF | Model selection | Boosting (sequential, bias-reduction) vs bagging (parallel, variance-reduction); LightGBM leaf-wise vs XGBoost level-wise |
| Randomized search + loguniform | Tuning | Better budget efficiency than grid; log-scale for multiplicative params like learning rate |
| Low LR + many trees | Final params (0.011, 1321) | Smaller steps + more of them = smoother convergence, better generalisation |
| Cyclical (sin/cos) encoding | Features | Maps hours onto a circle so 23:00 and 00:00 are neighbours |
| SHAP / TreeExplainer | Explainability | Shapley values = each feature's fair contribution to a prediction; TreeExplainer computes them exactly & fast for trees; global = mean(|SHAP|) |
| Decision threshold as product knob | Threshold page | Model outputs probabilities; where to cut is a business decision → we made it a UI control |
| Train/validation/test discipline | Splits | Tune on validation, touch test once; 15/42.5/42.5 stratified |
| Weight of Evidence | EDA (explored) | log(P(category\|good)/P(category\|bad)) — credit-scoring categorical encoding |
| Pickle/version compatibility | requirements.txt | Serialized models must be loaded with compatible library versions |

---

## 11. Likely Interview Questions (with answers)

**Q: Why LightGBM?**
Best PR-AUC (0.929) in a 7-model stratified-CV benchmark, best balance of precision and recall, *and* ~7× faster than RandomForest and ~35× faster than KNN end-to-end. It handles categoricals natively (no one-hot blow-up) and produces a small artifact (4.7MB) that's fast at inference — which matters for a real-time fraud check. Runner-ups: RF had the best precision but 3× lower recall; XGBoost had the best recall but 25% precision (too many false alarms).

**Q: How did you handle class imbalance?**
Three levers: (1) stratified splitting/CV so every fold sees the true fraud rate; (2) class weighting — `scale_pos_weight = neg/pos` for boosting, `class_weight='balanced'` for sklearn models — which upweights fraud in the loss instead of resampling data; (3) evaluation via PR-AUC rather than accuracy or ROC-AUC. We prototyped SMOTE but rejected it (compute at 4M+ rows, synthetic-sample realism, fold-leakage complexity).

**Q: 84% recall means you miss 16% of fraud. Is that acceptable?**
It's one operating point, not the model's limit. The PR curve shows what recall we can buy at what precision cost, and the app exposes the threshold as a control precisely so the business can choose — e.g. drop the threshold to catch more fraud and staff up manual review for the extra false positives. The right threshold depends on the cost ratio of a missed fraud vs a false alarm, which is a business input, not an ML output.

**Q: How would you productionise this beyond a demo?**
Split serving from UI (FastAPI model service + this dashboard as a client); batch/stream scoring rather than CSV loads; model registry & versioning (MLflow) instead of a pickled file in git; monitoring for data/label drift and threshold-level alert-volume tracking; retraining pipeline; feature store for account-level aggregates; A/B or shadow deployment for model updates.

**Q: What features would you add with real data?**
Account-level history (transaction velocity, deviation from account's normal amounts), graph features on the origin→destination network (fraud rings), balance-consistency features (e.g. `oldbalanceOrg - amount - newbalanceOrig` mismatch, which is famously predictive on PaySim-style data), device/geo signals, merchant category.

**Q: Limitations you're aware of?**
Synthetic data — patterns (like fraud living almost entirely in TRANSFER/CASH_OUT) are cleaner than reality, and the fraud rate/mechanisms are simulated; no temporal validation split (real fraud drifts, so I'd use time-based splits in production); the app scores a 10k sample, not a live stream; SHAP explanation surfaces only the top contributor (a full waterfall would be richer); no CI/CD or tests around the model contract.

**Q: Explain SHAP to a non-technical stakeholder.**
For any single flagged transaction, SHAP fairly divides the "blame" for the fraud score among the input features — like splitting a bill based on what each person ordered. So we can say "this was flagged mainly because the entire origin balance was emptied in a TRANSFER at 3 AM" instead of "the black box said so." That matters for analyst trust and for regulatory explainability requirements.

**Q: Why Streamlit and not Flask/FastAPI + React?**
Right tool for the deliverable: an interactive ML demo where the audience is exploring model behaviour. Streamlit gave us multipage routing, widgets, and Plotly integration with zero frontend code, letting us spend the effort on modelling and the caching/state architecture. The utils layer is UI-agnostic, so swapping the front end for an API later wouldn't touch the model code.

---

## 12. Numbers to Memorise

- **6.36M** rows; **~0.13%** fraud rate (~1 in 770)
- **9 features**; 5 transaction types
- Splits: **15% / 42.5% / 42.5%** (benchmark / validation / test), stratified
- **7 models** benchmarked, **5-fold** stratified CV; primary metric **PR-AUC**
- LightGBM CV PR-AUC: **0.929**; tuning: **100 candidates × 3 folds** randomized search
- Final params: **lr ≈ 0.011, 1321 trees**
- Test set (~2.7M rows): **precision 96.4%, recall 83.6%, F1 89.5%**; confusion matrix **107 FP / 571 FN / 2,902 TP**
- App sample: **10k rows**; default threshold **50%**; PR curve precomputed at **101 thresholds**
- Model artifact: **4.7MB** joblib pickle; Docker base **python:3.11-slim**; port **8501**
