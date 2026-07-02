# Healthcare Provider Fraud Detection Pipeline

A four-layer fraud detection system built on the [Kaggle Healthcare Provider Fraud Detection dataset](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis).
Built as part of the Cotiviti *Temporary Intern — Agentic AI & Neural-Symbolic Systems (Healthcare)* assessment.

---

## Running the Web App

The project includes a local Streamlit application that lets you investigate any provider interactively. The agent autonomously calls ML models and billing-rule checks, then streams a structured investigation memo.

### 1. Install Python dependencies

```bash
pip install -r fraud_agent/requirements.txt
```

### 2. Install Ollama and pull the model

```bash
# macOS
brew install ollama

# Then pull the model (one-time, ~5 GB)
ollama pull qwen3:8b
```

### 3. Start the Ollama daemon

```bash
ollama serve
```

Leave this running in a separate terminal tab.

### 4. Launch the app

Run this from the project root (the folder containing the CSV files):

```bash
streamlit run fraud_agent/app.py
```

Open **http://localhost:8501** in your browser.

**First launch takes 60–90 seconds** while the models train on the full claims data. After that, provider lookups are instant.

### Demo providers

The app pre-loads a dropdown of the six highest-volume known-fraud and six highest-volume known non-fraud providers for easy screen-recording. You can also type any Provider ID manually.

If you enter a provider ID that isn't in the training data, the app switches to **single-claim evaluation mode** — described at the bottom of this document.

---

## Project Overview

The pipeline processes three CSV files (`Train_IP.csv`, `Train_OP.csv`, `Train_Beneficiary.csv`) merged with fraud ground-truth labels (`Train.csv`). It builds four successive detection layers, each asking a different question about the same underlying data.

| Layer | File | Method | Signal |
|---|---|---|---|
| 1 | `provider_analysis.Rmd`, `provider-level.ipynb` | Ridge logistic regression + Isolation Forest | Provider-level fraud probability |
| 2 | `claim_analysis.ipynb` | Z-score ensemble across 36 groupings | Per-claim anomaly fraction |
| 3 | `time_series_analysis.Rmd` | Rolling z-score spike detector + OLS ramp | Temporal billing patterns |
| 4 | `fraud_agent/` | Agentic LLM + symbolic billing rules | Investigation memo |

---

## Layer 1 — Provider-Level Machine Learning

**Files:** `provider-level.ipynb` (feature engineering + Isolation Forest), `provider_analysis.Rmd` (ridge logistic regression)

### Feature engineering

Inpatient and outpatient claims are merged and joined to the beneficiary table to form a 558,211-row claims table. This is then aggregated to a provider-level dataset with 27 engineered features, including:

- Billing volume: total claims, unique patients, claims per patient, total and average reimbursement
- Inpatient intensity: % inpatient claims, average length of stay, total stay days
- Physician diversity: number of unique attending/operating/other physicians, claims-to-physician ratio
- Coding complexity: unique diagnosis codes, unique procedure codes, average codes per claim
- Patient demographics: average patient age, % deceased patients, average annual IP/OP reimbursement

A correlation check identified four features with |r| > 0.95 against others (`total_deductible_paid`, `avg_deductible_paid`, `num_unique_attending_physicians`, `num_unique_procedure_codes`) and dropped them, leaving 23 features for modelling.

### Ridge logistic regression

A ridge-penalised logistic regression model (`glmnet`, α = 0) is trained on the 23-feature provider set with lambda selected by validation-set AUC. Scaling is applied before fitting to ensure regularisation is applied uniformly. The model produces a **fraud probability per provider** (0–1).

### Isolation Forest

An Isolation Forest is trained on the same scaled features with hyperparameters selected by grid search (n_estimators, max_features, max_samples, contamination) evaluated on validation AUC. It produces an **anomaly score per provider** where higher values indicate a provider is more unusual relative to the population.

Both models are re-trained from scratch at app startup and produce a combined Layer 1 signal: the ridge probability reflects the learned fraud pattern while the Isolation Forest is fully unsupervised and can detect novel anomaly types not captured by the labelled training distribution.

---

## Layer 2 — Claim-Level Z-Score Ensemble

**File:** `claim_analysis.ipynb`

Instead of flagging providers directly, Layer 2 asks: *is this individual claim unusual relative to comparable claims?*

### Method

For each of 36 **grouping columns** (attending physician, diagnosis codes 1–10, procedure codes 1–6, claim type, state, county, chronic condition flags, etc.), we compute the within-group z-score of `log1p(InscClaimAmtReimbursed)` — the log-transformed reimbursement amount. A claim is flagged in a grouping if |z| > 3.

To avoid noisy z-scores from tiny groups, an **adaptive minimum group size** is computed per column: we walk the group-size distribution until 80% of claims are covered by groups above the cutoff, capped at 1,000 (with a fallback of 30). This means a common diagnosis code like `4280` (heart failure) gets a tighter minimum than a rare secondary procedure code.

Each claim ends up with a **risk fraction** — the share of eligible groupings in which it was flagged. Claims with a risk fraction above 0.25 are labelled anomalous.

### Results

| | Non-Fraud Providers | Fraud Providers |
|---|---|---|
| % anomalous claims | 0.13% | 0.38% |
| Fraud/non-fraud ratio | — | **3.05×** |
| χ² p-value | — | 3.7 × 10⁻⁸⁷ |

The ensemble flags roughly 1,250 of 558,000 claims. Despite being entirely unsupervised (no labels used), claims from fraud-labelled providers are flagged at three times the rate of claims from clean providers.

---

## Layer 3 — Temporal Analysis

**File:** `time_series_analysis.Rmd`

This layer asks whether fraudulent providers show unusual *temporal* billing patterns — spikes or ramps in their month-by-month reimbursement totals.

### Spike detector

A rolling z-score is computed against a 3-month lagged baseline. After recalibration (requiring ≥ 3 anomalous months to flag a provider, which brings the overall flag rate to ~2.7%), the spike detector shows **no discrimination**: the fraud/non-fraud flag-rate ratio is 0.64× (backwards) with p = 0.24.

### Ramp detector (honest null result)

An OLS slope is fitted through each provider's monthly reimbursement series (month index 1, 2, … as x). The raw-dollar slope z-score flagged 7.06× more fraud providers than non-fraud providers (p < 2×10⁻³⁶), which appeared to be a strong signal.

Two robustness checks revealed this was a **scale artifact**:

| Method | Fraud flag rate | Non-fraud flag rate | Ratio |
|---|---|---|---|
| Raw-dollar OLS slope | 11.6% | 1.6% | 7.06× |
| Log-scale slope | 0.8% | 2.4% | 0.34× |
| Mean-normalised slope | 0.0% | 2.9% | 0.00× |
| Edge-trimmed (log, full months only) | 0.8% | 2.5% | 0.32× |

Once the dollar scale is removed, the signal reverses: non-fraud providers are *more* likely to be flagged. The raw-dollar ramp was detecting "is this a high-billing provider?" — and high-billing providers happen to be over-represented in the fraud set. The temporal layer adds no independent signal beyond what Layer 1 already captures from total reimbursement features. This is reported honestly in the analysis rather than discarded.

---

## Layer 4 — Agentic Neural-Symbolic Investigation

**Directory:** `fraud_agent/`

The fourth layer adds an agentic reasoning loop that combines the learned ML scores (neural) with deterministic billing-policy rules (symbolic) to produce a human-readable investigation memo.

### Symbolic rule engine (`src/symbolic.py`)

Five hard billing-policy rules are implemented as deterministic Python functions over a provider's claims DataFrame. Each is illustrative but domain-grounded, with a comment naming the real CMS policy that would back it in production (the kind of work Cotiviti does in post-payment audits):

| Rule | What it checks | CMS policy analogue |
|---|---|---|
| 1 | Procedure code billed with no diagnosis code | NCCI edit tables |
| 2 | Duplicate procedure codes on the same claim | NCCI unbundling edits |
| 3 | Reimbursement exceeds ceiling for claim type (IP > $150 K, OP > $15 K) | DRG / APC fee schedules |
| 4 | Temporal impossibilities (negative LOS; overlapping inpatient stays; physician attending > 15 claims/day) | Medicare benefit policy / PECOS |
| 5 | Deductible paid exceeds reimbursement amount | CMS MLN deductible structure |

### Agentic loop (`src/agent.py`)

A local `qwen3:8b` model (via Ollama's OpenAI-compatible API) runs a tool-calling loop with a hard cap of six steps. The model autonomously decides which tools to call and in what order — it is not scripted. Available tools:

- `get_claims_summary` — billing portfolio overview
- `ml_risk_score` — Layer 1 + Layer 2 scores
- `check_billing_rules` — all five symbolic rule results

After gathering evidence from both neural scores and symbolic violations, the model streams a structured investigation memo that fuses both sources of evidence. The LLM client is isolated behind a single interface, so switching from Ollama to any OpenAI-compatible API (Groq, Gemini via OpenRouter, etc.) requires changing three lines.

### Single-claim mode (cold-start)

If a provider ID has no training history, the app automatically offers single-claim evaluation. The user fills out one claim form (pre-populated from a real training example). Layer 1 is shown as N/A with an explicit explanation. Layer 2 runs by comparing the claim's reimbursement against the reference group distributions pre-computed from training data. All five symbolic rules run on the single claim. The agent is scoped to only claim-level tools and produces a claim-level assessment memo rather than a provider fraud determination.

This mode demonstrates the cold-start strength of the neural-symbolic architecture: when learned models cannot operate (no provider history), hard policy rules still provide immediate, explainable fraud-detection capability.

---

## Repository Structure

```
├── provider-level.ipynb       # Layer 1 feature engineering + Isolation Forest
├── provider_analysis.Rmd      # Layer 1 ridge logistic regression (R)
├── claim_analysis.ipynb       # Layer 2 claim-level z-score ensemble
├── time_series_analysis.Rmd   # Layer 3 temporal analysis (honest null result)
└── fraud_agent/
    ├── app.py                 # Streamlit web app
    ├── requirements.txt
    └── src/
        ├── data.py            # CSV loading + feature engineering
        ├── neural.py          # Layer 1 + 2 models, train at startup
        ├── symbolic.py        # 5 billing rules
        ├── tools.py           # Agent-callable tool wrappers
        └── agent.py           # Ollama tool-calling loop
```
