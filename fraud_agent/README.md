# FraudGuard — Neural-Symbolic Provider Investigation

Local, free, fully agentic fraud-detection demo for the Cotiviti intern assessment.

## What it does

A four-layer healthcare provider fraud detection system:

| Layer | What | How |
|---|---|---|
| 1a | Ridge logistic regression | Provider-level fraud probability |
| 1b | Isolation Forest | Provider-level anomaly score |
| 2 | Z-score ensemble | Per-claim risk fraction across 36 groupings |
| 3 | Symbolic rule engine | 5 hard billing-policy checks |
| Agent | Agentic loop | Local LLM autonomously calls tools, streams a memo |

## Setup

### 1. Install Python dependencies

```bash
cd fraud_agent/
pip install -r requirements.txt
```

### 2. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. Pull the model

```bash
ollama pull qwen3:8b
```

`qwen3:8b` (~5.2 GB) is already pulled if you see it in `ollama list`.

### 4. Start the Ollama daemon

```bash
ollama serve
```

Leave this running in a terminal tab. The app checks for it at startup.

### 5. Run the app

From the `Cotiviti/` directory (where the CSV files live):

```bash
streamlit run fraud_agent/app.py
```

Open http://localhost:8501 in your browser.

---

## First run

**Model training takes ~60–90 seconds on first launch** (ridge regression, isolation
forest grid search, z-score ensemble over 558 K claims). Streamlit caches the result
so subsequent provider lookups are instant.

---

## Demo provider IDs

The app pre-populates a dropdown of the 6 highest-volume fraud and 6 highest-volume
non-fraud providers from the training labels. Pick any to demo:

- **Known fraud, high volume** — providers with many claims and `PotentialFraud=Yes`
- **Known non-fraud** — for comparison; the agent should produce a LOW risk memo

---

## Swapping the LLM

To point at a different backend (Gemini via OpenRouter, Groq, etc.), edit three
constants at the top of `src/agent.py`:

```python
LLM_BASE_URL = "https://openrouter.ai/api/v1"   # or Groq, etc.
LLM_API_KEY  = "your-api-key"
LLM_MODEL    = "google/gemini-flash-1.5"
```

No other code changes required — everything else uses the OpenAI SDK interface.

---

## Project structure

```
fraud_agent/
  src/
    data.py       # CSV loading, claims merge, provider feature engineering
    neural.py     # Layer 1 (ridge + IF) + Layer 2 (z-score ensemble), train on startup
    symbolic.py   # 5 billing rules — deterministic, fully explainable
    tools.py      # JSON-serializable wrappers exposed to the agent
    agent.py      # Ollama tool-calling loop (max 6 steps, streaming, swappable LLM)
  app.py          # Streamlit UI
  requirements.txt
  README.md
```

## Notes on symbolic rules

All 5 rules are illustrative but domain-grounded. Each rule comment names the real
CMS policy that would back it in production (NCCI edits, DRG fee schedules, PECOS,
MLN deductible tables) — mirroring Cotiviti's actual post-payment audit business.
