"""
FraudGuard — Provider Investigation UI
Streamlit app: pick a provider, hit Investigate, watch the agent work live.
Supports an optional single-claim evaluation mode for unknown providers.
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from src.agent import LLM_MODEL, check_ollama, run_claim_investigation, run_investigation
from src.data import DIAG_COLS, PROC_COLS, get_provider_claims, load_claims, load_labels
from src.neural import score_provider, score_single_claim
from src.symbolic import run_all_rules
from src.tools import _dict_to_claim_df, check_single_claim_rules

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudGuard · Provider Investigation",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Cached startup work ───────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training models on claims data — ~60 s on first run…")
def _load_models():
    from src.neural import get_models
    return get_models()


@st.cache_data(show_spinner=False)
def _example_providers():
    labels = load_labels()
    claims = load_claims()
    counts = claims.groupby("Provider").size().reset_index(name="n_claims")
    merged = labels.merge(counts, on="Provider", how="left").fillna(0)
    fraud = merged[merged["PotentialFraud"] == "Yes"].nlargest(6, "n_claims")
    legit = merged[merged["PotentialFraud"] == "No"].nlargest(6, "n_claims")
    return fraud, legit


@st.cache_data(show_spinner=False)
def _get_example_claim() -> dict:
    """Return a pre-filled example claim dict (Python native types) for the form."""
    claims = load_claims()
    labels = load_labels()
    fraud_ids = set(labels[labels["PotentialFraud"] == "Yes"]["Provider"])

    mask = (
        claims["Provider"].isin(fraud_ids)
        & (claims["claim_type"] == 1)
        & claims["AdmissionDt"].notna()
        & claims["DischargeDt"].notna()
        & claims["ClmProcedureCode_1"].notna()
        & claims["ClmDiagnosisCode_1"].notna()
        & (claims["InscClaimAmtReimbursed"] >= 10_000)
    )
    cands = claims[mask]
    if cands.empty:
        cands = claims[claims["claim_type"] == 1].dropna(subset=["AdmissionDt"])

    row = cands.sample(1, random_state=42).iloc[0]

    def _clean(v):
        if isinstance(v, float) and np.isnan(v):
            return None
        if isinstance(v, (pd.Timestamp, datetime)):
            return v.date()
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            f = float(v)
            return None if np.isnan(f) else f
        return v

    return {k: _clean(v) for k, v in row.items()}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("FraudGuard")
    st.caption("Neural-Symbolic Provider Investigation")
    st.markdown("---")
    st.markdown(f"""
**Detection pipeline**

**Layer 1 — Provider ML**
Ridge LR (fraud prob) + Isolation Forest (anomaly score)

**Layer 2 — Claim Ensemble**
Z-score across 36 grouping dimensions → per-claim risk fraction

**Layer 3 — Billing Rules**
5 hard symbolic policy checks (deterministic)

**Agent** · `{LLM_MODEL}` via Ollama
Calls tools autonomously, streams an investigation memo
""")
    st.markdown("---")
    ok, msg = check_ollama()
    if ok:
        st.success(f"Ollama ready · `{LLM_MODEL}`")
    else:
        st.error(msg)
        st.info("Start Ollama, then reload this page.")

# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in [
    ("investigated_provider", None),
    ("claim_mode_active", False),
    ("submitted_claim", None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Load models at startup ────────────────────────────────────────────────────
_load_models()
fraud_provs, legit_provs = _example_providers()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("Provider Fraud Investigation")
st.caption(
    "Select a provider and click **Investigate**. "
    "The agent autonomously calls ML models and billing-rule checks, "
    "then streams a structured memo."
)
st.divider()

# ── Provider selection ────────────────────────────────────────────────────────
col_sel, col_go = st.columns([4, 1])

with col_sel:
    fraud_opts = [
        f"{r.Provider}  —  Known Fraud  ({int(r.n_claims):,} claims)"
        for _, r in fraud_provs.iterrows()
    ]
    legit_opts = [
        f"{r.Provider}  —  Non-Fraud  ({int(r.n_claims):,} claims)"
        for _, r in legit_provs.iterrows()
    ]
    choice = st.selectbox(
        "Select a provider",
        options=fraud_opts + legit_opts,
        index=0,
        help="Top 6 fraud + top 6 non-fraud providers by claim volume.",
    )
    provider_id = choice.split()[0]

    manual = st.text_input(
        "…or type any Provider ID",
        placeholder="e.g. PRV55912",
        label_visibility="collapsed",
    )
    if manual.strip():
        provider_id = manual.strip()

with col_go:
    st.write("")  # vertical spacer
    st.write("")
    ollama_ok, _ = check_ollama()
    investigate = st.button(
        "Investigate",
        type="primary",
        disabled=not ollama_ok,
        use_container_width=True,
    )

if investigate:
    st.session_state.investigated_provider = provider_id
    st.session_state.claim_mode_active = False
    st.session_state.submitted_claim = None

if not ollama_ok:
    st.warning("Ollama is not running — start it with `ollama serve`, then reload.")

# ── Investigation results ─────────────────────────────────────────────────────
current_provider = st.session_state.investigated_provider
if not current_provider:
    st.stop()

st.divider()

# ── Compute scores ────────────────────────────────────────────────────────────
with st.spinner("Computing scores…"):
    scores = score_provider(current_provider)

# ── Unknown provider → single-claim mode ────────────────────────────────────
if "error" in scores:
    st.subheader(f"Provider not found: {current_provider}")
    st.info(
        f"**{current_provider}** has no billing history in the training data. "
        "**Layer 1** (provider-level ridge regression + isolation forest) requires "
        "aggregated historical data and **cannot run** here."
    )
    st.markdown("""
> **Cold-start strength of the neural-symbolic approach:**
> Even without provider history, the **symbolic rule engine** provides immediate
> fraud-detection capability — it is deterministic, claim-level, and requires
> zero prior data. This is exactly the scenario where hard policy rules outperform
> purely learned models.
""")

    if not st.session_state.claim_mode_active:
        if st.button("Evaluate a single claim from this provider →", type="secondary"):
            st.session_state.claim_mode_active = True
            st.rerun()

    if not st.session_state.claim_mode_active:
        st.stop()

    # ── Single-claim form ─────────────────────────────────────────────────────
    ex = _get_example_claim()   # pre-filled example from a fraud provider

    st.subheader("Single-Claim Evaluation")
    st.caption(
        "Fields are pre-filled with a real training example — edit any value. "
        "Only Layer 2 (z-score) and Layer 3 (billing rules) will run. "
        "Layer 1 is shown as N/A."
    )

    with st.form("claim_eval_form", clear_on_submit=False):
        fc1, fc2 = st.columns(2)
        with fc1:
            claim_type_str = st.radio(
                "Claim type",
                ["Inpatient", "Outpatient"],
                index=0 if ex.get("claim_type", 1) == 1 else 1,
                horizontal=True,
            )
            claim_type = 1 if claim_type_str == "Inpatient" else 0

            reimb = st.number_input(
                "Reimbursement ($)",
                min_value=0.0, step=100.0,
                value=float(ex.get("InscClaimAmtReimbursed") or 10000),
            )
            deduct = st.number_input(
                "Deductible paid ($)",
                min_value=0.0, step=10.0,
                value=float(ex.get("DeductibleAmtPaid") or 0),
            )

        with fc2:
            claim_start = st.date_input(
                "Claim start date",
                value=ex.get("ClaimStartDt") or date(2009, 6, 1),
            )
            if claim_type == 1:
                admit_dt = st.date_input(
                    "Admission date",
                    value=ex.get("AdmissionDt") or date(2009, 6, 1),
                )
                disch_dt = st.date_input(
                    "Discharge date",
                    value=ex.get("DischargeDt") or date(2009, 6, 5),
                )
            else:
                admit_dt = disch_dt = None

        st.markdown("**Codes** *(primary diagnosis required; others optional)*")
        cc1, cc2, cc3, cc4 = st.columns(4)
        diag1 = cc1.text_input(
            "Diagnosis 1 *",
            value=str(ex.get("ClmDiagnosisCode_1") or ""),
        )
        diag2 = cc2.text_input(
            "Diagnosis 2",
            value=str(ex.get("ClmDiagnosisCode_2") or ""),
        )
        proc1_raw = ex.get("ClmProcedureCode_1")
        proc1_val = "" if proc1_raw is None else str(int(proc1_raw)) if isinstance(proc1_raw, float) else str(proc1_raw)
        proc1 = cc3.text_input("Procedure 1", value=proc1_val)
        proc2_raw = ex.get("ClmProcedureCode_2")
        proc2_val = "" if proc2_raw is None else str(int(proc2_raw)) if isinstance(proc2_raw, float) else str(proc2_raw)
        proc2 = cc4.text_input("Procedure 2", value=proc2_val)

        st.markdown("**Context** *(affects Layer 2 peer-group matching)*")
        cx1, cx2 = st.columns(2)
        attending = cx1.text_input(
            "Attending physician",
            value=str(ex.get("AttendingPhysician") or ""),
            placeholder="PHY390922",
        )
        state = cx2.text_input(
            "State code",
            value=str(ex.get("State") or ""),
            max_chars=5,
            placeholder="5",
        )

        submitted = st.form_submit_button("Evaluate Claim", type="primary")

    # ── Validate and store on submit ──────────────────────────────────────────
    if submitted:
        errors = []
        if not diag1.strip():
            errors.append("Primary diagnosis code is required.")
        if errors:
            for e in errors:
                st.error(e)
            st.stop()

        def _or_none(v):
            return v.strip() if isinstance(v, str) and v.strip() else None

        claim_dict: dict = {
            "ClaimID":               "MANUAL_CLAIM_001",
            "BeneID":                "BENE_MANUAL",
            "Provider":              current_provider,
            "claim_type":            claim_type,
            "InscClaimAmtReimbursed": float(reimb),
            "DeductibleAmtPaid":     float(deduct),
            "ClaimStartDt":          pd.Timestamp(claim_start),
            "ClaimEndDt":            pd.Timestamp(disch_dt or claim_start),
            "AdmissionDt":           pd.Timestamp(admit_dt) if admit_dt else None,
            "DischargeDt":           pd.Timestamp(disch_dt) if disch_dt else None,
            "AttendingPhysician":    _or_none(attending),
            "OperatingPhysician":    None,
            "OtherPhysician":        None,
            "ClmDiagnosisCode_1":    _or_none(diag1),
            "ClmDiagnosisCode_2":    _or_none(diag2),
            # fill remaining diag/proc slots from the example claim for L2 grouping depth
            **{c: ex.get(c) for c in DIAG_COLS[2:]},
            "ClmProcedureCode_1":    float(proc1) if proc1.strip().replace(".", "").isdigit() else _or_none(proc1),
            "ClmProcedureCode_2":    float(proc2) if proc2.strip().replace(".", "").isdigit() else _or_none(proc2),
            **{c: ex.get(c) for c in PROC_COLS[2:]},
            "State":                 _or_none(state),
            # pull remaining beneficiary/chronic-condition fields from example for L2
            **{c: ex.get(c) for c in [
                "County", "RenalDiseaseIndicator", "DiagnosisGroupCode",
                "ClmAdmitDiagnosisCode",
                "ChronicCond_Alzheimer", "ChronicCond_Heartfailure",
                "ChronicCond_KidneyDisease", "ChronicCond_Cancer",
                "ChronicCond_ObstrPulmonary", "ChronicCond_Depression",
                "ChronicCond_Diabetes", "ChronicCond_IschemicHeart",
                "ChronicCond_Osteoporasis", "ChronicCond_rheumatoidarthritis",
                "ChronicCond_stroke",
            ]},
        }
        st.session_state.submitted_claim = claim_dict

    # ── Single-claim results ──────────────────────────────────────────────────
    if st.session_state.submitted_claim is None:
        st.stop()

    claim_dict = st.session_state.submitted_claim
    st.divider()
    st.subheader("Single-Claim Results")

    # Layer 1 — explicitly N/A
    # Layer 2 — run now
    # Layer 3 — run now
    with st.spinner("Running Layer 2 + billing rules…"):
        l2_result   = score_single_claim(claim_dict)
        rule_result = check_single_claim_rules(claim_dict)

    # Metric row
    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Layer 1 — Ridge / Isolation Forest",
        "N/A",
        help="Requires aggregated provider history. Not available for unknown providers.",
    )
    l2_frac = l2_result.get("claim_risk_fraction", 0)
    m2.metric(
        "Layer 2 — Claim Risk Fraction",
        f"{l2_frac:.3f}",
        delta="ANOMALOUS" if l2_result.get("is_claim_anomalous") else "Normal",
        delta_color="inverse" if l2_result.get("is_claim_anomalous") else "normal",
    )
    triggered = rule_result.get("summary", {}).get("rules_triggered", 0)
    m3.metric(
        "Layer 3 — Rules Triggered",
        str(triggered),
        delta=f"out of 5 rules" if triggered > 0 else "No violations",
        delta_color="inverse" if triggered > 0 else "normal",
    )

    # Layer 2 detail
    with st.expander("Layer 2 — Z-Score Anomaly Detail", expanded=True):
        st.write(l2_result.get("interpretation", ""))
        n_elig = l2_result.get("n_groupings_eligible", 0)
        n_flag = l2_result.get("n_groupings_flagged", 0)
        st.caption(
            f"{n_elig} peer-group comparisons ran · {n_flag} flagged |z| > 3 · "
            f"risk fraction = {l2_frac:.1%}"
        )
        if l2_result.get("flagged_groupings"):
            st.markdown("**Flagged groupings:**")
            for fg in l2_result["flagged_groupings"]:
                st.caption(
                    f"  • `{fg['grouping']}` = {fg['group_value']} → z = {fg['z_score']}"
                )

    # Billing rules
    with st.expander("Layer 3 — Billing Rule Violations", expanded=True):
        if triggered == 0:
            st.success("No billing rule violations detected.")
        else:
            st.warning(f"{triggered} rule(s) triggered")
        for key in ["rule_1", "rule_2", "rule_3", "rule_4", "rule_5"]:
            r = rule_result.get(key)
            if r is None:
                continue
            vc = r.get("violation_count", 0)
            icon = "🔴" if vc > 0 else "🟢"
            st.markdown(f"{icon} **{r['rule']}** — {vc} violation(s)")
            if vc > 0:
                st.caption(r.get("explanation", ""))
                if r.get("findings"):
                    for f in r["findings"]:
                        st.caption(f"  • {f}")

    # Agent memo
    st.subheader("Agent Assessment Memo")
    st.caption(
        "The agent runs only claim-level tools (Layer 2 + rules). "
        "Layer 1 is explicitly omitted."
    )
    tool_log_exp = st.expander("Agent Tool-Call Log", expanded=True)
    with tool_log_exp:
        tool_log_ph = st.empty()
    memo_ph = st.empty()

    tool_log_md = ""
    memo_text = ""
    step_label = st.empty()

    for event in run_claim_investigation(claim_dict):
        etype = event["type"]
        if etype == "step":
            step_label.caption(f"Step {event['step']}: {event['action']}")
        elif etype == "tool_call":
            tool_log_md += (
                f"**`{event['tool']}`**\n\n"
                f"> {event['result'][:300]}{'…' if len(event['result']) > 300 else ''}\n\n---\n\n"
            )
            tool_log_ph.markdown(tool_log_md)
        elif etype == "memo_chunk":
            memo_text += event["text"]
            memo_ph.markdown(memo_text + "▌")
        elif etype == "error":
            st.error(f"Agent error: {event['message']}")
            break
        elif etype == "done":
            memo_ph.markdown(memo_text)
            step_label.caption("Assessment complete.")
            break

    st.stop()   # don't fall through to the provider results block below


# ═══════════════════════════════════════════════════════════════════════════════
# Known-provider investigation (normal flow)
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader(f"Investigation: {current_provider}")

with st.spinner("Loading claims and rule checks…"):
    claims_df   = get_provider_claims(current_provider)
    rule_result = run_all_rules(claims_df) if not claims_df.empty else {}

# ── Metric row ────────────────────────────────────────────────────────────────
ridge_p = scores["layer1_ridge_fraud_probability"]
iso_s   = scores["layer1_isolation_forest_score"]
l2_frac = scores["layer2_mean_claim_risk_fraction"]
l2_pct  = scores["layer2_pct_anomalous_claims"]

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    "Ridge Fraud Prob (L1)",
    f"{ridge_p:.1%}",
    delta="HIGH" if ridge_p > 0.6 else "MODERATE" if ridge_p > 0.35 else "LOW",
    delta_color="inverse" if ridge_p > 0.6 else "normal",
)
m2.metric(
    "Isolation Forest (L1)",
    f"{iso_s:.3f}",
    help="Higher = more anomalous vs all providers.",
)
m3.metric(
    "Mean Claim Risk Fraction (L2)",
    f"{l2_frac:.3f}",
    help="Average z-score ensemble risk across all claims (0–1).",
)
m4.metric(
    "Anomalous Claims (L2)",
    f"{l2_pct:.1%}",
    help="% of claims with risk fraction > 0.25.",
)

# ── Claims portfolio ──────────────────────────────────────────────────────────
with st.expander("Claims Portfolio", expanded=False):
    if claims_df.empty:
        st.warning("No claims found.")
    else:
        ip = claims_df[claims_df["claim_type"] == 1]
        op = claims_df[claims_df["claim_type"] == 0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total claims", f"{len(claims_df):,}")
        c2.metric("Inpatient", f"{len(ip):,}")
        c3.metric("Outpatient", f"{len(op):,}")
        c4.metric("Unique patients", f"{claims_df['BeneID'].nunique():,}")
        reimb = claims_df["InscClaimAmtReimbursed"]
        r1, r2, r3 = st.columns(3)
        r1.metric("Total reimbursed", f"${reimb.sum():,.0f}")
        r2.metric("Mean per claim", f"${reimb.mean():,.0f}")
        r3.metric("Max single claim", f"${reimb.max():,.0f}")

# ── Billing rule violations ───────────────────────────────────────────────────
with st.expander("Billing Rule Violations", expanded=True):
    if not rule_result:
        st.info("No claims data.")
    else:
        summary = rule_result.get("summary", {})
        triggered = summary.get("rules_triggered", 0)
        total_viol = summary.get("total_violations", 0)
        if triggered == 0:
            st.success("No billing rule violations detected.")
        else:
            st.warning(f"{triggered} rule(s) triggered · {total_viol} total violation(s)")
        for key in ["rule_1", "rule_2", "rule_3", "rule_4", "rule_5"]:
            r = rule_result.get(key)
            if r is None:
                continue
            vc = r.get("violation_count", 0)
            icon = "🔴" if vc > 0 else "🟢"
            st.markdown(f"{icon} **{r['rule']}** — {vc} violation(s)")
            if vc > 0:
                st.caption(r.get("explanation", ""))
                if r.get("findings"):
                    for f in r["findings"]:
                        st.caption(f"  • {f}")
                if r.get("claim_ids"):
                    st.caption(
                        "Sample claim IDs: "
                        + ", ".join(str(x) for x in r["claim_ids"][:5])
                    )

# ── Agent memo ────────────────────────────────────────────────────────────────
st.subheader("Agent Investigation Memo")
st.caption(
    f"The agent ({LLM_MODEL}) calls tools autonomously. "
    "Expect 30–90 s on first tool call."
)

tool_log_exp = st.expander("Agent Tool-Call Log", expanded=True)
with tool_log_exp:
    tool_log_ph = st.empty()
memo_ph = st.empty()

tool_log_md = ""
memo_text = ""
step_label = st.empty()

for event in run_investigation(current_provider):
    etype = event["type"]
    if etype == "step":
        step_label.caption(f"Step {event['step']}: {event['action']}")
    elif etype == "tool_call":
        tool = event["tool"]
        result_preview = event["result"][:300].replace("\n", " ")
        tool_log_md += (
            f"**`{tool}`** `{json.dumps(event['args'])}`\n\n"
            f"> {result_preview}{'…' if len(event['result']) > 300 else ''}\n\n---\n\n"
        )
        tool_log_ph.markdown(tool_log_md)
    elif etype == "memo_chunk":
        memo_text += event["text"]
        memo_ph.markdown(memo_text + "▌")
    elif etype == "error":
        st.error(f"Agent error: {event['message']}")
        break
    elif etype == "done":
        memo_ph.markdown(memo_text)
        step_label.caption("Investigation complete.")
        break
