"""
Thin JSON-serializable wrappers around neural.py, symbolic.py, and data.py.
These are the functions the agent calls; they must accept and return plain dicts.
"""
from __future__ import annotations

import pandas as pd

from .data import get_provider_claims, DIAG_COLS, PROC_COLS
from .neural import score_provider, score_single_claim
from .symbolic import run_all_rules


# ── Tool implementations ──────────────────────────────────────────────────────

def ml_risk_score(provider_id: str) -> dict:
    """
    Return ML risk scores for a provider:
      - layer1_ridge_fraud_probability  (0-1, higher = more likely fraud)
      - layer1_isolation_forest_score   (higher = more anomalous)
      - layer2_mean_claim_risk_fraction (average over all claims)
      - layer2_pct_anomalous_claims     (fraction of claims flagged anomalous)
    """
    return score_provider(provider_id)


def check_billing_rules(provider_id: str) -> dict:
    """
    Run 5 symbolic billing rules on a provider's claims.
    Returns violation counts and plain-English explanations for each rule.
    """
    df = get_provider_claims(provider_id)
    if df.empty:
        return {"error": f"No claims found for provider {provider_id}"}
    result = run_all_rules(df)
    # Truncate claim_id lists to keep the LLM context manageable
    for v in result.values():
        if isinstance(v, dict) and "claim_ids" in v:
            v["claim_ids"] = v["claim_ids"][:5]
    return result


def get_claims_summary(provider_id: str) -> dict:
    """
    Return a statistical overview of a provider's claims portfolio:
    claim counts by type, reimbursement distribution, unique patients/physicians/codes.
    """
    df = get_provider_claims(provider_id)
    if df.empty:
        return {"error": f"No claims found for provider {provider_id}"}

    ip = df[df["claim_type"] == 1].copy()
    if not ip.empty and "AdmissionDt" in ip.columns and "DischargeDt" in ip.columns:
        ip["los"] = (ip["DischargeDt"] - ip["AdmissionDt"]).dt.days
        avg_los = round(float(ip["los"].mean()), 2)
    else:
        avg_los = None

    reimb = df["InscClaimAmtReimbursed"]
    return {
        "provider_id": provider_id,
        "total_claims": int(len(df)),
        "inpatient_claims": int((df["claim_type"] == 1).sum()),
        "outpatient_claims": int((df["claim_type"] == 0).sum()),
        "unique_patients": int(df["BeneID"].nunique()),
        "date_range": {
            "first": str(df["ClaimStartDt"].min().date()) if df["ClaimStartDt"].notna().any() else None,
            "last": str(df["ClaimStartDt"].max().date()) if df["ClaimStartDt"].notna().any() else None,
        },
        "reimbursement": {
            "total": round(float(reimb.sum()), 2),
            "mean": round(float(reimb.mean()), 2),
            "median": round(float(reimb.median()), 2),
            "max": round(float(reimb.max()), 2),
            "p99": round(float(reimb.quantile(0.99)), 2),
        },
        "avg_length_of_stay_days": avg_los,
        "unique_attending_physicians": int(df["AttendingPhysician"].nunique()),
        "unique_diagnosis_codes": int(df[DIAG_COLS].stack().nunique()),
        "unique_procedure_codes": int(df[PROC_COLS].stack().nunique()),
        "pct_claims_with_procedure": round(
            float(df[PROC_COLS].notna().any(axis=1).mean()), 4
        ),
        "claims_per_patient": round(float(len(df) / df["BeneID"].nunique()), 2),
    }


# ── Tool schema for Ollama tool-calling API (OpenAI format) ──────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ml_risk_score",
            "description": (
                "Get machine-learning fraud risk scores for a healthcare provider. "
                "Returns: ridge logistic regression fraud probability (0-1, higher means "
                "more fraud risk), isolation forest anomaly score (higher means more "
                "anomalous compared to peers), and claim-level z-score ensemble stats "
                "(mean risk fraction and % anomalous claims across all groupings)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "provider_id": {
                        "type": "string",
                        "description": "Provider ID, e.g. PRV55912",
                    }
                },
                "required": ["provider_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_billing_rules",
            "description": (
                "Run 5 symbolic billing-policy rule checks on a provider's claims. "
                "Rules check: (1) procedure without diagnosis, (2) duplicate procedure "
                "codes, (3) reimbursement ceiling breach, (4) temporal impossibilities "
                "(negative LOS, overlapping stays, physician day overload), "
                "(5) deductible exceeds reimbursement. Returns violation counts and "
                "plain-English explanations for each rule."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "provider_id": {
                        "type": "string",
                        "description": "Provider ID, e.g. PRV55912",
                    }
                },
                "required": ["provider_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_claims_summary",
            "description": (
                "Get a statistical overview of a provider's claims portfolio: "
                "claim counts (inpatient vs outpatient), reimbursement statistics, "
                "unique patients, physicians, diagnosis codes, procedure codes, "
                "and average length of stay. Call this first to understand the provider's scale."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "provider_id": {
                        "type": "string",
                        "description": "Provider ID, e.g. PRV55912",
                    }
                },
                "required": ["provider_id"],
            },
        },
    },
]

TOOL_MAP: dict[str, callable] = {
    "ml_risk_score": ml_risk_score,
    "check_billing_rules": check_billing_rules,
    "get_claims_summary": get_claims_summary,
}


# ── Claim-level tools (for single-claim / unknown-provider mode) ─────────────

def _dict_to_claim_df(claim_dict: dict) -> pd.DataFrame:
    """Build a 1-row DataFrame from a claim dict, ensuring all expected columns exist."""
    row = {**claim_dict}
    required_cols = (
        ["ClaimID", "BeneID", "claim_type", "InscClaimAmtReimbursed",
         "DeductibleAmtPaid", "ClaimStartDt", "ClaimEndDt",
         "AdmissionDt", "DischargeDt", "AttendingPhysician",
         "OperatingPhysician", "OtherPhysician"]
        + DIAG_COLS + PROC_COLS
    )
    for col in required_cols:
        if col not in row:
            row[col] = None
    df = pd.DataFrame([row])
    for col in ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def score_single_claim_layer2(claim_dict: dict) -> dict:
    """
    Layer 2 z-score anomaly check for a single claim.
    Compares this claim's reimbursement against the training reference
    distributions (group means/stds pre-computed at startup).
    Layer 1 provider-level scoring is NOT available — it requires aggregated
    provider history that does not exist for new/unknown providers.
    """
    return score_single_claim(claim_dict)


def check_single_claim_rules(claim_dict: dict) -> dict:
    """
    Run all 5 symbolic billing rules on a single claim.
    Note: rules 4b (overlapping stays) and 4c (physician day-load) require
    multiple claims and will not fire on a single-claim submission.
    """
    df = _dict_to_claim_df(claim_dict)
    return run_all_rules(df)


CLAIM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "score_single_claim_layer2",
            "description": (
                "Score this claim's reimbursement amount against training reference "
                "distributions using the z-score ensemble (Layer 2). "
                "Returns: risk fraction, anomaly flag, and which peer-group comparisons "
                "flagged it. Note: Layer 1 provider-level models are NOT available for "
                "unknown providers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_dict": {
                        "type": "object",
                        "description": "The claim data dict (passed automatically by the system).",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_single_claim_rules",
            "description": (
                "Run 5 symbolic billing-rule checks on this single claim. "
                "Rules: (1) procedure without diagnosis, (2) duplicate procedure codes, "
                "(3) reimbursement ceiling breach, (4a) discharge before admission, "
                "(5) deductible exceeds reimbursement. "
                "Rules 4b/4c (overlapping stays, physician day-load) require multiple "
                "claims and will not fire here."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_dict": {
                        "type": "object",
                        "description": "The claim data dict (passed automatically by the system).",
                    }
                },
                "required": [],
            },
        },
    },
]

CLAIM_TOOL_MAP: dict[str, callable] = {
    "score_single_claim_layer2": score_single_claim_layer2,
    "check_single_claim_rules": check_single_claim_rules,
}
