"""
Load and merge the Kaggle healthcare claims CSVs.
Mirrors the exact data-prep logic from provider-level.ipynb and claim_analysis.ipynb.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

# CSVs sit two levels above this file (Cotiviti/)
DATA_DIR = Path(__file__).resolve().parents[2]

DIAG_COLS = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PROC_COLS = [f"ClmProcedureCode_{i}" for i in range(1, 7)]

# Correlated features dropped before modelling (same as notebooks)
DROP_FEATURES = [
    "total_deductible_paid",
    "avg_deductible_paid",
    "num_unique_attending_physicians",
    "num_unique_procedure_codes",
]


@functools.lru_cache(maxsize=1)
def load_claims() -> pd.DataFrame:
    """Merge IP + OP + Beneficiary into a single claims DataFrame."""
    ip = pd.read_csv(DATA_DIR / "Train_IP.csv")
    op = pd.read_csv(DATA_DIR / "Train_OP.csv")
    ben = pd.read_csv(DATA_DIR / "Train_Beneficiary.csv")

    ip["claim_type"] = 1
    op["claim_type"] = 0
    claims = pd.concat([ip, op], ignore_index=True)
    claims = claims.merge(ben, on="BeneID", how="left")

    for col in ["ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt", "DOB", "DOD"]:
        if col in claims.columns:
            claims[col] = pd.to_datetime(claims[col], errors="coerce")

    return claims


@functools.lru_cache(maxsize=1)
def load_labels() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "Train.csv")


def get_provider_claims(provider_id: str) -> pd.DataFrame:
    return load_claims().query("Provider == @provider_id").copy()


@functools.lru_cache(maxsize=1)
def build_provider_features() -> pd.DataFrame:
    """
    Reproduce the full feature-engineering pipeline from provider-level.ipynb.
    Returns a DataFrame with one row per Provider, 23 numeric features
    (after dropping the 4 correlated columns), plus Provider + PotentialFraud.
    """
    claims = load_claims()
    labels = load_labels()

    ip = claims[claims["claim_type"] == 1].copy()
    ip["length_of_stay"] = (ip["DischargeDt"] - ip["AdmissionDt"]).dt.days

    p = pd.DataFrame(index=claims["Provider"].unique())
    p.index.name = "Provider"

    p["total_claims"] = claims.groupby("Provider").size()
    p["unique_patients"] = claims.groupby("Provider")["BeneID"].nunique()
    p["claims_per_patient"] = p["total_claims"] / p["unique_patients"]

    p["total_reimbursement"] = claims.groupby("Provider")["InscClaimAmtReimbursed"].sum()
    p["avg_reimbursement"] = claims.groupby("Provider")["InscClaimAmtReimbursed"].mean()
    p["std_reimbursement"] = claims.groupby("Provider")["InscClaimAmtReimbursed"].std()
    p["max_reimbursement"] = claims.groupby("Provider")["InscClaimAmtReimbursed"].max()

    p["total_deductible_paid"] = claims.groupby("Provider")["DeductibleAmtPaid"].sum()
    p["avg_deductible_paid"] = claims.groupby("Provider")["DeductibleAmtPaid"].mean()

    p["pct_inpatient_claims"] = claims.groupby("Provider")["claim_type"].mean()

    p["avg_length_of_stay"] = ip.groupby("Provider")["length_of_stay"].mean()
    p["total_stay_days"] = ip.groupby("Provider")["length_of_stay"].sum()

    p["num_unique_attending_physicians"] = (
        claims.groupby("Provider")["AttendingPhysician"].nunique()
    )
    p["num_unique_operating_physicians"] = (
        claims.groupby("Provider")["OperatingPhysician"].nunique()
    )
    p["num_unique_other_physicians"] = (
        claims.groupby("Provider")["OtherPhysician"].nunique()
    )

    diag_long = claims.melt(
        id_vars="Provider", value_vars=DIAG_COLS, value_name="diagnosis_code"
    ).dropna(subset=["diagnosis_code"])
    p["num_unique_diagnosis_codes"] = (
        diag_long.groupby("Provider")["diagnosis_code"].nunique()
    )

    proc_long = claims.melt(
        id_vars="Provider", value_vars=PROC_COLS, value_name="procedure_code"
    ).dropna(subset=["procedure_code"])
    p["num_unique_procedure_codes"] = (
        proc_long.groupby("Provider")["procedure_code"].nunique()
    )

    claims = claims.copy()
    claims["num_diag_codes_this_claim"] = claims[DIAG_COLS].notna().sum(axis=1)
    claims["num_proc_codes_this_claim"] = claims[PROC_COLS].notna().sum(axis=1)
    p["avg_num_diagnosis_codes_per_claim"] = (
        claims.groupby("Provider")["num_diag_codes_this_claim"].mean()
    )
    p["avg_num_procedure_codes_per_claim"] = (
        claims.groupby("Provider")["num_proc_codes_this_claim"].mean()
    )

    p["avg_claim_duration"] = (
        (claims["ClaimEndDt"] - claims["ClaimStartDt"])
        .dt.days.groupby(claims["Provider"]).mean()
    )
    p["avg_patient_age"] = (
        ((claims["ClaimEndDt"] - claims["DOB"]).dt.days / 365.25)
        .groupby(claims["Provider"]).mean()
    )

    patient_level = claims.drop_duplicates(subset=["Provider", "BeneID"])
    p["pct_patients_deceased"] = patient_level.groupby("Provider")["DOD"].apply(
        lambda x: x.notna().mean()
    )
    p["avg_patient_IPAnnualReimbursement"] = patient_level.groupby("Provider")[
        "IPAnnualReimbursementAmt"
    ].mean()
    p["avg_patient_OPAnnualReimbursement"] = patient_level.groupby("Provider")[
        "OPAnnualReimbursementAmt"
    ].mean()

    p["reimbursement_per_diagnosis_code"] = (
        p["total_reimbursement"] / p["num_unique_diagnosis_codes"]
    )

    phys_long = claims.melt(
        id_vars="Provider",
        value_vars=["AttendingPhysician", "OperatingPhysician", "OtherPhysician"],
        value_name="physician_id",
    ).dropna(subset=["physician_id"])
    p["num_unique_physicians"] = phys_long.groupby("Provider")["physician_id"].nunique()
    p["claims_to_physician_ratio"] = p["total_claims"] / p["num_unique_physicians"]

    p = p.reset_index()
    p = p.merge(labels[["Provider", "PotentialFraud"]], on="Provider", how="left")
    p["PotentialFraud"] = p["PotentialFraud"].map({"Yes": 1, "No": 0})
    p = p.drop(columns=DROP_FEATURES, errors="ignore")
    p = p.fillna(0)

    return p
