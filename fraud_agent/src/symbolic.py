"""
Symbolic billing-rule engine — the deterministic half of the neural-symbolic system.

Each rule is a hard logical check over a provider's claims DataFrame.
Rules are ILLUSTRATIVE but domain-grounded; comments note the real CMS policies
that would back them in production (Cotiviti's core business).

Rule 1: No-diagnosis procedure billing      [CMS NCCI edit tables]
Rule 2: Duplicate procedure codes           [CMS NCCI unbundling edits]
Rule 3: Reimbursement ceiling breach        [CMS DRG/APC fee schedules]
Rule 4: Temporal impossibility              [Medicare benefit policy / PECOS]
Rule 5: Deductible-reimbursement inversion  [CMS MLN deductible structure]
"""
from __future__ import annotations

import pandas as pd
import numpy as np

DIAG_COLS = [f"ClmDiagnosisCode_{i}" for i in range(1, 11)]
PROC_COLS = [f"ClmProcedureCode_{i}" for i in range(1, 7)]

# Rule 3 illustrative thresholds (production: per-DRG CMS fee schedules)
IP_CEILING = 150_000
OP_CEILING = 15_000

# Rule 4c: max distinct claims per physician per calendar day
# (production: cross-reference PECOS provider enrollment capacity)
PHYSICIAN_DAY_CEILING = 15


def rule_no_diagnosis_procedure(df: pd.DataFrame) -> dict:
    """
    Rule 1: Claim bills ≥1 procedure code but lists zero diagnosis codes.
    CMS NCCI requires every procedure to be paired with an ICD-9 diagnosis;
    a claim without any diagnosis code cannot be clinically justified.
    [Illustrative — production: CMS NCCI edit tables]
    """
    has_proc = df[PROC_COLS].notna().any(axis=1)
    has_diag = df[DIAG_COLS].notna().any(axis=1)
    viol = df[has_proc & ~has_diag]
    ids = viol["ClaimID"].tolist()
    return {
        "rule": "Rule 1 — No-diagnosis procedure billing",
        "violation_count": len(ids),
        "claim_ids": ids[:10],
        "explanation": (
            f"{len(ids)} claim(s) bill a procedure code with no diagnosis code. "
            "Every procedure must be clinically justified by an ICD-9 diagnosis "
            "(CMS NCCI). Unbacked procedure billing is a common upcoding vector. "
            "[Illustrative — production: CMS NCCI edit tables]"
        ),
    }


def rule_duplicate_procedures(df: pd.DataFrame) -> dict:
    """
    Rule 2: Two or more positions in ClmProcedureCode_1–6 hold the same non-null code.
    Equivalent to an NCCI unbundling violation: billing the same procedure twice on
    one claim inflates reimbursement without clinical justification.
    [Illustrative — production: CMS NCCI unbundling edit tables]
    """
    def _has_dup(row: pd.Series) -> bool:
        codes = row[PROC_COLS].dropna().tolist()
        return len(codes) > 1 and len(codes) != len(set(codes))

    mask = df.apply(_has_dup, axis=1)
    ids = df[mask]["ClaimID"].tolist()
    return {
        "rule": "Rule 2 — Duplicate procedure codes on a single claim",
        "violation_count": len(ids),
        "claim_ids": ids[:10],
        "explanation": (
            f"{len(ids)} claim(s) list the same procedure code in multiple slots. "
            "Duplicate procedure billing on one claim is a form of double-billing "
            "analogous to NCCI unbundling violations. "
            "[Illustrative — production: CMS NCCI edit tables]"
        ),
    }


def rule_reimbursement_ceiling(df: pd.DataFrame) -> dict:
    """
    Rule 3: InscClaimAmtReimbursed exceeds claim-type-specific ceiling.
    IP > $150 K or OP > $15 K are implausible without extraordinary clinical docs.
    [Illustrative — production: per-DRG/APC CMS fee schedules]
    """
    ip_flag = (df["claim_type"] == 1) & (df["InscClaimAmtReimbursed"] > IP_CEILING)
    op_flag = (df["claim_type"] == 0) & (df["InscClaimAmtReimbursed"] > OP_CEILING)
    viol = df[ip_flag | op_flag][
        ["ClaimID", "claim_type", "InscClaimAmtReimbursed"]
    ].copy()
    viol["claim_type_label"] = viol["claim_type"].map({1: "Inpatient", 0: "Outpatient"})
    details = viol[["ClaimID", "claim_type_label", "InscClaimAmtReimbursed"]].to_dict("records")
    return {
        "rule": "Rule 3 — Reimbursement ceiling breach",
        "violation_count": len(viol),
        "claim_ids": viol["ClaimID"].tolist()[:10],
        "details": details[:10],
        "explanation": (
            f"{len(viol)} claim(s) exceed the reimbursement ceiling "
            f"(IP > ${IP_CEILING:,}, OP > ${OP_CEILING:,}). "
            "These amounts are implausible without extraordinary clinical justification. "
            "[Illustrative — production: CMS DRG/APC fee schedules]"
        ),
    }


def rule_temporal_impossibility(df: pd.DataFrame) -> dict:
    """
    Rule 4: Three temporal checks:
      4a — IP claim: DischargeDt < AdmissionDt (negative length of stay)
      4b — Same beneficiary has two overlapping inpatient admissions
      4c — AttendingPhysician appears on > PHYSICIAN_DAY_CEILING distinct claims
           on the same calendar day (implausible patient throughput)
    [Illustrative — production: 4c cross-referenced with PECOS enrollment]
    """
    findings: list[str] = []
    violation_ids: list[str] = []

    ip = df[(df["claim_type"] == 1) & df["AdmissionDt"].notna() & df["DischargeDt"].notna()]

    # 4a — negative LOS
    neg = ip[ip["DischargeDt"] < ip["AdmissionDt"]]
    if not neg.empty:
        findings.append(
            f"4a: {len(neg)} inpatient claim(s) with DischargeDt < AdmissionDt"
        )
        violation_ids.extend(neg["ClaimID"].tolist())

    # 4b — overlapping stays for same beneficiary
    overlap_ids: list[str] = []
    if not ip.empty and "BeneID" in ip.columns:
        for _, grp in ip.sort_values(["BeneID", "AdmissionDt"]).groupby("BeneID"):
            rows = grp.reset_index(drop=True)
            for i in range(len(rows) - 1):
                if rows.loc[i + 1, "AdmissionDt"] < rows.loc[i, "DischargeDt"]:
                    overlap_ids += [rows.loc[i, "ClaimID"], rows.loc[i + 1, "ClaimID"]]
    overlap_ids = list(set(overlap_ids))
    if overlap_ids:
        findings.append(
            f"4b: {len(overlap_ids)} claim(s) part of overlapping IP stays "
            "for the same beneficiary"
        )
        violation_ids.extend(overlap_ids)

    # 4c — physician day-load
    overloaded: list[str] = []
    if "AttendingPhysician" in df.columns and df["ClaimStartDt"].notna().any():
        tmp = df.dropna(subset=["AttendingPhysician"]).copy()
        tmp["_day"] = tmp["ClaimStartDt"].dt.date
        daily = tmp.groupby(["AttendingPhysician", "_day"])["ClaimID"].count()
        over = daily[daily > PHYSICIAN_DAY_CEILING]
        if not over.empty:
            overloaded = list(over.index.get_level_values("AttendingPhysician").unique())
            findings.append(
                f"4c: {len(overloaded)} physician(s) on >{PHYSICIAN_DAY_CEILING} "
                f"claims in a single day "
                f"({', '.join(str(p) for p in overloaded[:3])}{'…' if len(overloaded) > 3 else ''})"
            )
            violation_ids.extend(
                df[df["AttendingPhysician"].isin(overloaded)]["ClaimID"].tolist()[:20]
            )

    violation_ids = list(set(violation_ids))
    return {
        "rule": "Rule 4 — Temporal impossibility",
        "violation_count": len(findings),
        "findings": findings,
        "claim_ids": violation_ids[:10],
        "explanation": (
            ("No temporal violations found."
             if not findings else
             "Temporal checks triggered: " + "; ".join(findings) + ".")
            + " [Illustrative — production: 4c via PECOS provider enrollment]"
        ),
    }


def rule_deductible_inversion(df: pd.DataFrame) -> dict:
    """
    Rule 5: DeductibleAmtPaid > InscClaimAmtReimbursed on any claim.
    Under Medicare Part A/B, the patient deductible cannot exceed the total
    Medicare reimbursement — this combination is structurally impossible.
    [Illustrative — production: CMS MLN deductible/coinsurance structure]
    """
    valid = df.dropna(subset=["DeductibleAmtPaid", "InscClaimAmtReimbursed"])
    viol = valid[valid["DeductibleAmtPaid"] > valid["InscClaimAmtReimbursed"]]
    ids = viol["ClaimID"].tolist()
    return {
        "rule": "Rule 5 — Deductible exceeds reimbursement",
        "violation_count": len(ids),
        "claim_ids": ids[:10],
        "explanation": (
            f"{len(ids)} claim(s) where DeductibleAmtPaid > InscClaimAmtReimbursed. "
            "A patient's deductible share cannot exceed what Medicare reimbursed — "
            "this is structurally impossible and indicates billing record manipulation. "
            "[Illustrative — production: CMS MLN deductible/coinsurance tables]"
        ),
    }


def run_all_rules(df: pd.DataFrame) -> dict:
    """Run all 5 rules and return a combined result dict."""
    results = {
        "rule_1": rule_no_diagnosis_procedure(df),
        "rule_2": rule_duplicate_procedures(df),
        "rule_3": rule_reimbursement_ceiling(df),
        "rule_4": rule_temporal_impossibility(df),
        "rule_5": rule_deductible_inversion(df),
    }
    triggered = sum(
        1 for r in results.values()
        if isinstance(r, dict) and r.get("violation_count", 0) > 0
    )
    results["summary"] = {
        "rules_triggered": triggered,
        "total_violations": sum(
            r.get("violation_count", 0)
            for r in results.values()
            if isinstance(r, dict) and "violation_count" in r
        ),
    }
    return results
