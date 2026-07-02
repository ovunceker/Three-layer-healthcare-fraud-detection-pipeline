"""
Layer 1: Ridge logistic regression + Isolation Forest (provider-level).
Layer 2: Z-score ensemble across 36 claim groupings (claim-level, aggregated to provider).

All models are trained from scratch at first call and cached in memory.
Re-creating the logic from provider-level.ipynb and claim_analysis.ipynb.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .data import build_provider_features, load_claims

logger = logging.getLogger(__name__)

_models: dict = {}
_feature_cols: list[str] = []


# ── public entry point ───────────────────────────────────────────────────────

def get_models() -> dict:
    if not _models:
        _train_all()
    return _models


def score_provider(provider_id: str) -> dict:
    """Return Layer 1 + Layer 2 scores for a single provider."""
    m = get_models()
    df = m["provider_df"]
    row = df[df["Provider"] == provider_id]

    if row.empty:
        return {"error": f"Provider {provider_id} not found in training data"}

    X = row[_feature_cols].values.astype(float)
    X_s = m["scaler"].transform(X)

    ridge_prob = float(m["ridge"].predict_proba(X_s)[0, 1])
    iso_score = float(-m["iso"].decision_function(X_s)[0])

    l2_row = m["layer2"][m["layer2"]["Provider"] == provider_id]
    mean_rfrac = float(l2_row["mean_claim_risk_fraction"].iloc[0]) if not l2_row.empty else 0.0
    pct_anom = float(l2_row["pct_anomalous_claims"].iloc[0]) if not l2_row.empty else 0.0

    return {
        "provider_id": provider_id,
        "layer1_ridge_fraud_probability": round(ridge_prob, 4),
        "layer1_isolation_forest_score": round(iso_score, 4),
        "layer2_mean_claim_risk_fraction": round(mean_rfrac, 4),
        "layer2_pct_anomalous_claims": round(pct_anom, 4),
    }


# ── private training ─────────────────────────────────────────────────────────

def _train_all() -> None:
    global _feature_cols

    logger.info("Building provider features…")
    df = build_provider_features()

    feature_cols = [
        c for c in df.columns
        if c not in ("Provider", "PotentialFraud")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    _feature_cols = feature_cols

    X = df[feature_cols].values.astype(float)
    y = df["PotentialFraud"].values.astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_va_s = scaler.transform(X_val)

    # ── Ridge Logistic Regression ────────────────────────────────────────────
    # LogisticRegressionCV picks the best C (=1/λ) by 5-fold CV AUC, mirroring
    # the glmnet lambda-selection via validation AUC in provider_analysis.Rmd.
    logger.info("Training ridge logistic regression…")
    ridge = LogisticRegressionCV(
        Cs=20, cv=5, penalty="l2", solver="lbfgs",
        max_iter=2000, scoring="roc_auc", random_state=42, n_jobs=-1,
    )
    ridge.fit(X_tr_s, y_train)
    val_auc = roc_auc_score(y_val, ridge.predict_proba(X_va_s)[:, 1])
    logger.info(f"Ridge LR  val AUC: {val_auc:.4f}  best C: {ridge.C_[0]:.4f}")

    # ── Isolation Forest ─────────────────────────────────────────────────────
    # Reduced grid (notebook best was n=100, f=0.75, s='auto', c=fraud_rate).
    # We search a small neighbourhood around that optimum for validation AUC.
    logger.info("Training Isolation Forest (grid search)…")
    fraud_rate = float(y_train.mean())
    best_auc, best_iso = -1.0, None
    for n in [100, 200]:
        for f in [0.75, 1.0]:
            for s in ["auto", 0.5]:
                for c in [fraud_rate, "auto"]:
                    m = IsolationForest(
                        n_estimators=n, max_features=f,
                        max_samples=s, contamination=c,
                        random_state=42,
                    )
                    m.fit(X_tr_s)
                    auc = roc_auc_score(y_val, -m.decision_function(X_va_s))
                    if auc > best_auc:
                        best_auc, best_iso = auc, m
    logger.info(f"Isolation Forest val AUC: {best_auc:.4f}")

    # ── Layer 2: Z-score ensemble ────────────────────────────────────────────
    logger.info("Computing Layer 2 z-score ensemble (this takes ~30 s)…")
    layer2, ref_stats = _compute_layer2()
    logger.info("All models ready.")

    _models.update({
        "ridge": ridge,
        "scaler": scaler,
        "iso": best_iso,
        "layer2": layer2,
        "layer2_ref_stats": ref_stats,
        "provider_df": df,
    })


def _norm_key(v) -> str | None:
    """Normalise a groupby value to a string key for ref_stats lookup.
    Converts numpy scalars and strips trailing '.0' from integer-valued floats
    so that e.g. claim_type=1 (int), 1.0 (float), and '1' all map to '1'.
    """
    if v is None:
        return None
    try:
        if (isinstance(v, float) and np.isnan(v)) or (
            hasattr(v, "__float__") and np.isnan(float(v))
        ):
            return None
    except (TypeError, ValueError):
        pass
    s = str(v)
    if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
        return s[:-2]
    return s


def _compute_layer2() -> tuple[pd.DataFrame, dict]:
    """
    Reproduce claim_analysis.ipynb exactly:
    36 grouping columns × log1p(InscClaimAmtReimbursed) z-score ensemble
    → per-claim risk fraction → aggregate to provider.

    Also builds ref_stats — the per-column group means/stds needed to score
    a single new claim against the training reference distributions.
    """
    claims = load_claims()

    OUTPUT_COLS = [
        "n_groupings_eligible", "n_groupings_flagged",
        "claim_risk_fraction", "is_claim_anomalous",
    ]
    exclude_cols = [
        "BeneID", "ClaimID", "Provider",
        "ClaimStartDt", "ClaimEndDt", "AdmissionDt", "DischargeDt", "DOB", "DOD",
        "InscClaimAmtReimbursed", "DeductibleAmtPaid",
        "IPAnnualReimbursementAmt", "IPAnnualDeductibleAmt",
        "OPAnnualReimbursementAmt", "OPAnnualDeductibleAmt",
        "NoOfMonths_PartACov", "NoOfMonths_PartBCov",
        "Gender", "Race",
    ] + OUTPUT_COLS

    candidate_cols = [c for c in claims.columns if c not in exclude_cols]

    COVERAGE = 0.8
    FALLBACK = 30
    CEILING = 1000
    Z_THRESH = 3.0
    RISK_THRESH = 0.25

    def adaptive_cutoff(series: pd.Series) -> int:
        counts = series.value_counts()
        if len(counts) == 0:
            return FALLBACK
        cum = counts.cumsum() / counts.sum()
        mask = cum >= COVERAGE
        cutoff = int(counts[mask].iloc[0]) if mask.any() else int(counts.iloc[-1])
        return cutoff if cutoff <= CEILING else FALLBACK

    min_sizes = {col: adaptive_cutoff(claims[col]) for col in candidate_cols}

    target = "InscClaimAmtReimbursed"
    log_vals = np.log1p(claims[target])

    flags = pd.DataFrame(index=claims.index)
    for col in candidate_cols:
        gs = claims.groupby(col)[target].transform("size")
        eligible = gs >= min_sizes[col]
        z = log_vals.groupby(claims[col]).transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else np.nan
        )
        z = z.where(eligible)
        flags[col] = (z.abs() > Z_THRESH).where(z.notna())

    claims = claims.copy()
    claims["n_groupings_eligible"] = flags.notna().sum(axis=1)
    claims["n_groupings_flagged"] = flags.sum(axis=1, skipna=True)
    claims["claim_risk_fraction"] = (
        claims["n_groupings_flagged"] / claims["n_groupings_eligible"]
    )
    claims["is_claim_anomalous"] = claims["claim_risk_fraction"] > RISK_THRESH

    agg = (
        claims.groupby("Provider")
        .agg(
            total_claims=("ClaimID", "count"),
            mean_claim_risk_fraction=("claim_risk_fraction", "mean"),
            pct_anomalous_claims=("is_claim_anomalous", "mean"),
            max_claim_risk_fraction=("claim_risk_fraction", "max"),
        )
        .reset_index()
    )

    # ── Build reference stats for single-claim scoring ───────────────────────
    # Keys are normalised to strings so lookup is type-safe.
    ref_stats: dict = {}
    for col in candidate_cols:
        grp_stats = log_vals.groupby(claims[col]).agg(["mean", "std", "count"])
        valid = grp_stats[grp_stats["count"] >= min_sizes[col]]
        if valid.empty:
            continue
        ref_stats[col] = {
            "means": {_norm_key(k): float(v) for k, v in valid["mean"].items()},
            "stds":  {_norm_key(k): float(v) for k, v in valid["std"].items()},
        }

    return agg, ref_stats


def score_single_claim(claim_dict: dict) -> dict:
    """
    Score one claim from an unknown provider using Layer 2 reference stats
    that were computed from the full training corpus.

    For each grouping column that (a) the claim has a value for and (b) that
    value's group appeared enough times in training to have valid statistics,
    we compute a log1p z-score and check whether |z| > 3.

    Layer 1 (provider-level ridge / IF) is intentionally NOT called here —
    it requires aggregated provider history that doesn't exist for new providers.
    """
    m = get_models()
    if "layer2_ref_stats" not in m:
        return {"error": "Layer 2 reference stats unavailable; call get_models() first"}

    ref_stats = m["layer2_ref_stats"]
    reimb = claim_dict.get("InscClaimAmtReimbursed")
    if reimb is None:
        return {"error": "InscClaimAmtReimbursed is required for Layer 2 scoring"}

    log_reimb = np.log1p(float(reimb))
    eligible = 0
    flagged = 0
    flagged_groupings: list[dict] = []

    for col, stats in ref_stats.items():
        raw_val = claim_dict.get(col)
        key = _norm_key(raw_val)
        if key is None:
            continue

        g_mean = stats["means"].get(key)
        g_std  = stats["stds"].get(key)
        if g_mean is None or g_std is None or g_std == 0:
            continue

        z = (log_reimb - g_mean) / g_std
        eligible += 1
        if abs(z) > 3.0:
            flagged += 1
            flagged_groupings.append({
                "grouping": col,
                "group_value": str(raw_val)[:40],
                "z_score": round(float(z), 3),
            })

    risk_fraction = flagged / eligible if eligible > 0 else 0.0
    return {
        "n_groupings_eligible": eligible,
        "n_groupings_flagged": flagged,
        "claim_risk_fraction": round(risk_fraction, 4),
        "is_claim_anomalous": risk_fraction > 0.25,
        "flagged_groupings": flagged_groupings[:5],
        "interpretation": (
            f"Risk fraction {risk_fraction:.1%} across {eligible} eligible peer groups. "
            + ("ANOMALOUS — reimbursement is an outlier in multiple peer-group comparisons."
               if risk_fraction > 0.25 else
               "Within normal range relative to comparable training claims.")
        ),
    }
