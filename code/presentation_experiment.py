from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
DERIVED = REPO_ROOT / "data" / "derived-data"
OUTPUTS = REPO_ROOT / "outputs"


def _build_pipelines() -> tuple[Pipeline, Pipeline]:
    cls = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=1)),
        ]
    )
    reg = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("reg", Ridge(alpha=1.0)),
        ]
    )
    return cls, reg


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DERIVED / "state_year_panel.csv")
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")

    # Only evaluate where we have sentiment built from headlines
    df = df.dropna(subset=["YEAR", "sent_neg_share", "bad_year", "severity"]).copy()
    if df.empty:
        raise SystemExit("No rows with sentiment available. Run preprocessing.py first.")

    years = sorted(int(y) for y in df["YEAR"].dropna().unique())
    test_year = max(years)
    train_years = [y for y in years if y < test_year]
    if not train_years:
        raise SystemExit("Not enough years for a time split evaluation.")

    train = df[df["YEAR"].isin(train_years)].copy()
    test = df[df["YEAR"] == test_year].copy()

    fdic_feats = [c for c in ["ROA_L1", "DROA_L1", "DASSET", "DDEP", "DBANKS", "NIM"] if c in df.columns]
    sent_feats = [c for c in ["sent_mean", "sent_neg_share", "news_count"] if c in df.columns]

    setups = [
        ("FDIC_only", fdic_feats),
        ("FDIC_plus_sentiment", fdic_feats + sent_feats),
    ]

    results = []
    for name, feats in setups:
        if not feats:
            continue

        Xtr = train[feats].to_numpy()
        Xte = test[feats].to_numpy()
        ytr = train["bad_year"].to_numpy()
        yte = test["bad_year"].to_numpy()

        cls, reg = _build_pipelines()
        cls.fit(Xtr, ytr)
        pte = cls.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, pte) if len(np.unique(yte)) > 1 else np.nan

        # Severity regression evaluated on all (but target is 0 when not bad)
        reg.fit(Xtr, train["severity"].to_numpy())
        sev_hat = np.maximum(reg.predict(Xte), 0.0)
        r2 = r2_score(test["severity"].to_numpy(), sev_hat)

        results.append(
            {
                "setup": name,
                "train_years": f"{min(train_years)}-{max(train_years)}",
                "test_year": test_year,
                "n_train": len(train),
                "n_test": len(test),
                "AUC_bad_year": auc,
                "R2_severity": r2,
            }
        )

    out = pd.DataFrame(results)
    out_path = OUTPUTS / "presentation_metrics.csv"
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()

