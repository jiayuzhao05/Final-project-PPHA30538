from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent
RAW_DIR = REPO_ROOT / "data" / "raw-data"
DERIVED_DIR = REPO_ROOT / "data" / "derived-data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
EXTERNAL_DATA_DIRS = [
    REPO_ROOT.parent / "DATA",
    REPO_ROOT.parent.parent / "DATA",
]
STUDY_MIN_YEAR = 2014
STUDY_MAX_YEAR = 2020
US_STATES_GEOJSON_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
US_STATE_NAMES = {
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "District Of Columbia",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
}
STATE_ABBR = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District Of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}
POSITIVE_WORDS = {
    "gain",
    "gains",
    "growth",
    "strong",
    "beat",
    "beats",
    "upgrade",
    "upgrades",
    "recovery",
    "profit",
    "profits",
    "improve",
    "improves",
    "improved",
    "positive",
    "surge",
    "surges",
    "rebound",
    "bullish",
    "stable",
    "support",
    "supports",
}
NEGATIVE_WORDS = {
    "loss",
    "losses",
    "drop",
    "drops",
    "decline",
    "declines",
    "declined",
    "fall",
    "falls",
    "fell",
    "down",
    "downgrade",
    "downgrades",
    "crisis",
    "stress",
    "default",
    "defaults",
    "risk",
    "risks",
    "bankruptcy",
    "fail",
    "fails",
    "failed",
    "bad",
    "negative",
    "warning",
    "warnings",
    "panic",
}


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _first_existing_path(candidates: list[Path], purpose: str) -> Path:
    for path in candidates:
        if path.exists():
            return path

    msg = [f"Missing required file for {purpose}. Checked:"]
    msg.extend([f"- {p}" for p in candidates])
    raise FileNotFoundError("\n".join(msg))


def _resolve_fdic_path() -> Path:
    candidates = [RAW_DIR / "B" / "Summary_data_states.csv"]
    candidates.extend([d / "Summary_of_Results_3_2_2026 (1).csv" for d in EXTERNAL_DATA_DIRS])
    candidates.extend([d / "Summary_data_states.csv" for d in EXTERNAL_DATA_DIRS])
    return _first_existing_path(candidates, "FDIC state-year panel")


def _resolve_news_path() -> Path:
    """
    Locate raw news / headline data.

    Preferred order:
    1. data/raw-data/A/raw_partner_headlines.csv  (partner headlines, if present)
    2. data/raw-data/A/raw_analyst_ratings.csv    (fallback: Benzinga analyst headlines)
    3. External DATA/raw_partner_headlines.csv    (for large files kept outside repo)
    """
    candidates = [
        RAW_DIR / "A" / "raw_partner_headlines.csv",
        RAW_DIR / "A" / "raw_analyst_ratings.csv",
    ]
    candidates.extend([d / "raw_partner_headlines.csv" for d in EXTERNAL_DATA_DIRS])
    return _first_existing_path(candidates, "raw news headlines")


def _resolve_labeled_sentiment_path() -> Path:
    candidates = [RAW_DIR / "A" / "all-data.csv"]
    candidates.extend([d / "all-data.csv" for d in EXTERNAL_DATA_DIRS])
    return _first_existing_path(candidates, "labeled sentiment training data")


def _lexicon_sentiment_scores(headlines: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    sent_scores: list[float] = []
    neg_shares: list[float] = []
    for text in headlines.astype(str):
        tokens = re.findall(r"[a-z]+", text.lower())
        if not tokens:
            sent_scores.append(0.0)
            neg_shares.append(0.0)
            continue

        pos_n = sum(1 for t in tokens if t in POSITIVE_WORDS)
        neg_n = sum(1 for t in tokens if t in NEGATIVE_WORDS)
        total = pos_n + neg_n
        denom = float(total + 1)

        sent_scores.append((pos_n - neg_n) / denom)
        neg_shares.append(neg_n / denom)

    return np.asarray(sent_scores, dtype=float), np.asarray(neg_shares, dtype=float)


def load_fdic_state_year() -> pd.DataFrame:
    """
    FDIC state-year aggregates.

    Expected columns (from Summary_data_states.csv):
    ASSET, NETINC, NIM, DEP, BANKS, STNAME, YEAR, ...
    """
    fdic_path = _resolve_fdic_path()
    df = pd.read_csv(fdic_path)

    # Normalize types
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    df["STNAME"] = df["STNAME"].astype(str).str.strip()

    for c in ["ASSET", "NETINC", "DEP", "BANKS", "NIM"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["YEAR", "STNAME", "ASSET", "NETINC"]).copy()
    df = df[df["STNAME"].isin(US_STATE_NAMES)].copy()

    # Derived fundamentals
    df["ROA"] = df["NETINC"] / df["ASSET"]
    df = df.sort_values(["STNAME", "YEAR"])
    df["ROA_L1"] = df.groupby("STNAME")["ROA"].shift(1)
    df["DROA"] = df["ROA"] - df["ROA_L1"]
    df["DROA_L1"] = df.groupby("STNAME")["DROA"].shift(1)

    for c in ["ASSET", "DEP", "BANKS", "NIM"]:
        if c in df.columns:
            df[f"{c}_L1"] = df.groupby("STNAME")[c].shift(1)
            denom = df[f"{c}_L1"].where(df[f"{c}_L1"].abs() > 0)
            df[f"D{c}"] = (df[c] - df[f"{c}_L1"]) / denom

    # Guardrails: remove infinities from early years / zero denominators
    df = df.replace([np.inf, -np.inf], np.nan)

    # Labels
    q20 = df.groupby("STNAME")["DROA"].transform(lambda s: s.quantile(0.2))
    df["bad_year"] = (df["DROA"] < q20).astype(int)
    df["severity"] = (-np.minimum(df["DROA"], 0.0)).astype(float)

    return df


def train_sentiment_model() -> tuple[TfidfVectorizer | None, LogisticRegression | None]:
    """
    Train a lightweight sentiment classifier from the bundled labeled text dataset.

    Uses data/raw-data/A/all-data.csv which appears to be in format:
      <label>,<text>
    and encoded as cp1252.
    """
    try:
        train_path = _resolve_labeled_sentiment_path()
    except FileNotFoundError:
        print("Labeled all-data.csv not found; using lexicon sentiment fallback.")
        return None, None
    train_df = pd.read_csv(
        train_path,
        encoding="cp1252",
        header=None,
        names=["label", "text"],
    )
    train_df["label"] = train_df["label"].astype(str).str.strip().str.lower()
    train_df["text"] = train_df["text"].astype(str)
    train_df = train_df.dropna(subset=["label", "text"])

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
    )
    X = vectorizer.fit_transform(train_df["text"])
    y = train_df["label"]

    # Use single-process to stay compatible with sandboxed environments.
    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=1,
    )
    clf.fit(X, y)
    return vectorizer, clf


def build_news_sentiment_yearly(
    vectorizer: TfidfVectorizer | None,
    clf: LogisticRegression | None,
    *,
    keywords: list[str],
    min_year: int,
    max_year: int,
    chunksize: int = 200_000,
) -> pd.DataFrame:
    """
    Build a year-level sentiment index from raw headlines.

    We filter headlines by regulatory/policy keywords first, then score.
    """
    news_path = _resolve_news_path()
    usecols = ["headline", "date"]

    kw_re = re.compile("|".join([re.escape(k) for k in keywords]), flags=re.IGNORECASE)

    rows = []
    for chunk in pd.read_csv(news_path, usecols=usecols, chunksize=chunksize):
        chunk["headline"] = chunk["headline"].astype(str)
        chunk = chunk[chunk["headline"].str.contains(kw_re, na=False)].copy()
        if chunk.empty:
            continue

        dt = pd.to_datetime(chunk["date"], errors="coerce")
        chunk["YEAR"] = dt.dt.year
        chunk = chunk.dropna(subset=["YEAR"])
        chunk["YEAR"] = chunk["YEAR"].astype(int)
        chunk = chunk[(chunk["YEAR"] >= min_year) & (chunk["YEAR"] <= max_year)]
        if chunk.empty:
            continue

        if vectorizer is not None and clf is not None:
            Xh = vectorizer.transform(chunk["headline"])
            proba = clf.predict_proba(Xh)
            classes = list(clf.classes_)

            def _col(name: str) -> int | None:
                name = name.lower()
                return classes.index(name) if name in classes else None

            neg_i = _col("negative")
            pos_i = _col("positive")

            neg_p = proba[:, neg_i] if neg_i is not None else np.zeros(proba.shape[0])
            pos_p = proba[:, pos_i] if pos_i is not None else np.zeros(proba.shape[0])
            score = pos_p - neg_p
        else:
            score, neg_p = _lexicon_sentiment_scores(chunk["headline"])

        out = pd.DataFrame(
            {
                "YEAR": chunk["YEAR"].to_numpy(),
                "sent_score": score,
                "sent_neg_p": neg_p,
            }
        )
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["YEAR", "sent_mean", "sent_neg_share", "news_count"])

    all_scored = pd.concat(rows, ignore_index=True)
    yearly = (
        all_scored.groupby("YEAR", as_index=False)
        .agg(
            sent_mean=("sent_score", "mean"),
            sent_neg_share=("sent_neg_p", "mean"),
            news_count=("sent_score", "size"),
        )
        .sort_values("YEAR")
    )
    return yearly


def fit_and_score_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Fit simple baseline models and generate StressScore for visualization.
    """
    df = panel.copy()

    feature_cols = [
        "ROA_L1",
        "DROA_L1",
        "DASSET",
        "DDEP",
        "DBANKS",
        "NIM",
        "sent_mean",
        "sent_neg_share",
        "news_count",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    model_df = df.dropna(subset=feature_cols + ["bad_year", "severity"]).copy()
    if model_df.empty:
        df["p_bad_year"] = np.nan
        df["sev_hat"] = np.nan
        df["StressScore"] = np.nan
        return df

    X = model_df[feature_cols].to_numpy()
    y_cls = model_df["bad_year"].to_numpy()
    y_reg = model_df["severity"].to_numpy()

    cls_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=1)),
        ]
    )
    cls_pipe.fit(X, y_cls)
    p = cls_pipe.predict_proba(X)[:, 1]

    reg_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("reg", Ridge(alpha=1.0)),
        ]
    )
    if (y_cls == 1).any():
        reg_pipe.fit(X[y_cls == 1], y_reg[y_cls == 1])
    else:
        reg_pipe.fit(X, y_reg)
    sev_hat = np.maximum(reg_pipe.predict(X), 0.0)

    model_df["p_bad_year"] = p
    model_df["sev_hat"] = sev_hat
    model_df["StressScore"] = model_df["p_bad_year"] * model_df["sev_hat"]

    df = df.merge(
        model_df[["STNAME", "YEAR", "p_bad_year", "sev_hat", "StressScore"]],
        on=["STNAME", "YEAR"],
        how="left",
    )
    return df


def _fit_simple_slope(x: pd.Series, y: pd.Series, min_obs: int = 4) -> tuple[float, float, int]:
    mask = x.notna() & y.notna()
    n_obs = int(mask.sum())
    if n_obs < min_obs:
        return np.nan, np.nan, n_obs

    x_raw = x[mask].astype(float).to_numpy()
    y_raw = y[mask].astype(float).to_numpy()

    x_centered = x_raw - x_raw.mean()
    y_centered = y_raw - y_raw.mean()

    denom = float((x_centered * x_centered).sum())
    if denom <= 0:
        return np.nan, np.nan, n_obs

    beta = float((x_centered * y_centered).sum() / denom)

    if float(np.std(x_centered)) <= 0 or float(np.std(y_centered)) <= 0:
        corr = np.nan
    else:
        corr = float(np.corrcoef(x_raw, y_raw)[0, 1])

    return beta, corr, n_obs


def build_state_analysis_tables(
    panel: pd.DataFrame,
    *,
    min_year: int,
    max_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = panel.copy()
    work["YEAR"] = pd.to_numeric(work["YEAR"], errors="coerce")
    work = work[(work["YEAR"] >= min_year) & (work["YEAR"] <= max_year)].copy()
    work = work.dropna(subset=["STNAME", "YEAR"]).copy()
    work = work[work["STNAME"].isin(US_STATE_NAMES)].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    work["YEAR"] = work["YEAR"].astype(int)

    stress = (
        work.groupby("STNAME", as_index=False)
        .agg(
            n_years=("YEAR", "nunique"),
            bad_year_count=("bad_year", "sum"),
            bad_year_share=("bad_year", "mean"),
            avg_roa=("ROA", "mean"),
            avg_droa=("DROA", "mean"),
            avg_severity=("severity", "mean"),
            cumulative_negative_droa=(
                "DROA",
                lambda s: float((-np.minimum(s.fillna(0.0).to_numpy(), 0.0)).sum()),
            ),
            worst_droa=("DROA", "min"),
        )
        .sort_values(["bad_year_share", "cumulative_negative_droa"], ascending=[False, False])
    )

    bad_only = (
        work.loc[work["bad_year"] == 1]
        .groupby("STNAME", as_index=False)["severity"]
        .mean()
        .rename(columns={"severity": "avg_severity_bad_year"})
    )
    stress = stress.merge(bad_only, on="STNAME", how="left")

    valid_worst = work.dropna(subset=["DROA"]).copy()
    if not valid_worst.empty:
        worst_idx = valid_worst.groupby("STNAME")["DROA"].idxmin()
        worst_rows = (
            valid_worst.loc[worst_idx, ["STNAME", "YEAR", "DROA"]]
            .rename(columns={"YEAR": "worst_year", "DROA": "worst_droa_year_value"})
            .drop_duplicates(subset=["STNAME"])
        )
        stress = stress.merge(worst_rows, on="STNAME", how="left")

    work = work.sort_values(["STNAME", "YEAR"])
    for c in ["sent_neg_share", "sent_mean"]:
        if c in work.columns:
            work[f"{c}_L1"] = work.groupby("STNAME")[c].shift(1)

    sent_rows: list[dict[str, float | int | str]] = []
    for state, g in work.groupby("STNAME"):
        row: dict[str, float | int | str] = {
            "STNAME": state,
            "n_years": int(g["YEAR"].nunique()),
        }

        specs = [
            ("sent_neg_share", "beta_neg_t", "corr_neg_t", "n_neg_t"),
            ("sent_neg_share_L1", "beta_neg_l1", "corr_neg_l1", "n_neg_l1"),
            ("sent_mean", "beta_pos_t", "corr_pos_t", "n_pos_t"),
            ("sent_mean_L1", "beta_pos_l1", "corr_pos_l1", "n_pos_l1"),
        ]
        for x_col, beta_col, corr_col, n_col in specs:
            if x_col in g.columns and "DROA" in g.columns:
                beta, corr, n_obs = _fit_simple_slope(g[x_col], g["DROA"])
            else:
                beta, corr, n_obs = np.nan, np.nan, 0
            row[beta_col] = beta
            row[corr_col] = corr
            row[n_col] = n_obs

        sent_rows.append(row)

    sentiment = pd.DataFrame(sent_rows)
    if sentiment.empty:
        return stress, sentiment

    sentiment["d_droa_if_neg_news_plus_10pp_l1"] = sentiment["beta_neg_l1"] * 0.10
    sentiment["d_droa_if_pos_sent_plus_0p1_t"] = sentiment["beta_pos_t"] * 0.10
    sentiment["early_positive_response_score"] = sentiment["beta_pos_t"] - sentiment["beta_pos_l1"]

    sentiment = sentiment.merge(
        stress[["STNAME", "bad_year_share", "cumulative_negative_droa", "avg_droa"]],
        on="STNAME",
        how="left",
    )

    sentiment = sentiment.sort_values(
        ["beta_neg_l1", "early_positive_response_score"],
        ascending=[True, False],
    )
    return stress, sentiment


def write_state_analysis_outputs(
    panel: pd.DataFrame,
    *,
    min_year: int,
    max_year: int,
) -> dict[str, Path]:
    _safe_mkdir(OUTPUTS_DIR)
    stress, sentiment = build_state_analysis_tables(panel, min_year=min_year, max_year=max_year)

    output_paths: dict[str, Path] = {}
    if stress.empty:
        return output_paths

    stress_path = OUTPUTS_DIR / f"state_stress_summary_{min_year}_{max_year}.csv"
    stress.to_csv(stress_path, index=False)
    output_paths["stress_summary"] = stress_path

    if not sentiment.empty:
        sens_path = OUTPUTS_DIR / f"state_sentiment_sensitivity_{min_year}_{max_year}.csv"
        sentiment.to_csv(sens_path, index=False)
        output_paths["sentiment_sensitivity"] = sens_path

        hurt = sentiment.dropna(subset=["beta_neg_l1"]).sort_values("beta_neg_l1", ascending=True).head(10)
        early = sentiment.dropna(subset=["beta_pos_t"]).sort_values(
            ["beta_pos_t", "early_positive_response_score"],
            ascending=[False, False],
        ).head(10)

        hurt_path = OUTPUTS_DIR / f"top_states_hurt_by_negative_sentiment_{min_year}_{max_year}.csv"
        early_path = OUTPUTS_DIR / f"top_states_early_positive_response_{min_year}_{max_year}.csv"
        hurt.to_csv(hurt_path, index=False)
        early.to_csv(early_path, index=False)
        output_paths["top_hurt"] = hurt_path
        output_paths["top_early_positive"] = early_path

    return output_paths


def make_static_plots(
    panel: pd.DataFrame,
    *,
    min_year: int,
    max_year: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm

    _safe_mkdir(OUTPUTS_DIR)

    plot_df = panel.dropna(subset=["sent_neg_share", "DROA", "YEAR"]).copy()
    plot_df["YEAR"] = pd.to_numeric(plot_df["YEAR"], errors="coerce")
    plot_df = plot_df[(plot_df["YEAR"] >= min_year) & (plot_df["YEAR"] <= max_year)]
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.scatter(plot_df["sent_neg_share"], plot_df["DROA"], alpha=0.25, s=12)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("Regulatory-news negative probability (yearly mean)")
        ax.set_ylabel("ΔROA (state-year)")
        ax.set_title(
            f"News negativity vs next-period ROA change (state-year, {min_year}-{max_year})"
        )
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "static_scatter_sent_vs_droa.png", dpi=200)
        plt.close(fig)

    # Map plot uses geopandas; keep it robust for presentation
    try:
        import geopandas as gpd

        # Choose a year where the mapped metric is actually available.
        year_candidates = pd.to_numeric(
            panel.dropna(subset=["StressScore", "YEAR"])["YEAR"],
            errors="coerce",
        ).dropna()
        year_candidates = year_candidates[
            (year_candidates >= min_year) & (year_candidates <= max_year)
        ].astype(int)
        if year_candidates.empty:
            return
        map_year = int(year_candidates.max())
        map_df = panel[(panel["YEAR"] == map_year)].dropna(subset=["StressScore"]).copy()
        if not map_df.empty:
            shp_zip = RAW_DIR / "C" / "shapefile" / "cb_2024_us_all_20m" / "cb_2024_us_state_20m.zip"
            if shp_zip.exists():
                gdf = gpd.read_file(f"zip://{shp_zip}")
                name_col = "NAME" if "NAME" in gdf.columns else "name"
            else:
                gdf = gpd.read_file(US_STATES_GEOJSON_URL)
                name_col = "name" if "name" in gdf.columns else "NAME"

            gdf["NAME"] = gdf[name_col].astype(str).str.strip()
            gdf = gdf[gdf["NAME"].isin(US_STATE_NAMES)].copy()
            gdf = gdf.merge(map_df[["STNAME", "StressScore"]], left_on="NAME", right_on="STNAME", how="left")

            fig, ax = plt.subplots(figsize=(13.5, 8))
            vals = gdf["StressScore"].dropna()
            quantile_bins = (
                np.quantile(vals, [0.0, 0.1, 0.25, 0.4, 0.6, 0.75, 0.9, 1.0])
                if not vals.empty
                else np.array([])
            )
            quantile_bins = np.unique(quantile_bins)

            if quantile_bins.size >= 3:
                norm = BoundaryNorm(quantile_bins, ncolors=256, clip=True)
                gdf.plot(
                    column="StressScore",
                    ax=ax,
                    cmap="YlOrRd",
                    norm=norm,
                    edgecolor="white",
                    linewidth=0.45,
                    missing_kwds={"color": "lightgrey", "label": "Missing"},
                )
                sm = ScalarMappable(norm=norm, cmap="YlOrRd")
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
                cbar.set_label("StressScore (sensitive quantile bins)")
            else:
                gdf.plot(
                    column="StressScore",
                    ax=ax,
                    legend=True,
                    cmap="YlOrRd",
                    edgecolor="white",
                    linewidth=0.45,
                    missing_kwds={"color": "lightgrey", "label": "Missing"},
                )

            label_gdf = gdf[gdf["StressScore"].notna()].copy()
            label_gdf["abbr"] = label_gdf["NAME"].map(STATE_ABBR)
            label_gdf = label_gdf.dropna(subset=["abbr"])
            for _, row in label_gdf.iterrows():
                point = row.geometry.representative_point()
                ax.text(
                    point.x,
                    point.y,
                    row["abbr"],
                    fontsize=6.8,
                    ha="center",
                    va="center",
                    color="#1d1d1d",
                    fontweight="bold",
                )

            ax.set_axis_off()
            ax.set_title(f"State StressScore (sensitive colors + state labels) — {map_year}")
            fig.tight_layout()
            fig.savefig(OUTPUTS_DIR / "static_map_stressscore.png", dpi=200)
            fig.savefig(OUTPUTS_DIR / "static_map_stressscore_labeled.png", dpi=240)
            plt.close(fig)
    except Exception:
        # Map is optional for early presentation; Streamlit map can be used instead.
        return


def main() -> None:
    _safe_mkdir(RAW_DIR)
    _safe_mkdir(DERIVED_DIR)
    _safe_mkdir(OUTPUTS_DIR)

    print("Loading FDIC state-year data.")
    fdic = load_fdic_state_year()
    fdic = fdic[(fdic["YEAR"] >= STUDY_MIN_YEAR) & (fdic["YEAR"] <= STUDY_MAX_YEAR)].copy()

    min_year = STUDY_MIN_YEAR
    max_year = STUDY_MAX_YEAR

    print("Training lightweight sentiment model.")
    vec, clf = train_sentiment_model()

    print("Scoring regulatory/policy news headlines by year.")
    keywords = ["fdic", "sec", "fed", "regulation", "bank run", "regional bank", "stress test"]
    yearly_sent = build_news_sentiment_yearly(
        vec,
        clf,
        keywords=keywords,
        min_year=min_year,
        max_year=max_year,
    )

    panel = fdic.merge(yearly_sent, on="YEAR", how="left")
    panel = fit_and_score_panel(panel)

    out_parquet = DERIVED_DIR / "state_year_panel.parquet"
    out_csv = DERIVED_DIR / "state_year_panel.csv"
    try:
        panel.to_parquet(out_parquet, index=False)
    except ImportError:
        out_parquet = None
    panel.to_csv(out_csv, index=False)

    analysis_paths = write_state_analysis_outputs(panel, min_year=min_year, max_year=max_year)
    make_static_plots(panel, min_year=min_year, max_year=max_year)

    print("Wrote derived panel:")
    if out_parquet is not None:
        print(f"- {out_parquet}")
    print(f"- {out_csv}")
    if analysis_paths:
        print("Wrote state-level stress/sentiment analysis:")
        for p in analysis_paths.values():
            print(f"- {p}")
    print("Wrote static plots to outputs/:")
    print("- static_scatter_sent_vs_droa.png")
    print("- static_map_stressscore.png (if available)")


if __name__ == "__main__":
    main()

