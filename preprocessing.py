from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
RAW_DIR = REPO_ROOT / "data" / "raw-data"
DERIVED_DIR = REPO_ROOT / "data" / "derived-data"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    # Placeholder: implement data pulls + cleaning here.
    # Requirements (per course instructions):
    # - Read raw inputs from data/raw-data/
    # - Write derived outputs to data/derived-data/
    #
    # Example targets you may create later:
    # - derived_market.parquet
    # - derived_news_sentiment.parquet
    # - derived_bank_hq.parquet
    print(f"Raw data dir: {RAW_DIR}")
    print(f"Derived data dir: {DERIVED_DIR}")
    print("preprocessing.py scaffold created. Add your ETL steps here.")


if __name__ == "__main__":
    main()

