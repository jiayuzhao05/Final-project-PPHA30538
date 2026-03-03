# Final Project (PPHA 30538)

**Can LLMs Quantify Market Reactions to Financial News and Regulation?**

This repository follows the course turn-in structure and satisfies the required components: **2 datasets**, **2 static plots** (altair/geopandas), and **1 Streamlit app** with a spatial visualization.

---

## Streamlit app

- **Local:** `streamlit run streamlit-app/app.py`
- **Streamlit Community Cloud:** [Deploy from this repo and add your app URL here once deployed.]

---

## Data sources and where to put them

All raw inputs go under `data/raw-data/`. You do not need to rename files; paths used in code match the structure below.

| Dataset | Source | Where to save | Notes |
|--------|--------|----------------|-------|
| **A – News / sentiment** | Kaggle (e.g. “Daily Financial News for 6000+ Stocks” or partner headlines); plus labeled sentences for training | `data/raw-data/A/raw_partner_headlines.csv`, `data/raw-data/A/all-data.csv` | `all-data.csv`: 2 columns `label`, `text` (encoding: cp1252). Headlines CSV: columns `headline`, `date` (and optionally `stock`). |
| **B – FDIC state-year** | FDIC (e.g. “Summary of Results” or State Tables for Commercial Banks – Financial) | `data/raw-data/B/Summary_data_states.csv` | Must include at least: `ASSET`, `NETINC`, `BANKS`, `NIM`, `STNAME`, `YEAR`. |
| **C – Geography** | US Census (TIGER/Line shapefiles) | `data/raw-data/C/shapefile/cb_2024_us_all_20m/cb_2024_us_state_20m.zip` | State boundary shapefile for choropleth. |

If any dataset is >100MB or not public, host it (e.g. Google Drive/Dropbox) and put the download link and exact save path in this README. `.gitignore` is set to ignore `data/raw-data/*` and `data/derived-data/*` as needed; keep `outputs/` (static plots and metrics CSV) committed so the writeup knits without re-running Python.

---

## How the data are processed

- **Script:** `preprocessing.py`  
  - **Inputs:** `data/raw-data/A/` (headlines + labeled text), `data/raw-data/B/Summary_data_states.csv`, `data/raw-data/C/.../cb_2024_us_state_20m.zip`.  
  - **Outputs:**  
    - `data/derived-data/state_year_panel.csv` (and `.parquet`): merged state–year panel with FDIC fundamentals, sentiment indices, labels (`bad_year`, `severity`), and model-based `StressScore`.  
    - `outputs/static_scatter_sent_vs_droa.png`: static plot 1 (sentiment vs ΔROA).  
    - `outputs/static_map_stressscore.png`: static plot 2 (state choropleth of StressScore).  

- **Optional metrics script:** `code/presentation_experiment.py`  
  - **Input:** `data/derived-data/state_year_panel.csv`.  
  - **Output:** `outputs/presentation_metrics.csv` (AUC and R² for FDIC-only vs FDIC+Sentiment).  

Merging and reshaping are done in `preprocessing.py`; the writeup (`final_project.qmd`) reads the derived tables and `outputs/` to display figures and tables.

---

## Reproducibility (to knit the writeup)

1. Install dependencies: `python -m pip install -r requirements.txt`
2. Run preprocessing: `python preprocessing.py`
3. (Optional) Run metrics: `python code/presentation_experiment.py`
4. Knit the writeup: open `final_project.qmd` in Quarto and render to HTML and PDF.

A TA or reviewer should be able to clone the repo, install dependencies, run steps 2–3 (and place raw data as above if not committed), then knit `final_project.qmd` to reproduce one version of the writeup without renaming any files.

---

## Repository structure

- `final_project.qmd`: main writeup (render to HTML + PDF for submission)
- `preprocessing.py`: ETL from `data/raw-data/` → `data/derived-data/` and `outputs/`
- `streamlit-app/`: Streamlit app (deploy to Streamlit Community Cloud)
- `code/`: helper scripts (e.g. `presentation_experiment.py`)
- `data/raw-data/`: raw inputs (A, B, C as above)
- `data/derived-data/`: cleaned/merged panel (generated)
- `outputs/`: static plots and metrics CSV (committed so writeup knits)

---

## Requirements

- `requirements.txt` at repo root (and `streamlit-app/requirements.txt` for Cloud deploy)
- `.gitignore` set to ignore venv, `__pycache__`, and optionally large data dirs; `outputs/` is **not** ignored so the writeup can be knitted after clone.
