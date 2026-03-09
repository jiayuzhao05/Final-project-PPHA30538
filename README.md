# Final Project

This repository follows the course turn-in structure for PPHA 30538.

## Repository structure

- `final_project.qmd`: main writeup source (knit to HTML/PDF)
- `preprocessing.py`: reproducible ETL step (reads from `data/raw-data/`, writes to `data/derived-data/`)
- `streamlit-app/`: Streamlit app code (deployable on Streamlit Community Cloud)
- `code/`: optional helper scripts (EDA, plotting, modeling)
- `data/raw-data/`: raw input data (downloaded or released separately)
- `data/derived-data/`: cleaned/merged datasets used by analysis/app

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Rebuild derived datasets

```bash
python preprocessing.py
```

This script reads from `data/raw-data/` and writes:

- `data/derived-data/state_year_panel.csv` (and `.parquet` if supported)
- Static figures under `outputs/` used in the slides/app

## Run Streamlit locally

```bash
streamlit run streamlit-app/app.py
```

## Deployed Streamlit app

The app is also deployed to Streamlit Community Cloud:

- Live demo: [`final-project-ppha30538`](https://final-project-ppha30538-r9vojv7vvmmmqehe82uikv.streamlit.app/)

Because the underlying datasets are relatively large, the cloud deployment cannot
load the full data at the same resolution as the local version. As a result, some
maps and plots may appear lower quality or simplified compared with running the
app locally with the full data. For the best visual quality and full dataset, we
recommend running the app locally.

## Knit writeup

Open `final_project.qmd` in Quarto and render to both HTML and PDF. These knitted
outputs, together with this repository, are intended to fully reproduce the main
figures and tables in the writeup.

## Responsiveness to presentation feedback

After our in-class presentation we made several changes based on instructor and
peer feedback:

- **Move data processing into the `.qmd` file**: we refactored the workflow so
  that all merging, reshaping, and model fitting (including the construction of
  `ROA`, `DROA`, `bad_year`, `severity`, and `StressScore`) now runs inside
  `final_project.qmd`, with `preprocessing.py` providing reusable helper
  functions only.
- **Strengthen static visualizations**: we rewrote the Results section to
  include two static plots built with Altair (a scatter of sentiment vs ΔROA and
  a spatial map of state-level StressScore), each accompanied by a short
  interpretation that links back to the research questions.
- **Clarify data access and app deployment**: we documented that large raw and
  derived datasets are hosted in GitHub Releases, and we added notes about the
  limitations of the Streamlit Cloud deployment together with instructions for
  running the full-resolution app locally.

## Data

Due to GitHub file size limits, some of the datasets used in this project are **not**
stored directly in this repository. Instead, they are provided in the GitHub
**Releases** section of this repo.

- Large raw data files: available in the latest release under the `data/raw-data/`
  folder structure.
- Derived / processed data: available in the latest release under `data/derived-data/`.

To reproduce the analysis:

1. Download the corresponding release assets.
2. Place them into the local `data/raw-data/` and `data/derived-data` directories
   following the same folder structure.
3. Run `python preprocessing.py` and then knit `final_project.qmd` as described above.

