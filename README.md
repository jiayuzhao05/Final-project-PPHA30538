# Final Project

## Repository structure
- `final_project.qmd`: main writeup source (knit to HTML/PDF)
- `preprocessing.py`: reproducible ETL step (reads from `data/raw-data/`, writes to `data/derived-data/`)
- `streamlit-app/`: Streamlit app code (deployable on Streamlit Community Cloud)
- `code/`: optional helper scripts (EDA, plotting, modeling)
- `data/raw-data/`: raw input data (downloaded)
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
app locally with the full data.I recommended to run locally.

## Knit writeup

Open `final_project.qmd` in Quarto and render to HTML/PDF.

Can LLMs Quantify Market Reactions to Financial News and Regulation?

## Data

Due to GitHub file size limits, some of the datasets used in this project are **not** stored directly in this repository. Instead, they are provided in the GitHub **Releases** section of this repo.

- Large raw data files: available in the latest release under the `data/raw-data/` folder structure.
- Derived / processed data: available in the latest release under `data/derived-data/`.

To reproduce the analysis:
1. Download the corresponding release assets.
2. Place them into the local `data/raw-data/` and `data/derived-data` directories following the same folder structure.
3. Run `python preprocessing.py` and the rest of the pipeline as described above.

# Final Project

This repository follows the course turn-in structure (template-like):

## Repository structure

- `final_project.qmd`: main writeup source (knit to HTML/PDF)
- `preprocessing.py`: reproducible ETL step (reads from `data/raw-data/`, writes to `data/derived-data/`)
- `streamlit-app/`: Streamlit app code (deployable on Streamlit Community Cloud)
- `code/`: optional helper scripts (EDA, plotting, modeling)
- `data/raw-data/`: raw input data (downloaded)
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

## Run Streamlit locally

```bash
streamlit run streamlit-app/app.py
```

## Knit writeup

Open `final_project.qmd` in Quarto and render to HTML/PDF.

Can LLMs Quantify Market Reactions to Financial News and Regulation?

## Data

Due to GitHub file size limits, some of the datasets used in this project are **not** stored directly in this repository. Instead, they are provided in the GitHub **Releases** section of this repo.

- Large raw data files: available in the latest release under the `data/raw-data/` folder structure.
- Derived / processed data: available in the latest release under `data/derived-data/`.

To reproduce the analysis:
1. Download the corresponding release assets.
2. Place them into the local `data/raw-data/` and `data/derived-data/` directories following the same folder structure.
3. run `python preprocessing.py` and the rest of the pipeline as described above.