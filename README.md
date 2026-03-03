# Final Project (PPHA 30538)

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
