# Student Outcome Prediction

A small project that trains a Random Forest model to predict student outcomes (Dropout, Enrolled, Graduate)
and exposes a Streamlit dashboard for interactive prediction.

## Files
- `dashboard.py` — Streamlit app that loads `optimized_rf_model.pkl` and provides a UI to predict outcomes.
- `feature sel.py` — Feature selection, training and evaluation script. Produces `optimized_rf_model.pkl` and reports.
- `data.xlsx` — (not checked in) source dataset used by `feature sel.py`.

## Quick Start
1. Create a Python virtual environment and install dependencies. Example (PowerShell):

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

2. If you have the dataset, run feature selection & training (this will produce `optimized_rf_model.pkl`):

```powershell
python "feature sel.py"
```

3. Run the Streamlit dashboard:

```powershell
streamlit run dashboard.py
```

Open the URL printed by Streamlit to use the interactive dashboard.

## Notes
- The dashboard expects `optimized_rf_model.pkl` in the same folder. If you don't have it, run `feature sel.py` to create it.
- If `data.xlsx` is large or private, keep it out of source control and provide it locally before training.

## Dependencies
- pandas, scikit-learn, joblib, streamlit (and other libs used in `feature sel.py` such as seaborn, statsmodels, python-docx)

Create a `requirements.txt` if you want reproducible installs.

---
If you want, I can add a `requirements.txt`, improve the Streamlit UI, or push these changes for you.