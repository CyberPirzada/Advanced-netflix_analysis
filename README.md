# Netflix Content Analysis (Streamlit)

Place your Netflix dataset (CSV or Excel) in this folder or use the upload widget in the app.

Quick start (PowerShell, with your venv activated):

```powershell
cd C:\Heckathiyon\Advanced_level
pip install -r requirements.txt
streamlit run netflix_streamlit.py
```

App features:
- Upload Netflix CSV/Excel.
- Clean missing `director` values and normalize date fields.
- Interactive Plotly charts: Movies vs TV Shows added per year, monthly release heatmap, genre trends.

When you upload the dataset I will run it and refine the visualizations.
