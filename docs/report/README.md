# PBL Report — AI-Powered Fake Product Review Detection Using NLP

`AI_Fake_Review_Detection_PBL_Report.pdf` is the print-ready A4 report (58 pages; poster on the last page).

## Rebuilding

From the repository root (needs `weasyprint`, `matplotlib`, `pypdf`, `reportlab` and the project requirements):

```bash
python docs/report/build/charts.py      # charts from the dataset and models/metrics.json
python docs/report/build/diagrams.py    # technical diagrams (SVG)
python docs/report/build/build_report.py
```

The builder runs WeasyPrint twice so the contents, figure and table lists carry real page numbers,
then adds the page-number footer and places the A3 poster (`assets/poster_A3.pdf`) upright, as vector content, on the final page.

## Sources

- `screenshots/` — captured from the running Streamlit app (`streamlit run app.py`); likelihoods shown are real model outputs for the built-in sample reviews.
- `figures/` — generated; every value comes from `dataset/fake reviews dataset.csv` or `models/metrics.json`.
- `assets/` — institutional header, watermark and logos taken from the institutional reference report, plus the project poster.
