# TOW Mapper (Cloud DB) — Runbook

**Repo:** `mkozina969/tow-mapper-cloud`  
**Purpose:** Map supplier invoice item codes to internal **TOW** codes using a **persistent Postgres (Neon)** database. Supports **Excel / CSV / PDF** invoices.  
**Persistence:** All inserts/updates live in Postgres and survive restarts.

---

## 1) Stack & Key Files
- **Frontend/App:** Streamlit
- **Lang/Libs:** Python, pandas (≥2.1), SQLAlchemy (≥2.0), pdfplumber (PDF), openpyxl/xlsxwriter (Excel)
- **DB:** Postgres (Neon)
- **Main app:** `streamlit_app.py`
- **Initial bulk loader (fast):** `bulk_copy_to_neon.py`
- **Incremental upserts:** `migrate_csv_to_db.py`
- **Deps:** `requirements.txt`
- **Ignored secrets:** `.streamlit/secrets.toml` (local only)

