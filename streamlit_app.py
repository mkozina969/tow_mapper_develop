from __future__ import annotations

import os
import csv
from io import BytesIO
from typing import Optional
from pathlib import Path

import pandas as pd
import streamlit as st
import pdfplumber
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

st.set_page_config(page_title="Supplier → TOW Mapper (Cloud DB)", layout="wide")

# =============================================================================
# Small utils
# =============================================================================
def _log(msg: str):
    if st.session_state.get("_debug", False):
        st.caption(f"🔎 {msg}")

def df_head(df: pd.DataFrame, n: int = 200):
    if isinstance(df, pd.DataFrame):
        return df.head(n)
    return df

def df_read_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params or {})

# =============================================================================
# Load DB engine
# =============================================================================
def get_engine() -> Engine:
    db_url = os.getenv("DATABASE_URL") or st.secrets.get("DATABASE_URL", "")
    if not db_url:
        st.error("DATABASE_URL not configured. Set it in Streamlit Secrets or environment.")
        st.stop()
    return create_engine(db_url, pool_pre_ping=True)

engine = get_engine()

# =============================================================================
# Schema ensure & helpers
# =============================================================================
def ensure_schema(engine: Engine) -> tuple[str, str, Optional[str]]:
    """
    Ensure crosswalk + queue tables exist. Returns info text for banner.
    """
    created = []
    with engine.begin() as conn:
        # crosswalk
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS crosswalk (
            tow_code    TEXT NOT NULL,
            supplier_id TEXT NOT NULL,
            vendor_id   TEXT NOT NULL
        )
        """))
        # Unique (vendor_id, supplier_id)
        conn.execute(text("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'uq_crosswalk_vendor_supplier'
                  AND conrelid = 'crosswalk'::regclass
            ) THEN
                ALTER TABLE crosswalk
                ADD CONSTRAINT uq_crosswalk_vendor_supplier
                UNIQUE (vendor_id, supplier_id);
            END IF;
        END
        $$;
        """))
        # queue table (optional workflow)
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS mapping_queue (
            qid SERIAL PRIMARY KEY,
            vendor_id   TEXT NOT NULL,
            supplier_id TEXT NOT NULL,
            tow_code    TEXT,
            created_at  TIMESTAMP DEFAULT NOW(),
            status      TEXT DEFAULT 'PENDING'
        )
        """))
    return ("crosswalk", "mapping_queue", None)

CROSSWALK_TBL, QUEUE_TBL, _ = ensure_schema(engine)

# =============================================================================
# Page header / How-to
# =============================================================================
st.title("Supplier → TOW Mapper (Cloud DB)")

with st.expander("How to use", expanded=True):
    st.markdown("""
1. This app maps **vendor + supplier code** to **TOW** using a cloud Postgres crosswalk.  
2. Upload supplier invoice (Excel / CSV / PDF). Choose **Vendor** and **Supplier code** column, then run mapping.  
3. Admin tab: add/queue mappings and run live search on the DB.  
4. Blank vendor is treated as **GLOBAL**, duplicates prevented by unique *(vendor_id, supplier_id)*.
    """)

# Debug toggle
st.toggle("Debug logs", key="_debug", value=False)

# =============================================================================
# 1) Crosswalk banner (approx count)
# =============================================================================
try:
    df_cnt = df_read_sql("SELECT COUNT(*) AS n FROM crosswalk")
    n_rows = int(df_cnt.iloc[0]["n"])
    st.success(f"Crosswalk loaded (rows: **{n_rows:,}**) ✅")
except Exception as e:
    st.warning(f"Could not read crosswalk count: {e}")

# =============================================================================
# 2) Vendor input + file uploader
# =============================================================================
vendor = st.selectbox("Vendor", options=["", "DOB0000025", "DOB0000001", "DOB0009999"], index=0)
st.caption("Leave empty for GLOBAL mappings (applies to all vendors).")

st.subheader("2) Upload supplier invoice (Excel / CSV / PDF)")
uploaded = st.file_uploader("Drag and drop the file here", type=["xlsx", "xls", "csv", "pdf"])

preview_df: pd.DataFrame | None = None
if uploaded is not None:
    try:
        suffix = Path(uploaded.name).suffix.lower()

        if suffix in [".xlsx", ".xls"]:
            preview_df = pd.read_excel(uploaded)
        elif suffix == ".csv":
            # try to sniff separator
            content = uploaded.getvalue().decode("utf-8", errors="replace")
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(content.splitlines()[0])
            sep = dialect.delimiter
            preview_df = pd.read_csv(BytesIO(uploaded.getvalue()), sep=sep)
        elif suffix == ".pdf":
            # very light extraction: try table extraction with pdfplumber
            tables = []
            with pdfplumber.open(BytesIO(uploaded.getvalue())) as pdf:
                for page in pdf.pages:
                    tbl = page.extract_table()
                    if tbl:
                        # first row headers
                        header, *rows = tbl
                        dfp = pd.DataFrame(rows, columns=[str(x) for x in header])
                        tables.append(dfp)
            if tables:
                preview_df = pd.concat(tables, ignore_index=True)
            else:
                st.error("No tables detected in PDF.")
                preview_df = None
        else:
            st.error(f"Unsupported file type: {suffix}")
            preview_df = None

        if preview_df is not None:
            st.caption(f"📄 Columns: {', '.join(map(str, preview_df.columns))}")
            st.dataframe(df_head(preview_df), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        preview_df = None

# =============================================================================
# 3) Map to TOW
# =============================================================================
st.subheader("3) Map to TOW")

if preview_df is not None and isinstance(preview_df, pd.DataFrame) and not preview_df.empty:
    # choose which column contains the SUPPLIER code
    supplier_col = st.selectbox(
        "Which column contains the SUPPLIER code?",
        options=list(preview_df.columns),
        index=0,
    )

    if st.button("Run mapping", type="primary"):
        try:
            df_sup = preview_df.rename(columns={supplier_col: "supplier_id"}).copy()
            df_sup["supplier_id"] = df_sup["supplier_id"].astype(str).str.strip()

            # Prepare params
            # empty vendor means global
            vparam = vendor.strip() if vendor else ""
            # Query join
            with engine.connect() as conn:
                # Try vendor-specific first; if not found use GLOBAL ('')
                # vendor hit
                q_vendor = text("""
                    SELECT c.vendor_id, c.supplier_id, c.tow_code
                    FROM crosswalk c
                    WHERE (c.vendor_id = :vparam OR c.vendor_id = '')
                """)
                df_map = pd.read_sql(q_vendor, conn, params={"vparam": vparam})

            # Build a mapping dict keyed by (vendor_id, supplier_id) and ('', supplier_id)
            # Priority: exact vendor, fallback to global ''
            vendor_dict = {(r["vendor_id"], r["supplier_id"]): r["tow_code"] for _, r in df_map.iterrows()}
            def resolve_tow(supp_code: str) -> str | None:
                s = str(supp_code)
                # exact vendor
                if (vparam, s) in vendor_dict:
                    return vendor_dict[(vparam, s)]
                # global
                if ("", s) in vendor_dict:
                    return vendor_dict[("", s)]
                return None

            df_out = df_sup.copy()
            df_out["tow"] = df_out["supplier_id"].map(resolve_tow)

            matched = df_out[df_out["tow"].notna()].copy()
            unmatched = df_out[df_out["tow"].isna()].copy()

            st.success(f"Mapping complete → matched: {len(matched):,} | unmatched: {len(unmatched):,}")

            with st.expander("Preview: Matched (first 200 rows)", expanded=False):
                st.dataframe(df_head(matched, 200), use_container_width=True)
            with st.expander("Preview: Unmatched (first 200 rows)", expanded=False):
                st.dataframe(df_head(unmatched, 200), use_container_width=True)

            # === Old/simple export (zadržavamo ga) ==============================
            def to_excel_bytes(d: dict) -> bytes:
                bio = BytesIO()
                with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
                    for sheet, df in d.items():
                        df.to_excel(w, index=False, sheet_name=sheet)
                return bio.getvalue()

            st.download_button(
                "Download Excel (Matched + Unmatched)",
                data=to_excel_bytes({"Matched": matched, "Unmatched": unmatched}),
                file_name="mapping_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # ==== 4) Custom Excel export (choose columns & order) =========================
            st.subheader("4) Custom Excel export")

            def _excel_from_dict(dfs: dict[str, pd.DataFrame]) -> bytes:
                """
                Build an XLSX from a dict of DataFrames {sheet_name: df}.
                """
                bio2 = BytesIO()
                with pd.ExcelWriter(bio2, engine="xlsxwriter") as w:
                    for sheet, df in dfs.items():
                        if isinstance(df, pd.DataFrame) and not df.empty:
                            df.reset_index(drop=True).to_excel(w, index=False, sheet_name=sheet)
                return bio2.getvalue()

            tabs = st.tabs(["Matched", "Unmatched", "Both (custom)"])

            # --- Tab 1: Matched only ---
            with tabs[0]:
                if isinstance(matched, pd.DataFrame) and not matched.empty:
                    all_cols_m = list(matched.columns)
                    st.caption("Odaberi kolone (redoslijed = redoslijed u multiselectu).")
                    cols_m = st.multiselect(
                        "Columns to export (Matched)",
                        options=all_cols_m,
                        default=all_cols_m,
                        key="custom_cols_matched",
                    )
                    dfm = matched[cols_m] if cols_m else matched
                    st.dataframe(dfm.head(30), use_container_width=True, height=240)
                    st.download_button(
                        "⬇️ Download Matched (custom columns)",
                        data=_excel_from_dict({"Matched": dfm}),
                        file_name="matched_custom.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_matched_custom",
                    )
                else:
                    st.info("Nema Matched podataka za export.")

            # --- Tab 2: Unmatched only ---
            with tabs[1]:
                if isinstance(unmatched, pd.DataFrame) and not unmatched.empty:
                    all_cols_u = list(unmatched.columns)
                    st.caption("Odaberi kolone (redoslijed = redoslijed u multiselectu).")
                    cols_u = st.multiselect(
                        "Columns to export (Unmatched)",
                        options=all_cols_u,
                        default=all_cols_u,
                        key="custom_cols_unmatched",
                    )
                    dfu = unmatched[cols_u] if cols_u else unmatched
                    st.dataframe(dfu.head(30), use_container_width=True, height=240)
                    st.download_button(
                        "⬇️ Download Unmatched (custom columns)",
                        data=_excel_from_dict({"Unmatched": dfu}),
                        file_name="unmatched_custom.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_unmatched_custom",
                    )
                else:
                    st.info("Nema Unmatched podataka za export.")

            # --- Tab 3: Both sheets (svaki sa svojim izborom kolona) ---
            with tabs[2]:
                has_any = (isinstance(matched, pd.DataFrame) and not matched.empty) or \
                          (isinstance(unmatched, pd.DataFrame) and not unmatched.empty)
                if has_any:
                    cols_m_both = st.session_state.get("custom_cols_matched", list(getattr(matched, "columns", [])))
                    cols_u_both = st.session_state.get("custom_cols_unmatched", list(getattr(unmatched, "columns", [])))

                    dfm_both = matched[cols_m_both] if isinstance(matched, pd.DataFrame) and not matched.empty and cols_m_both else matched
                    dfu_both = unmatched[cols_u_both] if isinstance(unmatched, pd.DataFrame) and not unmatched.empty and cols_u_both else unmatched

                    data_dict = {}
                    if isinstance(dfm_both, pd.DataFrame) and dfm_both is not None and not dfm_both.empty:
                        data_dict["Matched"] = dfm_both
                    if isinstance(dfu_both, pd.DataFrame) and dfu_both is not None and not dfu_both.empty:
                        data_dict["Unmatched"] = dfu_both

                    st.download_button(
                        "⬇️ Download Both (custom columns & order)",
                        data=_excel_from_dict(data_dict),
                        file_name="mapping_custom.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="dl_both_custom",
                    )
                else:
                    st.info("Nema podataka za zajednički export.")

        except Exception as e:
            st.error(f"Mapping failed: {e}")
else:
    st.info("Upload your supplier invoice to enable mapping.")

# =============================================================================
# Admin helpers (upsert/queue)
# =============================================================================
st.divider()
st.subheader("Admin: Add / Queue / Apply Mappings + Live search")

# --- Add a single mapping (upsert) -------------------------------------------
with st.form("add_single"):
    colA, colB, colC = st.columns(3)
    with colA:
        vendor_in = st.text_input("Vendor ('' = GLOBAL)", value=st.session_state.get("prefill_vendor_id", vendor or ""))
    with colB:
        supp_in = st.text_input("Supplier code", value=st.session_state.get("prefill_supplier_id", ""))
    with colC:
        tow_in = st.text_input("TOW code", value=st.session_state.get("prefill_tow_code", ""))

    submitted = st.form_submit_button("Upsert mapping")
    if submitted:
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO crosswalk (vendor_id, supplier_id, tow_code)
                    VALUES (:v, :s, :t)
                    ON CONFLICT (vendor_id, supplier_id)
                    DO UPDATE SET tow_code = EXCLUDED.tow_code
                """), {"v": vendor_in.strip(), "s": supp_in.strip(), "t": tow_in.strip()})
            st.success("Mapping upserted.")
        except Exception as e:
            st.error(f"Failed to upsert: {e}")

# --- Queue multiple -----------------------------------------------------------
st.caption("Queue multiple mappings from CSV (columns: vendor_id, supplier_id, tow_code)")
queue_file = st.file_uploader("Upload CSV to queue mappings", type=["csv"], key="queue_csv")
if queue_file is not None:
    try:
        dfq = pd.read_csv(queue_file)
        st.dataframe(df_head(dfq), use_container_width=True, height=200)
        if st.button("Insert into queue"):
            with engine.begin() as conn:
                for _, r in dfq.iterrows():
                    conn.execute(text("""
                        INSERT INTO mapping_queue (vendor_id, supplier_id, tow_code, status)
                        VALUES (:v, :s, :t, 'PENDING')
                    """), {"v": str(r.get("vendor_id","")), "s": str(r.get("supplier_id","")), "t": str(r.get("tow_code",""))})
            st.success("Queued.")
    except Exception as e:
        st.error(f"Queue import failed: {e}")

# --- Apply queue --------------------------------------------------------------
if st.button("Apply queue → crosswalk"):
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO crosswalk (vendor_id, supplier_id, tow_code)
                SELECT vendor_id, supplier_id, COALESCE(tow_code, '')
                FROM mapping_queue
                WHERE status = 'PENDING'
                ON CONFLICT (vendor_id, supplier_id)
                DO UPDATE SET tow_code = EXCLUDED.tow_code
            """))
            conn.execute(text("UPDATE mapping_queue SET status='APPLIED' WHERE status='PENDING'"))
        st.success("Queue applied.")
    except Exception as e:
        st.error(f"Failed applying queue: {e}")

# --- Live search --------------------------------------------------------------
with st.expander("Live search crosswalk", expanded=False):
    q_vendor = st.text_input("Vendor ('' = GLOBAL) — filter", value="")
    q_supplier = st.text_input("Supplier contains — filter", value="")
    q_tow = st.text_input("TOW contains — filter", value="")

    clauses = []
    params = {}
    if q_vendor != "":
        clauses.append("vendor_id = :v1")
        params["v1"] = q_vendor
    if q_supplier != "":
        clauses.append("supplier_id ILIKE :s1")
        params["s1"] = f"%{q_supplier}%"
    if q_tow != "":
        clauses.append("tow_code ILIKE :t1")
        params["t1"] = f"%{q_tow}%"

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    q = f"SELECT vendor_id, supplier_id, tow_code FROM crosswalk{where} ORDER BY vendor_id, supplier_id LIMIT 500"
    df_res = df_read_sql(q, params)
    st.caption(f"{len(df_res)} result(s) shown (max 500)")
    st.dataframe(df_res, use_container_width=True, height=260)

    if not df_res.empty:
        with st.form("prefill_form"):
            idx = st.number_input("Pick row # to prefill", min_value=0, max_value=len(df_res)-1, step=1, value=0)
            if st.form_submit_button("Prefill Admin form from row"):
                row = df_res.iloc[int(idx)]
                st.session_state["prefill_vendor_id"] = str(row.get("vendor_id","") or "")
                st.session_state["prefill_supplier_id"] = str(row.get("supplier_id","") or "")
                st.session_state["prefill_tow_code"] = str(row.get("tow_code","") or "")
                st.success("Prefilled. Scroll up to 'Add a single mapping'.")
