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
# =============================================================================
# Vendor select (dynamic from DB, '' = GLOBAL at the top)
# =============================================================================
def _load_vendor_list() -> list[str]:
    try:
        with engine.connect() as conn:
            dfv = pd.read_sql(text("SELECT DISTINCT vendor_id FROM crosswalk"), conn)
        vals = sorted(set(str(x or "") for x in dfv["vendor_id"]))
    except Exception:
        vals = []
    # '' (GLOBAL) stavi na vrh
    if "" in vals:
        vals.remove("")
    return [""] + vals  # prvi je GLOBAL

if "vendor_list" not in st.session_state:
    st.session_state.vendor_list = _load_vendor_list()

vendor = st.selectbox(
    "Vendor",
    options=st.session_state.vendor_list,
    format_func=lambda v: "GLOBAL (blank)" if v == "" else v,
    index=0 if "" in st.session_state.vendor_list else 0,
    key="vendor_select",
)
st.caption("Ostavi prazno za GLOBAL mapiranja (vrijedi za sve vendore).")

# =============================================================================

# 3) Map to TOW  (persistent in session_state)
# =============================================================================
st.subheader("3) Map to TOW")

if preview_df is not None and isinstance(preview_df, pd.DataFrame) and not preview_df.empty:
    supplier_col = st.selectbox(
        "Which column contains the SUPPLIER code?",
        options=list(preview_df.columns),
        index=0,
        key="supplier_col_select",
    )

    if st.button("Run mapping", type="primary", key="btn_run_mapping"):
        try:
            df_sup = preview_df.rename(columns={supplier_col: "supplier_id"}).copy()
            df_sup["supplier_id"] = df_sup["supplier_id"].astype(str).str.strip()

            vparam = vendor.strip() if vendor is not None else ""

            with engine.connect() as conn:
                df_map = pd.read_sql(
                    text("""
                        SELECT vendor_id, supplier_id, tow_code
                        FROM crosswalk
                        WHERE vendor_id = :v OR vendor_id = ''
                    """),
                    conn,
                    params={"v": vparam},
                )

            mapp = {(r["vendor_id"], r["supplier_id"]): r["tow_code"] for _, r in df_map.iterrows()}

            def resolve_tow(supp_code: str) -> Optional[str]:
                s = str(supp_code)
                return (
                    mapp.get((vparam, s))
                    or mapp.get(("", s))
                    or None
                )

            df_out = df_sup.copy()
            df_out["tow"] = df_out["supplier_id"].map(resolve_tow)

            st.session_state.matched = df_out[df_out["tow"].notna()].copy()
            st.session_state.unmatched = df_out[df_out["tow"].isna()].copy()
            st.session_state.mapped_ready = True
            st.session_state.matched_cols = list(st.session_state.matched.columns)
            st.session_state.unmatched_cols = list(st.session_state.unmatched.columns)
            st.toast("Mapping complete ✅", icon="✅")
        except Exception as e:
            st.session_state.mapped_ready = False
            st.error(f"Mapping failed: {e}")

# ——— PRIKAZ ako već imamo rezultat u session_state (sprječava reset na widget promjenama)
if st.session_state.get("mapped_ready", False):
    matched = st.session_state.matched
    unmatched = st.session_state.unmatched

    st.success(f"Mapping complete → matched: {len(matched):,} | unmatched: {len(unmatched):,}")

    with st.expander("Preview: Matched (first 200 rows)", expanded=False):
        st.dataframe(matched.head(200), use_container_width=True)
    with st.expander("Preview: Unmatched (first 200 rows)", expanded=False):
        st.dataframe(unmatched.head(200), use_container_width=True)

    # — stari/simpler export ostaje —
    def _to_excel_bytes(dfs: dict[str, pd.DataFrame]) -> bytes:
        bio = BytesIO()
        with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
            for name, df in dfs.items():
                df.to_excel(w, index=False, sheet_name=name)
        return bio.getvalue()

    st.download_button(
        "Download Excel (Matched + Unmatched)",
        data=_to_excel_bytes({"Matched": matched, "Unmatched": unmatched}),
        file_name="mapping_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_simple_both",
    )



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
