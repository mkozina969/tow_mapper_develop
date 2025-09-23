from __future__ import annotations

import os
import csv
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import streamlit as st

# PDF parsing (optional)
try:
    import pdfplumber  # type: ignore
    _HAS_PDF = True
except Exception:
    _HAS_PDF = False

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

import streamlit_sortables

st.set_page_config(page_title="Supplier → TOW Mapper (Developer DB)", layout="wide")

# =============================================================================
# Helpers (PURE DATA ONLY; cache allowed)
# =============================================================================
@st.cache_data(show_spinner=False)
def _crosswalk_count() -> int:
    return int(_read_sql("SELECT COUNT(*) AS n FROM crosswalk").iloc[0]["n"])

@st.cache_data(show_spinner=False)
def _fetch_vendors(filter_q: str = "", limit: int = 500) -> List[str]:
    filter_q = (filter_q or "").strip()
    q = """
        SELECT DISTINCT vendor_id
        FROM crosswalk
        WHERE vendor_id ILIKE :q OR :q = ''
        ORDER BY vendor_id
        LIMIT :limit
    """
    df = _read_sql(q, {"q": f"%{filter_q}%", "limit": int(limit)})
    vals = [str(x or "") for x in df["vendor_id"].tolist()]
    if "" not in vals:
        vals.insert(0, "")
    else:
        vals = [""] + [v for v in vals if v != ""]
    return vals

@st.cache_data(show_spinner=False)
def _load_vendor_names() -> Dict[str, str]:
    try:
        df = _read_sql("SELECT vendor_id, vendor_name FROM vendors")
    except Exception:
        return {}
    df["vendor_id"] = df["vendor_id"].astype(str).str.strip().str.upper()
    df["vendor_name"] = df["vendor_name"].astype(str).str.strip()
    return dict(zip(df["vendor_id"], df["vendor_name"]))

@st.cache_data(show_spinner=False)
def _read_sql(q: str, params: Optional[dict] = None) -> pd.DataFrame:
    engine = _get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(q), conn, params=params or {})

@st.cache_resource(show_spinner=False)
def _get_engine() -> Engine:
    dsn = os.getenv("DB_URL", "postgresql+psycopg2://dev:dev@localhost:5432/tow_dev")
    return create_engine(dsn, future=True, pool_pre_ping=True)

def _excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            safe = (name or "Sheet1")[:31]
            df.to_excel(writer, sheet_name=safe, index=False)
    return bio.getvalue()

def _normalize_suppliers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].apply(lambda x: x if pd.isna(x) else str(x).strip())
    return out

def _merge_mapping(supplier_df: pd.DataFrame, crosswalk_df: pd.DataFrame) -> pd.DataFrame:
    supplier_df = supplier_df.copy()
    # normalize keys
    if "supplier_id" in supplier_df.columns:
        supplier_df["supplier_id"] = supplier_df["supplier_id"].astype(str).str.strip()
    if "vendor_id" in supplier_df.columns:
        supplier_df["vendor_id"] = supplier_df["vendor_id"].astype(str).str.strip().str.upper()
    crosswalk_df = crosswalk_df.copy()
    crosswalk_df["supplier_id"] = crosswalk_df["supplier_id"].astype(str).str.strip()
    crosswalk_df["vendor_id"] = crosswalk_df["vendor_id"].astype(str).str.strip().str.upper()

    # left join to get TOW codes for matches
    merged = supplier_df.merge(
        crosswalk_df[["vendor_id", "supplier_id", "tow_code"]],
        how="left",
        on=["vendor_id", "supplier_id"],
        suffixes=("", "_cw"),
    )
    return merged

def _parse_uploaded_file(file) -> pd.DataFrame:
    suffix = Path(file.name).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file)
    elif suffix == ".csv":
        return pd.read_csv(file)
    elif suffix == ".pdf" and _HAS_PDF:
        rows: List[dict] = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                try:
                    tbls = page.extract_tables()
                    for tbl in tbls or []:
                        if not tbl or len(tbl) < 2:
                            continue
                        headers = [str(h or "").strip() for h in tbl[0]]
                        for row in tbl[1:]:
                            d = {headers[i] if i < len(headers) else f"col{i}": row[i] for i in range(len(row))}
                            rows.append(d)
                except Exception:
                    continue
        return pd.DataFrame(rows)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

# =============================================================================
# UI (STATEFUL)
# =============================================================================
st.title("Supplier → TOW Mapper (Developer DB)")

with st.sidebar:
    st.header("DB & Info")
    st.caption(f"Crosswalk rows: **{_crosswalk_count()}**")
    vnames = _load_vendor_names()
    if vnames:
        st.caption(f"Vendors in DB: {len(vnames)}")
    st.caption("Upload: XLSX, XLS, CSV, (PDF optional)")

# ---- Upload ---------------------------------------------------------------
uploaded = st.file_uploader("Drag & drop or browse a file", type=["xlsx", "xls", "csv", "pdf"], accept_multiple_files=False)

if uploaded:
    try:
        raw_df = _parse_uploaded_file(uploaded)
        st.success(f"Loaded {len(raw_df):,} rows.")
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
        st.stop()

    # Ask user which columns are supplier_id and vendor_id
    st.markdown("### 1) Identify key columns")
    col1, col2, col3 = st.columns([1.2, 1.2, 1])
    with col1:
        supplier_col = st.selectbox("supplier_id column", options=list(raw_df.columns), index=0)
    with col2:
        vendor_col = st.selectbox("vendor_id column", options=list(raw_df.columns), index=min(1, len(raw_df.columns)-1))
    with col3:
        # Optional additional filters or notes
        pass

    df_in = raw_df.rename(columns={supplier_col: "supplier_id", vendor_col: "vendor_id"})
    df_in = _normalize_suppliers(df_in)

    # ---- Do mapping --------------------------------------------------------
    st.markdown("### 2) Map to TOW (DB crosswalk)")
    # fetch crosswalk slice for vendors present
    vendors_present = df_in["vendor_id"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    params = {"vendors": tuple(vendors_present) if vendors_present else tuple([""])}
    cw_q = """
        SELECT vendor_id, supplier_id, tow_code
        FROM crosswalk
        WHERE vendor_id = ANY(:vendors)
    """
    try:
        cw = _read_sql(cw_q, {"vendors": vendors_present})
    except Exception:
        # fallback: whole table if driver can't handle ANY with tuples
        cw = _read_sql("SELECT vendor_id, supplier_id, tow_code FROM crosswalk")
        cw = cw[cw["vendor_id"].astype(str).str.upper().isin(vendors_present)]

    merged = _merge_mapping(df_in, cw)

    matched = merged[~merged["tow_code"].isna()].copy()
    unmatched = merged[merged["tow_code"].isna()].copy()

    st.write("**Matched**", len(matched))
    st.dataframe(matched.head(200), use_container_width=True)
    st.write("**Unmatched**", len(unmatched))
    st.dataframe(unmatched.head(200), use_container_width=True)

    # -----------------------------------------------------------------------------
    # 2a) SIMPLE EXPORT
    # -----------------------------------------------------------------------------
    st.download_button(
        "Download Excel (Matched + Unmatched)",
        data=_excel_bytes({"Matched": matched, "Unmatched": unmatched}),
        file_name="mapping_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_simple_both",
    )

    # -----------------------------------------------------------------------------
    # 2b) Custom export (add Invoice/Item/vendor_id/date/location + choose columns & order)
    # -----------------------------------------------------------------------------
    st.subheader("3) Custom export (add columns + choose order)")

    invoice_label = "Invoice"
    item_label = "Item"
    vendor_to_stamp = (vendors_present[0] if len(vendors_present) == 1 else "GLOBAL") or "GLOBAL"
    location_text = "H00"
    all_cols_now = list(dict.fromkeys(list(matched.columns) + list(unmatched.columns)))
    date_choice = "(none)"
    date_manual = ""

    with st.expander("Add optional columns (Invoice, Item, vendor_id, location, date)", expanded=True):
        cA, cB, cC, cD, cE = st.columns([1, 1, 1.2, 1.2, 1.2])
        with cA:
            invoice_label = st.text_input("Constant value for 'Invoice'", value="Invoice")
        with cB:
            item_label = st.text_input("Constant value for 'Item'", value="Item")
        with cC:
            vendor_to_stamp = st.text_input(
                "vendor_id to stamp on export",
                value=vendor_to_stamp
            ).strip().upper() or "GLOBAL"
        with cD:
            location_text = st.text_input("Constant value for 'Location'", value="H00")
        with cE:
            date_manual = st.text_input("Manual Date (YYYY-MM-DD)", value="")

        add_invoice = st.checkbox("Add 'Invoice' column", value=True)
        add_item = st.checkbox("Add 'Item' column", value=True)
        add_vendor = st.checkbox("Add 'vendor_id' column", value=True)
        add_location = st.checkbox("Add 'Location' column", value=True)
        add_date = st.checkbox("Add 'date' column (manual text)", value=True)

    matched_en = matched.copy()
    unmatched_en = unmatched.copy()

    if add_invoice:
        matched_en["Invoice"] = invoice_label
        unmatched_en["Invoice"] = invoice_label
    if add_item:
        matched_en["Item"] = item_label
        unmatched_en["Item"] = item_label
    if add_vendor:
        matched_en["vendor_id"] = vendor_to_stamp
        unmatched_en["vendor_id"] = vendor_to_stamp
    if add_location:
        matched_en["Location"] = location_text
        unmatched_en["Location"] = location_text
    if add_date:
        matched_en["date"] = date_manual if date_manual else ""
        unmatched_en["date"] = date_manual if date_manual else ""

    all_cols = list(dict.fromkeys(list(matched_en.columns) + list(unmatched_en.columns)))
    if "date" not in all_cols and add_date:
        all_cols.append("date")

    preferred_first = [c for c in ["Invoice", "Item", "Location", "date", "vendor_id", "tow_code", "supplier_id", "vendor_id"] if c in all_cols]
    rest = [c for c in all_cols if c not in preferred_first]
    default_order = preferred_first + rest

    # --- remember/rebuild ordering UI state -------------------------
    if "pending_export_cols" not in st.session_state:
        st.session_state["pending_export_cols"] = default_order.copy()
    if "export_cols" not in st.session_state:
        st.session_state["export_cols"] = default_order.copy()
    if "columns_applied" not in st.session_state:
        st.session_state["columns_applied"] = True

    def columns_sortable_with_apply(preferred_order: List[str]) -> List[str]:
        st.markdown("#### Choose columns and order")
        st.caption("Drag horizontally to reorder. Then use the selector to add/remove.")
        sorted_cols = streamlit_sortables.sort_items(
            st.session_state["pending_export_cols"],
            direction="horizontal",
            key="sortable_export_cols"
        )
        # limit options to known columns; preserve preferred order
        all_options = preferred_order.copy()

        def filter_to_options(sel: List[str], options: List[str]) -> List[str]:
            seen = set()
            out = []
            for x in sel:
                if x in options and x not in seen:
                    out.append(x); seen.add(x)
            for x in options:
                if x not in seen:
                    out.append(x); seen.add(x)
            return out

        default_selected = filter_to_options(sorted_cols, all_options)
        selected = st.multiselect(
            "Add/remove columns (order preserved above):",
            options=all_options,
            default=default_selected,
            key="export_cols_selector"
        )

        if st.button("Apply changes", key="btn_apply_cols"):
            st.session_state["pending_export_cols"] = filter_to_options(sorted_cols, all_options)
            st.session_state["export_cols"] = filter_to_options(selected, all_options)
            st.session_state["columns_applied"] = True

        if st.session_state.get("columns_applied", False):
            return st.session_state["export_cols"]
        return default_selected

    if default_order:
        export_cols = columns_sortable_with_apply(default_order)
    else:
        st.warning("No columns available for export/reordering.")
        export_cols = None

    if export_cols:
        def _apply_selection(df: pd.DataFrame) -> pd.DataFrame:
            cols_in_df = [c for c in export_cols if c in df.columns]
            return df[cols_in_df] if cols_in_df else df

        matched_out = _apply_selection(matched_en)
        unmatched_out = _apply_selection(unmatched_en)

        # --- NEW: allow forcing selected columns to TEXT (string) for export ---
        text_cols = st.multiselect(
            "Force these columns to TEXT (strings) in the exported Excel",
            options=export_cols,
            help="Useful for long IDs, product numbers, postal codes, etc. Values will be written as strings."
        )
        def _force_text(df: pd.DataFrame) -> pd.DataFrame:
            if not text_cols:
                return df
            df2 = df.copy()
            for c in text_cols:
                if c in df2.columns:
                    df2[c] = df2[c].astype("string").fillna("")
            return df2
        matched_out = _force_text(matched_out)
        unmatched_out = _force_text(unmatched_out)

        with st.expander("Preview (custom): Matched", expanded=False):
            st.dataframe(matched_out.head(200), use_container_width=True)
        with st.expander("Preview (custom): Unmatched", expanded=False):
            st.dataframe(unmatched_out.head(200), use_container_width=True)

        st.download_button(
            "⬇️ Download Excel (custom columns & order)",
            data=_excel_bytes({"Matched": matched_out, "Unmatched": unmatched_out}),
            file_name="mapping_custom.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_both_custom",
        )
    else:
        st.info("Select columns, drag to reorder, and press 'Apply changes' to update preview/export.")
else:
    st.info("Učitaj datoteku i pokreni mapping.")

# =============================================================================
# Admin (PIN -> unlock) + Queue/Direct upsert
# =============================================================================
st.markdown("---")
st.markdown("### Admin: manage crosswalk")

pin_ok = False
with st.expander("Unlock admin (PIN)"):
    pin = st.text_input("PIN", type="password")
    if st.button("Unlock"):
        if pin == os.getenv("ADMIN_PIN", "1234"):
            pin_ok = True
            st.success("Admin unlocked.")
        else:
            st.error("Wrong PIN.")

if pin_ok:
    st.markdown("#### Add a single mapping")
    with st.form("admin_add_single"):
        c1, c2, c3 = st.columns(3)
        with c1:
            vendor_in = st.text_input("vendor_id (leave blank for GLOBAL)", value=st.session_state.get("vendor_prefill", ""))
        with c2:
            supp_in = st.text_input("supplier_id", value=st.session_state.get("supplier_prefill", ""))
        with c3:
            tow_in = st.text_input("tow_code", value=st.session_state.get("tow_prefill", ""))

        st.caption("Action…")
        action = st.radio(
            label="Action…",
            options=["Queue (downloadable CSV)", "Directly to DB (upsert)"],
            horizontal=True,
            key="admin_add_action"
        )

        submitted = st.form_submit_button("Add")
        if submitted:
            v = (vendor_in or "").strip() or "GLOBAL"
            s = (supp_in or "").strip()
            t = (tow_in or "").strip()

            if not s:
                st.error("supplier_id is required.")
            else:
                if action.startswith("Queue"):
                    queue_cols = ["vendor_id", "supplier_id", "tow_code"]
                    if "updates_df" not in st.session_state:
                        st.session_state["updates_df"] = pd.DataFrame(columns=queue_cols)
                    st.session_state["updates_df"] = pd.concat([
                        st.session_state["updates_df"],
                        pd.DataFrame([{"vendor_id": v, "supplier_id": s, "tow_code": t}])
                    ], ignore_index=True)
                    st.success("Queued. See 'Review & Apply queued updates'.")
                else:
                    try:
                        eng = _get_engine()
                        with eng.begin() as conn:
                            conn.execute(text("""
                                INSERT INTO crosswalk (vendor_id, supplier_id, tow_code)
                                VALUES (:v, :s, :t)
                                ON CONFLICT (vendor_id, supplier_id)
                                DO UPDATE SET tow_code = EXCLUDED.tow_code
                            """), {"v": v, "s": s, "t": t})
                        st.success("Upserted to DB.")
                    except Exception as e:
                        st.error(f"DB upsert failed: {e}")

    st.markdown("#### Review & Apply queued updates")
    c1, c_apply, c_apply2 = st.columns([2, 1, 1])
    with c1:
        upd = st.session_state.get("updates_df", pd.DataFrame(columns=["vendor_id", "supplier_id", "tow_code"]))
        st.dataframe(upd, use_container_width=True)
        st.download_button(
            "Download updates.csv",
            data=upd.to_csv(index=False).encode("utf-8"),
            file_name="updates.csv",
            mime="text/csv",
            key="dl_updates_csv"
        )
    with c_apply:
        if st.button("Apply updates.csv to DB"):
            try:
                eng = _get_engine()
                with eng.begin() as conn:
                    for _, r in upd.iterrows():
                        conn.execute(text("""
                            INSERT INTO crosswalk (vendor_id, supplier_id, tow_code)
                            VALUES (:v, :s, :t)
                            ON CONFLICT (vendor_id, supplier_id)
                            DO UPDATE SET tow_code = EXCLUDED.tow_code
                        """), {
                            "v": str(r.get("vendor_id", "")),
                            "s": str(r.get("supplier_id", "")),
                            "t": str(r.get("tow_code", "")),
                        })
                st.success("updates.csv applied to DB.")
            except Exception as e:
                st.error(f"Apply failed: {e}")
    with c_apply2:
        if st.button("Clear queued items", key="btn_clear_updates"):
            st.session_state["updates_df"] = pd.DataFrame(columns=["vendor_id", "supplier_id", "tow_code"])
            st.success("Cleared.")


# =============================================================================
# Search helper (prefill form)
# =============================================================================
st.markdown("---")
st.markdown("### Quick search (DB) → prefill admin form")
with st.form("search_form"):
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    with c1:
        vendor = st.text_input("vendor_id filter", "")
    with c2:
        supplier = st.text_input("supplier_id filter", "")
    with c3:
        limit = st.number_input("Limit", min_value=1, max_value=1000, value=50)
    with c4:
        search = st.form_submit_button("Search")

if "vendor_prefill" not in st.session_state:
    st.session_state["vendor_prefill"] = ""
if "supplier_prefill" not in st.session_state:
    st.session_state["supplier_prefill"] = ""
if "tow_prefill" not in st.session_state:
    st.session_state["tow_prefill"] = ""

if search:
    try:
        q = """
            SELECT vendor_id, supplier_id, tow_code
            FROM crosswalk
            WHERE (:v = '' OR vendor_id ILIKE :v_like)
              AND (:s = '' OR supplier_id ILIKE :s_like)
            ORDER BY vendor_id, supplier_id
            LIMIT :lim
        """
        params = {
            "v": vendor.strip(),
            "s": supplier.strip(),
            "v_like": f"%{vendor.strip()}%",
            "s_like": f"%{supplier.strip()}%",
            "lim": int(limit),
        }
        df_res = _read_sql(q, params)
        st.dataframe(df_res, use_container_width=True)
        st.markdown("##### Pick row as prefill")
        c_idx, c_btn = st.columns([1, 2])
        with c_idx:
            idx = st.number_input(" ", min_value=0, max_value=max(len(df_res)-1, 0), value=0, step=1, label_visibility="collapsed")
        with c_btn:
            if st.button("Prefill Admin form from row", key="btn_prefill"):
                if not df_res.empty:
                    row = df_res.iloc[int(idx)]
                    st.session_state["vendor_prefill"] = str(row.get("vendor_id", "") or "")
                    st.session_state["supplier_prefill"] = str(row.get("supplier_id", "") or "")
                    st.session_state["tow_prefill"] = str(row.get("tow_code", "") or "")
                    st.success("Prefilled — skrolaj gore do 'Add a single mapping'.")
    except Exception as e:
        st.error(f"Search failed: {e}")
