from __future__ import annotations

import os
import csv
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import streamlit as st

# PDF ekstrakcija je opcionalna
try:
    import pdfplumber  # type: ignore
    _HAS_PDF = True
except Exception:
    _HAS_PDF = False

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

st.set_page_config(page_title="Supplier → TOW Mapper (Cloud DB)", layout="wide")


# =============================================================================
# Utility
# =============================================================================
def _excel_bytes(dfs: Dict[str, pd.DataFrame]) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
        for sheet, df in dfs.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.reset_index(drop=True).to_excel(w, index=False, sheet_name=sheet)
    return bio.getvalue()


def _engine() -> Engine:
    db_url = os.getenv("DATABASE_URL") or st.secrets.get("DATABASE_URL", "")
    if not db_url:
        st.error("DATABASE_URL nije postavljen (Secrets ili env).")
        st.stop()
    return create_engine(db_url, pool_pre_ping=True)


engine = _engine()


@st.cache_data(show_spinner=False)
def _crosswalk_count() -> int:
    with engine.connect() as conn:
        n = pd.read_sql(text("SELECT COUNT(*) AS n FROM crosswalk"), conn).iloc[0]["n"]
    return int(n)


@st.cache_data(show_spinner=False)
def _fetch_vendors(filter_q: str = "", limit: int = 500) -> List[str]:
    q = """
        SELECT DISTINCT vendor_id
        FROM crosswalk
        WHERE vendor_id IS NOT NULL
          AND ( :q = '' OR vendor_id ILIKE :like )
        ORDER BY vendor_id
        LIMIT :lim
    """
    params = {"q": filter_q.strip(), "like": f"%{filter_q.strip()}%", "lim": limit}
    with engine.connect() as conn:
        df = pd.read_sql(text(q), conn, params=params)
    vals = [str(x or "") for x in df["vendor_id"].tolist()]
    # GLOBAL (blank) always first
    if "" not in vals:
        vals.insert(0, "")
    else:
        # move '' to front
        vals = [""] + [v for v in vals if v != ""]
    return vals


def _read_sql(query: str, params: dict | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params or {})


# =============================================================================
# Header
# =============================================================================
st.title("Supplier → TOW Mapper (Cloud DB)")
with st.expander("How to use", expanded=False):
    st.markdown("""
1) Učitaj račun (Excel/CSV/PDF).  
2) Odaberi kolonu sa **supplier code** i pokreni mapping.  
3) Export: klasični (Matched+Unmatched) ili **Custom** (odabir kolona i redoslijeda).  
4) Admin (PIN): upsert, queue, apply i live search.
""")

st.toggle("Debug logs", key="_debug", value=False)

try:
    st.success(f"Crosswalk loaded (rows: { _crosswalk_count():, }) ✅")
except Exception as e:
    st.warning(f"Ne mogu pročitati broj redaka crosswalka: {e}")


# =============================================================================
# Vendor select: filter + refresh (dinamički iz baze)
# =============================================================================
st.markdown("**Vendor**")
cc1, cc2 = st.columns([3, 1])
with cc1:
    vendor_filter = st.text_input("Filter vendors (substring / prefix)", value="", key="vendor_filter")
with cc2:
    if st.button("Refresh list", key="btn_refresh_vendors"):
        st.cache_data.clear()  # osvježi cache za vendor listu

vendors = _fetch_vendors(vendor_filter)
# zadrži prethodni odabir ako još postoji
prev_vendor = st.session_state.get("vendor_select", "")
if prev_vendor not in vendors:
    prev_vendor = ""  # pad na GLOBAL

vendor = st.selectbox(
    " ", options=vendors,
    index=vendors.index(prev_vendor) if prev_vendor in vendors else 0,
    key="vendor_select",
    format_func=lambda v: "GLOBAL (blank)" if v == "" else v,
    label_visibility="collapsed",
)
st.caption("Ostavi prazno za GLOBAL mapiranja (vrijedi za sve vendore).")


# =============================================================================
# Upload datoteke
# =============================================================================
st.subheader("1) Upload supplier invoice (Excel / CSV / PDF)")
uploaded = st.file_uploader("Drag & drop ili odaberi datoteku", type=["xlsx", "xls", "csv", "pdf"], key="uploader")

preview_df: pd.DataFrame | None = None
if uploaded is not None:
    try:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix in [".xlsx", ".xls"]:
            preview_df = pd.read_excel(uploaded)
        elif suffix == ".csv":
            content = uploaded.getvalue().decode("utf-8", errors="replace")
            try:
                dialect = csv.Sniffer().sniff(content.splitlines()[0])
                sep = dialect.delimiter
            except Exception:
                sep = ","
            preview_df = pd.read_csv(BytesIO(uploaded.getvalue()), sep=sep)
        elif suffix == ".pdf":
            if not _HAS_PDF:
                st.error("PDF parsing nije omogućen (pdfplumber nije instaliran).")
            else:
                tables = []
                with pdfplumber.open(BytesIO(uploaded.getvalue())) as pdf:
                    for page in pdf.pages:
                        try:
                            tbl = page.extract_table()
                            if tbl:
                                header, *rows = tbl
                                tables.append(pd.DataFrame(rows, columns=[str(x) for x in header]))
                        except Exception:
                            continue
                if tables:
                    preview_df = pd.concat(tables, ignore_index=True)
                else:
                    st.error("Nije detektirana tablica u PDF-u.")
        if isinstance(preview_df, pd.DataFrame):
            st.caption(f"📄 Kolone: {', '.join(map(str, preview_df.columns))}")
            st.dataframe(preview_df.head(200), use_container_width=True)
    except Exception as e:
        st.error(f"Greška pri čitanju datoteke: {e}")
        preview_df = None


# =============================================================================
# 2) Map to TOW  (rezultat zaključavamo u session_state)
# =============================================================================
st.subheader("2) Map to TOW")

# Tipke za kontrolu stanja rezultata
c1, c2 = st.columns([1, 1])
with c1:
    if st.button("Clear result", key="btn_clear_map"):
        for k in ["matched", "unmatched", "matched_cols", "unmatched_cols", "mapped_ready", "map_locked"]:
            st.session_state.pop(k, None)
with c2:
    st.session_state["map_locked"] = st.checkbox(
        "Lock mapping result (prevent reruns from clearing)",
        value=st.session_state.get("map_locked", True),
        key="chk_lock",
    )

if isinstance(preview_df, pd.DataFrame) and not preview_df.empty:
    supplier_col = st.selectbox(
        "Koja kolona sadrži SUPPLIER code?",
        options=list(preview_df.columns),
        index=0,
        key="supplier_col_select",
    )

    if st.button("Run mapping", type="primary", key="btn_run_mapping"):
        try:
            df_sup = preview_df.rename(columns={supplier_col: "supplier_id"}).copy()
            df_sup["supplier_id"] = df_sup["supplier_id"].astype(str).str.strip()
            vparam = (vendor or "").strip()

            df_map = _read_sql("""
                SELECT vendor_id, supplier_id, tow_code
                FROM crosswalk
                WHERE vendor_id = :v OR vendor_id = ''
            """, {"v": vparam})

            lookup = {(r["vendor_id"], r["supplier_id"]): r["tow_code"] for _, r in df_map.iterrows()}

            def resolve_tow(supp_code: str) -> Optional[str]:
                s = str(supp_code)
                return lookup.get((vparam, s)) or lookup.get(("", s)) or None

            df_out = df_sup.copy()
            df_out["tow"] = df_out["supplier_id"].map(resolve_tow)

            st.session_state["matched"] = df_out[df_out["tow"].notna()].copy()
            st.session_state["unmatched"] = df_out[df_out["tow"].isna()].copy()
            st.session_state["matched_cols"] = list(st.session_state["matched"].columns)
            st.session_state["unmatched_cols"] = list(st.session_state["unmatched"].columns)
            st.session_state["mapped_ready"] = True
            st.session_state["map_locked"] = True  # auto-lock nakon mapiranja
            st.toast("Mapping complete ✅", icon="✅")
        except Exception as e:
            st.error(f"Mapping failed: {e}")
            st.session_state["mapped_ready"] = False

# Prikaz rezultata (ostaje vidljiv dok god je mapped_ready i lock aktivan)
if st.session_state.get("mapped_ready", False):
    matched = st.session_state["matched"]
    unmatched = st.session_state["unmatched"]

    st.success(f"Mapping complete → matched: {len(matched):,} | unmatched: {len(unmatched):,}")

    with st.expander("Preview: Matched (first 200 rows)", expanded=False):
        st.dataframe(matched.head(200), use_container_width=True)
    with st.expander("Preview: Unmatched (first 200 rows)", expanded=False):
        st.dataframe(unmatched.head(200), use_container_width=True)

    # Stari export
    st.download_button(
        "Download Excel (Matched + Unmatched)",
        data=_excel_bytes({"Matched": matched, "Unmatched": unmatched}),
        file_name="mapping_result.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="dl_simple_both",
    )

    # Custom export (multiselecti ne ruše rezultat)
    st.subheader("3) Custom Excel export")

    tabs = st.tabs(["Matched", "Unmatched", "Both (custom)"])

    with tabs[0]:
        all_cols_m = st.session_state["matched_cols"]
        cols_m = st.multiselect(
            "Columns to export (Matched)",
            options=all_cols_m,
            default=st.session_state.get("sel_cols_matched", all_cols_m),
            key="sel_cols_matched",
        )
        dfm = matched[cols_m] if cols_m else matched
        st.dataframe(dfm.head(30), use_container_width=True, height=240)
        st.download_button(
            "⬇️ Download Matched (custom columns)",
            data=_excel_bytes({"Matched": dfm}),
            file_name="matched_custom.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_matched_custom",
        )

    with tabs[1]:
        all_cols_u = st.session_state["unmatched_cols"]
        cols_u = st.multiselect(
            "Columns to export (Unmatched)",
            options=all_cols_u,
            default=st.session_state.get("sel_cols_unmatched", all_cols_u),
            key="sel_cols_unmatched",
        )
        dfu = unmatched[cols_u] if cols_u else unmatched
        st.dataframe(dfu.head(30), use_container_width=True, height=240)
        st.download_button(
            "⬇️ Download Unmatched (custom columns)",
            data=_excel_bytes({"Unmatched": dfu}),
            file_name="unmatched_custom.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_unmatched_custom",
        )

    with tabs[2]:
        cols_m_both = st.session_state.get("sel_cols_matched", all_cols_m)
        cols_u_both = st.session_state.get("sel_cols_unmatched", all_cols_u)
        data_dict: Dict[str, pd.DataFrame] = {}
        if not matched.empty:
            data_dict["Matched"] = matched[cols_m_both] if cols_m_both else matched
        if not unmatched.empty:
            data_dict["Unmatched"] = unmatched[cols_u_both] if cols_u_both else unmatched

        st.download_button(
            "⬇️ Download Both (custom columns & order)",
            data=_excel_bytes(data_dict),
            file_name="mapping_custom.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_both_custom",
        )
else:
    st.info("Učitaj datoteku i pokreni mapping.")


# =============================================================================
# Admin (PIN gate)
# =============================================================================
st.divider()
st.subheader("Admin - Add / Queue / Apply Mappings - Live search")

# --- PIN gate (kao u originalu: input + Unlock gumb) ---
c_pin1, c_pin2 = st.columns([5, 1])
with c_pin1:
    pin_in = st.text_input("Admin PIN", type="password", key="admin_pin_input")
with c_pin2:
    if st.button("Unlock", key="btn_unlock"):
        st.session_state["admin_unlocked"] = (pin_in == st.secrets.get("ADMIN_PIN", ""))

if not st.session_state.get("admin_unlocked", False):
    st.info("Admin locked. Unesi PIN i klikni **Unlock**.")
    st.stop()

st.success("Admin unlocked.")

# ============== ADD A SINGLE MAPPING (Queue CSV ili Direct upsert) ===========
st.markdown("### Add a single mapping")

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
        v = (vendor_in or "").strip()
        s = (supp_in or "").strip()
        t = (tow_in or "").strip()

        if not s:
            st.error("supplier_id is required.")
        else:
            if action.startswith("Queue"):
                # držimo queue u session_state i nudimo CSV download – kao u originalu
                queue_cols = ["vendor_id", "supplier_id", "tow_code"]
                if "updates_df" not in st.session_state:
                    st.session_state["updates_df"] = pd.DataFrame(columns=queue_cols)
                st.session_state["updates_df"] = pd.concat(
                    [st.session_state["updates_df"], pd.DataFrame([{"vendor_id": v, "supplier_id": s, "tow_code": t}])],
                    ignore_index=True
                )
                st.success("Queued. (updates.csv u dnu sekcije)")
            else:
                # direktno u NEON (upsert)
                try:
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO crosswalk (vendor_id, supplier_id, tow_code)
                            VALUES (:v, :s, :t)
                            ON CONFLICT (vendor_id, supplier_id)
                            DO UPDATE SET tow_code = EXCLUDED.tow_code
                        """), {"v": v, "s": s, "t": t})
                    st.success("Upsert OK.")
                except Exception as e:
                    st.error(f"DB error: {e}")

# --- Queue CSV “widget” (download / clear / apply) ---------------------------
st.markdown("### Apply queued CSV to DB")

if "updates_df" in st.session_state and not st.session_state["updates_df"].empty:
    dfq = st.session_state["updates_df"].copy()
    st.dataframe(dfq, use_container_width=True, height=200)

    # Download updates.csv
    qcsv = dfq.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download updates.csv",
        data=qcsv,
        file_name="updates.csv",
        mime="text/csv",
        key="dl_updates_csv"
    )

    c_apply1, c_apply2 = st.columns([1, 1])
    with c_apply1:
        if st.button("Apply updates.csv to DB", key="btn_apply_updates"):
            try:
                with engine.begin() as conn:
                    for _, r in dfq.iterrows():
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
                # po želji: očisti queue nakon apply-a
                # st.session_state["updates_df"] = pd.DataFrame(columns=["vendor_id","supplier_id","tow_code"])
            except Exception as e:
                st.error(f"Apply failed: {e}")
    with c_apply2:
        if st.button("Clear queued items", key="btn_clear_updates"):
            st.session_state["updates_df"] = pd.DataFrame(columns=["vendor_id", "supplier_id", "tow_code"])
            st.info("Queue cleared.")
else:
    st.caption("No updates.csv yet.")

# ======================== LIVE SEARCH / INSPECT ==============================
st.markdown("### Live search / inspect")

c_f1, c_f2, c_f3 = st.columns([2.2, 2.2, 1.2])
with c_f1:
    vendor_filter = st.text_input("vendor_id filter (blank = ALL)", value=st.session_state.get("ls_vendor", ""))
with c_f2:
    supp_filter = st.text_input("supplier_id search (exact or contains)", value=st.session_state.get("ls_supplier", ""))
with c_f3:
    exact = st.checkbox("Exact supplier match", value=st.session_state.get("ls_exact", False))

st.session_state["ls_vendor"] = vendor_filter
st.session_state["ls_supplier"] = supp_filter
st.session_state["ls_exact"] = exact

# build WHERE
clauses, params = [], {}
if vendor_filter != "":
    clauses.append("vendor_id = :v1")
    params["v1"] = vendor_filter
if supp_filter != "":
    if exact:
        clauses.append("supplier_id = :s1")
        params["s1"] = supp_filter
    else:
        clauses.append("supplier_id ILIKE :s1")
        params["s1"] = f"%{supp_filter}%"

where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
q = f"""
    SELECT vendor_id, supplier_id, tow_code
    FROM crosswalk
    {where}
    ORDER BY vendor_id, supplier_id
    LIMIT 500
"""

try:
    df_res = _read_sql(q, params)
    st.caption(f"500 results shown (max 500) — shown: {len(df_res):,}")
    st.dataframe(df_res, use_container_width=True, height=260)

    # Prefill admin form
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
                st.success("Prefilled. Scroll up to 'Add a single mapping'.")
except Exception as e:
    st.error(f"Search failed: {e}")
