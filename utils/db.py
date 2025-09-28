from __future__ import annotations
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import pandas as pd
from typing import Iterable
from .config import settings

_engine: Engine | None = None

def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(settings.database_url, pool_pre_ping=True)
    return _engine

def fetch_df(sql: str, params: dict | None = None) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

def execute(sql: str, params: dict | None = None):
    with get_engine().begin() as conn:
        conn.execute(text(sql), params or {})

def batch_upsert_crosswalk(rows: list[dict], current_user: str = "system"):
    if not rows:
        return 0
    # Expect keys: vendor_id, supplier_id, tow_code
    with get_engine().begin() as conn:
        conn.execute(text("SET LOCAL app.current_user = :u"), {"u": current_user})
        conn.execute(
            text("""
            INSERT INTO crosswalk (vendor_id, supplier_id, tow_code)
            VALUES
            """ + ",\n".join(
                f"(:v{i}, :s{i}, :t{i})" for i in range(len(rows))
            ) +
            """
            ON CONFLICT (vendor_id, supplier_id)
            DO UPDATE SET tow_code = EXCLUDED.tow_code, updated_at = now()
            """),
            {
                **{f"v{i}": r.get("vendor_id","") for i,r in enumerate(rows)},
                **{f"s{i}": r.get("supplier_id","") for i,r in enumerate(rows)},
                **{f"t{i}": r.get("tow_code","") for i,r in enumerate(rows)},
            }
        )
    return len(rows)
