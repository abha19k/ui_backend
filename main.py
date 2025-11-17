# main.py
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from starlette.staticfiles import StaticFiles

# === Load forecast modules for all periods ===================================

# Daily: 7-day horizon script (assumed similar API to monthly)
_DAILY_PATH = Path(__file__).with_name("forecast_7d_multi_models_backtest_with_plots_pi.py")
daily_mod = None
try:
    daily_mod = SourceFileLoader("batch7d_mod", str(_DAILY_PATH)).load_module()  # type: ignore
except Exception as _e:
    logging.getLogger("uvicorn.error").warning("Daily batch script import failed: %s", _e)

# Weekly: 13-week horizon script
_WEEKLY_PATH = Path(__file__).with_name("forecast_13w_multi_models_backtest_with_plots_pi.py")
weekly_mod = None
try:
    weekly_mod = SourceFileLoader("batch13w_mod", str(_WEEKLY_PATH)).load_module()  # type: ignore
except Exception as _e:
    logging.getLogger("uvicorn.error").warning("Weekly batch script import failed: %s", _e)

# Monthly: 18-month horizon script
_MONTHLY_PATH = Path(__file__).with_name("forecast_18m_multi_models_backtest_with_plots_pi.py")
bf = None  # monthly module
try:
    bf = SourceFileLoader("batch18m_mod", str(_MONTHLY_PATH)).load_module()  # type: ignore
except Exception as _e:
    logging.getLogger("uvicorn.error").warning("Monthly batch script import failed: %s", _e)

# Helper: map period -> module
def _get_forecast_module(period: Literal["daily", "weekly", "monthly"]):
    if period == "daily":
        if daily_mod is None:
            raise HTTPException(status_code=400, detail="Daily forecast module not available on server.")
        return daily_mod
    if period == "weekly":
        if weekly_mod is None:
            raise HTTPException(status_code=400, detail="Weekly forecast module not available on server.")
        return weekly_mod
    # monthly
    if bf is None:
        raise HTTPException(status_code=400, detail="Monthly forecast module not available on server.")
    return bf


# === App & CORS ===============================================================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === DB ======================================================================
DB_URI = "postgresql://abha:planwise123@localhost:5432/planwise"
engine = create_engine(DB_URI, pool_pre_ping=True)

def fetch_all(query: str, params: dict | None = None):
    with engine.begin() as conn:
        res = conn.execute(text(query), params or {})
        if not res.returns_rows:
            return []
        return [dict(row._mapping) for row in res]


# === Helpers =================================================================
def _tbl_freq(period: Literal["daily", "weekly", "monthly"]) -> Tuple[str, str, str]:
    if period == "daily":
        return "history_daily", "D", "day"
    if period == "weekly":
        return "history_weekly", "W-MON", "week"
    return "history_monthly", "MS", "month"


def _forecast_table(period: Literal["daily", "weekly", "monthly"]) -> str:
    return {"daily": "forecast_daily", "weekly": "forecast_weekly", "monthly": "forecast_monthly"}[period]


def _period_delta(period: Literal["daily", "weekly", "monthly"]) -> timedelta:
    return timedelta(days=1 if period == "daily" else 7 if period == "weekly" else 30)


def _period_end(sdt: datetime, period: Literal["daily", "weekly", "monthly"]) -> datetime:
    if period == "daily":
        return sdt
    if period == "weekly":
        return sdt + timedelta(days=6)
    # monthly
    return (sdt.replace(day=1) + timedelta(days=40)).replace(day=1) - timedelta(days=1)


def _ts_expr(alias: str) -> str:
    """Robust timestamp expression (works for TEXT/DATE/TIMESTAMP)."""
    return f"""
    CASE
      WHEN {alias}."StartDate" IS NULL THEN NULL
      WHEN ({alias}."StartDate")::text ~ '^[0-9]{{2}}/[0-9]{{2}}/[0-9]{{4}}$'
        THEN to_timestamp(({alias}."StartDate")::text, 'DD/MM/YYYY')
      WHEN ({alias}."StartDate")::text ~ '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}'
        THEN ({alias}."StartDate")::timestamp
      ELSE
        ({alias}."StartDate")::timestamp
    END
    """


def _make_field_map(main_alias: str, p_alias: str, c_alias: str, l_alias: str) -> Dict[str, str]:
    return {
        "productid": f'COALESCE({p_alias}."ProductID", {main_alias}."ProductID")',
        "productdescr": f'{p_alias}."ProductDescr"',
        "businessunit": f'{p_alias}."BusinessUnit"',
        "isdailyforecastrequired": f'CAST({p_alias}."IsDailyForecastRequired" AS TEXT)',
        "isnew": f'CAST({p_alias}."IsNew" AS TEXT)',
        "productfamily": f'{p_alias}."ProductFamily"',
        "productlevel": f'CAST({p_alias}."Level" AS TEXT)',
        "channelid": f'COALESCE({c_alias}."ChannelID", {main_alias}."ChannelID")',
        "channeldescr": f'{c_alias}."ChannelDescr"',
        "channellevel": f'CAST({c_alias}."Level" AS TEXT)',
        "locationid": f'COALESCE({l_alias}."LocationID", {main_alias}."LocationID")',
        "locationdescr": f'{l_alias}."Location_Descr"',
        "locationlevel": f'CAST({l_alias}."Level" AS TEXT)',
        "geography": f'{l_alias}."Geography"',
    }


def _to_ilike_pattern(term: str) -> str:
    t = (term or "").replace("%", r"\%").replace("_", r"\_").replace("*", "%")
    return t if ("%" in t or t.startswith("%")) else f"%{t}%"


def _build_where_clause_with_map(q: str, params: dict, field_map: Dict[str, str], bind_prefix: str) -> str:
    import shlex

    def _add(sql_parts: List[str], sql: str, op_next: str):
        if sql:
            if sql_parts:
                sql_parts.append(op_next)
            sql_parts.append(f"({sql})")

    try:
        tokens = shlex.split(q) if q else []
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid query syntax: {e}")

    sql_parts: List[str] = []
    op_next = "AND"
    bind_i = 0

    for t in tokens:
        up = t.upper()
        if up in ("AND", "OR"):
            op_next = up
            continue

        if ":" in t:
            field, raw_value = t.split(":", 1)
            col = field_map.get(field.lower())
            if not col:
                continue
            value = raw_value.strip().strip('"').strip("'")
            if value in ("", "all", "*"):
                _add(sql_parts, "TRUE", op_next)
                continue
            values = [v.strip() for v in value.split(",") if v.strip()]
            if any(v.lower() == "all" or v == "*" for v in values):
                _add(sql_parts, "TRUE", op_next)
                continue
            if len(values) == 1:
                bind_i += 1
                k = f"{bind_prefix}{bind_i}"
                params[k] = _to_ilike_pattern(values[0])
                _add(sql_parts, f"""{col} ILIKE :{k} ESCAPE '\\'""", op_next)
            else:
                ors = []
                for v in values:
                    bind_i += 1
                    k = f"{bind_prefix}{bind_i}"
                    params[k] = _to_ilike_pattern(v)
                    ors.append(f"""{col} ILIKE :{k} ESCAPE '\\'""")
                _add(sql_parts, " OR ".join(ors), op_next)
        else:
            if t.strip().lower() in ("all", "*"):
                _add(sql_parts, "TRUE", op_next)
                continue
            cols = [
                field_map.get("productid"), field_map.get("productdescr"), field_map.get("businessunit"),
                field_map.get("productfamily"), field_map.get("channelid"), field_map.get("channeldescr"),
                field_map.get("locationid"), field_map.get("locationdescr"), field_map.get("geography"),
            ]
            cols = [c for c in cols if c]
            bind_i += 1
            k = f"{bind_prefix}{bind_i}"
            params[k] = _to_ilike_pattern(t)
            ors = [f"""{c} ILIKE :{k} ESCAPE '\\'""" for c in cols]
            _add(sql_parts, " OR ".join(ors), op_next)

    return " ".join(sql_parts) if sql_parts else "TRUE"


# === Startup DDL & indexes ====================================================
@app.on_event("startup")
def startup_ddl():
    logging.getLogger("uvicorn.error").info("Daily module: %s", getattr(daily_mod, "__name__", "unavailable"))
    logging.getLogger("uvicorn.error").info("Weekly module: %s", getattr(weekly_mod, "__name__", "unavailable"))
    logging.getLogger("uvicorn.error").info("Monthly module: %s", getattr(bf, "__name__", "unavailable"))
    with engine.begin() as c:
        # saved searches
        c.execute(text("""
        CREATE TABLE IF NOT EXISTS saved_search(
          id SERIAL PRIMARY KEY,
          name TEXT NOT NULL,
          query TEXT NOT NULL,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """))
        # history indexes
        c.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_hist_daily_pcl   ON history_daily("ProductID","ChannelID","LocationID");
        CREATE INDEX IF NOT EXISTS ix_hist_weekly_pcl  ON history_weekly("ProductID","ChannelID","LocationID");
        CREATE INDEX IF NOT EXISTS ix_hist_monthly_pcl ON history_monthly("ProductID","ChannelID","LocationID");
        CREATE INDEX IF NOT EXISTS brin_hist_daily_start   ON history_daily USING BRIN("StartDate");
        CREATE INDEX IF NOT EXISTS brin_hist_weekly_start  ON history_weekly USING BRIN("StartDate");
        CREATE INDEX IF NOT EXISTS brin_hist_monthly_start ON history_monthly USING BRIN("StartDate");
        """))
        # forecast created_at + unique
        c.execute(text("""
        ALTER TABLE IF EXISTS forecast_daily   ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now();
        ALTER TABLE IF EXISTS forecast_weekly  ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now();
        ALTER TABLE IF EXISTS forecast_monthly ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now();
        """))
        for tbl in ("forecast_daily", "forecast_weekly", "forecast_monthly"):
            c.execute(text(f"""
                WITH dupe AS (
                  SELECT ctid FROM (
                    SELECT ctid,
                           ROW_NUMBER() OVER (
                             PARTITION BY "ProductID","ChannelID","LocationID","StartDate","EndDate","Method"
                             ORDER BY created_at DESC, ctid DESC
                           ) rn
                    FROM {tbl}
                  ) t WHERE rn > 1
                )
                DELETE FROM {tbl} f USING dupe d WHERE f.ctid = d.ctid;
            """))
        c.execute(text("""
        CREATE UNIQUE INDEX IF NOT EXISTS ux_forecast_daily_xgb
          ON forecast_daily("ProductID","ChannelID","LocationID","StartDate","EndDate","Method");
        CREATE UNIQUE INDEX IF NOT EXISTS ux_forecast_weekly_xgb
          ON forecast_weekly("ProductID","ChannelID","LocationID","StartDate","EndDate","Method");
        CREATE UNIQUE INDEX IF NOT EXISTS ux_forecast_monthly_xgb
          ON forecast_monthly("ProductID","ChannelID","LocationID","StartDate","EndDate","Method");
        """))
        # cleanse_profile
        c.execute(text("""
        CREATE TABLE IF NOT EXISTS cleanse_profile (
          id SERIAL PRIMARY KEY,
          name TEXT UNIQUE NOT NULL,
          settings JSONB NOT NULL DEFAULT '{}'::jsonb,
          is_active BOOLEAN NOT NULL DEFAULT FALSE,
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """))
        c.execute(text("""
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name='cleanse_profile' AND column_name='config'
          ) THEN
            ALTER TABLE cleanse_profile ADD COLUMN config JSONB;
          END IF;
        END$$;
        """))
        c.execute(text("""
          ALTER TABLE cleanse_profile
            ALTER COLUMN config SET DEFAULT '{}'::jsonb;
          UPDATE cleanse_profile SET config='{}'::jsonb WHERE config IS NULL;
          ALTER TABLE cleanse_profile
            ALTER COLUMN config SET NOT NULL;
        """))
        # classification results
        c.execute(text("""
        CREATE TABLE IF NOT EXISTS forecast_element_classification (
          ProductID TEXT NOT NULL,
          ChannelID TEXT NOT NULL,
          LocationID TEXT NOT NULL,
          Period TEXT NOT NULL,
          Label TEXT NOT NULL,
          Score DOUBLE PRECISION NOT NULL,
          ComputedAt TIMESTAMPTZ NOT NULL DEFAULT now(),
          IsActive BOOLEAN NOT NULL DEFAULT TRUE,
          PRIMARY KEY (ProductID,ChannelID,LocationID,Period)
        );
        """))

# serve plots dir if present
try:
    app.mount("/plots", StaticFiles(directory="plots"), name="plots")
except Exception:
    pass

# === Models ===================================================================
class Product(BaseModel):
    ProductID: str
    ProductDescr: Optional[str] = None
    Level: Optional[int] = None
    BusinessUnit: Optional[str] = None
    IsDailyForecastRequired: Optional[bool] = None
    IsNew: Optional[bool] = None
    ProductFamily: Optional[str] = None


class Channel(BaseModel):
    ChannelID: str
    ChannelDescr: Optional[str] = None
    Level: Optional[int] = None


class Location(BaseModel):
    LocationID: str
    LocationDescr: Optional[str] = None
    Level: Optional[int] = None
    Geography: Optional[str] = None


class KeyTriplet(BaseModel):
    ProductID: str
    ChannelID: str
    LocationID: str


class SeriesPoint(BaseModel):
    StartDate: str
    Qty: float


# === Basic dropdowns ==========================================================
@app.get("/")
def root():
    return {"message": "Planwise API"}


@app.get("/api/products", response_model=List[Product])
def get_products():
    return fetch_all('SELECT * FROM product')


@app.get("/api/channels", response_model=List[Channel])
def get_channels():
    return fetch_all('SELECT * FROM channel')


@app.get("/api/locations", response_model=List[Location])
def get_locations():
    return fetch_all('SELECT "LocationID","Location_Descr" AS "LocationDescr","Level","Geography" FROM location')


# === Saved searches ===========================================================
@app.get("/api/saved-searches")
def list_saved():
    return fetch_all("""
        SELECT id, name, query, created_at
        FROM saved_search
        ORDER BY created_at DESC
    """)

@app.post("/api/saved-searches")
def save_search(item: dict = Body(...)):
    name = (item.get("name") or "").strip()
    query = (item.get("query") or "").strip()
    if not name or not query:
        raise HTTPException(status_code=400, detail="name and query are required")

    # Use a plain execute for INSERT (no rows expected)
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO saved_search(name, query) VALUES (:name, :query)"),
            {"name": name, "query": query},
        )

    return {"ok": True}


# === Search (keys & values) ===================================================
FIELD_MAP = _make_field_map("fe", "p", "c", "l")


class SearchResult(BaseModel):
    query: str
    count: int
    keys: List[KeyTriplet]


@app.get("/api/search", response_model=SearchResult)
def search(q: Optional[str] = None, limit: int = 5000, offset: int = 0):
    MAX_LIMIT = 20000
    limit = min(limit, MAX_LIMIT)
    params: dict = {}
    where_sql = _build_where_clause_with_map(q or "", params, FIELD_MAP, bind_prefix="b")
    params.update({"limit": limit, "offset": offset})

    sql = f"""
    WITH base AS (
      SELECT DISTINCT
        fe."ProductID", fe."ChannelID", fe."LocationID"
      FROM forecast_element fe
      LEFT JOIN product  p ON p."ProductID" = fe."ProductID"
      LEFT JOIN channel  c ON c."ChannelID" = fe."ChannelID"
      LEFT JOIN location l ON l."LocationID" = fe."LocationID"
      WHERE {where_sql}
      ORDER BY fe."ProductID", fe."ChannelID", fe."LocationID"
      LIMIT :limit OFFSET :offset
    )
    SELECT * FROM base;
    """
    count_sql = f"""
      SELECT COUNT(*) FROM (
        SELECT 1
        FROM forecast_element fe
        LEFT JOIN product  p ON p."ProductID" = fe."ProductID"
        LEFT JOIN channel  c ON c."ChannelID" = fe."ChannelID"
        LEFT JOIN location l ON l."LocationID" = fe."LocationID"
        WHERE {where_sql}
        GROUP BY fe."ProductID", fe."ChannelID", fe."LocationID"
      ) t;
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
        total = conn.execute(text(count_sql), params).scalar_one()
    keys = [KeyTriplet(**dict(r)) for r in rows]
    return SearchResult(query=q or "", count=total, keys=keys)


@app.get("/api/search/values")
def search_values(field: str, like: Optional[str] = None, limit: int = 500):
    fmap = FIELD_MAP
    col = fmap.get(field.lower())
    if not col:
        raise HTTPException(status_code=400, detail=f"Unknown field: {field}")

    params: Dict[str, Any] = {"limit": min(limit, 5000)}
    like_sql = ""
    if like:
        params["like"] = _to_ilike_pattern(like)
        like_sql = """AND val ILIKE :like ESCAPE '\\'"""

    sql = f"""
    WITH base AS (
      SELECT DISTINCT fe."ProductID", fe."ChannelID", fe."LocationID"
      FROM forecast_element fe
      LEFT JOIN product  p ON p."ProductID" = fe."ProductID"
      LEFT JOIN channel  c ON c."ChannelID" = fe."ChannelID"
      LEFT JOIN location l ON l."LocationID" = fe."LocationID"
    )
    SELECT DISTINCT val AS value
    FROM (
      SELECT {col} AS val
      FROM base
      LEFT JOIN product  p ON p."ProductID" = base."ProductID"
      LEFT JOIN channel  c ON c."ChannelID" = base."ChannelID"
      LEFT JOIN location l ON l."LocationID" = base."LocationID"
    ) s
    WHERE val IS NOT NULL {like_sql}
    ORDER BY value
    LIMIT :limit;
    """
    rows = fetch_all(sql, params)
    return {"field": field, "values": [str(r["value"]) for r in rows if r.get("value") is not None]}


# === History series by query (charts) ========================================
@app.get("/api/history/{bucket}-series-by-query")
def history_series_by_query(
    bucket: Literal["daily", "weekly", "monthly"],
    q: Optional[str] = None,
    max_points: int = 800,
    key_limit: int = 5000,
):
    if max_points < 50:
        max_points = 50
    if max_points > 5000:
        max_points = 5000

    hist_table, _, trunc_part = _tbl_freq(bucket)
    params: Dict[str, Any] = {}
    where_sql = _build_where_clause_with_map(q or "", params, _make_field_map("fe", "p", "c", "l"), bind_prefix="fe_b")
    params.update({"trunc_part": trunc_part, "max_points": max_points, "key_limit": key_limit})

    sql = f"""
    WITH keys AS (
      SELECT DISTINCT fe."ProductID", fe."ChannelID", fe."LocationID"
      FROM forecast_element fe
      LEFT JOIN product  p ON p."ProductID"  = fe."ProductID"
      LEFT JOIN channel  c ON c."ChannelID"  = fe."ChannelID"
      LEFT JOIN location l ON l."LocationID" = fe."LocationID"
      WHERE {where_sql}
      LIMIT :key_limit
    ),
    raw AS (
      SELECT date_trunc(:trunc_part, {_ts_expr('h')}) AS dt, SUM(h."Qty") AS qty
      FROM {hist_table} h
      JOIN keys k
        ON h."ProductID" = k."ProductID"
       AND h."ChannelID" = k."ChannelID"
       AND h."LocationID" = k."LocationID"
      GROUP BY 1
      ORDER BY 1
    ),
    stats AS (
      SELECT COUNT(*)::int cnt,
             MIN(EXTRACT(EPOCH FROM dt)) min_ep,
             MAX(EXTRACT(EPOCH FROM dt)) max_ep
      FROM raw
    ),
    ds AS (
      SELECT r.dt, r.qty FROM raw r
      WHERE (SELECT cnt FROM stats) <= :max_points
         OR (SELECT min_ep = max_ep FROM stats)
      UNION ALL
      SELECT MIN(t.dt) AS dt, SUM(t.qty) AS qty
      FROM (
        SELECT r.dt, r.qty,
               width_bucket(EXTRACT(EPOCH FROM r.dt), s.min_ep, s.max_ep, GREATEST(1, :max_points)) AS bid
        FROM raw r CROSS JOIN stats s
      ) t
      WHERE (SELECT cnt FROM stats) > :max_points
        AND (SELECT min_ep <> max_ep FROM stats)
      GROUP BY t.bid
    )
    SELECT to_char(dt, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS "StartDate",
           qty::float AS "Qty"
    FROM ds
    ORDER BY dt;
    """
    return fetch_all(sql, params)


# === Forecast series by query (charts, using forecast_* tables) ==============
@app.get("/api/forecast/{bucket}-series-by-query")
def forecast_series_by_query(
    bucket: Literal["daily", "weekly", "monthly"],
    q: Optional[str] = None,
    max_points: int = 800,
    method: Optional[str] = None,
    type_filter: Optional[str] = None,
):
    if max_points < 50:
        max_points = 50
    if max_points > 5000:
        max_points = 5000

    ftable = _forecast_table(bucket)
    ts = _ts_expr("f")
    trunc_part = {"daily": "day", "weekly": "week", "monthly": "month"}[bucket]

    params_fe: Dict[str, Any] = {}
    params_fc: Dict[str, Any] = {}
    where_fe = _build_where_clause_with_map(q or "", params_fe, _make_field_map("fe", "p", "c", "l"), "fe_b")
    where_fc = _build_where_clause_with_map(q or "", params_fc, _make_field_map("f", "p", "c", "l"), "fc_b")

    meth_sql = ' AND f."Method" = :m' if method else ""
    type_sql = ' AND COALESCE(f."Type","") = :t' if type_filter else ""
    if method:
        params_fc["m"] = method
    if type_filter:
        params_fc["t"] = type_filter
    params_common: Dict[str, Any] = {"trunc_part": trunc_part, "max_points": max_points}

    sql = f"""
    WITH keys_fe AS (
      SELECT DISTINCT fe."ProductID", fe."ChannelID", fe."LocationID"
      FROM forecast_element fe
      LEFT JOIN product  p ON p."ProductID"  = fe."ProductID"
      LEFT JOIN channel  c ON c."ChannelID"  = fe."ChannelID"
      LEFT JOIN location l ON l."LocationID" = fe."LocationID"
      WHERE {where_fe}
    ),
    keys_fc AS (
      SELECT DISTINCT f."ProductID", f."ChannelID", f."LocationID"
      FROM {ftable} f
      LEFT JOIN product  p ON p."ProductID"  = f."ProductID"
      LEFT JOIN channel  c ON c."ChannelID"  = f."ChannelID"
      LEFT JOIN location l ON l."LocationID" = f."LocationID"
      WHERE {where_fc} {meth_sql} {type_sql}
    ),
    keys AS (SELECT * FROM keys_fe UNION SELECT * FROM keys_fc),
    raw AS (
      SELECT date_trunc(:trunc_part, {ts}) AS dt, SUM(f."Qty") AS qty
      FROM {ftable} f
      JOIN keys k
        ON f."ProductID" = k."ProductID"
       AND f."ChannelID" = k."ChannelID"
       AND f."LocationID" = k."LocationID"
      WHERE {ts} IS NOT NULL {meth_sql} {type_sql}
      GROUP BY 1
      ORDER BY 1
    ),
    stats AS (
      SELECT COUNT(*)::int cnt,
             MIN(EXTRACT(EPOCH FROM dt)) min_ep,
             MAX(EXTRACT(EPOCH FROM dt)) max_ep
      FROM raw
    ),
    ds AS (
      SELECT r.dt, r.qty
      FROM raw r
      WHERE (SELECT cnt FROM stats) <= :max_points
         OR (SELECT min_ep = max_ep FROM stats)
      UNION ALL
      SELECT MIN(t.dt) AS dt, SUM(t.qty) AS qty
      FROM (
        SELECT r.dt, r.qty,
               width_bucket(EXTRACT(EPOCH FROM r.dt), s.min_ep, s.max_ep, GREATEST(1,:max_points)) AS bid
        FROM raw r CROSS JOIN stats s
      ) t
      WHERE (SELECT cnt FROM stats) > :max_points
        AND (SELECT min_ep <> max_ep FROM stats)
      GROUP BY t.bid
    )
    SELECT to_char(dt, 'YYYY-MM-DD"T"HH24:MI:SS"Z"') AS "StartDate",
           qty::float AS "Qty"
    FROM ds
    ORDER BY 1;
    """
    params = {**params_common, **params_fe, **params_fc}
    return fetch_all(sql, params)


# === GeneratedPeriods helper ==================================================
def _load_generated_periods(period: Literal["daily", "weekly", "monthly"], start_after: datetime) -> List[Tuple[datetime, datetime]]:
    csv_path = Path(__file__).with_name("GeneratedPeriods.csv")
    if not csv_path.exists():
        return []
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return []
    if not {"StartDate", "EndDate", "Period"}.issubset(df.columns):
        return []
    df["StartDate"] = pd.to_datetime(df["StartDate"], dayfirst=True, errors="coerce", utc=False)
    df["EndDate"] = pd.to_datetime(df["EndDate"], dayfirst=True, errors="coerce", utc=False)
    df = df.dropna(subset=["StartDate", "EndDate", "Period"])
    prefix = f"Future_{period.capitalize()}"
    df = df[df["Period"].astype(str).str.startswith(prefix)]
    df = df[df["StartDate"] > start_after].sort_values("StartDate")
    return [(pd.Timestamp(s).to_pydatetime(), pd.Timestamp(e).to_pydatetime()) for s, e in zip(df["StartDate"], df["EndDate"]) ]


# === Load single-key history ==================================================
def _load_series_for_key(
    period: Literal["daily", "weekly", "monthly"],
    pid: str, cid: str, lid: str,
    want_cleansed: bool,
) -> Tuple[pd.Series, bool, str, str]:
    hist_table, pd_freq, trunc_part = _tbl_freq(period)

    def run_query(extra_where: str):
        sql = f"""
        WITH src AS (
          SELECT {_ts_expr('h')} AS ts, h."Qty"
          FROM {hist_table} h
          WHERE h."ProductID"=:pid AND h."ChannelID"=:cid AND h."LocationID"=:lid
          {extra_where}
        ),
        raw AS (
          SELECT date_trunc(:trunc_part, ts) AS dt, SUM("Qty") AS qty
          FROM src WHERE ts IS NOT NULL
          GROUP BY 1 ORDER BY 1
        )
        SELECT dt, qty::float FROM raw ORDER BY dt;
        """
        return fetch_all(sql, {"pid": pid, "cid": cid, "lid": lid, "trunc_part": trunc_part})

    rows = run_query("")
    used_cleansed = False
    if not rows and want_cleansed:
        rows = run_query("")
        used_cleansed = False
    if not rows:
        raise HTTPException(status_code=400, detail="No history for the selected key/period.")

    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["dt"], utc=True, errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["dt"]).sort_values("dt")
    s = pd.to_numeric(df["qty"], errors="coerce").dropna()
    s.index = df["dt"].values
    s = s.astype(float)
    return s, used_cleansed, pd_freq, trunc_part


# === Cleanse Profiles (list + upsert) ========================================
class CleanseProfile(BaseModel):
    id: Optional[int] = None
    name: str
    settings: Dict[str, Any] = {}
    is_active: bool = False


@app.get("/api/cleanse/profiles", response_model=List[CleanseProfile])
def cleanse_profiles_list():
    rows = fetch_all("""
      SELECT id, name, settings, is_active
      FROM cleanse_profile
      ORDER BY is_active DESC, name
    """)
    for r in rows:
        if isinstance(r.get("settings"), str):
            try: r["settings"] = json.loads(r["settings"])
            except Exception: r["settings"] = {}
    return rows


@app.post("/api/cleanse/profiles", response_model=CleanseProfile)
def cleanse_profiles_upsert(profile: CleanseProfile = Body(...)):
    name = (profile.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    settings_json = json.dumps(profile.settings or {})
    params = {
        "name": name,
        "settings": settings_json,
        "config": settings_json,
        "is_active": bool(profile.is_active),
    }

    row = fetch_all("""
        INSERT INTO cleanse_profile(name, settings, config, is_active, updated_at)
        VALUES (:name, CAST(:settings AS jsonb), CAST(:config AS jsonb), :is_active, now())
        ON CONFLICT(name) DO UPDATE
          SET settings  = EXCLUDED.settings,
              config    = EXCLUDED.config,
              is_active = EXCLUDED.is_active,
              updated_at = now()
        RETURNING id, name, settings, is_active
    """, params)
    if not row:
        raise HTTPException(status_code=500, detail="upsert failed")
    r = row[0]
    if isinstance(r.get("settings"), str):
        try: r["settings"] = json.loads(r["settings"])
        except Exception: r["settings"] = {}
    return r


# (Optional) Ingest cleansed history rows
class IngestRow(BaseModel):
    ProductID: str
    ChannelID: str
    LocationID: str
    StartDate: str
    Qty: float
    Period: Literal["Daily","Weekly","Monthly"] = "Daily"

@app.post("/api/history/ingest-cleansed")
def history_ingest_cleansed(body: Union[List[IngestRow], IngestRow, dict] = Body(...)):
    if isinstance(body, list):
        raw_items = body
    elif isinstance(body, dict):
        raw_items = body.get("rows") if isinstance(body.get("rows"), list) else [body]
    else:
        raw_items = [body]  # type: ignore

    if not raw_items:
        raise HTTPException(status_code=400, detail="No rows provided.")

    normalized: List[IngestRow] = []
    errors: List[str] = []

    def _norm_period(p: Optional[str]) -> str:
        if not p:
            return "Daily"
        p2 = str(p).strip().lower()
        if p2.startswith("d"): return "Daily"
        if p2.startswith("w"): return "Weekly"
        if p2.startswith("m"): return "Monthly"
        return "Daily"

    for i, item in enumerate(raw_items):
        try:
            d = item.dict() if isinstance(item, IngestRow) else dict(item)  # type: ignore
            if "startdate" in d and "StartDate" not in d:
                d["StartDate"] = d.pop("startdate")
            if "qty" in d and "Qty" not in d:
                d["Qty"] = d.pop("qty")

            req = ["ProductID", "ChannelID", "LocationID", "StartDate", "Qty"]
            missing = [k for k in req if k not in d or d[k] in (None, "")]
            if missing:
                raise ValueError(f"missing fields: {', '.join(missing)}")

            try:
                d["Qty"] = float(d["Qty"])
            except Exception:
                raise ValueError("Qty must be numeric")

            d["Period"] = _norm_period(d.get("Period"))

            normalized.append(IngestRow(**d))
        except Exception as e:
            errors.append(f"row {i}: {e}")

    if errors and not normalized:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    groups: Dict[str, List[Dict[str, Any]]] = {"daily": [], "weekly": [], "monthly": []}
    for r in normalized:
        bucket = r.Period.lower()
        groups[bucket].append({
            "ProductID": r.ProductID,
            "ChannelID": r.ChannelID,
            "LocationID": r.LocationID,
            "StartDate": r.StartDate,
            "Qty": r.Qty
        })

    inserted = 0
    with engine.begin() as c:
        for bucket, batch in groups.items():
            if not batch:
                continue
            table = _tbl_freq(bucket)[0]
            c.execute(text(f"""
                INSERT INTO {table}("ProductID","ChannelID","LocationID","StartDate","Qty")
                VALUES (:ProductID,:ChannelID,:LocationID,:StartDate,:Qty)
            """), batch)
            inserted += len(batch)

    return {"inserted": inserted, "errors": errors}


# === Classify forecast elements ==============================================
class ClassifyRequest(BaseModel):
    period: Literal["daily","weekly","monthly"] = "daily"
    lookback_buckets: int = 8
    min_sum: float = 1.0
    include_inactive: bool = False  # for GET only


@app.post("/api/classify/compute")
def classify_compute(body: ClassifyRequest):
    hist_table, _, trunc_part = _tbl_freq(body.period)
    unit = {"daily": "day", "weekly": "week", "monthly": "month"}[body.period]

    raw = fetch_all(
        f"""
      WITH raw AS (
        SELECT h."ProductID", h."ChannelID", h."LocationID",
               date_trunc(:trunc_part, {_ts_expr('h')}) AS dt,
               SUM(h."Qty")::float AS qty
        FROM {hist_table} h
        WHERE {_ts_expr('h')} IS NOT NULL
        GROUP BY 1,2,3,4
      ),
      recent AS (
        SELECT r."ProductID", r."ChannelID", r."LocationID",
               SUM(r.qty) AS s,
               MAX(r.dt)  AS last_dt
        FROM raw r
        WHERE r.dt >= (SELECT MAX(dt) FROM raw) - (:lb - 1) * INTERVAL '1 {unit}'
        GROUP BY 1,2,3
      )
      SELECT * FROM recent
        """,
        {"trunc_part": trunc_part, "lb": int(max(1, body.lookback_buckets))},
    )

    upserts = []
    for r in raw:
        label = "Active" if (r.get("s") or 0) >= body.min_sum else "Inactive"
        score = float(r.get("s") or 0.0)
        upserts.append({
            "ProductID": r["ProductID"], "ChannelID": r["ChannelID"], "LocationID": r["LocationID"],
            "Period": body.period, "Label": label, "Score": score, "IsActive": (label == "Active")
        })

    if upserts:
        with engine.begin() as c:
            c.execute(text("""
              INSERT INTO forecast_element_classification
                ("ProductID","ChannelID","LocationID","Period","Label","Score","IsActive","ComputedAt")
              VALUES
                (:ProductID,:ChannelID,:LocationID,:Period,:Label,:Score,:IsActive, now())
              ON CONFLICT ("ProductID","ChannelID","LocationID","Period")
              DO UPDATE SET
                "Label"=EXCLUDED."Label",
                "Score"=EXCLUDED."Score",
                "IsActive"=EXCLUDED."IsActive",
                "ComputedAt"=now()
            """), upserts)

    return {"updated": len(upserts), "period": body.period}


@app.get("/api/classify/results")
def classify_results(period: Literal["daily","weekly","monthly"], include_inactive: bool = False):
    sql = """
      SELECT "ProductID","ChannelID","LocationID","Period","Label","Score","IsActive","ComputedAt"
      FROM forecast_element_classification
      WHERE "Period" = :p {filt}
      ORDER BY "IsActive" DESC, "Score" DESC, "ProductID","ChannelID","LocationID"
    """
    filt = "" if include_inactive else 'AND "IsActive" = TRUE'
    return fetch_all(sql.format(filt=filt), {"p": period})


# === Batch 18M (monthly CSV-driven) ==========================================
def _parse_dt_dmy(s: pd.Series | str):
    if isinstance(s, str):
        return pd.to_datetime(s, dayfirst=True, errors="coerce")
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def _read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


class BatchRunBody(BaseModel):
    history_file: Optional[str] = None
    metric: Optional[Literal["WMAPE","MAE"]] = None
    fast_mode: Optional[bool] = None
    forward_horizon: Optional[int] = None
    max_plots: Optional[int] = None


class BatchRunResult(BaseModel):
    backtest_file: Optional[str]
    summary_file: Optional[str]
    forward_file: Optional[str]
    plots_backtest_dir: Optional[str]
    plots_history_forecast_dir: Optional[str]
    rows_backtest: int
    rows_summary: int
    rows_forward: int


class Combo(BaseModel):
    ProductID: str
    ChannelID: str
    LocationID: str


@app.post("/api/batch/monthly/run", response_model=BatchRunResult)
def api_batch_monthly_run(body: BatchRunBody):
    if bf is None:
        raise HTTPException(status_code=400, detail="Batch script not available on server.")
    if body.history_file is not None: bf.HISTORY_FILE = body.history_file
    if body.metric is not None: bf.METRIC = body.metric
    if body.fast_mode is not None: bf.FAST_MODE = bool(body.fast_mode)
    if body.forward_horizon is not None: bf.FORWARD_HORIZON = int(body.forward_horizon)
    if body.max_plots is not None: bf.MAX_PLOTS = int(body.max_plots)
    res = bf.run()
    bt = _read_csv_safe(Path(res.get("backtest_file","")))
    sm = _read_csv_safe(Path(res.get("summary_file","")))
    fw = _read_csv_safe(Path(res.get("forward_file","")))
    return BatchRunResult(
        backtest_file=res.get("backtest_file"),
        summary_file=res.get("summary_file"),
        forward_file=res.get("forward_file"),
        plots_backtest_dir=res.get("plots_backtest_dir"),
        plots_history_forecast_dir=res.get("plots_history_forecast_dir"),
        rows_backtest=int(len(bt)), rows_summary=int(len(sm)), rows_forward=int(len(fw)),
    )


@app.get("/api/batch/monthly/combos", response_model=List[Combo])
def api_batch_monthly_combos():
    if bf is None: return []
    hist = _read_csv_safe(Path(bf.HISTORY_FILE))
    if hist.empty: return []
    needed = {"ProductID","ChannelID","LocationID"}
    if not needed.issubset(hist.columns):
        raise HTTPException(status_code=400, detail="History file missing required columns.")
    out = (hist[["ProductID","ChannelID","LocationID"]].astype(str)
           .drop_duplicates().sort_values(["ProductID","ChannelID","LocationID"]))
    return [Combo(**dict(r)) for _, r in out.iterrows()]


@app.get("/api/batch/monthly/backtest")
def api_batch_backtest(pid: Optional[str] = None, cid: Optional[str] = None, lid: Optional[str] = None):
    if bf is None: return []
    df = _read_csv_safe(Path(bf.OUT_BACKTEST_CSV))
    if df.empty: return []
    for c in ("ProductID","ChannelID","LocationID"):
        if c in df.columns: df[c] = df[c].astype(str)
    if pid: df = df[df["ProductID"]==pid]
    if cid: df = df[df["ChannelID"]==cid]
    if lid: df = df[df["LocationID"]==lid]
    if "Date" in df.columns:
        df["Date"] = _parse_dt_dmy(df["Date"]).dt.strftime("%Y-%m-%dT00:00:00Z")
    return df.to_dict(orient="records")


@app.get("/api/batch/monthly/summary")
def api_batch_summary():
    if bf is None: return []
    df = _read_csv_safe(Path(bf.OUT_SUMMARY_CSV))
    if df.empty: return []
    for c in ("ProductID","ChannelID","LocationID"):
        if c in df.columns: df[c] = df[c].astype(str)
    return df.to_dict(orient="records")


@app.get("/api/batch/monthly/forward")
def api_batch_forward(pid: Optional[str] = None, cid: Optional[str] = None, lid: Optional[str] = None):
    if bf is None: return []
    df = _read_csv_safe(Path(bf.OUT_FORWARD_CSV))
    if df.empty: return []
    for c in ("ProductID","ChannelID","LocationID"):
        if c in df.columns: df[c] = df[c].astype(str)
    if pid: df = df[df["ProductID"]==pid]
    if cid: df = df[df["ChannelID"]==cid]
    if lid: df = df[df["LocationID"]==lid]
    for col in ("StartDate","EndDate","Forecast Date","History End Date"):
        if col in df.columns:
            df[col] = _parse_dt_dmy(df[col]).dt.strftime("%Y-%m-%dT00:00:00Z")
    return df.rename(columns={"Forecast Qty":"ForecastQty"}).to_dict(orient="records")


@app.get("/api/batch/monthly/history-series", response_model=List[SeriesPoint])
def api_batch_history_series(pid: str, cid: str, lid: str):
    if bf is None: return []
    hist = _read_csv_safe(Path(bf.HISTORY_FILE))
    if hist.empty: return []
    need = {"ProductID","ChannelID","LocationID","StartDate","Qty"}
    if not need.issubset(hist.columns):
        raise HTTPException(status_code=400, detail="History file missing required columns.")
    sub = hist[(hist["ProductID"].astype(str)==pid) &
               (hist["ChannelID"].astype(str)==cid) &
               (hist["LocationID"].astype(str)==lid)]
    if sub.empty: return []
    sub = sub.copy()
    sub["StartDate"] = _parse_dt_dmy(sub["StartDate"]).dt.strftime("%Y-%m-%dT00:00:00Z")
    grp = (sub.groupby("StartDate", as_index=False)["Qty"].sum()
           .sort_values("StartDate"))
    return [SeriesPoint(StartDate=r["StartDate"], Qty=float(r["Qty"])) for _, r in grp.iterrows()]


@app.get("/api/batch/monthly/forward-series", response_model=List[SeriesPoint])
def api_batch_forward_series(pid: str, cid: str, lid: str):
    if bf is None: return []
    fw = _read_csv_safe(Path(bf.OUT_FORWARD_CSV))
    if fw.empty: return []
    sub = fw[(fw["ProductID"].astype(str)==pid) &
             (fw["ChannelID"].astype(str)==cid) &
             (fw["LocationID"].astype(str)==lid)]
    if sub.empty: return []
    sub = sub.copy()
    sub["StartDate"] = _parse_dt_dmy(sub["StartDate"]).dt.strftime("%Y-%m-%dT00:00:00Z")
    sub["ForecastQty"] = pd.to_numeric(sub["Forecast Qty"], errors="coerce")
    grp = (sub.groupby("StartDate", as_index=False)["ForecastQty"].sum()
           .sort_values("StartDate"))
    return [SeriesPoint(StartDate=r["StartDate"], Qty=float(r["ForecastQty"])) for _, r in grp.iterrows()]


# === Shared horizon defaults per period ======================================
DEFAULT_HORIZON: Dict[str, int] = {
    "daily": 7,      # next 7 days
    "weekly": 13,    # next 13 weeks
    "monthly": 18,   # next 18 months
}


# === NEW: Per-key forecast from DB history (all periods) =====================
class Run18mBody(BaseModel):
    key: KeyTriplet
    period: Literal["daily", "weekly", "monthly"] = "monthly"
    horizon: int = 18         # interpreted as "buckets"; we override default per period if needed
    save: bool = False        # if true, upsert into forecast_* table
    use_cleansed: bool = False  # reserved for future use


class Run18mPrediction(BaseModel):
    ProductID: str
    ChannelID: str
    LocationID: str
    StartDate: str
    EndDate: str
    Qty: float
    Method: str
    Period: str
    Type: str


class Run18mResult(BaseModel):
    key: KeyTriplet
    period: Literal["daily", "weekly", "monthly"] = "monthly"
    model: str
    horizon: int
    history_points: int
    predictions: List[Run18mPrediction]


# === NEW: Forecast aggregated by saved-search query (all periods) ============
class Run18mByQueryBody(BaseModel):
    q: Optional[str] = None
    period: Literal["daily", "weekly", "monthly"] = "monthly"
    horizon: int = 18
    max_keys: int = 500
    use_cleansed: bool = False  # reserved / future use
    save: bool = False          # NEW: save per-key forecasts into forecast_*


class Run18mByQueryResult(BaseModel):
    query: str
    period: Literal["daily", "weekly", "monthly"] = "monthly"
    horizon: int
    keys_scanned: int
    keys_forecasted: int
    skipped: int
    series: List[SeriesPoint]


def _step_dates(last_dt: datetime, period: str, step_index: int) -> datetime:
    """Return forecast bucket StartDate for step_index (1-based) after last_dt."""
    if period == "daily":
        return last_dt + timedelta(days=step_index)
    if period == "weekly":
        return last_dt + timedelta(days=7 * step_index)
    # monthly
    return (last_dt + pd.DateOffset(months=step_index)).to_pydatetime()


@app.post("/api/forecast/18m/run-by-key", response_model=Run18mResult)
def run_18m_by_key(body: Run18mBody):
    """
    Per-key interactive forecast using the appropriate module for the period:
      - daily_mod for daily (7d horizon typical)
      - weekly_mod for weekly (13w horizon typical)
      - bf (monthly_mod) for monthly (18m horizon typical)
    """
    period: Literal["daily", "weekly", "monthly"] = body.period or "monthly"
    mod = _get_forecast_module(period)

    # Use reasonable default horizon per period if caller passes <=0 or default 18
    default_H = DEFAULT_HORIZON[period]
    raw_H = body.horizon or default_H
    H = int(raw_H if raw_H > 0 else default_H)

    pid, cid, lid = body.key.ProductID, body.key.ChannelID, body.key.LocationID

    # Load series from the appropriate history table
    series, _, _, _ = _load_series_for_key(period, pid, cid, lid, body.use_cleansed)

    MIN_TRAIN = int(max(10, getattr(mod, "MIN_TRAIN_POINTS", 10)))
    if len(series) < MIN_TRAIN:
        raise HTTPException(status_code=400, detail=f"Not enough {period} history (have {len(series)}).")

    # Model selection knobs from the module
    METRIC = getattr(mod, "METRIC", "WMAPE")
    FAST_MODE = getattr(mod, "FAST_MODE", True)
    CV_STRIDE = getattr(mod, "CV_STRIDE", 2)
    SNAIVE_PREF = getattr(mod, "SNAIVE_PREFERENCE_MARGIN", 0.05)

    # Rolling backtest with registry models + snaive
    preds_dict, scores, _ = mod.rolling_backtest(series, METRIC, CV_STRIDE if FAST_MODE else 1)

    # Pick best model with SNaive preference rule
    metric_key = "WMAPE" if (METRIC or "").upper() == "WMAPE" else "MAE"
    best_model = min(scores.items(), key=lambda kv: kv[1][metric_key])[0]
    best_score = scores[best_model][metric_key]
    if "snaive" in scores and best_model != "snaive":
        snaive_score = scores["snaive"][metric_key]
        if not np.isinf(snaive_score):
            gain = snaive_score - best_score
            if gain < SNAIVE_PREF * snaive_score:
                best_model = "snaive"

    # Forward forecast using that module
    fut_vals = mod.forward_forecast_best(series, best_model, horizon=H)

    # Build time windows according to period
    last_dt = series.index.max().to_pydatetime()
    preds: List[Run18mPrediction] = []
    for i in range(1, H + 1):
        sdt = _step_dates(last_dt, period, i)
        edt = _period_end(sdt, period)
        preds.append(Run18mPrediction(
            ProductID=pid, ChannelID=cid, LocationID=lid,
            StartDate=sdt.strftime("%Y-%m-%dT00:00:00Z"),
            EndDate=edt.strftime("%Y-%m-%dT23:59:59Z"),
            Qty=float(fut_vals[i-1]),
            Method=best_model,
            Period=period.capitalize(),
            Type="Algorithm-Forecast",
        ))

    # Optional: save to forecast_* table
    if body.save and preds:
        tbl = _forecast_table(period)
        sql_ins = f"""
        INSERT INTO {tbl}
          ("ProductID","ChannelID","LocationID","Method","Period","StartDate","EndDate","Type","Qty","Level")
        VALUES
          (:ProductID,:ChannelID,:LocationID,:Method,:Period,:StartDate,:EndDate,:Type,:Qty,'Item')
        ON CONFLICT ("ProductID","ChannelID","LocationID","StartDate","EndDate","Method")
        DO UPDATE SET
          "Qty" = EXCLUDED."Qty",
          "Type" = EXCLUDED."Type",
          "created_at" = now();
        """
        with engine.begin() as c:
            c.execute(text(sql_ins), [p.dict() for p in preds])



    return Run18mResult(
        key=body.key,
        period=period,
        model=best_model,
        horizon=H,
        history_points=int(len(series)),
        predictions=preds
    )


@app.post("/api/forecast/18m/run-by-query", response_model=Run18mByQueryResult)
def run_18m_by_query(body: Run18mByQueryBody):
    """
    For a query like 'productid:*' or 'productid:AztecWrap geography:Gelderland',
    find all matching keys, run per-key forecasts using the same logic
    for the chosen period (daily/weekly/monthly), and return the aggregate
    (sum) series by StartDate.

    If save=True, per-key forecasts are also upserted into forecast_daily /
    forecast_weekly / forecast_monthly with ProductID, ChannelID, LocationID
    and StartDate/EndDate/Method.
    """
    period: Literal["daily", "weekly", "monthly"] = body.period or "monthly"
    mod = _get_forecast_module(period)

    default_H = DEFAULT_HORIZON[period]
    raw_H = body.horizon or default_H
    H = int(raw_H if raw_H > 0 else default_H)

    # 1) Collect keys matching the query (limit by max_keys)
    params: Dict[str, Any] = {}
    where_sql = _build_where_clause_with_map(body.q or "", params, FIELD_MAP, bind_prefix="b")
    params.update({"limit": int(max(1, min(body.max_keys, 5000)))})
    keys_sql = f"""
      SELECT DISTINCT fe."ProductID", fe."ChannelID", fe."LocationID"
      FROM forecast_element fe
      LEFT JOIN product  p ON p."ProductID"  = fe."ProductID"
      LEFT JOIN channel  c ON c."ChannelID"  = fe."ChannelID"
      LEFT JOIN location l ON l."LocationID" = fe."LocationID"
      WHERE {where_sql}
      ORDER BY fe."ProductID", fe."ChannelID", fe."LocationID"
      LIMIT :limit
    """
    with engine.begin() as conn:
        key_rows = conn.execute(text(keys_sql), params).mappings().all()

    keys_scanned = len(key_rows)
    if not keys_scanned:
        return Run18mByQueryResult(
            query=body.q or "",
            period=period,
            horizon=H,
            keys_scanned=0, keys_forecasted=0, skipped=0,
            series=[]
        )

    # 2) Per-key forecast, then aggregate by StartDate
    METRIC = getattr(mod, "METRIC", "WMAPE")
    FAST_MODE = getattr(mod, "FAST_MODE", True)
    CV_STRIDE = getattr(mod, "CV_STRIDE", 2)
    SNAIVE_PREF = getattr(mod, "SNAIVE_PREFERENCE_MARGIN", 0.05)
    MIN_TRAIN = int(max(10, getattr(mod, "MIN_TRAIN_POINTS", 10)))

    agg: Dict[str, float] = {}
    keys_ok = 0
    skipped = 0
    all_save_rows: List[Dict[str, Any]] = []

    for r in key_rows:
        pid, cid, lid = str(r["ProductID"]), str(r["ChannelID"]), str(r["LocationID"])
        try:
            series, _, _, _ = _load_series_for_key(period, pid, cid, lid, body.use_cleansed)
            if len(series) < MIN_TRAIN:
                skipped += 1
                continue

            # pick best model via rolling backtest
            preds_dict, scores, _ = mod.rolling_backtest(series, METRIC, CV_STRIDE if FAST_MODE else 1)
            metric_key = "WMAPE" if (METRIC or "").upper() == "WMAPE" else "MAE"
            best_model = min(scores.items(), key=lambda kv: kv[1][metric_key])[0]
            best_score = scores[best_model][metric_key]
            if "snaive" in scores and best_model != "snaive":
                snaive_score = scores["snaive"][metric_key]
                if not np.isinf(snaive_score):
                    gain = snaive_score - best_score
                    if gain < SNAIVE_PREF * snaive_score:
                        best_model = "snaive"

            fut_vals = mod.forward_forecast_best(series, best_model, horizon=H)

            # Anchor buckets from this key's last history point
            last_dt = series.index.max().to_pydatetime()
            for i in range(1, H + 1):
                sdt = _step_dates(last_dt, period, i)
                edt = _period_end(sdt, period)
                start_str = sdt.strftime("%Y-%m-%dT00:00:00Z")
                agg[start_str] = agg.get(start_str, 0.0) + float(fut_vals[i-1])

                if body.save:
                    all_save_rows.append({
                        "ProductID": pid,
                        "ChannelID": cid,
                        "LocationID": lid,
                        "StartDate": start_str,
                        "EndDate": edt.strftime("%Y-%m-%dT23:59:59Z"),
                        "Qty": float(fut_vals[i-1]),
                        "Period": period.capitalize(),
                        "Method": best_model,
                        "Type": "Algorithm-Forecast",
                    })

            keys_ok += 1

        except Exception:
            skipped += 1
            continue

    # 2b) Save per-key forecasts if requested
    if body.save and all_save_rows:
        tbl = _forecast_table(period)
        sql_ins = f"""
        INSERT INTO {tbl}
          ("ProductID","ChannelID","LocationID","Method","Period","StartDate","EndDate","Type","Qty","Level")
        VALUES
          (:ProductID,:ChannelID,:LocationID,:Method,:Period,:StartDate,:EndDate,:Type,:Qty,'Item')
        ON CONFLICT ("ProductID","ChannelID","LocationID","StartDate","EndDate","Method")
        DO UPDATE SET
          "Qty" = EXCLUDED."Qty",
          "Type" = EXCLUDED."Type",
          "created_at" = now();
        """
        with engine.begin() as c:
            c.execute(text(sql_ins), all_save_rows)


    # 3) Build aggregated series (sorted)
    out_series = [
        SeriesPoint(StartDate=k, Qty=v) for k, v in sorted(agg.items(), key=lambda kv: kv[0])
    ]

    return Run18mByQueryResult(
        query=body.q or "",
        period=period,
        horizon=H,
        keys_scanned=keys_scanned,
        keys_forecasted=keys_ok,
        skipped=skipped,
        series=out_series
    )
