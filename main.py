from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, date

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
import math

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


# === Weather / Sales correlation helpers =====================================

def load_weather_daily() -> pd.DataFrame:
    """
    Load weather_daily from DB in a robust way and normalize columns to:

      LocationID, Date, TempAvg, TempMin, TempMax, RainMm, SnowCm

    Date is returned as datetime.date.
    """
    rows = fetch_all("SELECT * FROM weather_daily")
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Map lowercase -> original col name
    lower_map = {c.lower(): c for c in df.columns}

    def find_col(*candidates: str) -> Optional[str]:
        for name in candidates:
            if name.lower() in lower_map:
                return lower_map[name.lower()]
        return None

    loc_col = find_col("LocationID", "locationid", "location_id", "storeid", "store_id")
    date_col = find_col("Date", "date", "StartDate", "startdate", "dt")
    tavg_col = find_col("TempAvg", "tempavg", "temp_avg", "temperatureavg", "temperature")
    tmin_col = find_col("TempMin", "tempmin", "temp_min")
    tmax_col = find_col("TempMax", "tempmax", "temp_max")
    rain_col = find_col("RainMm", "rainmm", "rain_mm", "rain")
    snow_col = find_col("SnowCm", "snowcm", "snow_cm", "snow")

    # Hard requirements
    if loc_col is None or date_col is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "weather_daily table missing required columns for LocationID/Date. "
                f"Actual columns: {list(df.columns)}"
            ),
        )

    if tavg_col is None and tmin_col is None and tmax_col is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "weather_daily table must have at least one temperature column "
                "(TempAvg/TempMin/TempMax or similar). "
                f"Actual columns: {list(df.columns)}"
            ),
        )

    # Build rename map → canonical names
    rename_map: Dict[str, str] = {}
    if loc_col != "LocationID":
        rename_map[loc_col] = "LocationID"
    if date_col != "Date":
        rename_map[date_col] = "Date"
    if tavg_col and tavg_col != "TempAvg":
        rename_map[tavg_col] = "TempAvg"
    if tmin_col and tmin_col != "TempMin":
        rename_map[tmin_col] = "TempMin"
    if tmax_col and tmax_col != "TempMax":
        rename_map[tmax_col] = "TempMax"
    if rain_col and rain_col != "RainMm":
        rename_map[rain_col] = "RainMm"
    if snow_col and snow_col != "SnowCm":
        rename_map[snow_col] = "SnowCm"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure optional columns exist
    for col in ["TempAvg", "TempMin", "TempMax", "RainMm", "SnowCm"]:
        if col not in df.columns:
            df[col] = np.nan

    # Parse Date to date (no time)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df = df.dropna(subset=["Date"])

    # Keep canonical columns only
    keep_cols = ["LocationID", "Date", "TempAvg", "TempMin", "TempMax", "RainMm", "SnowCm"]
    df = df[[c for c in keep_cols if c in df.columns]]

    return df


def load_history_daily_for_key(product_id: str, channel_id: str, location_id: str) -> pd.DataFrame:
    """
    Load daily history for a single Product–Channel–Location from history_daily.

    Normalizes to:
      Date (datetime.date), Qty (float)
    """
    rows = fetch_all(
        """
        SELECT
          "ProductID",
          "ChannelID",
          "LocationID",
          "StartDate",
          "Qty"
        FROM history_daily
        WHERE "ProductID" = :pid
          AND "ChannelID" = :cid
          AND "LocationID" = :lid
        """,
        {"pid": product_id, "cid": channel_id, "lid": location_id},
    )
    df = pd.DataFrame(rows)

    if df.empty:
        return df

    df = df.rename(columns={"StartDate": "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df = df.dropna(subset=["Date"])

    df["Qty"] = pd.to_numeric(df["Qty"], errors="coerce")
    df = df.dropna(subset=["Qty"])

    return df[["Date", "Qty"]]


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
                field_map.get("productid"),
                field_map.get("productdescr"),
                field_map.get("businessunit"),
                field_map.get("productfamily"),
                field_map.get("channelid"),
                field_map.get("channeldescr"),
                field_map.get("locationid"),
                field_map.get("locationdescr"),
                field_map.get("geography"),
            ]
            cols = [c for c in cols if c]
            bind_i += 1
            k = f"{bind_prefix}{bind_i}"
            params[k] = _to_ilike_pattern(t)
            ors = [f"""{c} ILIKE :{k} ESCAPE '\\'""" for c in cols]
            _add(sql_parts, " OR ".join(ors), op_next)

    return " ".join(sql_parts) if sql_parts else "TRUE"


def _normalize_date_str(v: Any) -> str:
    """
    Take a StartDate/EndDate value from SQL (could be text, date, datetime)
    and return a clean 'YYYY-MM-DD' string.

    If parsing fails, returns the original string.
    """
    if v is None:
        return ""

    # Already a date/datetime
    if isinstance(v, datetime):
        return v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()

    s = str(v).strip()
    if not s:
        return ""

    # Try a few common formats first
    fmts = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y",
        "%d/%m/%y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.date().isoformat()
        except ValueError:
            continue

    # Fallback: let pandas guess, with dayfirst=True
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="raise")
        # pd.to_datetime can return Timestamp or DatetimeIndex, but we expect scalar here
        if isinstance(dt, pd.Timestamp):
            return dt.date().isoformat()
    except Exception:
        pass

    # Last resort: return original string
    return s


# === Startup DDL & indexes ====================================================
@app.on_event("startup")
def startup_ddl():
    logging.getLogger("uvicorn.error").info("Daily module: %s", getattr(daily_mod, "__name__", "unavailable"))
    logging.getLogger("uvicorn.error").info("Weekly module: %s", getattr(weekly_mod, "__name__", "unavailable"))
    logging.getLogger("uvicorn.error").info("Monthly module: %s", getattr(bf, "__name__", "unavailable"))
    with engine.begin() as c:
        # saved searches
        c.execute(
            text(
                """
        CREATE TABLE IF NOT EXISTS saved_search(
          id SERIAL PRIMARY KEY,
          name TEXT NOT NULL,
          query TEXT NOT NULL,
          created_at TIMESTAMPTZ DEFAULT now()
        );
        """
            )
        )
        # history indexes
        c.execute(
            text(
                """
        CREATE INDEX IF NOT EXISTS ix_hist_daily_pcl   ON history_daily("ProductID","ChannelID","LocationID");
        CREATE INDEX IF NOT EXISTS ix_hist_weekly_pcl  ON history_weekly("ProductID","ChannelID","LocationID");
        CREATE INDEX IF NOT EXISTS ix_hist_monthly_pcl ON history_monthly("ProductID","ChannelID","LocationID");
        CREATE INDEX IF NOT EXISTS brin_hist_daily_start   ON history_daily USING BRIN("StartDate");
        CREATE INDEX IF NOT EXISTS brin_hist_weekly_start  ON history_weekly USING BRIN("StartDate");
        CREATE INDEX IF NOT EXISTS brin_hist_monthly_start ON history_monthly USING BRIN("StartDate");
        """
            )
        )
        # cleansed history tables (structure cloned from raw history)
        c.execute(
            text(
                """
        CREATE TABLE IF NOT EXISTS history_cleansed_daily   (LIKE history_daily   INCLUDING ALL);
        CREATE TABLE IF NOT EXISTS history_cleansed_weekly  (LIKE history_weekly  INCLUDING ALL);
        CREATE TABLE IF NOT EXISTS history_cleansed_monthly (LIKE history_monthly INCLUDING ALL);
        """
            )
        )

        # forecast created_at + unique
        c.execute(
            text(
                """
        ALTER TABLE IF EXISTS forecast_daily   ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now();
        ALTER TABLE IF EXISTS forecast_weekly  ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now();
        ALTER TABLE IF EXISTS forecast_monthly ADD COLUMN IF NOT EXISTS created_at timestamptz DEFAULT now();
        """
            )
        )
        for tbl in ("forecast_daily", "forecast_weekly", "forecast_monthly"):
            c.execute(
                text(
                    f"""
                WITH dupe AS (
                  SELECT ctid FROM (
                    SELECT ctid,
                           ROW_NUMBER() OVER (
                             PARTITION BY "ProductID","ChannelID","LocationID","StartDate","EndDate","Method","Type"
                             ORDER BY created_at DESC, ctid DESC
                           ) rn
                    FROM {tbl}
                  ) t WHERE rn > 1
                )
                DELETE FROM {tbl} f USING dupe d WHERE f.ctid = d.ctid;
            """
                )
            )
        c.execute(
            text(
                """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_forecast_daily_xgb
          ON forecast_daily("ProductID","ChannelID","LocationID","StartDate","EndDate","Method","Type");
        CREATE UNIQUE INDEX IF NOT EXISTS ux_forecast_weekly_xgb
          ON forecast_weekly("ProductID","ChannelID","LocationID","StartDate","EndDate","Method","Type");
        CREATE UNIQUE INDEX IF NOT EXISTS ux_forecast_monthly_xgb
          ON forecast_monthly("ProductID","ChannelID","LocationID","StartDate","EndDate","Method","Type");
        """
            )
        )
        # cleanse_profile
        c.execute(
            text(
                """
        CREATE TABLE IF NOT EXISTS cleanse_profile (
          id SERIAL PRIMARY KEY,
          name TEXT UNIQUE NOT NULL,
          settings JSONB NOT NULL DEFAULT '{}'::jsonb,
          is_active BOOLEAN NOT NULL DEFAULT FALSE,
          updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
            )
        )
        c.execute(
            text(
                """
        DO $$
        BEGIN
          IF NOT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name='cleanse_profile' AND column_name='config'
          ) THEN
            ALTER TABLE cleanse_profile ADD COLUMN config JSONB;
          END IF;
        END$$;
        """
            )
        )
        c.execute(
            text(
                """
          ALTER TABLE cleanse_profile
            ALTER COLUMN config SET DEFAULT '{}'::jsonb;
          UPDATE cleanse_profile SET config='{}'::jsonb WHERE config IS NULL;
          ALTER TABLE cleanse_profile
            ALTER COLUMN config SET NOT NULL;
        """
            )
        )

        # Bill of Materials table (will be skipped if bom already exists)
        c.execute(
            text(
                """
        CREATE TABLE IF NOT EXISTS bom (
          id SERIAL PRIMARY KEY,
          "ProductID"         TEXT NOT NULL,
          "ProductName"       TEXT NOT NULL,
          "ItemID"            TEXT NOT NULL,
          "ItemName"          TEXT NOT NULL,
          "ItemQty"           DOUBLE PRECISION NOT NULL,
          "UnitofMeasurement" TEXT NOT NULL
        );
        """
            )
        )

        # classification results
        c.execute(
            text(
                """
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
        """
            )
        )


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


class KeysBody(BaseModel):  # <<< NEW
    keys: List[KeyTriplet]


class SeriesPoint(BaseModel):
    StartDate: str
    Qty: float


class KpiByComboRow(BaseModel):
    ProductID: str
    ChannelID: str
    LocationID: str
    Period: str
    LagConfig: str
    WMAPE: float
    WAPE: float
    MAE: float
    RMSE: float
    MAPE: float
    sMAPE: float
    Bias: float


# === BOM + Supply Models ======================================================

class BOMItem(BaseModel):
    ProductID: str
    ProductName: str
    ItemID: str
    ItemName: str
    ItemQty: float
    UnitofMeasurement: str


class BOMProduct(BaseModel):
    ProductID: str
    ProductName: str


class LotSizeItem(BaseModel):
    ItemID: str
    ItemName: str
    LotSize: float
    UnitofMeasurement: str


class SafetyStockItem(BaseModel):
    ItemID: str
    ItemName: str
    SafetyStockRule: str
    DaysOfSafetyStock: float


class InventoryItem(BaseModel):
    ItemID: str
    LocationID: str
    Qty: float
    UnitofMeasurement: str


class SourcingOrderItem(BaseModel):
    ItemID: str
    LocationID: str
    ArrivalDate: str
    Qty: float
    UnitofMeasurement: str


class SupplyForecastRow(BaseModel):
    ProductID: str
    LocationID: str
    StartDate: str
    EndDate: str
    Qty: float


class ItemForecastRow(BaseModel):
    ItemID: str
    ItemName: str
    LocationID: str
    StartDate: str
    EndDate: str
    Qty: float
    UnitofMeasurement: str


class RecommendedSourcingRow(BaseModel):
    ItemID: str
    ItemName: str
    LocationID: str
    StartDate: str
    EndDate: str
    Qty: float
    UnitofMeasurement: str

class InventoryProfilePoint(BaseModel):
    ItemID: str
    LocationID: str
    Date: str
    DemandQty: float
    SafetyStockQty: float
    EndInventoryQty: float
    InboundConfirmedQty: float
    RecommendedOrderQty: float




# === Basic dropdowns ==========================================================
@app.get("/")
def root():
    return {"message": "Planwise API"}


@app.get("/api/products", response_model=List[Product])
def get_products():
    return fetch_all("SELECT * FROM product")


@app.get("/api/channels", response_model=List[Channel])
def get_channels():
    return fetch_all("SELECT * FROM channel")


@app.get("/api/locations", response_model=List[Location])
def get_locations():
    return fetch_all(
        'SELECT "LocationID","Location_Descr" AS "LocationDescr","Level","Geography" FROM location'
    )


# === Bill of Materials (BOM) ==================================================
@app.get("/api/bom/products", response_model=List[BOMProduct])
def get_bom_products():
    """
    Return distinct products that have a BOM defined.
    """
    sql = """
      SELECT DISTINCT
        "ProductID",
        "ProductName"
      FROM bom
      ORDER BY "ProductID","ProductName";
    """
    return fetch_all(sql, {})


@app.get("/api/bom", response_model=List[BOMItem])
def get_bom(productId: Optional[str] = None):
    """
    Return BOM rows.
    - If productId is provided, filter BOM for that product.
    - Otherwise returns all BOM entries.
    """
    params: Dict[str, Any] = {}
    where_sql = ""

    if productId:
        where_sql = 'WHERE "ProductID" = :pid'
        params["pid"] = productId

    sql = f"""
      SELECT
        "ProductID",
        "ProductName",
        "ItemID",
        "ItemName",
        "ItemQty",
        "UnitofMeasurement"
      FROM bom
      {where_sql}
      ORDER BY "ProductID","ItemID";
    """
    return fetch_all(sql, params)


# === Lot Size ================================================================
@app.get("/api/lotsize", response_model=List[LotSizeItem])
def get_lotsize(itemId: Optional[str] = None):
    """
    Return LotSize rows.
    - If itemId is provided, filter by that ItemID.
    """
    params: Dict[str, Any] = {}
    where_sql = ""
    if itemId:
        where_sql = 'WHERE "ItemID" = :iid'
        params["iid"] = itemId

    sql = f'''
      SELECT "ItemID","ItemName","LotSize","UnitofMeasurement"
      FROM lotsize
      {where_sql}
      ORDER BY "ItemID";
    '''
    return fetch_all(sql, params)


# === Safety Stock ============================================================
@app.get("/api/safetystock", response_model=List[SafetyStockItem])
def get_safetystock(itemId: Optional[str] = None):
    """
    Return SafetyStock rows.
    - If itemId is provided, filter by that ItemID.
    """
    params: Dict[str, Any] = {}
    where_sql = ""
    if itemId:
        where_sql = 'WHERE "ItemID" = :iid'
        params["iid"] = itemId

    sql = f'''
      SELECT "ItemID","ItemName","SafetyStockRule","DaysOfSafetyStock"
      FROM safetystock
      {where_sql}
      ORDER BY "ItemID";
    '''
    return fetch_all(sql, params)


# === Inventory ===============================================================
@app.get("/api/inventory", response_model=List[InventoryItem])
def get_inventory(
    itemId: Optional[str] = None,
    locationId: Optional[str] = None,
):
    """
    Return Inventory rows.
    Optional filters:
      - itemId
      - locationId
    """
    params: Dict[str, Any] = {}
    where_parts: List[str] = []

    if itemId:
        where_parts.append('"ItemID" = :iid')
        params["iid"] = itemId
    if locationId:
        where_parts.append('"LocationID" = :lid')
        params["lid"] = locationId

    where_sql = ""
    if where_parts:
        where_sql = "WHERE " + " AND ".join(where_parts)

    sql = f'''
      SELECT "ItemID","LocationID","Qty","UnitofMeasurement"
      FROM inventory
      {where_sql}
      ORDER BY "LocationID","ItemID";
    '''
    return fetch_all(sql, params)


# === Sourcing Orders =========================================================
@app.get("/api/sourcingorder", response_model=List[SourcingOrderItem])
def get_sourcing_order(
    itemId: Optional[str] = None,
    locationId: Optional[str] = None,
    fromDate: Optional[str] = None,
    toDate: Optional[str] = None,
):
    """
    Return SourcingOrder rows.
    Optional filters:
      - itemId
      - locationId
      - fromDate / toDate (on ArrivalDate, inclusive)
    Dates should be in YYYY-MM-DD format if used.
    """
    params: Dict[str, Any] = {}
    where_parts: List[str] = []

    if itemId:
        where_parts.append('"ItemID" = :iid')
        params["iid"] = itemId
    if locationId:
        where_parts.append('"LocationID" = :lid')
        params["lid"] = locationId
    if fromDate:
        where_parts.append('"ArrivalDate" >= :fromd')
        params["fromd"] = fromDate
    if toDate:
        where_parts.append('"ArrivalDate" <= :tod')
        params["tod"] = toDate

    where_sql = ""
    if where_parts:
        where_sql = "WHERE " + " AND ".join(where_parts)

    sql = f'''
      SELECT "ItemID","LocationID","ArrivalDate","Qty","UnitofMeasurement"
      FROM sourcingorder
      {where_sql}
      ORDER BY "ArrivalDate","LocationID","ItemID";
    '''
    return fetch_all(sql, params)


#===========Forecast Aggregate================

# === Forecast aggregate for Supply Data ======================================

@app.get("/api/supply-forecast", response_model=List[SupplyForecastRow])
def api_supply_forecast(
    period: Literal["daily", "weekly", "monthly"] = "daily",
    method: str = "XGBoost",
):
    """
    Aggregate forecast for Supply Data.

    Step 1:
      For each ProductID + ChannelID + LocationID + StartDate + EndDate,
      sum Qty for Type 1 and 2 only (for the given Period label and Method).

    Step 2:
      For each ProductID + LocationID + StartDate + EndDate,
      sum across all Channels.

    Final output columns:
      ProductID, LocationID, StartDate, EndDate, Qty
    """

    table = {
        "daily": "forecast_daily",
        "weekly": "forecast_weekly",
        "monthly": "forecast_monthly",
    }[period]

    period_label = {
        "daily": "Daily",
        "weekly": "Weekly",
        "monthly": "Monthly",
    }[period]

    params = {
        "period_label": period_label,
        "method": method,
    }

    sql = f"""
      WITH base AS (
        SELECT
          "ProductID",
          "ChannelID",
          "LocationID",
          "StartDate",
          "EndDate",
          "Type",
          "Qty"
        FROM {table}
        WHERE TRIM("Type"::text) IN ('1','2')
          AND COALESCE("Period",'') = :period_label
          AND "Method" = :method
      ),
      per_key AS (
        -- sum Type 1 + 2 per Product + Channel + Location + StartDate + EndDate
        SELECT
          "ProductID",
          "ChannelID",
          "LocationID",
          "StartDate",
          "EndDate",
          SUM("Qty")::float AS qty_type12
        FROM base
        GROUP BY
          "ProductID",
          "ChannelID",
          "LocationID",
          "StartDate",
          "EndDate"
      ),
      per_location AS (
        -- then sum across Channels for same Product + Location + StartDate + EndDate
        SELECT
          "ProductID",
          "LocationID",
          "StartDate",
          "EndDate",
          SUM(qty_type12)::float AS "Qty"
        FROM per_key
        GROUP BY
          "ProductID",
          "LocationID",
          "StartDate",
          "EndDate"
      )
      SELECT
        "ProductID",
        "LocationID",
        "StartDate",
        "EndDate",
        "Qty"
      FROM per_location
      ORDER BY
        "ProductID",
        "LocationID",
        "StartDate",
        "EndDate";
    """

    rows = fetch_all(sql, params)

    # Normalize for JSON / Pydantic
    for r in rows:
        r["ProductID"] = str(r.get("ProductID", ""))
        r["LocationID"] = str(r.get("LocationID", ""))
        r["StartDate"] = _normalize_date_str(r.get("StartDate"))
        r["EndDate"] = _normalize_date_str(r.get("EndDate"))
        r["Qty"] = float(r.get("Qty") or 0.0)

    return rows


@app.get("/api/item-forecast", response_model=List[ItemForecastRow])
def api_item_forecast(
    period: Literal["daily", "weekly", "monthly"] = "daily",
    method: str = "XGBoost",
):
    """
    Item-level forecast based on:
    - Forecast_* tables (sum of Type 1 + 2 across channels)
    - BOM (ItemQty per product unit)

    Steps:
      1) Same as /api/supply-forecast:
         For each ProductID + LocationID + StartDate + EndDate,
         sum Qty over Type in (1,2) and all channels (for given Period+Method).
      2) Join with BOM on ProductID and multiply:
           item_qty = product_forecast_qty * ItemQty
      3) Aggregate by ItemID + ItemName + LocationID + StartDate + EndDate.
    """

    table = {
        "daily": "forecast_daily",
        "weekly": "forecast_weekly",
        "monthly": "forecast_monthly",
    }[period]

    period_label = {
        "daily": "Daily",
        "weekly": "Weekly",
        "monthly": "Monthly",
    }[period]

    params = {
        "period_label": period_label,
        "method": method,
    }

    sql = f"""
      WITH base AS (
        SELECT
          "ProductID",
          "ChannelID",
          "LocationID",
          "StartDate",
          "EndDate",
          "Type",
          "Qty"
        FROM {table}
        WHERE TRIM("Type"::text) IN ('1','2')
          AND COALESCE("Period",'') = :period_label
          AND "Method" = :method
      ),
      per_key AS (
        -- sum Type 1 + 2 per Product + Channel + Location + StartDate + EndDate
        SELECT
          "ProductID",
          "ChannelID",
          "LocationID",
          "StartDate",
          "EndDate",
          SUM("Qty")::float AS qty_type12
        FROM base
        GROUP BY
          "ProductID",
          "ChannelID",
          "LocationID",
          "StartDate",
          "EndDate"
      ),
      per_location AS (
        -- sum across Channels for same Product + Location + StartDate + EndDate
        SELECT
          "ProductID",
          "LocationID",
          "StartDate",
          "EndDate",
          SUM(qty_type12)::float AS prod_qty
        FROM per_key
        GROUP BY
          "ProductID",
          "LocationID",
          "StartDate",
          "EndDate"
      ),
      per_item AS (
        -- join BOM and convert product forecast to item quantities
        SELECT
          b."ItemID",
          b."ItemName",
          b."UnitofMeasurement",
          f."LocationID",
          f."StartDate",
          f."EndDate",
          SUM(f.prod_qty * b."ItemQty")::float AS "Qty"
        FROM per_location f
        JOIN bom b
          ON b."ProductID" = f."ProductID"
        GROUP BY
          b."ItemID",
          b."ItemName",
          b."UnitofMeasurement",
          f."LocationID",
          f."StartDate",
          f."EndDate"
      )
      SELECT
        "ItemID",
        "ItemName",
        "UnitofMeasurement",
        "LocationID",
        "StartDate",
        "EndDate",
        "Qty"
      FROM per_item
      ORDER BY
        "ItemID",
        "LocationID",
        "StartDate",
        "EndDate";
    """

    rows = fetch_all(sql, params)

    # Normalize for JSON / Pydantic (including date formatting)
    for r in rows:
        r["ItemID"] = str(r.get("ItemID", ""))
        r["ItemName"] = str(r.get("ItemName", ""))
        r["UnitofMeasurement"] = str(r.get("UnitofMeasurement") or "")
        r["LocationID"] = str(r.get("LocationID", ""))
        r["StartDate"] = _normalize_date_str(r.get("StartDate"))
        r["EndDate"] = _normalize_date_str(r.get("EndDate"))
        r["Qty"] = float(r.get("Qty") or 0.0)

    return rows

@app.get("/api/recommended-sourcing", response_model=List[RecommendedSourcingRow])
def api_recommended_sourcing(
    period: Literal["daily", "weekly", "monthly"] = "daily",
    method: str = "XGBoost",
):
    """
    Recommended sourcing order (simple MRP-style) using:
      - Item-level forecast (from /api/item-forecast)
      - Inventory (starting on-hand, in grams)
      - SourcingOrder (future inbound, in grams)
      - LotSize (round up to this lot, same unit as demand)
      - SafetyStock (DaysOfSafetyStock → buffer per day)

    Logic (per ItemID + LocationID, per date, **all in grams**):

      available_before = on_hand + inbound_today
      base_eoh        = available_before - demand_today

      safety_stock_qty = DaysOfSafetyStock * demand_today

      If base_eoh >= safety_stock_qty:
          rec_qty = 0
          end_of_day = base_eoh
      Else:
          needed   = safety_stock_qty - base_eoh
          rec_qty  = ceil(needed / lot_size) * lot_size
          end_of_day = base_eoh + rec_qty

      Carry forward:
          on_hand_next_day = end_of_day
    """

    # 1) Get item-level forecast using existing logic
    item_fc = api_item_forecast(period=period, method=method)
    if not item_fc:
        return []

    fc_df = pd.DataFrame(item_fc)
    if fc_df.empty:
        return []

    # Parse dates
    fc_df["StartDate"] = pd.to_datetime(fc_df["StartDate"], errors="coerce")
    fc_df["EndDate"] = pd.to_datetime(fc_df["EndDate"], errors="coerce")
    fc_df = fc_df.dropna(subset=["StartDate"])

    # 2) Load supporting tables
    inv_rows = fetch_all(
        'SELECT "ItemID","LocationID","Qty" FROM inventory',
        {},
    )
    inv_df = pd.DataFrame(inv_rows) if inv_rows else pd.DataFrame(columns=["ItemID", "LocationID", "Qty"])

    lot_rows = fetch_all(
        'SELECT "ItemID","LotSize" FROM lotsize',
        {},
    )
    lot_df = pd.DataFrame(lot_rows) if lot_rows else pd.DataFrame(columns=["ItemID", "LotSize"])

    ss_rows = fetch_all(
        'SELECT "ItemID","DaysOfSafetyStock" FROM safetystock',
        {},
    )
    ss_df = pd.DataFrame(ss_rows) if ss_rows else pd.DataFrame(columns=["ItemID", "DaysOfSafetyStock"])

    so_rows = fetch_all(
        'SELECT "ItemID","LocationID","ArrivalDate","Qty" FROM sourcingorder',
        {},
    )
    so_df = pd.DataFrame(so_rows) if so_rows else pd.DataFrame(columns=["ItemID", "LocationID", "ArrivalDate", "Qty"])

    # === Normalise units (everything in grams) ================================
    if not inv_df.empty:
        # inventory in kg → grams
        inv_df["Qty"] = pd.to_numeric(inv_df["Qty"], errors="coerce").fillna(0.0) * 1000.0

    if not lot_df.empty:
        lot_df["LotSize"] = pd.to_numeric(lot_df["LotSize"], errors="coerce").fillna(1.0)

    if not ss_df.empty:
        ss_df["DaysOfSafetyStock"] = pd.to_numeric(ss_df["DaysOfSafetyStock"], errors="coerce").fillna(0.0)

    if not so_df.empty:
        # inbound sourcing orders in kg → grams
        so_df["Qty"] = pd.to_numeric(so_df["Qty"], errors="coerce").fillna(0.0) * 1000.0
        so_df["ArrivalDate"] = pd.to_datetime(so_df["ArrivalDate"], errors="coerce")

    # Build lookup maps
    inv_map: dict[tuple[str, str], float] = {}
    if not inv_df.empty:
        inv_group = inv_df.groupby(["ItemID", "LocationID"], as_index=False)["Qty"].sum()
        for _, row in inv_group.iterrows():
            inv_map[(str(row["ItemID"]), str(row["LocationID"]))] = float(row["Qty"])

    lot_map: dict[str, float] = {}
    if not lot_df.empty:
        for _, row in lot_df.iterrows():
            lot_map[str(row["ItemID"])] = float(row["LotSize"]) if row["LotSize"] else 1.0

    ss_map: dict[str, float] = {}
    if not ss_df.empty:
        for _, row in ss_df.iterrows():
            ss_map[str(row["ItemID"])] = float(row["DaysOfSafetyStock"]) if row["DaysOfSafetyStock"] else 0.0

    inbound_map: dict[tuple[str, str, date], float] = {}
    if not so_df.empty:
        so_df = so_df.dropna(subset=["ArrivalDate"])
        for _, row in so_df.iterrows():
            key = (str(row["ItemID"]), str(row["LocationID"]), row["ArrivalDate"].date())
            inbound_map[key] = inbound_map.get(key, 0.0) + float(row["Qty"])

    results: List[RecommendedSourcingRow] = []

    # Work per (ItemID, ItemName, LocationID)
    fc_df["ItemID"] = fc_df["ItemID"].astype(str)
    fc_df["LocationID"] = fc_df["LocationID"].astype(str)

    for (item_id, item_name, loc_id), grp in fc_df.groupby(["ItemID", "ItemName", "LocationID"], as_index=False):
        grp = grp.sort_values("StartDate").reset_index(drop=True)

        # starting on-hand inventory (grams)
        on_hand = inv_map.get((item_id, loc_id), 0.0)
        safety_days = ss_map.get(item_id, 0.0)
        lot_size = lot_map.get(item_id, 1.0) or 1.0
        if lot_size <= 0:
            lot_size = 1.0  # guard
        uom = str(grp["UnitofMeasurement"].iloc[0] if "UnitofMeasurement" in grp.columns else "")

        for _, row in grp.iterrows():
            dt = row["StartDate"].date()
            demand = float(row["Qty"] or 0.0)  # already in grams from /api/item-forecast

            inbound_today = inbound_map.get((item_id, loc_id, dt), 0.0)
            available_before = on_hand + inbound_today

            # Safety stock in grams (DaysOfCoverage * today's demand)
            safety_stock_qty = safety_days * demand

            # End-of-day inventory *before* any new order
            base_eoh = available_before - demand

            if base_eoh >= safety_stock_qty:
                # no order needed; we are above safety stock
                rec_qty = 0.0
                end_of_day = base_eoh
            else:
                # order enough to reach safety stock, rounded up to lot size
                needed = max(0.0, safety_stock_qty - base_eoh)
                rec_qty = math.ceil(needed / lot_size) * lot_size if needed > 0 else 0.0
                end_of_day = base_eoh + rec_qty

                if rec_qty > 0:
                    results.append(
                        RecommendedSourcingRow(
                            ItemID=item_id,
                            ItemName=item_name,
                            LocationID=loc_id,
                            StartDate=_normalize_date_str(dt),
                            EndDate=_normalize_date_str(row["EndDate"]),
                            Qty=float(rec_qty),
                            UnitofMeasurement=uom,
                        )
                    )

            # carry inventory forward to next day
            on_hand = end_of_day

    # Sorted output
    results_sorted = sorted(
        results,
        key=lambda r: (r.ItemID, r.LocationID, r.StartDate),
    )
    return results_sorted



from typing import List  # already imported at top

@app.get("/api/inventory-profile", response_model=List[InventoryProfilePoint])
def api_inventory_profile(
    itemId: str,
    locationId: str,
    period: Literal["daily", "weekly", "monthly"] = "daily",
    method: str = "XGBoost",
):
    """
    Inventory profile for a single ItemID + LocationID.

    All quantities are handled in the same unit as the forecast/lot size
    (in your setup: grams). We convert inventory and sourcing orders from kg → g.

    For each forecast bucket (e.g. next 7 days when daily):

      available_before = on_hand + inbound_today
      base_eoh        = available_before - demand_today

      safety_stock_qty = DaysOfSafetyStock * demand_today   (days of coverage)

      If base_eoh >= safety_stock_qty:
          rec_qty    = 0
          end_of_day = base_eoh
      Else:
          needed     = safety_stock_qty - base_eoh
          rec_qty    = ceil(needed / lot_size) * lot_size
          end_of_day = base_eoh + rec_qty

      Carry forward:
          on_hand_next_day = end_of_day
    """

    # 1) Item-level forecast (reuse existing logic)
    item_fc = api_item_forecast(period=period, method=method)
    if not item_fc:
        return []

    fc_df = pd.DataFrame(item_fc)
    if fc_df.empty:
        return []

    fc_df["ItemID"] = fc_df["ItemID"].astype(str)
    fc_df["LocationID"] = fc_df["LocationID"].astype(str)

    # filter to requested item + location
    fc_df = fc_df[(fc_df["ItemID"] == itemId) & (fc_df["LocationID"] == locationId)]
    if fc_df.empty:
        return []

    fc_df["StartDate"] = pd.to_datetime(fc_df["StartDate"], errors="coerce")
    fc_df = fc_df.dropna(subset=["StartDate"]).sort_values("StartDate")

    # 2) Inventory, lot size, safety stock and confirmed sourcing orders
    inv_rows = fetch_all(
        'SELECT "ItemID","LocationID","Qty" FROM inventory '
        'WHERE "ItemID" = :iid AND "LocationID" = :lid',
        {"iid": itemId, "lid": locationId},
    )
    inv_df = pd.DataFrame(inv_rows) if inv_rows else pd.DataFrame(columns=["ItemID", "LocationID", "Qty"])

    # inventory in kg → grams
    on_hand = 0.0
    if not inv_df.empty:
        inv_df["Qty"] = pd.to_numeric(inv_df["Qty"], errors="coerce").fillna(0.0) * 1000.0
        on_hand = float(inv_df["Qty"].sum())

    lot_rows = fetch_all(
        'SELECT "ItemID","LotSize" FROM lotsize WHERE "ItemID" = :iid',
        {"iid": itemId},
    )
    lot_df = pd.DataFrame(lot_rows) if lot_rows else pd.DataFrame(columns=["ItemID", "LotSize"])
    lot_size = 1.0
    if not lot_df.empty:
        lot_size = float(pd.to_numeric(lot_df["LotSize"], errors="coerce").fillna(1.0).iloc[0])
        if lot_size <= 0:
            lot_size = 1.0

    ss_rows = fetch_all(
        'SELECT "ItemID","DaysOfSafetyStock" FROM safetystock WHERE "ItemID" = :iid',
        {"iid": itemId},
    )
    ss_df = pd.DataFrame(ss_rows) if ss_rows else pd.DataFrame(columns=["ItemID", "DaysOfSafetyStock"])
    safety_days = 0.0
    if not ss_df.empty:
        safety_days = float(pd.to_numeric(ss_df["DaysOfSafetyStock"], errors="coerce").fillna(0.0).iloc[0])

    so_rows = fetch_all(
        'SELECT "ItemID","LocationID","ArrivalDate","Qty" FROM sourcingorder '
        'WHERE "ItemID" = :iid AND "LocationID" = :lid',
        {"iid": itemId, "lid": locationId},
    )
    so_df = pd.DataFrame(so_rows) if so_rows else pd.DataFrame(columns=["ItemID", "LocationID", "ArrivalDate", "Qty"])

    inbound_map: dict[tuple[str, str, date], float] = {}
    if not so_df.empty:
        # dd/mm/yyyy → datetime
        so_df["ArrivalDate"] = pd.to_datetime(so_df["ArrivalDate"], errors="coerce", dayfirst=True)
        # inbound in kg → grams
        so_df["Qty"] = pd.to_numeric(so_df["Qty"], errors="coerce").fillna(0.0) * 1000.0
        so_df = so_df.dropna(subset=["ArrivalDate"])
        for _, row in so_df.iterrows():
            key = (str(row["ItemID"]), str(row["LocationID"]), row["ArrivalDate"].date())
            inbound_map[key] = inbound_map.get(key, 0.0) + float(row["Qty"])

    # 3) Walk through the forecast horizon and build profile rows
    results: List[InventoryProfilePoint] = []

    for _, row in fc_df.iterrows():
        dt = row["StartDate"].date()
        demand = float(row["Qty"] or 0.0)  # already in grams from /api/item-forecast

        inbound_today = inbound_map.get((itemId, locationId, dt), 0.0)
        available_before = on_hand + inbound_today

        # safety stock in grams (DaysOfCoverage * today's demand)
        safety_stock_qty = safety_days * demand

        # end-of-day inventory BEFORE any new order
        base_eoh = available_before - demand

        if base_eoh >= safety_stock_qty:
            rec_qty = 0.0
            end_of_day = base_eoh
        else:
            needed = max(0.0, safety_stock_qty - base_eoh)
            rec_qty = math.ceil(needed / lot_size) * lot_size if needed > 0 else 0.0
            end_of_day = base_eoh + rec_qty

        results.append(
            InventoryProfilePoint(
                ItemID=itemId,
                LocationID=locationId,
                Date=dt.isoformat(),
                DemandQty=demand,
                SafetyStockQty=safety_stock_qty,
                InboundConfirmedQty=inbound_today,
                RecommendedOrderQty=rec_qty,
                EndInventoryQty=end_of_day,
            )
        )

        # carry to next day
        on_hand = end_of_day

    # (if you want only first N points, slice here, e.g. results = results[:7])
    return results





# === Saved searches ===========================================================
FIELD_MAP = _make_field_map("fe", "p", "c", "l")


class SearchResult(BaseModel):
    query: str
    count: int
    keys: List[KeyTriplet]


@app.get("/api/saved-searches")
def list_saved():
    return fetch_all(
        """
        SELECT id, name, query, created_at
        FROM saved_search
        ORDER BY created_at DESC
    """
    )


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
    where_sql = _build_where_clause_with_map(
        q or "", params, _make_field_map("fe", "p", "c", "l"), bind_prefix="fe_b"
    )
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


# === KPI BY COMBO (Monthly lag variants, from DB) ============================
@app.get("/api/kpi/monthly-bycombo")
def get_kpi_monthly_bycombo():
    """
    Returns rows with columns:
      ProductID, ChannelID, LocationID,
      Period, LagConfig,
      WMAPE, WAPE, MAE, RMSE, MAPE, sMAPE, Bias
    """

    sql = """
    SELECT
      "ProductID",
      "ChannelID",
      "LocationID",
      "Period",
      "LagConfig",
      "WMAPE",
      "WAPE",
      "MAE",
      "RMSE",
      "MAPE",
      "sMAPE",
      "Bias"
    FROM (
      -- lags_oldest
      SELECT
        "ProductID",
        "ChannelID",
        "LocationID",
        COALESCE("Period", 'Monthly')      AS "Period",
        COALESCE("LagConfig", 'lag0')   AS "LagConfig",
        "WMAPE",
        "WAPE",
        "MAE",
        "RMSE",
        "MAPE",
        "sMAPE",
        "Bias"
      FROM kpi_bycombo_monthly_18m_lags_oldest

      UNION ALL

      -- lags_oldest_mid
      SELECT
        "ProductID",
        "ChannelID",
        "LocationID",
        COALESCE("Period", 'Monthly')          AS "Period",
        COALESCE("LagConfig", 'lags01')   AS "LagConfig",
        "WMAPE",
        "WAPE",
        "MAE",
        "RMSE",
        "MAPE",
        "sMAPE",
        "Bias"
      FROM kpi_bycombo_monthly_18m_lags_oldest_mid

      UNION ALL

      -- lags_all3
      SELECT
        "ProductID",
        "ChannelID",
        "LocationID",
        COALESCE("Period", 'Monthly')        AS "Period",
        COALESCE("LagConfig", 'lag012')       AS "LagConfig",
        "WMAPE",
        "WAPE",
        "MAE",
        "RMSE",
        "MAPE",
        "sMAPE",
        "Bias"
      FROM kpi_bycombo_monthly_18m_lags_all3
    ) t
    ORDER BY "Period","LagConfig","LocationID","ProductID","ChannelID";
    """

    rows = fetch_all(sql, {})
    # Make sure everything is JSON-friendly
    for r in rows:
        r["ProductID"] = str(r.get("ProductID", ""))
        r["ChannelID"] = str(r.get("ChannelID", ""))
        r["LocationID"] = str(r.get("LocationID", ""))
        r["Period"] = str(r.get("Period", "Monthly"))
        r["LagConfig"] = str(r.get("LagConfig", ""))

    return rows


# === NEW: Weather–Sales endpoints ============================================

@app.get("/api/weather-sales")
def api_weather_sales(
    productId: str,
    channelId: str,
    locationId: str,
    period: Literal["daily", "weekly", "monthly"] = "daily",
    metric: Literal["TempAvg", "TempMin", "TempMax", "RainMm", "SnowCm"] = "TempAvg",
):
    """
    Return joined weather + sales series for a given P–C–L key.

    Response: [{ date: 'YYYY-MM-DD', weatherMetric: float, quantity: float }, ...]
    Period:
      - daily   → daily data
      - weekly  → grouped by week (weather averaged, qty summed)
      - monthly → grouped by month (weather averaged, qty summed)
    """
    weather_df = load_weather_daily()
    hist_df = load_history_daily_for_key(productId, channelId, locationId)

    if weather_df.empty:
        raise HTTPException(status_code=400, detail="No weather data in weather_daily.")

    if hist_df.empty:
        raise HTTPException(
            status_code=400,
            detail="No history data for this Product–Channel–Location.",
        )

    if metric not in weather_df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Requested metric '{metric}' not found in weather data. Available: {list(weather_df.columns)}",
        )

    # Filter weather by location
    w = weather_df[weather_df["LocationID"] == locationId].copy()
    if w.empty:
        raise HTTPException(
            status_code=400,
            detail=f"No weather data for LocationID={locationId}.",
        )

    # Convert Date to datetime for grouping
    w["Date"] = pd.to_datetime(w["Date"])
    h = hist_df.copy()
    h["Date"] = pd.to_datetime(h["Date"])

    # Align period
    if period == "daily":
        w_grp = w[["Date", metric]].groupby("Date", as_index=False)[metric].mean()
        h_grp = h.groupby("Date", as_index=False)["Qty"].sum()
    else:
        if period == "weekly":
            w["PeriodKey"] = w["Date"].dt.to_period("W-MON").dt.start_time
            h["PeriodKey"] = h["Date"].dt.to_period("W-MON").dt.start_time
        else:  # monthly
            w["PeriodKey"] = w["Date"].dt.to_period("M").dt.start_time
            h["PeriodKey"] = h["Date"].dt.to_period("M").dt.start_time

        w_grp = w.groupby("PeriodKey", as_index=False)[metric].mean().rename(columns={"PeriodKey": "Date"})
        h_grp = h.groupby("PeriodKey", as_index=False)["Qty"].sum().rename(columns={"PeriodKey": "Date"})

    merged = pd.merge(w_grp, h_grp, on="Date", how="inner")

    if merged.empty:
        return []

    merged = merged.sort_values("Date")

    out = []
    for _, row in merged.iterrows():
        out.append(
            {
                "date": row["Date"].date().isoformat(),
                "weatherMetric": float(row[metric]) if pd.notna(row[metric]) else None,
                "quantity": float(row["Qty"]) if pd.notna(row["Qty"]) else None,
            }
        )

    return out


@app.get("/api/weather-correlations")
def api_weather_correlations(
    productId: str,
    channelId: str,
    locationId: str,
    period: Literal["daily", "weekly", "monthly"] = "daily",
):
    """
    Compute Pearson correlation between weather variables and Qty
    for a given Product–Channel–Location and period.

    Returns dict like:
      {
        "TempAvg": 0.42,
        "TempMin": 0.38,
        "TempMax": 0.45,
        "RainMm": -0.10,
        "SnowCm": 0.01
      }
    (values can be null if not computable)
    """
    weather_df = load_weather_daily()
    hist_df = load_history_daily_for_key(productId, channelId, locationId)

    base = {
        "TempAvg": None,
        "TempMin": None,
        "TempMax": None,
        "RainMm": None,
        "SnowCm": None,
    }

    if weather_df.empty or hist_df.empty:
        return base

    w = weather_df[weather_df["LocationID"] == locationId].copy()
    if w.empty:
        return base

    w["Date"] = pd.to_datetime(w["Date"])
    h = hist_df.copy()
    h["Date"] = pd.to_datetime(h["Date"])

    if period == "daily":
        w_grp = w.groupby("Date", as_index=False)[["TempAvg", "TempMin", "TempMax", "RainMm", "SnowCm"]].mean()
        h_grp = h.groupby("Date", as_index=False)["Qty"].sum()
    else:
        if period == "weekly":
            w["PeriodKey"] = w["Date"].dt.to_period("W-MON").dt.start_time
            h["PeriodKey"] = h["Date"].dt.to_period("W-MON").dt.start_time
        else:  # monthly
            w["PeriodKey"] = w["Date"].dt.to_period("M").dt.start_time
            h["PeriodKey"] = h["Date"].dt.to_period("M").dt.start_time

        w_grp = (
            w.groupby("PeriodKey", as_index=False)[["TempAvg", "TempMin", "TempMax", "RainMm", "SnowCm"]]
            .mean()
            .rename(columns={"PeriodKey": "Date"})
        )
        h_grp = (
            h.groupby("PeriodKey", as_index=False)["Qty"]
            .sum()
            .rename(columns={"PeriodKey": "Date"})
        )

    merged = pd.merge(w_grp, h_grp, on="Date", how="inner")
    if merged.empty:
        return base

    results: Dict[str, Optional[float]] = {}
    for col in ["TempAvg", "TempMin", "TempMax", "RainMm", "SnowCm"]:
        if col not in merged.columns:
            results[col] = None
            continue

        sub = merged[[col, "Qty"]].dropna()
        if len(sub) < 3:
            results[col] = None
        else:
            r = sub[col].corr(sub["Qty"])
            results[col] = float(r) if pd.notna(r) else None

    # ensure all keys exist
    for k in base:
        results.setdefault(k, None)

    return results


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
    where_fe = _build_where_clause_with_map(
        q or "", params_fe, _make_field_map("fe", "p", "c", "l"), "fe_b"
    )
    where_fc = _build_where_clause_with_map(
        q or "", params_fc, _make_field_map("f", "p", "c", "l"), "fc_b"
    )

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

# === NEW: Forecast rows by keys (used by Angular Forecast page) ==============
@app.post("/api/forecast/{bucket}-by-keys")  # <<< NEW
def forecast_by_keys(
    bucket: Literal["daily", "weekly", "monthly"],
    body: KeysBody = Body(...)
):
    """
    Return raw forecast rows for the given list of Product–Channel–Location keys
    and selected bucket (daily/weekly/monthly).

    This matches the Angular /forecast page which posts:
      POST /api/forecast/{bucket}-by-keys
      { "keys": [ { ProductID, ChannelID, LocationID }, ... ] }
    """
    keys = body.keys or []
    if not keys:
        return []

    ftable = _forecast_table(bucket)

    period_label = {
        "daily": "Daily",
        "weekly": "Weekly",
        "monthly": "Monthly",
    }[bucket]

    params: Dict[str, Any] = {"period_label": period_label}
    clauses: List[str] = []

    for i, k in enumerate(keys):
        pid_k = f"p{i}"
        cid_k = f"c{i}"
        lid_k = f"l{i}"
        params[pid_k] = k.ProductID
        params[cid_k] = k.ChannelID
        params[lid_k] = k.LocationID
        clauses.append(
            f'("ProductID","ChannelID","LocationID") = (:{pid_k},:{cid_k},:{lid_k})'
        )

    where_keys = " OR ".join(clauses)
    
    sql = f"""
      SELECT
        "ProductID",
        "ChannelID",
        "LocationID",
        "Method",
        COALESCE("Period",'')          AS "Period",
        "StartDate",
        "EndDate",
        COALESCE("Type"::text,'')      AS "Type",
        COALESCE("Qty",0)::float       AS "Qty",
        COALESCE("Level"::text,'')     AS "Level"
      FROM {ftable}
      WHERE COALESCE("Period",'') = :period_label
        AND ({where_keys})
      ORDER BY
        "ProductID",
        "ChannelID",
        "LocationID",
        "StartDate",
        "EndDate",
        "Method",
        "Type";
    """


    rows = fetch_all(sql, params)

    # Normalize for JSON / Angular table
    for r in rows:
        r["ProductID"] = str(r.get("ProductID", ""))
        r["ChannelID"] = str(r.get("ChannelID", ""))
        r["LocationID"] = str(r.get("LocationID", ""))
        r["Method"] = str(r.get("Method", ""))
        r["Period"] = str(r.get("Period", ""))
        r["Type"] = str(r.get("Type", ""))
        r["Level"] = str(r.get("Level", ""))

        r["StartDate"] = _normalize_date_str(r.get("StartDate"))
        r["EndDate"] = _normalize_date_str(r.get("EndDate"))

        r["Qty"] = float(r.get("Qty") or 0.0)

    return rows

@app.post("/api/history/{bucket}-by-keys")
def history_by_keys(
    bucket: Literal["daily", "weekly", "monthly"],
    body: KeysBody = Body(...),
):
    """
    Return raw history rows for the given list of Product–Channel–Location keys
    and selected bucket (daily/weekly/monthly).

    Angular posts:
      POST /api/history/{bucket}-by-keys
      { "keys": [ { ProductID, ChannelID, LocationID }, ... ] }
    """
    keys = body.keys or []
    if not keys:
        return []

    # history_daily / history_weekly / history_monthly
    hist_table, _, _ = _tbl_freq(bucket)

    params: Dict[str, Any] = {}
    clauses: List[str] = []

    for i, k in enumerate(keys):
        pid_k = f"p{i}"
        cid_k = f"c{i}"
        lid_k = f"l{i}"
        params[pid_k] = k.ProductID
        params[cid_k] = k.ChannelID
        params[lid_k] = k.LocationID
        clauses.append(
            f'("ProductID","ChannelID","LocationID") = (:{pid_k},:{cid_k},:{lid_k})'
        )

    where_keys = " OR ".join(clauses)

    sql = f"""
      SELECT
        "ProductID",
        "ChannelID",
        "LocationID",
        COALESCE("Period",'')          AS "Period",
        "StartDate",
        "EndDate",
        COALESCE("Qty",0)::float       AS "Qty",
        COALESCE("Level"::text,'')     AS "Level"
      FROM {hist_table}
      WHERE ({where_keys})
      ORDER BY
        "ProductID",
        "ChannelID",
        "LocationID",
        "StartDate",
        "EndDate";
    """

    rows = fetch_all(sql, params)

    # Normalize for JSON / Angular table
    for r in rows:
        r["ProductID"] = str(r.get("ProductID", ""))
        r["ChannelID"] = str(r.get("ChannelID", ""))
        r["LocationID"] = str(r.get("LocationID", ""))
        r["Period"] = str(r.get("Period", ""))
        r["Level"] = str(r.get("Level", ""))

        r["StartDate"] = _normalize_date_str(r.get("StartDate"))
        r["EndDate"] = _normalize_date_str(r.get("EndDate"))
        r["Qty"] = float(r.get("Qty") or 0.0)

    return rows



# === GeneratedPeriods helper ==================================================
def _load_generated_periods(
    period: Literal["daily", "weekly", "monthly"], start_after: datetime
) -> List[Tuple[datetime, datetime]]:
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
    return [
        (pd.Timestamp(s).to_pydatetime(), pd.Timestamp(e).to_pydatetime())
        for s, e in zip(df["StartDate"], df["EndDate"])
    ]


# === Load single-key history ==================================================
def _load_series_for_key(
    period: Literal["daily", "weekly", "monthly"],
    pid: str,
    cid: str,
    lid: str,
    want_cleansed: bool,
) -> Tuple[pd.Series, bool, str, str]:
    base_table, pd_freq, trunc_part = _tbl_freq(period)
    hist_table = f"history_cleansed_{period}" if want_cleansed else base_table

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
    rows = fetch_all(
        """
      SELECT id, name, settings, is_active
      FROM cleanse_profile
      ORDER BY is_active DESC, name
    """
    )
    for r in rows:
        if isinstance(r.get("settings"), str):
            try:
                r["settings"] = json.loads(r["settings"])
            except Exception:
                r["settings"] = {}
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

    row = fetch_all(
        """
        INSERT INTO cleanse_profile(name, settings, config, is_active, updated_at)
        VALUES (:name, CAST(:settings AS jsonb), CAST(:config AS jsonb), :is_active, now())
        ON CONFLICT(name) DO UPDATE
          SET settings  = EXCLUDED.settings,
              config    = EXCLUDED.config,
              is_active = EXCLUDED.is_active,
              updated_at = now()
        RETURNING id, name, settings, is_active
    """,
        params,
    )
    if not row:
        raise HTTPException(status_code=500, detail="upsert failed")
    r = row[0]
    if isinstance(r.get("settings"), str):
        try:
            r["settings"] = json.loads(r["settings"])
        except Exception:
            r["settings"] = {}
    return r


# (Optional) Ingest cleansed history rows
class IngestRow(BaseModel):
    ProductID: str
    ChannelID: str
    LocationID: str
    StartDate: str
    Qty: float
    Period: Literal["Daily", "Weekly", "Monthly"] = "Daily"


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
        if p2.startswith("d"):
            return "Daily"
        if p2.startswith("w"):
            return "Weekly"
        if p2.startswith("m"):
            return "Monthly"
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
        groups[bucket].append(
            {
                "ProductID": r.ProductID,
                "ChannelID": r.ChannelID,
                "LocationID": r.LocationID,
                "StartDate": r.StartDate,
                "Qty": r.Qty,
            }
        )

    inserted = 0
    with engine.begin() as c:
        for bucket, batch in groups.items():
            if not batch:
                continue

            raw_table = _tbl_freq(bucket)[0]                  # history_daily / weekly / monthly
            cleansed_table = f"history_cleansed_{bucket}"     # history_cleansed_daily / ...

            # 1) insert into cleansed history (used by classification)
            c.execute(
                text(
                    f"""
                INSERT INTO {cleansed_table}
                  ("ProductID","ChannelID","LocationID","StartDate","Qty")
                VALUES (:ProductID,:ChannelID,:LocationID,:StartDate,:Qty)
                """
                ),
                batch,
            )

            # 2) (optional) also store into raw history, if you want it there
            c.execute(
                text(
                    f"""
                INSERT INTO {raw_table}
                  ("ProductID","ChannelID","LocationID","StartDate","Qty")
                VALUES (:ProductID,:ChannelID,:LocationID,:StartDate,:Qty)
                """
                ),
                batch,
            )

            inserted += len(batch)

    return {"inserted": inserted, "errors": errors}


# === Classify forecast elements ==============================================
class ClassifyRequest(BaseModel):
    period: Literal["daily", "weekly", "monthly"] = "daily"
    lookback_buckets: int = 8
    min_sum: float = 1.0
    include_inactive: bool = False  # for GET only

@app.post("/api/classify/compute")
def classify_compute(body: ClassifyRequest):
    base_table, _, trunc_part = _tbl_freq(body.period)  # base history_* table + trunc_part
    hist_table_cleansed = f"history_cleansed_{body.period}"
    unit = {"daily": "day", "weekly": "week", "monthly": "month"}[body.period]

    def _run_recent_from(table_name: str):
        return fetch_all(
            f"""
          WITH raw AS (
            SELECT h."ProductID", h."ChannelID", h."LocationID",
                   date_trunc(:trunc_part, {_ts_expr('h')}) AS dt,
                   SUM(h."Qty")::float AS qty
            FROM {table_name} h
            WHERE {_ts_expr('h')} IS NOT NULL
            GROUP BY 1,2,3,4
          ),
          recent AS (
            SELECT r."ProductID", r."ChannelID", r."LocationID",
                   SUM(r.qty) AS s,
                   MAX(r.dt)  AS last_dt
            FROM raw r
            WHERE r.dt >= (
                SELECT MAX(dt) FROM raw
            ) - (:lb - 1) * INTERVAL '1 {unit}'
            GROUP BY 1,2,3
          )
          SELECT * FROM recent
            """,
            {"trunc_part": trunc_part, "lb": int(max(1, body.lookback_buckets))},
        )

    # 1) Try cleansed table first
    raw = _run_recent_from(hist_table_cleansed)

    # 2) If no cleansed history, fall back to base history table
    if not raw:
        raw = _run_recent_from(base_table)

    # 3) If still nothing, then really no history for this period
    if not raw:
        raise HTTPException(
            status_code=400,
            detail="No Cleansed-History found for this period. Please run Cleanse History first.",
        )

    upserts = []
    for r in raw:
        label = "Active" if (r.get("s") or 0) >= body.min_sum else "Inactive"
        score = float(r.get("s") or 0.0)
        upserts.append(
            {
                "ProductID": r["ProductID"],
                "ChannelID": r["ChannelID"],
                "LocationID": r["LocationID"],
                "Period": body.period,
                "Label": label,
                "Score": score,
                "IsActive": (label == "Active"),
            }
        )

    if upserts:
        with engine.begin() as c:
            c.execute(
                text("""
          INSERT INTO forecast_element_classification
            ("ProductID","ChannelID","LocationID","Period","Label","Score","IsActive")
          VALUES
            (:ProductID,:ChannelID,:LocationID,:Period,:Label,:Score,:IsActive)
          ON CONFLICT ("ProductID","ChannelID","LocationID","Period")
          DO UPDATE SET
            "Label"     = EXCLUDED."Label",
            "Score"     = EXCLUDED."Score",
            "IsActive"  = EXCLUDED."IsActive",
            "ComputedAt" = now()
                          """),
                upserts,
            )

    return {"updated": len(upserts), "period": body.period}



@app.get("/api/classify/results")
def classify_results(period: Literal["daily", "weekly", "monthly"], include_inactive: bool = False):
    sql = """
      SELECT
        "ProductID",
        "ChannelID",
        "LocationID",
        "Period",
        "Label"     AS "Label",
        "Score"     AS "Score",
        "IsActive"  AS "IsActive",
        "ComputedAt"
      FROM forecast_element_classification
      WHERE "Period" = :p
    """
    if not include_inactive:
        sql += " AND \"IsActive\" = TRUE"

    sql += """
      ORDER BY "IsActive" DESC, "Score" DESC,
               "ProductID","ChannelID","LocationID"
    """

    return fetch_all(sql, {"p": period})





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
    metric: Optional[Literal["WMAPE", "MAE"]] = None
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
    if body.history_file is not None:
        bf.HISTORY_FILE = body.history_file
    if body.metric is not None:
        bf.METRIC = body.metric
    if body.fast_mode is not None:
        bf.FAST_MODE = bool(body.fast_mode)
    if body.forward_horizon is not None:
        bf.FORWARD_HORIZON = int(body.forward_horizon)
    if body.max_plots is not None:
        bf.MAX_PLOTS = int(body.max_plots)
    res = bf.run()
    bt = _read_csv_safe(Path(res.get("backtest_file", "")))
    sm = _read_csv_safe(Path(res.get("summary_file", "")))
    fw = _read_csv_safe(Path(res.get("forward_file", "")))
    return BatchRunResult(
        backtest_file=res.get("backtest_file"),
        summary_file=res.get("summary_file"),
        forward_file=res.get("forward_file"),
        plots_backtest_dir=res.get("plots_backtest_dir"),
        plots_history_forecast_dir=res.get("plots_history_forecast_dir"),
        rows_backtest=int(len(bt)),
        rows_summary=int(len(sm)),
        rows_forward=int(len(fw)),
    )


@app.get("/api/batch/monthly/combos", response_model=List[Combo])
def api_batch_monthly_combos():
    if bf is None:
        return []
    hist = _read_csv_safe(Path(bf.HISTORY_FILE))
    if hist.empty:
        return []
    needed = {"ProductID", "ChannelID", "LocationID"}
    if not needed.issubset(hist.columns):
        raise HTTPException(status_code=400, detail="History file missing required columns.")
    out = (
        hist[["ProductID", "ChannelID", "LocationID"]]
        .astype(str)
        .drop_duplicates()
        .sort_values(["ProductID", "ChannelID", "LocationID"])
    )
    return [Combo(**dict(r)) for _, r in out.iterrows()]


@app.get("/api/batch/monthly/backtest")
def api_batch_backtest(pid: Optional[str] = None, cid: Optional[str] = None, lid: Optional[str] = None):
    if bf is None:
        return []
    df = _read_csv_safe(Path(bf.OUT_BACKTEST_CSV))
    if df.empty:
        return []
    for c in ("ProductID", "ChannelID", "LocationID"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    if pid:
        df = df[df["ProductID"] == pid]
    if cid:
        df = df[df["ChannelID"] == cid]
    if lid:
        df = df[df["LocationID"] == lid]
    if "Date" in df.columns:
        df["Date"] = _parse_dt_dmy(df["Date"]).dt.strftime("%Y-%m-%dT00:00:00Z")
    return df.to_dict(orient="records")


@app.get("/api/batch/monthly/summary")
def api_batch_summary():
    if bf is None:
        return []
    df = _read_csv_safe(Path(bf.OUT_SUMMARY_CSV))
    if df.empty:
        return []
    for c in ("ProductID", "ChannelID", "LocationID"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df.to_dict(orient="records")


@app.get("/api/batch/monthly/forward")
def api_batch_forward(pid: Optional[str] = None, cid: Optional[str] = None, lid: Optional[str] = None):
    if bf is None:
        return []
    df = _read_csv_safe(Path(bf.OUT_FORWARD_CSV))
    if df.empty:
        return []
    for c in ("ProductID", "ChannelID", "LocationID"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    if pid:
        df = df[df["ProductID"] == pid]
    if cid:
        df = df[df["ChannelID"] == cid]
    if lid:
        df = df[df["LocationID"] == lid]
    for col in ("StartDate", "EndDate", "Forecast Date", "History End Date"):
        if col in df.columns:
            df[col] = _parse_dt_dmy(df[col]).dt.strftime("%Y-%m-%dT00:00:00Z")
    return df.rename(columns={"Forecast Qty": "ForecastQty"}).to_dict(orient="records")


@app.get("/api/batch/monthly/history-series", response_model=List[SeriesPoint])
def api_batch_history_series(pid: str, cid: str, lid: str):
    if bf is None:
        return []
    hist = _read_csv_safe(Path(bf.HISTORY_FILE))
    if hist.empty:
        return []
    need = {"ProductID", "ChannelID", "LocationID", "StartDate", "Qty"}
    if not need.issubset(hist.columns):
        raise HTTPException(status_code=400, detail="History file missing required columns.")
    sub = hist[
        (hist["ProductID"].astype(str) == pid)
        & (hist["ChannelID"].astype(str) == cid)
        & (hist["LocationID"].astype(str) == lid)
    ]
    if sub.empty:
        return []
    sub = sub.copy()
    sub["StartDate"] = _parse_dt_dmy(sub["StartDate"]).dt.strftime("%Y-%m-%dT00:00:00Z")
    grp = sub.groupby("StartDate", as_index=False)["Qty"].sum().sort_values("StartDate")
    return [SeriesPoint(StartDate=r["StartDate"], Qty=float(r["Qty"])) for _, r in grp.iterrows()]


@app.get("/api/batch/monthly/forward-series", response_model=List[SeriesPoint])
def api_batch_forward_series(pid: str, cid: str, lid: str):
    if bf is None:
        return []
    fw = _read_csv_safe(Path(bf.OUT_FORWARD_CSV))
    if fw.empty:
        return []
    sub = fw[
        (fw["ProductID"].astype(str) == pid)
        & (fw["ChannelID"].astype(str) == cid)
        & (fw["LocationID"].astype(str) == lid)
    ]
    if sub.empty:
        return []
    sub = sub.copy()
    sub["StartDate"] = _parse_dt_dmy(sub["StartDate"]).dt.strftime("%Y-%m-%dT00:00:00Z")
    sub["ForecastQty"] = pd.to_numeric(sub["Forecast Qty"], errors="coerce")
    grp = sub.groupby("StartDate", as_index=False)["ForecastQty"].sum().sort_values("StartDate")
    return [SeriesPoint(StartDate=r["StartDate"], Qty=float(r["ForecastQty"])) for _, r in grp.iterrows()]


# === Shared horizon defaults per period ======================================
DEFAULT_HORIZON: Dict[str, int] = {
    "daily": 7,   # next 7 days
    "weekly": 13, # next 13 weeks
    "monthly": 18,  # next 18 months
}


# === NEW: Per-key forecast from DB history (all periods) =====================
class Run18mBody(BaseModel):
    key: KeyTriplet
    period: Literal["daily", "weekly", "monthly"] = "monthly"
    horizon: int = 18  # interpreted as "buckets"; we override default per period if needed
    save: bool = False  # if true, upsert into forecast_* table
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
        raise HTTPException(
            status_code=400,
            detail=f"Not enough {period} history (have {len(series)}).",
        )

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
        preds.append(
            Run18mPrediction(
                ProductID=pid,
                ChannelID=cid,
                LocationID=lid,
                StartDate=sdt.strftime("%Y-%m-%dT00:00:00Z"),
                EndDate=edt.strftime("%Y-%m-%dT23:59:59Z"),
                Qty=float(fut_vals[i - 1]),
                Method=best_model,
                Period=period.capitalize(),
                Type="Algorithm-Forecast",
            )
        )

    # Optional: save to forecast_* table
    if body.save and preds:
        tbl = _forecast_table(period)
        sql_ins = """
        INSERT INTO {tbl}
          ("ProductID","ChannelID","LocationID","Method","Period","StartDate","EndDate","Type","Qty","Level")
        VALUES
          (:ProductID,:ChannelID,:LocationID,:Method,:Period,:StartDate,:EndDate,:Type,:Qty,'Item')
        ON CONFLICT ("ProductID","ChannelID","LocationID","StartDate","EndDate","Method","Type")
        DO UPDATE SET
          "Qty" = EXCLUDED."Qty",
          "Type" = EXCLUDED."Type",
          "created_at" = now();
        """.format(
            tbl=tbl
        )
        with engine.begin() as c:
            c.execute(text(sql_ins), [p.dict() for p in preds])

    return Run18mResult(
        key=body.key,
        period=period,
        model=best_model,
        horizon=H,
        history_points=int(len(series)),
        predictions=preds,
    )


# === NEW: Forecast aggregated by saved-search query (all periods) ============
class Run18mByQueryBody(BaseModel):
    q: Optional[str] = None
    period: Literal["daily", "weekly", "monthly"] = "monthly"
    horizon: int = 18
    max_keys: int = 500
    use_cleansed: bool = False  # reserved / future use
    save: bool = False  # save per-key forecasts into forecast_*


class Run18mByQueryResult(BaseModel):
    query: str
    period: Literal["daily", "weekly", "monthly"] = "monthly"
    horizon: int
    keys_scanned: int
    keys_forecasted: int
    skipped: int
    series: List[SeriesPoint]


@app.post("/api/forecast/18m/run-by-query", response_model=Run18mByQueryResult)
def run_18m_by_query(body: Run18mByQueryBody):
    """
    For a query like 'productid:*' or 'productid:AztecWrap geography:Gelderland',
    find all matching keys, run per-key forecasts using the same logic
    for the chosen period (daily/weekly/monthly), and return the aggregate
    (sum) series by StartDate.
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
            keys_scanned=0,
            keys_forecasted=0,
            skipped=0,
            series=[],
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
                agg[start_str] = agg.get(start_str, 0.0) + float(fut_vals[i - 1])

                if body.save:
                    all_save_rows.append(
                        {
                            "ProductID": pid,
                            "ChannelID": cid,
                            "LocationID": lid,
                            "StartDate": start_str,
                            "EndDate": edt.strftime("%Y-%m-%dT23:59:59Z"),
                            "Qty": float(fut_vals[i - 1]),
                            "Period": period.capitalize(),
                            "Method": best_model,
                            "Type": "Algorithm-Forecast",
                        }
                    )

            keys_ok += 1

        except Exception:
            skipped += 1
            continue

    # 2b) Save per-key forecasts if requested
    if body.save and all_save_rows:
        tbl = _forecast_table(period)
        sql_ins = """
        INSERT INTO {tbl}
          ("ProductID","ChannelID","LocationID","Method","Period","StartDate","EndDate","Type","Qty","Level")
        VALUES
          (:ProductID,:ChannelID,:LocationID,:Method,:Period,:StartDate,:EndDate,:Type,:Qty,'Item')
        ON CONFLICT ("ProductID","ChannelID","LocationID","StartDate","EndDate","Method","Type")
        DO UPDATE SET
          "Qty" = EXCLUDED."Qty",
          "Type" = EXCLUDED."Type",
          "created_at" = now();
        """.format(
            tbl=tbl
        )
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
        series=out_series,
    )
