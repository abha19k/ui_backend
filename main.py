from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Literal
from sqlalchemy import create_engine, text
import json

app = FastAPI()

# --- CORS: explicit origins + credentials (dev) --------------------------------
origins = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,     # explicit origins when credentials=True
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database ------------------------------------------------------------------
DB_URI = "postgresql://abha:planwise123@localhost:5432/planwise"
engine = create_engine(DB_URI, pool_pre_ping=True)

# --- Create saved_search on startup -------------------------------------------
@app.on_event("startup")
def ensure_saved_search_table():
    ddl = """
    CREATE TABLE IF NOT EXISTS saved_search (
      id SERIAL PRIMARY KEY,
      name TEXT NOT NULL,
      query TEXT NOT NULL,
      created_at TIMESTAMPTZ DEFAULT now()
    );
    """
    with engine.begin() as c:
        c.execute(text(ddl))

# --- Cleanse: ensure Type on history tables + profiles ------------------------
@app.on_event("startup")
def ensure_cleanse_support():
    ddl = """
    -- Ensure Type column on history tables (defaults to 'Normal-History')
    ALTER TABLE IF EXISTS history_daily   ADD COLUMN IF NOT EXISTS "Type" TEXT DEFAULT 'Normal-History';
    ALTER TABLE IF EXISTS history_weekly  ADD COLUMN IF NOT EXISTS "Type" TEXT DEFAULT 'Normal-History';
    ALTER TABLE IF EXISTS history_monthly ADD COLUMN IF NOT EXISTS "Type" TEXT DEFAULT 'Normal-History';

    -- Helpful unique indexes so we can upsert (one row per period & type)
    CREATE UNIQUE INDEX IF NOT EXISTS ux_hist_daily_cleansed
      ON history_daily("ProductID","ChannelID","LocationID","StartDate","EndDate","Type");
    CREATE UNIQUE INDEX IF NOT EXISTS ux_hist_weekly_cleansed
      ON history_weekly("ProductID","ChannelID","LocationID","StartDate","EndDate","Type");
    CREATE UNIQUE INDEX IF NOT EXISTS ux_hist_monthly_cleansed
      ON history_monthly("ProductID","ChannelID","LocationID","StartDate","EndDate","Type");

    -- Store named cleanse settings
    CREATE TABLE IF NOT EXISTS cleanse_profile (
      id SERIAL PRIMARY KEY,
      name TEXT UNIQUE NOT NULL,
      config JSONB NOT NULL,
      created_at TIMESTAMPTZ DEFAULT now()
    );
    """
    with engine.begin() as c:
        c.execute(text(ddl))

class CleanseProfileIn(BaseModel):
    name: str
    config: dict

@app.get("/api/cleanse/profiles")
def list_cleanse_profiles():
    return fetch_all("""SELECT id, name, config, created_at
                        FROM cleanse_profile ORDER BY created_at DESC""")

@app.post("/api/cleanse/profiles")
def create_cleanse_profile(p: CleanseProfileIn):
    if not p.name.strip():
        raise HTTPException(status_code=400, detail="Profile name required")
    with engine.begin() as c:
        c.execute(text("""
            INSERT INTO cleanse_profile(name, config)
            VALUES (:n, :cfg)
            ON CONFLICT (name) DO UPDATE SET config=EXCLUDED.config
        """), {"n": p.name.strip(), "cfg": p.config})
    return {"ok": True}

# --- Pydantic models -----------------------------------------------------------
class Product(BaseModel):
    ProductID: str
    ProductDescr: str
    Level: int
    BusinessUnit: str
    IsDailyForecastRequired: bool
    IsNew: bool
    ProductFamily: str

class Channel(BaseModel):
    ChannelID: str
    ChannelDescr: str
    Level: int

class Location(BaseModel):
    LocationID: str
    LocationDescr: str
    Level: int
    Geography: str

class ForecastElement(BaseModel):
    ProductID: str
    ChannelID: str
    LocationID: str

class History(BaseModel):
    ProductID: str
    ChannelID: str
    LocationID: str
    StartDate: str
    EndDate: str
    Qty: float
    Level: str
    Period: str
    Type: str

class Forecast(History):
    Method: str
    # Type already inherited from History

# --- Helpers -------------------------------------------------------------------
def fetch_all(query: str, params: dict | None = None):
    with engine.begin() as conn:
        result = conn.execute(text(query), params or {})
        return [dict(row._mapping) for row in result]

def _jsonable(model):
    # Pydantic v2 uses model_dump(); v1 uses dict()
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()

# --- Root ----------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to Planwise API 🚀"}

# --- Existing API routes -------------------------------------------------------
@app.get("/api/products", response_model=List[Product])
def get_products():
    return fetch_all("SELECT * FROM product")

@app.get("/api/channels", response_model=List[Channel])
def get_channels():
    return fetch_all("SELECT * FROM channel")

@app.get("/api/locations", response_model=List[Location])
def get_locations():
    return fetch_all("""
        SELECT 
            "LocationID", 
            "Location_Descr" AS "LocationDescr", 
            "Level", 
            "Geography" 
        FROM location
    """)

@app.get("/api/forecast_elements", response_model=List[ForecastElement])
def get_forecast_elements():
    return fetch_all("SELECT * FROM forecast_element")

@app.get("/api/history/daily", response_model=List[History])
def get_history_daily():
    return fetch_all("""
        SELECT "ProductID", "ChannelID", "LocationID", "StartDate", "EndDate", "Qty", "Type",
               CAST("Level" AS TEXT) AS "Level", "Period"
        FROM history_daily
    """)

@app.get("/api/history/weekly", response_model=List[History])
def get_history_weekly():
    return fetch_all("""
        SELECT "ProductID", "ChannelID", "LocationID", "StartDate", "EndDate", "Qty", "Type",
               CAST("Level" AS TEXT) AS "Level", "Period"
        FROM history_weekly
    """)

@app.get("/api/history/monthly", response_model=List[History])
def get_history_monthly():
    return fetch_all("""
        SELECT "ProductID", "ChannelID", "LocationID", "StartDate", "EndDate", "Qty", "Type",
               CAST("Level" AS TEXT) AS "Level", "Period"
        FROM history_monthly
    """)

@app.get("/api/forecast/daily", response_model=List[Forecast])
def get_forecast_daily():
    return fetch_all("""
        SELECT "ProductID", "ChannelID", "LocationID", "StartDate", "EndDate", "Qty",
               CAST("Level" AS TEXT) AS "Level", "Period", "Method", CAST("Type" AS TEXT) AS "Type"
        FROM forecast_daily
    """)

@app.get("/api/forecast/weekly", response_model=List[Forecast])
def get_forecast_weekly():
    return fetch_all("""
        SELECT "ProductID", "ChannelID", "LocationID", "StartDate", "EndDate", "Qty",
               CAST("Level" AS TEXT) AS "Level", "Period", "Method", CAST("Type" AS TEXT) AS "Type"
        FROM forecast_weekly
    """)

@app.get("/api/forecast/monthly", response_model=List[Forecast])
def get_forecast_monthly():
    return fetch_all("""
        SELECT "ProductID", "ChannelID", "LocationID", "StartDate", "EndDate", "Qty",
               CAST("Level" AS TEXT) AS "Level", "Period", "Method", CAST("Type" AS TEXT) AS "Type"
        FROM forecast_monthly
    """)

# --- Search section ------------------------------------------------------------
class KeyTriplet(BaseModel):
    ProductID: str
    ChannelID: str
    LocationID: str

class SearchResult(BaseModel):
    query: str
    count: int
    keys: List[KeyTriplet]

# Map query fields to SQL columns (via joins)
FIELD_MAP: Dict[str, str] = {
    # Product
    "productid": 'p."ProductID"',
    "productdescr": 'p."ProductDescr"',
    "businessunit": 'p."BusinessUnit"',
    "isdailyforecastrequired": 'CAST(p."IsDailyForecastRequired" AS TEXT)',
    "isnew": 'CAST(p."IsNew" AS TEXT)',
    "productfamily": 'p."ProductFamily"',
    "productlevel": 'CAST(p."Level" AS TEXT)',

    # Channel
    "channelid": 'c."ChannelID"',
    "channeldescr": 'c."ChannelDescr"',
    "channellevel": 'CAST(c."Level" AS TEXT)',

    # Location
    "locationid": 'l."LocationID"',
    "locationdescr": 'l."Location_Descr"',
    "locationlevel": 'CAST(l."Level" AS TEXT)',
    "geography": 'l."Geography"',
}

def _to_ilike_pattern(term: str) -> str:
    # Support * wildcards; escape % and _
    t = term.replace('%', r'\%').replace('_', r'\_').replace('*', '%')
    return t if ('%' in t or t.startswith('%')) else f"%{t}%"

def _build_where_clause(q: str, params: dict) -> str:
    """
    Tiny safe-ish parser: free text, field:value, AND/OR, quotes (shlex).
    Unknown fields are ignored. Raises a clear 400 on bad quotes.
    """
    import shlex
    try:
        tokens = shlex.split(q) if q else []
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid query syntax: {e}")

    sql_parts: List[str] = []
    op_next = "AND"
    bind_i = 0

    def add_condition(sql: str):
        nonlocal op_next
        if not sql:
            return
        if sql_parts:
            sql_parts.append(op_next)
        sql_parts.append(f"({sql})")

    for t in tokens:
        up = t.upper()
        if up in ("AND", "OR"):
            op_next = up
            continue

        if ":" in t:
            field, value = t.split(":", 1)
            col = FIELD_MAP.get(field.lower())
            if not col:
                continue
            bind_i += 1
            key = f"b{bind_i}"
            params[key] = _to_ilike_pattern(value)
            add_condition(f"""{col} ILIKE :{key} ESCAPE '\\'""")
        else:
            cols = [
                'p."ProductID"', 'p."ProductDescr"', 'p."BusinessUnit"', 'p."ProductFamily"',
                'c."ChannelID"', 'c."ChannelDescr"',
                'l."LocationID"', 'l."Location_Descr"', 'l."Geography"',
            ]
            bind_i += 1
            key = f"b{bind_i}"
            params[key] = _to_ilike_pattern(t)
            ors = [f"""{c} ILIKE :{key} ESCAPE '\\'""" for c in cols]
            add_condition(" OR ".join(ors))
    return " ".join(sql_parts) if sql_parts else "TRUE"

@app.get("/api/search", response_model=SearchResult)
def search(q: Optional[str] = None, limit: int = 5000, offset: int = 0):
    MAX_LIMIT = 20000
    limit = min(limit, MAX_LIMIT)

    params: dict = {}
    where_sql = _build_where_clause(q or "", params)
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

    try:
        with engine.begin() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
            total = conn.execute(text(count_sql), params).scalar_one()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Search failed for q='{q}': {e}")

    keys = [KeyTriplet(**dict(r)) for r in rows]
    return SearchResult(query=q or "", count=total, keys=keys)

# --- Use search keys with data endpoints --------------------------------------
class KeysBody(BaseModel):
    keys: List[KeyTriplet]

def _keys_param(keys: List[KeyTriplet]) -> dict:
    return {"keys": json.dumps([_jsonable(k) if isinstance(k, KeyTriplet) else k for k in keys])}

def _ensure_keys(body: KeysBody):
    if not body.keys:
        raise HTTPException(status_code=400, detail="keys[] is required and cannot be empty")

def _select_forecast(table_name: str) -> str:
    return f"""
    WITH keys AS (
      SELECT
        (elem->>'ProductID')::text   AS "ProductID",
        (elem->>'ChannelID')::text   AS "ChannelID",
        (elem->>'LocationID')::text  AS "LocationID"
      FROM jsonb_array_elements((:keys)::jsonb) AS elem
    )
    SELECT
      f."ProductID", f."ChannelID", f."LocationID",
      f."StartDate", f."EndDate", f."Qty",
      CAST(f."Level" AS TEXT) AS "Level",
      f."Period", f."Method", CAST(f."Type" AS TEXT) AS "Type"
    FROM {table_name} f
    WHERE (f."ProductID", f."ChannelID", f."LocationID")
          IN (SELECT "ProductID","ChannelID","LocationID" FROM keys)
    ORDER BY f."ProductID", f."ChannelID", f."LocationID", f."StartDate";
    """

def _select_history(table_name: str) -> str:
    return f"""
    WITH keys AS (
      SELECT
        (elem->>'ProductID')::text   AS "ProductID",
        (elem->>'ChannelID')::text   AS "ChannelID",
        (elem->>'LocationID')::text  AS "LocationID"
      FROM jsonb_array_elements((:keys)::jsonb) AS elem
    )
    SELECT
      h."ProductID", h."ChannelID", h."LocationID",
      h."StartDate", h."EndDate", h."Qty",
      CAST(h."Level" AS TEXT) AS "Level",
      h."Period",
      CAST(h."Type" AS TEXT) AS "Type"
    FROM {table_name} h
    WHERE (h."ProductID", h."ChannelID", h."LocationID")
          IN (SELECT "ProductID","ChannelID","LocationID" FROM keys)
    ORDER BY h."ProductID", h."ChannelID", h."LocationID", h."StartDate";
    """

@app.post("/api/forecast/daily-by-keys", response_model=List[Forecast])
def forecast_daily_by_keys(body: KeysBody):
    _ensure_keys(body)
    return fetch_all(_select_forecast("forecast_daily"), _keys_param(body.keys))

@app.post("/api/forecast/weekly-by-keys", response_model=List[Forecast])
def forecast_weekly_by_keys(body: KeysBody):
    _ensure_keys(body)
    return fetch_all(_select_forecast("forecast_weekly"), _keys_param(body.keys))

@app.post("/api/forecast/monthly-by-keys", response_model=List[Forecast])
def forecast_monthly_by_keys(body: KeysBody):
    _ensure_keys(body)
    return fetch_all(_select_forecast("forecast_monthly"), _keys_param(body.keys))

@app.post("/api/history/daily-by-keys", response_model=List[History])
def history_daily_by_keys(body: KeysBody):
    _ensure_keys(body)
    return fetch_all(_select_history("history_daily"), _keys_param(body.keys))

@app.post("/api/history/weekly-by-keys", response_model=List[History])
def history_weekly_by_keys(body: KeysBody):
    _ensure_keys(body)
    return fetch_all(_select_history("history_weekly"), _keys_param(body.keys))

@app.post("/api/history/monthly-by-keys", response_model=List[History])
def history_monthly_by_keys(body: KeysBody):
    _ensure_keys(body)
    return fetch_all(_select_history("history_monthly"), _keys_param(body.keys))

# --- Ingest cleansed rows -----------------------------------------------------
class HistoryRowIn(BaseModel):
    ProductID: str
    ChannelID: str
    LocationID: str
    StartDate: str
    EndDate: str
    Qty: float
    Level: str
    Period: str

PeriodLiteral = Literal["daily","weekly","monthly"]

class IngestCleansedBody(BaseModel):
    period: PeriodLiteral
    rows: List[HistoryRowIn]

@app.post("/api/history/ingest-cleansed")
def ingest_cleansed(body: IngestCleansedBody):
    if not body.rows:
        return {"inserted": 0, "updated": 0}
    table = {"daily":"history_daily","weekly":"history_weekly","monthly":"history_monthly"}[body.period]

    sql = f"""
    INSERT INTO {table} 
      ("ProductID","ChannelID","LocationID","StartDate","EndDate","Qty","Level","Period","Type")
    VALUES 
      (:ProductID,:ChannelID,:LocationID,:StartDate,:EndDate,:Qty,:Level,:Period,'Cleansed-History')
    ON CONFLICT ("ProductID","ChannelID","LocationID","StartDate","EndDate","Type")
    DO UPDATE SET "Qty"=EXCLUDED."Qty", "Level"=EXCLUDED."Level", "Period"=EXCLUDED."Period";
    """
    params = [r.dict() for r in body.rows]
    with engine.begin() as c:
        c.execute(text(sql), params)
    return {"ok": True, "count": len(params)}

# --- Saved searches ------------------------------------------------------------
@app.get("/api/saved-searches")
def list_saved():
    with engine.begin() as c:
        try:
            rows = c.execute(text("""
                SELECT id, name, query, created_at
                FROM saved_search
                ORDER BY created_at DESC
            """)).mappings().all()
        except Exception:
            return []
    return [dict(r) for r in rows]

@app.post("/api/saved-searches")
def save_search(item: dict = Body(...)):
    name = (item.get("name") or "").strip()
    query = (item.get("query") or "").strip()
    if not name or not query:
        raise HTTPException(status_code=400, detail="name and query are required")
    with engine.begin() as c:
        c.execute(
            text("""INSERT INTO saved_search(name, query) VALUES (:name, :query)"""),
            {"name": name, "query": query},
        )
    return {"ok": True}
