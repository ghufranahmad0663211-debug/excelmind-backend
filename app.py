from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response, ORJSONResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.concurrency import run_in_threadpool
import pandas as pd
import numpy as np
import io, re, math, json, os, hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    orjson = None
    ORJSON_AVAILABLE = False


# Try to import scipy for p-values; fallback to manual approximation if unavailable
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

app = FastAPI(title="ExcelMind Analyst API", version="1.3.0")

# Compress large JSON responses to reduce transfer time and payload size.
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # The UI uses same-origin, credential-free requests. Wildcard origins and
    # credentialed CORS are both unsafe and incompatible in browsers.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_MB = 25
CHANGE_THRESHOLD = 1.00
RATE_CHANGE_THRESHOLD_POINTS = 15.0

# Simple in-memory analysis cache keyed by the uploaded file's MD5 hash.
# Values are final JSON response bytes, so cache hits skip the entire analysis
# pipeline and remain immutable/safe to reuse across requests.
ANALYZE_CACHE: Dict[str, bytes] = {}
ANALYZE_CACHE_MAX_ENTRIES = 8

@app.get("/api/health")
async def health():
    return {"ok": True, "service": "ExcelMind Analyst API", "version": app.version}


FRONTEND_FILENAME = "data analyst 2.html"

@app.get("/")
async def root():
    page = Path(__file__).with_name(FRONTEND_FILENAME)
    if not page.exists():
        raise HTTPException(404, f"Frontend HTML file not found. Place '{FRONTEND_FILENAME}' next to app.py.")
    return FileResponse(page, media_type="text/html", headers={"Cache-Control": "public, max-age=3600"})

def numeric_series(series: pd.Series) -> pd.Series:
    """Fast numeric coercion with a cheap fast-path before string cleanup."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    # Most clean numeric CSV columns can be converted directly. This avoids
    # the expensive string-replacement pipeline for the common case.
    direct = pd.to_numeric(series, errors="coerce")
    non_empty = series.notna()
    total = int(non_empty.sum())
    direct_valid = int(direct[non_empty].notna().sum()) if total else 0
    if total == 0 or direct_valid == total:
        return direct

    # If a representative sample contains no numeric-looking content, the
    # column is almost certainly categorical/text; avoid 10+ full string ops.
    sample = series[non_empty].astype("string").head(256)
    if len(sample) and not sample.str.contains(r"\d", regex=True, na=False).any():
        return direct

    s = series.astype("string").str.strip()
    empty_tokens = {"", "nan", "none", "null", "n/a", "na", "-", "—"}
    s = s.mask(s.str.lower().isin(empty_tokens))
    negative_parentheses = s.str.match(r"^\(.*\)$", na=False)
    s = (s.str.replace("\u00a0", "", regex=False)
           .str.replace(",", "", regex=False)
           .str.replace(" ", "", regex=False)
           .str.replace("Rs.", "", regex=False)
           .str.replace("Rs", "", regex=False)
           .str.replace("PKR", "", regex=False)
           .str.replace("USD", "", regex=False)
           .str.replace("GBP", "", regex=False)
           .str.replace("EUR", "", regex=False)
           .str.replace("$", "", regex=False)
           .str.replace("€", "", regex=False)
           .str.replace("£", "", regex=False)
           .str.replace("%", "", regex=False)
           .str.replace("(", "", regex=False)
           .str.replace(")", "", regex=False))
    result = pd.to_numeric(s, errors="coerce")
    result.loc[negative_parentheses & result.notna()] *= -1
    return result

def numeric_detection_ratio(series: pd.Series) -> float:
    non_empty = series.notna()
    total = int(non_empty.sum())
    if total == 0:
        return 0.0
    converted = numeric_series(series)
    return float(converted[non_empty].notna().mean())

def is_numeric_column(series: pd.Series, threshold: float = 0.80) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    return numeric_detection_ratio(series) >= threshold

def humanize(col: str) -> str:
    s = str(col).strip()
    s = re.sub(r"[_\-]+", " ", s)
    s = re.sub(r"\b(usd|pkr|gbp|eur)\s*m\b", r"\1 M", s, flags=re.I)
    s = re.sub(r"\b000s\b", "Thousands", s, flags=re.I)
    s = re.sub(r"\bgwh\b", "GWh", s, flags=re.I)
    s = re.sub(r"\b(mwh)\b", "MWh", s, flags=re.I)
    s = re.sub(r"\b(%)\b", "%", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:1].upper() + s[1:] if s else ""

def role(col: str, series=None, numeric_hint: bool | None = None) -> str:
    s = str(col).lower().replace("_", " ")
    if re.search(r"\b(id|uuid|identifier|sku|order number|customer id|transaction id)\b", s):
        return "identifier"
    if re.search(r"\b(date|timestamp|datetime|quarter|month|week|year|period)\b", s):
        return "time"
    if re.search(r"\b(product|item|model|brand)\b", s) and not re.search(r"\bproduction\b", s):
        if series is None or not (is_numeric_column(series) if numeric_hint is None else numeric_hint):
            return "product"
    if re.search(r"\b(region|country|city|state|category|segment|department|channel|status|type|method)\b", s):
        return "category"
    if re.search(r"\b(deliver(y|ies)|production|produced|units|quantity|volume|deployments?|headcount|orders?)\b", s):
        return "quantity"
    if re.search(r"\b(margin|rate|ratio|percent|percentage)\b|%", s):
        return "rate"
    if re.search(r"\b(revenue|sales|income|profit|earnings|cost|expense|cash flow|price|amount|value)\b", s):
        return "financial"
    return "numeric" if series is not None and (is_numeric_column(series) if numeric_hint is None else numeric_hint) else "text"

def quarter_parse_series(series: pd.Series) -> pd.Series:
    """Vectorized quarter parser; avoids Python row-by-row iteration."""
    s = series.astype("string").str.strip().str.upper()
    out = pd.Series(pd.NaT, index=series.index, dtype="period[Q]")

    m = s.str.extract(r"^(?:(\d{4})[- /]?Q([1-4])|Q([1-4])[- /]?(\d{4}))$", expand=True)
    y = m[0].fillna(m[3])
    q = m[1].fillna(m[2])
    valid = y.notna() & q.notna()
    if valid.any():
        labels = y[valid].astype("int64").astype(str) + "Q" + q[valid].astype("int64").astype(str)
        out.loc[valid] = pd.PeriodIndex(labels, freq="Q")
    return out

def is_time_col(col, series):
    """Detect time columns without repeatedly scanning 100k+ rows."""
    name = str(col).lower()
    if re.search(r"\b(quarter|month|week|year|period)\b", name):
        return True
    if re.search(r"\b(date|timestamp|datetime)\b", name):
        # Semantic names are strong signals; validate only a bounded sample.
        sample = series.dropna().head(2000)
        if len(sample) == 0:
            return False
        parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return True
        return bool(quarter_parse_series(sample).notna().mean() >= 0.8)

    # For unnamed time fields, sample only. Full-column parsing here was one
    # of the largest avoidable costs on large datasets.
    sample = series.dropna().head(1000)
    if len(sample) == 0:
        return False
    qratio = quarter_parse_series(sample).notna().mean()
    if qratio >= 0.8:
        return True
    parsed = pd.to_datetime(sample, errors="coerce")
    return bool(parsed.notna().mean() >= 0.8)

def sanitize_json(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_json(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, pd.Period):
        return str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v=float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj

def safe_value(v):
    if pd.isna(v):
        return None
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(v).isoformat()
    if isinstance(v, (pd.Period,)):
        return str(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v.item() if hasattr(v, "item") else v

def detect_unit(col):
    s = re.sub(r"[_\-]+", " ", str(col).lower())
    is_millions = bool(re.search(r"\b(millions?|m)\b", s))
    is_thousands = bool(re.search(r"\b(000s|thousands?|k)\b", s))
    scale = " (millions)" if is_millions else " (thousands)" if is_thousands else ""
    if re.search(r"\b(usd|us\$|dollar|dollars)\b", s): return "USD" + scale
    if re.search(r"\b(pkr|rs|rupee|rupees)\b", s): return "PKR" + scale
    if re.search(r"\b(gbp|pound|pounds)\b", s): return "GBP" + scale
    if re.search(r"\b(eur|euro|euros)\b", s): return "EUR" + scale
    if "gwh" in s: return "GWh"
    if is_thousands: return "thousands"
    if "%" in s or re.search(r"\b(percent|percentage|margin|rate)\b", s): return "%"
    if is_millions: return "millions"
    return None

def load_dataframe(data: bytes, filename: str):
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".csv":
            # pyarrow is substantially faster for large CSVs when installed.
            # Keep a pandas-engine fallback so the API remains portable.
            try:
                return pd.read_csv(io.BytesIO(data), engine="pyarrow")
            except (ImportError, ValueError, NotImplementedError):
                try:
                    return pd.read_csv(io.BytesIO(data))
                except UnicodeDecodeError:
                    return pd.read_csv(io.BytesIO(data), encoding="latin-1")
            except UnicodeDecodeError:
                try:
                    return pd.read_csv(io.BytesIO(data), engine="pyarrow", encoding="latin-1")
                except Exception:
                    return pd.read_csv(io.BytesIO(data), encoding="latin-1")
        if ext in (".xlsx", ".xls"):
            # python-calamine is often faster than openpyxl for read-only Excel ingestion.
            try:
                return pd.read_excel(io.BytesIO(data), engine="calamine")
            except (ImportError, ValueError, NotImplementedError):
                return pd.read_excel(io.BytesIO(data))
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {e}")
    raise HTTPException(400, "Only CSV, XLSX, and XLS files are supported.")

# ----------------------------------------------------------------------
# ADVANCED FORECASTING ENGINE (backend version)
# ----------------------------------------------------------------------

def _mean(a):
    return sum(a) / len(a) if len(a) else 0.0

def _std(a, ddof=1):
    if len(a) < 2:
        return 0.0
    m = _mean(a)
    return math.sqrt(sum((x - m) ** 2 for x in a) / (len(a) - ddof))

def _linear_trend(y, steps=3):
    """Linear trend with t-test for slope significance."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2:
        return None
    x = np.arange(n, dtype=float)
    sx = float(x.sum())
    sy = float(y.sum())
    sxx = float((x * x).sum())
    sxy = float((x * y).sum())
    den = n * sxx - sx * sx
    if abs(den) < 1e-12:
        return None
    slope = (n * sxy - sx * sy) / den
    intercept = (sy - slope * sx) / n
    fitted = intercept + slope * x
    residuals = y - fitted
    sse = float((residuals ** 2).sum())
    # R²
    ss_tot = float(((y - _mean(y)) ** 2).sum())
    r2 = 1.0 - sse / ss_tot if ss_tot > 0 else 0.0
    # AIC
    n_params = 2
    aic = n * math.log(sse / n) + 2 * n_params if sse > 0 else np.inf
    # Standard error of slope
    sigma_sq = sse / (n - n_params) if n > n_params else 0.0
    se_slope = math.sqrt(sigma_sq / (sxx - sx * sx / n)) if sigma_sq > 0 else 0.0
    t_stat = slope / se_slope if se_slope > 0 else 0.0
    # p-value (two-tailed)
    if SCIPY_AVAILABLE:
        p_value = float(stats.t.sf(abs(t_stat), n - 2) * 2)
    else:
        # rough normal approximation
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))
    # 95% CI for slope
    if SCIPY_AVAILABLE and n > 2:
        t_crit = stats.t.ppf(0.975, n - 2)
    else:
        t_crit = 1.96  # normal approximation
    ci_slope = [slope - t_crit * se_slope, slope + t_crit * se_slope]
    # forecast
    forecast = [float(intercept + slope * (n - 1 + i)) for i in range(1, steps + 1)]
    # confidence intervals for forecast (approximate)
    sigma = math.sqrt(sigma_sq) if sigma_sq > 0 else 0.0
    se_forecast = sigma * np.sqrt(1 + 1/n + ((np.arange(steps) + 1) * 0.1))  # simplified
    ci80 = [[f - 1.282 * se, f + 1.282 * se] for f, se in zip(forecast, se_forecast)]
    ci95 = [[f - 1.96 * se, f + 1.96 * se] for f, se in zip(forecast, se_forecast)]
    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r2,
        "sse": sse,
        "aic": aic,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_slope": ci_slope,
        "forecast": forecast,
        "ci80": ci80,
        "ci95": ci95,
        "sigma": sigma,
        "n": n,
        "fitted": fitted.tolist(),
        "residuals": residuals.tolist()
    }

def _holt_exponential_smoothing(y, steps=3):
    """Holt's linear trend method (optimized alpha,beta)."""
    n = len(y)
    if n < 2:
        return None
    best = None
    best_aic = float('inf')
    for alpha in np.arange(0.1, 0.95, 0.1):
        for beta in np.arange(0.1, 0.95, 0.1):
            level = y[0]
            trend = y[1] - y[0] if n > 1 else 0.0
            fitted = [y[0]]
            for i in range(1, n):
                prev_level = level
                level = alpha * y[i] + (1 - alpha) * (level + trend)
                trend = beta * (level - prev_level) + (1 - beta) * trend
                fitted.append(level)
            residuals = np.array(y) - np.array(fitted)
            sse = float((residuals ** 2).sum())
            n_params = 2
            aic = n * math.log(sse / n) + 2 * n_params if sse > 0 else float('inf')
            if aic < best_aic:
                best_aic = aic
                best = {
                    "alpha": alpha,
                    "beta": beta,
                    "level": level,
                    "trend": trend,
                    "fitted": fitted,
                    "residuals": residuals.tolist(),
                    "sse": sse,
                    "aic": aic,
                    "sigma": math.sqrt(sse / (n - n_params)) if n > n_params else 0.0,
                }
    if not best:
        return None
    # forecast
    forecast = [best["level"] + i * best["trend"] for i in range(1, steps + 1)]
    sigma = best["sigma"]
    se_forecast = sigma * np.sqrt(1 + 1/n + (np.arange(steps) + 1) * 0.1)
    ci80 = [[f - 1.282 * se, f + 1.282 * se] for f, se in zip(forecast, se_forecast)]
    ci95 = [[f - 1.96 * se, f + 1.96 * se] for f, se in zip(forecast, se_forecast)]
    best["forecast"] = forecast
    best["ci80"] = ci80
    best["ci95"] = ci95
    best["n"] = n
    return best

def _arima_111(y, steps=3):
    """Simple ARIMA(1,1,1) approximation."""
    n = len(y)
    if n < 4:
        return None
    # first difference
    diff = [y[i] - y[i-1] for i in range(1, n)]
    m = len(diff)
    # estimate phi via lag-1 autocorrelation of diff
    if m < 2:
        return None
    d_mean = _mean(diff)
    d_centered = [v - d_mean for v in diff]
    num = sum(d_centered[i] * d_centered[i+1] for i in range(m-1))
    den = sum(d_centered[i] ** 2 for i in range(m))
    phi = num / den if den != 0 else 0.0
    phi = max(-0.99, min(0.99, phi))
    # simple MA(1) fixed theta
    theta = 0.3
    # fitted on differenced series
    fitted_diff = [0.0] * m
    residuals_diff = [0.0] * m
    for i in range(m):
        prev_f = fitted_diff[i-1] if i > 0 else 0.0
        prev_res = residuals_diff[i-1] if i > 0 else 0.0
        fitted_diff[i] = phi * prev_f + theta * prev_res
        residuals_diff[i] = diff[i] - fitted_diff[i]
    # reconstruct fitted original
    fitted_orig = [y[0]]
    for i in range(m):
        fitted_orig.append(fitted_orig[-1] + fitted_diff[i])
    residuals = np.array(y) - np.array(fitted_orig)
    sse = float((residuals ** 2).sum())
    n_params = 2
    aic = n * math.log(sse / n) + 2 * n_params if sse > 0 else float('inf')
    sigma = math.sqrt(sse / (n - n_params)) if n > n_params else 0.0
    # forecast
    last_diff = diff[-1] if diff else 0.0
    last_res = residuals_diff[-1] if residuals_diff else 0.0
    forecast = []
    current = y[-1]
    for _ in range(steps):
        pred_diff = phi * last_diff + theta * last_res
        current += pred_diff
        forecast.append(current)
        last_diff = pred_diff
        # last_res remains constant for multi-step
    se_forecast = sigma * np.sqrt(1 + 1/n + (np.arange(steps) + 1) * 0.1)
    ci80 = [[f - 1.282 * se, f + 1.282 * se] for f, se in zip(forecast, se_forecast)]
    ci95 = [[f - 1.96 * se, f + 1.96 * se] for f, se in zip(forecast, se_forecast)]
    return {
        "phi": phi,
        "theta": theta,
        "fitted": fitted_orig,
        "residuals": residuals.tolist(),
        "sse": sse,
        "aic": aic,
        "sigma": sigma,
        "forecast": forecast,
        "ci80": ci80,
        "ci95": ci95,
        "n": n
    }

def advanced_forecast(y_values, steps=3):
    """Run all methods and return best by AIC."""
    if len(y_values) < 4:
        return None
    methods = {}
    lin = _linear_trend(y_values, steps)
    if lin:
        methods["linear"] = {"label": "Linear Trend", "short": "Linear", **lin}
    holt = _holt_exponential_smoothing(y_values, steps)
    if holt:
        methods["holt"] = {"label": "Holt's Exponential Smoothing", "short": "Exponential", **holt}
    ar = _arima_111(y_values, steps)
    if ar:
        methods["arima"] = {"label": "ARIMA(1,1,1)", "short": "ARIMA", **ar}
    if not methods:
        return None
    # pick best by AIC
    best_key = min(methods, key=lambda k: methods[k]["aic"])
    return {
        "methods": methods,
        "best": best_key,
        "steps": steps,
        "n": len(y_values)
    }

# ----------------------------------------------------------------------
# END ADVANCED FORECASTING
# ----------------------------------------------------------------------

def infer_trend_granularity(parsed_items):
    """Choose reporting grain from the actual calendar span.

    Business rule:
      * multi-year history -> quarterly
      * up to one year -> monthly
      * one month or a multi-month history -> weekly
    Explicit quarter/week data is preserved.
    """
    if not parsed_items:
        return None

    explicit_kinds = {x[1] for x in parsed_items}
    if "quarter" in explicit_kinds:
        return "quarter"
    if "week" in explicit_kinds:
        return "week"

    dates = sorted({pd.Timestamp(x[0]).normalize() for x in parsed_items})
    if not dates:
        return None

    span_days = (dates[-1] - dates[0]).days

    # More than one calendar year of history is presented quarter-by-quarter.
    if span_days > 366:
        return "quarter"

    # A one-year-or-shorter history is presented month-by-month.
    # Very short / monthly histories are presented week-by-week.
    if span_days <= 31:
        return "week"
    if span_days <= 366:
        # If the source is already monthly (e.g. YYYY-MM), the user wants
        # weekly reporting as well. Sparse monthly observations may therefore
        # produce only the weeks represented by the source.
        source_month_like = all(
            x[1] == "date" and pd.Timestamp(x[0]).day == 1
            for x in parsed_items
        )
        return "week" if source_month_like and len(dates) <= 13 else "month"

    return "quarter"


def prepare_time_periods(series: pd.Series):
    """Parse the time column once and cache the reusable period metadata."""
    raw = series.astype("string").str.strip()
    upper = raw.str.upper()

    qm = upper.str.extract(
        r"^(?:(\d{4})[- /]?Q([1-4])|Q([1-4])[- /]?(\d{4}))$", expand=True
    )
    q_year = pd.to_numeric(qm[0].fillna(qm[3]), errors="coerce")
    q_num = pd.to_numeric(qm[1].fillna(qm[2]), errors="coerce")
    quarter_mask = q_year.notna() & q_num.notna()

    wm = upper.str.extract(r"^(\d{4})[- /]?W(\d{1,2})$", expand=True)
    w_year = pd.to_numeric(wm[0], errors="coerce")
    w_num = pd.to_numeric(wm[1], errors="coerce")
    week_mask = (~quarter_mask) & w_year.notna() & w_num.notna() & w_num.between(1, 53)

    date_mask = ~(quarter_mask | week_mask)
    parsed_dt = pd.to_datetime(raw.where(date_mask), errors="coerce")

    frames = []
    if quarter_mask.any():
        idx = raw.index[quarter_mask]
        qdates = pd.PeriodIndex(
            q_year[quarter_mask].astype(int).astype(str) + "Q" +
            q_num[quarter_mask].astype(int).astype(str), freq="Q"
        ).to_timestamp()
        frames.append(pd.DataFrame({"dt": qdates, "source_kind": "quarter"}, index=idx))

    if week_mask.any():
        idx = raw.index[week_mask]
        iso = pd.DataFrame({"year": w_year[week_mask], "week": w_num[week_mask]}, index=idx)
        try:
            iso_dates = pd.to_datetime(
                iso["year"].astype(int).astype(str) + "-W" +
                iso["week"].astype(int).astype(str).str.zfill(2) + "-1",
                format="%G-W%V-%u", errors="coerce"
            )
        except Exception:
            iso_dates = pd.Series(pd.NaT, index=idx)
        frames.append(pd.DataFrame({"dt": iso_dates, "source_kind": "week"}, index=idx))

    if date_mask.any():
        idx = raw.index[date_mask]
        frames.append(pd.DataFrame({"dt": parsed_dt[date_mask], "source_kind": "date"}, index=idx))

    if not frames:
        return None

    work = pd.concat(frames, axis=0)
    work = work[work["dt"].notna()].copy()
    if work.empty:
        return None

    # Granularity is a property of the time column, not the metric, so calculate it once.
    parsed_items = list(zip(work["dt"], work["source_kind"], [None] * len(work),
                            [None] * len(work), [None] * len(work)))
    granularity = infer_trend_granularity(parsed_items)
    work["dt"] = pd.to_datetime(work["dt"])
    return {"work": work, "granularity": granularity}


def period_values(df: pd.DataFrame, time_col: str, metric_col: str,
                  prepared_time=None, metric_values: pd.Series | None = None):
    """Group a metric using pre-parsed time metadata whenever available."""
    if not time_col or not metric_col:
        return []

    prepared = prepared_time if prepared_time is not None else prepare_time_periods(df[time_col])
    if not prepared:
        return []

    work = prepared["work"].copy()
    vals = metric_values if metric_values is not None else numeric_series(df[metric_col])
    work["value"] = vals.reindex(work.index).to_numpy()
    work = work[work["value"].notna()].copy()
    if work.empty:
        return []

    granularity = prepared["granularity"]
    price_like = bool(re.search(
        r"\b(open|high|low|close|price|rate|index|nav|share|stock)\b",
        str(metric_col), re.I
    ))

    if granularity == "quarter":
        work["period_obj"] = work["dt"].dt.to_period("Q")
        work["key"] = work["period_obj"].astype(str)
        work["order"] = work["period_obj"].dt.start_time
    elif granularity == "month":
        work["period_obj"] = work["dt"].dt.to_period("M")
        work["key"] = work["period_obj"].astype(str)
        work["order"] = work["period_obj"].dt.start_time
    elif granularity == "week":
        work["period_obj"] = work["dt"].dt.to_period("W-SUN")
        iso = work["period_obj"].dt.start_time.dt.isocalendar()
        work["key"] = iso.year.astype("string") + "-W" + iso.week.astype("string").str.zfill(2)
        work["order"] = work["period_obj"].dt.start_time
    else:
        work["key"] = work["dt"].dt.strftime("%Y-%m-%d")
        work["order"] = work["dt"]

    if price_like:
        result_df = work.sort_values("dt").groupby("key", sort=False).tail(1).sort_values("order")
        return [{"period": str(r.key), "value": float(r.value)}
                for r in result_df[["key", "value"]].itertuples(index=False)]

    result_df = (work.groupby(["key", "order"], as_index=False, sort=True)["value"]
                 .sum().sort_values("order"))
    return [{"period": str(r.key), "value": float(r.value)}
            for r in result_df[["key", "value"]].itertuples(index=False)]

def trend_forecast(df: pd.DataFrame, time_col: str, metric_col: str, prepared_time=None, metric_values=None):
    points = period_values(df, time_col, metric_col, prepared_time, metric_values)
    n = len(points)
    if n < 2:
        return {"available": False, "reason": "At least 2 comparable time periods are required.", "periods": points}

    y = np.array([p["value"] for p in points], dtype=float)
    # use advanced forecasting
    adv = advanced_forecast(y.tolist(), steps=3)
    if adv:
        methods = adv["methods"]
        best = adv["best"]
        best_method = methods[best]
        # extract common info
        slope = methods.get("linear", {}).get("slope", None)
        r2 = methods.get("linear", {}).get("r2", None)
        growth = float(y[-1] / y[0] - 1) if y[0] != 0 else None
        direction = "Increasing" if slope and slope > 0 else "Decreasing" if slope and slope < 0 else "Stable"
        # build forecast list from best method
        forecast = best_method.get("forecast", [])
        ci80 = best_method.get("ci80", [])
        ci95 = best_method.get("ci95", [])
        # p-value from linear
        p_value = methods.get("linear", {}).get("p_value", None)
        can_forecast = n >= 4
        granularity = ("quarter" if any(re.search(r"-Q[1-4]$", p["period"]) for p in points)
                       else "month" if any(re.fullmatch(r"\d{4}-\d{2}", p["period"]) for p in points)
                       else "date")
        return {
            "available": True,
            "metric": metric_col,
            "metric_label": humanize(metric_col),
            "granularity": granularity,
            "periods": points,
            "observations": n,
            "trend": direction,
            "slope": float(slope) if slope is not None else None,
            "r2": float(r2) if r2 is not None else None,
            "first_to_last_growth": growth,
            "can_forecast": can_forecast,
            "forecast": [float(v) for v in forecast],
            "ci80": [[float(a), float(b)] for a, b in ci80] if ci80 else [],
            "ci95": [[float(a), float(b)] for a, b in ci95] if ci95 else [],
            "p_value": float(p_value) if p_value is not None else None,
            "best_method": best,
            "methods": {k: {
                "label": m.get("label", k),
                "aic": float(m["aic"]) if m.get("aic") is not None else None,
                "forecast": [float(v) for v in m.get("forecast", [])],
                "ci80": [[float(a), float(b)] for a, b in m.get("ci80", [])],
                "ci95": [[float(a), float(b)] for a, b in m.get("ci95", [])],
                "p_value": float(m.get("p_value")) if m.get("p_value") is not None else None,
                "slope": float(m["slope"]) if m.get("slope") is not None else None,
                "r2": float(m["r2"]) if m.get("r2") is not None else None,
            } for k, m in methods.items()},
            "forecast_note": "Forecast uses the best method selected by AIC among Linear, Holt, ARIMA(1,1,1)."
        }
    else:
        # fallback to simple linear if advanced fails
        x = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        ss_res = float(np.sum((y - fitted) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 if ss_tot == 0 else max(0.0, 1 - ss_res / ss_tot)
        growth = float(y[-1] / y[0] - 1) if y[0] != 0 else None
        direction = "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable"
        can_forecast = n >= 4
        forecast = []
        if can_forecast:
            for step in range(1, 4):
                value = float(intercept + slope * (n - 1 + step))
                forecast.append(value)
        granularity = ("quarter" if any(re.search(r"-Q[1-4]$", p["period"]) for p in points)
                       else "month" if any(re.fullmatch(r"\d{4}-\d{2}", p["period"]) for p in points)
                       else "date")
        return {
            "available": True,
            "metric": metric_col,
            "metric_label": humanize(metric_col),
            "granularity": granularity,
            "periods": points,
            "observations": n,
            "trend": direction,
            "slope": float(slope),
            "r2": r2,
            "first_to_last_growth": growth,
            "can_forecast": can_forecast,
            "forecast": forecast,
            "forecast_note": "Forecast uses a validated linear trend baseline from historical periods." if can_forecast else "Forecast withheld because fewer than 4 comparable periods are available."
        }


def detect_dataset_type(df: pd.DataFrame, headers: list[str], roles: dict, business_roles: dict) -> dict:
    """Classify the dataset using semantic signatures plus schema structure.

    The classifier deliberately gives stronger weight to distinctive fields
    and reduces false positives caused by generic words such as "order",
    "customer", "product" or "account".
    """
    text = " ".join(str(c).lower().replace("_", " ") for c in headers)

    signatures = {
        "Sales": {
            "strong": [r"\bsalesperson\b", r"\bsales rep\b", r"\binvoice\b", r"\border value\b"],
            "normal": [r"\bsales\b", r"\brevenue\b", r"\border\b", r"\bcustomer\b", r"\bproduct\b"],
        },
        "E-commerce": {
            "strong": [r"\bcart\b", r"\bcheckout\b", r"\bshipping\b", r"\bsku\b"],
            "normal": [r"\border\b", r"\bproduct\b", r"\bcustomer\b", r"\bpayment\b"],
        },
        "Finance": {
            "strong": [r"\bprofit\b", r"\bexpense\b", r"\bbudget\b", r"\bcash flow\b", r"\bebitda\b"],
            "normal": [r"\brevenue\b", r"\bcost\b", r"\bincome\b", r"\baccount\b"],
        },
        "Marketing": {
            "strong": [r"\bctr\b", r"\broas\b", r"\bad spend\b", r"\bimpressions?\b", r"\bclicks?\b"],
            "normal": [r"\bcampaign\b", r"\bconversion\b", r"\bleads?\b", r"\bengagement\b"],
        },
        "HR": {
            "strong": [r"\bemployee id\b", r"\bsalary\b", r"\bdesignation\b", r"\bjoining\b", r"\battendance\b"],
            "normal": [r"\bemployee\b", r"\bdepartment\b", r"\bhire\b", r"\bperformance\b"],
        },
        "Education": {
            "strong": [r"\bstudent id\b", r"\bmarks?\b", r"\bgrade\b", r"\bexam\b"],
            "normal": [r"\bstudent\b", r"\bsubject\b", r"\bteacher\b", r"\bcourse\b", r"\battendance\b"],
        },
        "Healthcare": {
            "strong": [r"\bdiagnosis\b", r"\bmedication\b", r"\badmission\b", r"\bpatient id\b"],
            "normal": [r"\bpatient\b", r"\btreatment\b", r"\bdoctor\b", r"\bhospital\b"],
        },
        "Inventory": {
            "strong": [r"\binventory\b", r"\bon hand\b", r"\breorder\b", r"\bwarehouse\b"],
            "normal": [r"\bstock\b", r"\bsku\b", r"\bquantity\b", r"\bproduct\b"],
        },
        "Manufacturing": {
            "strong": [r"\bdowntime\b", r"\byield\b", r"\bdefect\b", r"\bbatch\b"],
            "normal": [r"\bmachine\b", r"\bproduction\b", r"\bfactory\b", r"\bunits produced\b"],
        },
        "Logistics": {
            "strong": [r"\bshipment\b", r"\bcarrier\b", r"\btracking\b", r"\btransit\b"],
            "normal": [r"\bdelivery\b", r"\broute\b", r"\bfreight\b", r"\bshipping\b"],
        },
        "Real Estate": {
            "strong": [r"\bmortgage\b", r"\bbedroom\b", r"\bsquare feet\b", r"\blisting\b"],
            "normal": [r"\bproperty\b", r"\brent\b", r"\barea\b"],
        },
        "Banking": {
            "strong": [r"\bwithdrawal\b", r"\bdeposit\b", r"\bloan\b", r"\binterest\b"],
            "normal": [r"\btransaction\b", r"\baccount\b", r"\bbalance\b"],
        },
        "Web Analytics": {
            "strong": [r"\bpageviews?\b", r"\bbounce\b", r"\bsessions?\b", r"\btraffic\b"],
            "normal": [r"\bvisitor\b", r"\bwebsite\b", r"\buser\b", r"\bconversion\b"],
        },
        "Social Media": {
            "strong": [r"\bfollowers?\b", r"\bshares?\b", r"\bcomments?\b", r"\breach\b"],
            "normal": [r"\blikes?\b", r"\bengagement\b", r"\bimpressions?\b", r"\bplatform\b"],
        },
        "Restaurant": {
            "strong": [r"\bmenu\b", r"\bwaiter\b", r"\btip\b", r"\bdish\b"],
            "normal": [r"\brestaurant\b", r"\bfood\b", r"\btable\b", r"\border\b"],
        },
        "Survey": {
            "strong": [r"\bsurvey\b", r"\bsatisfaction\b", r"\bfeedback\b"],
            "normal": [r"\bresponse\b", r"\brating\b", r"\bquestion\b"],
        },
        "Project Management": {
            "strong": [r"\bmilestone\b", r"\bdeadline\b", r"\bassignee\b"],
            "normal": [r"\bproject\b", r"\btask\b", r"\bcompletion\b", r"\bstatus\b"],
        },
        "Energy": {
            "strong": [r"\bkwh\b", r"\bmwh\b", r"\bgwh\b", r"\belectricity\b"],
            "normal": [r"\benergy\b", r"\bconsumption\b", r"\bgeneration\b"],
        },
        "Stock Market / Financial Time Series": {
            "strong": [],
            "normal": [],
        },
    }

    score = {k: 0.0 for k in signatures}
    matched = {k: [] for k in signatures}

    for kind, sig in signatures.items():
        for pattern in sig["strong"]:
            if re.search(pattern, text, re.I):
                score[kind] += 5.0
                matched[kind].append(pattern)
        for pattern in sig["normal"]:
            if re.search(pattern, text, re.I):
                score[kind] += 1.5
                matched[kind].append(pattern)

    # Schema-role evidence.
    if business_roles.get("revenue"):
        score["Sales"] += 1.5
        score["Finance"] += 1.0
    if business_roles.get("profit"):
        score["Finance"] += 3.0
    if business_roles.get("cost"):
        score["Finance"] += 1.5
    if business_roles.get("product"):
        score["Sales"] += 0.75
        score["E-commerce"] += 1.0
        score["Inventory"] += 1.0
    if business_roles.get("quantity"):
        score["Inventory"] += 1.25
    if business_roles.get("payment"):
        score["E-commerce"] += 1.5
    if business_roles.get("customer"):
        score["Sales"] += 0.75
        score["E-commerce"] += 0.75

    # Strong OHLCV signature.
    open_col = next((c for c in headers if re.search(r"\b(open|opening)(?:\s+price)?\b", str(c), re.I)), None)
    high_col = next((c for c in headers if re.search(r"\b(high|highest)(?:\s+price)?\b", str(c), re.I)), None)
    low_col = next((c for c in headers if re.search(r"\b(low|lowest)(?:\s+price)?\b", str(c), re.I)), None)
    close_col = next((c for c in headers if re.search(r"\b(close|closing)(?:\s+price)?\b", str(c), re.I)), None)
    volume_col = next((c for c in headers if re.search(r"\b(trading\s+)?volume\b", str(c), re.I)), None)
    time_col = next((c for c in headers if roles.get(c) == "time"), None)
    ohlcv_count = sum(bool(x) for x in [open_col, high_col, low_col, close_col, volume_col])

    if ohlcv_count >= 4 and time_col:
        score["Stock Market / Financial Time Series"] = 20 + ohlcv_count
        matched["Stock Market / Financial Time Series"].append("OHLCV")
    elif ohlcv_count >= 3 and time_col:
        score["Stock Market / Financial Time Series"] = 12 + ohlcv_count
        matched["Stock Market / Financial Time Series"].append("OHLC")

    ranked = sorted(score.items(), key=lambda x: x[1], reverse=True)
    best, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if best_score < 2:
        best = "General Analytics"
        confidence = "low"
    elif best_score >= second_score + 3:
        confidence = "high"
    elif best_score >= second_score + 1:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "type": best,
        "confidence": confidence,
        "score": round(float(best_score), 2),
        "alternatives": [
            {"type": k, "score": round(float(v), 2)}
            for k, v in ranked[1:4] if v > 0
        ],
        "matched_signals": matched.get(best, [])[:8],
    }


def build_graph_specs(df: pd.DataFrame, headers: list[str], business_roles: dict, dataset_type: str, time_col: str | None, numeric: list[str]) -> list[dict]:
    """Create only charts whose required fields exist. These are recommendations, not hard-coded sample charts."""
    specs=[]
    rev=business_roles.get("revenue")
    quantity=business_roles.get("quantity")
    profit=business_roles.get("profit")
    cost=business_roles.get("cost")
    product=business_roles.get("product")
    category=business_roles.get("category")
    city=business_roles.get("city")
    payment=business_roles.get("payment")
    customer=business_roles.get("customer")

    def add(kind,x,y,title,limit=40,reason=""):
        if x and y and x in headers and y in headers and x != y:
            specs.append({"type":kind,"x":x,"y":y,"title":humanize(title or y),"titleSource":y,"limit":limit,"reason":reason})

    primary=rev or profit or quantity or (numeric[0] if numeric else None)
    if time_col and primary:
        add("line",time_col,primary,humanize(primary)+" Trend",60,"Time-series trend")
        if profit and profit != primary: add("line",time_col,profit,humanize(profit)+" Trend",60,"Profit trend")
        elif quantity and quantity != primary: add("line",time_col,quantity,humanize(quantity)+" Trend",60,"Volume trend")

    if category and primary: add("bar",category,primary,humanize(primary)+" by "+humanize(category),20,"Category comparison")
    if product and primary:
        # A Top-10 chart is only generated when at least 10 distinct valid products exist.
        groups=valid_group_count(df,product,primary)
        if groups >= 10: add("bar",product,primary,"Top 10 "+humanize(product)+" by "+humanize(primary),10,"Top-10 product ranking")
    if city and primary: add("bar",city,primary,humanize(primary)+" by "+humanize(city),15,"Geographic comparison")
    if payment and primary: add("doughnut",payment,primary,humanize(primary)+" by "+humanize(payment),10,"Mix analysis")
    if customer and primary and dataset_type in {"Sales","E-commerce","Banking"}: add("bar",customer,primary,humanize(primary)+" by "+humanize(customer),15,"Customer value comparison")

    # One relationship chart when two genuinely numeric measures are available.
    candidates=[c for c in numeric if c not in {cost}]
    if len(candidates)>=2:
        y=primary; x=next((c for c in candidates if c!=y),None)
        if x and y: add("scatter",x,y,humanize(y)+" vs "+humanize(x),40,"Numeric relationship")

    # Domain-specific useful chart if no revenue-like field exists.
    if dataset_type=="Marketing":
        spend=next((c for c in numeric if re.search(r"spend|cost",c,re.I)),None)
        conv=next((c for c in numeric if re.search(r"conversion|lead|click",c,re.I)),None)
        if category and spend: add("bar",category,spend,"Marketing Spend by "+humanize(category),20,"Campaign/channel comparison")
        if time_col and conv and conv!=primary: add("line",time_col,conv,humanize(conv)+" Trend",60,"Conversion trend")
    elif dataset_type=="HR":
        salary=next((c for c in numeric if re.search(r"salary|wage|compensation",c,re.I)),None)
        if salary and category: add("bar",category,salary,"Average "+humanize(salary)+" by "+humanize(category),20,"Compensation comparison")
    elif dataset_type=="Education":
        marks=next((c for c in numeric if re.search(r"mark|score|grade|result",c,re.I)),None)
        if marks and category: add("bar",category,marks,humanize(marks)+" by "+humanize(category),20,"Academic comparison")
    elif dataset_type=="Manufacturing":
        defects=next((c for c in numeric if re.search(r"defect|reject|downtime",c,re.I)),None)
        if time_col and defects: add("line",time_col,defects,humanize(defects)+" Trend",60,"Quality/operations trend")
    elif dataset_type=="Inventory":
        stock=next((c for c in numeric if re.search(r"stock|on.?hand|inventory",c,re.I)),None)
        if product and stock: add("bar",product,stock,"Stock by "+humanize(product),20,"Inventory position")

    # Deduplicate while preserving intentional chart order.
    seen=set(); out=[]
    for g in specs:
        key=(g["type"],g["x"],g["y"])
        if key not in seen:
            seen.add(key); out.append(g)
    return out[:10]

def valid_group_count(df: pd.DataFrame, dimension: str | None, metric: str | None = None) -> int:
    if not dimension or dimension not in df.columns:
        return 0
    if metric:
        if metric not in df.columns:
            return 0
        work = pd.DataFrame({"dim": df[dimension].astype("string").str.strip(), "metric": numeric_series(df[metric])})
        work = work[work["dim"].notna() & work["dim"].ne("") & work["metric"].notna()]
    else:
        work = df[[dimension]].copy()
        work[dimension] = work[dimension].astype("string").str.strip()
        work = work[work[dimension].notna() & work[dimension].ne("")]
    return int(work["dim" if metric else dimension].nunique())


def build_capabilities(df: pd.DataFrame, headers: list[str], business_roles: dict, time_col: str | None, numeric: list[str], numeric_data: dict | None = None, prepared_time=None) -> dict:
    revenue = business_roles.get("revenue")
    profit = business_roles.get("profit")
    cost = business_roles.get("cost")
    product = business_roles.get("product")
    category = business_roles.get("category")
    city = business_roles.get("city")
    salesperson = business_roles.get("salesperson")
    payment = business_roles.get("payment")

    def usable_metric(col):
        if not col or col not in df.columns:
            return False
        s = numeric_data.get(col) if numeric_data is not None and col in numeric_data else numeric_series(df[col])
        return bool(s.notna().sum() >= 1)

    def period_count(metric):
        if not time_col or not usable_metric(metric):
            return 0
        points = period_values(df, time_col, metric, prepared_time, numeric_data.get(metric) if numeric_data and metric in numeric_data else None)
        return len(points)

    product_groups = valid_group_count(df, product, revenue)
    category_groups = valid_group_count(df, category, revenue)
    city_groups = valid_group_count(df, city, revenue)
    salesperson_groups = valid_group_count(df, salesperson, revenue)
    payment_groups = valid_group_count(df, payment, revenue)

    usable_numeric = [c for c in numeric if (numeric_data.get(c) if numeric_data is not None and c in numeric_data else numeric_series(df[c])).notna().sum() >= 3]

    return {
        "dashboard": bool(headers),
        "revenue_kpi": usable_metric(revenue),
        "profit_kpi": bool((usable_metric(profit)) or (usable_metric(revenue) and usable_metric(cost))),
        "avg_order_value": usable_metric(revenue),
        "monthly_revenue_trend": period_count(revenue) >= 2,
        "monthly_profit_trend": period_count(profit) >= 2 or (usable_metric(revenue) and period_count(cost) >= 2),
        "category_performance": bool(category_groups and revenue and usable_metric(revenue)),
        "city_performance": bool(city_groups and revenue and usable_metric(revenue)),
        # A real Top-10 section is shown only when at least 10 distinct products have valid revenue.
        "top_10_products": bool(product_groups >= 10 and revenue and usable_metric(revenue)),
        "salesperson_performance": bool(salesperson_groups and revenue and usable_metric(revenue)),
        "payment_method_mix": bool(payment_groups and revenue and usable_metric(revenue)),
        "profit_margin": bool(usable_metric(revenue) and (usable_metric(profit) or usable_metric(cost))),
        "descriptive_statistics": bool(usable_numeric),
        "correlation": bool(len(usable_numeric) >= 2),
        "covariance": bool(len(usable_numeric) >= 2),
        "time_series": bool(time_col and period_count(revenue) >= 2),
        "outlier_review": bool(usable_numeric),
        "business_performance": bool(any([
            category_groups and revenue,
            city_groups and revenue,
            product_groups and revenue,
            payment_groups and revenue,
        ])),
        "cross_dimension": bool(revenue and sum(x > 0 for x in [category_groups, city_groups, product_groups, payment_groups]) >= 2),
        "what_if": usable_metric(revenue),
        "trend_forecast": bool(time_col and period_count(revenue) >= 2),
    }

# ----------------------------------------------------------------------
# DATA QUALITY SCORE (backend)
# ----------------------------------------------------------------------
def compute_quality_score(df: pd.DataFrame, headers: list[str], numeric_flags: dict | None = None, numeric_data: dict | None = None) -> dict:
    """Calculate a diagnostic quality score (0-100) and breakdown."""
    total_cells = df.shape[0] * df.shape[1]
    missing = int(df.isna().sum().sum())
    completeness = 1.0 - missing / total_cells if total_cells > 0 else 1.0

    # validity: check for invalid dates, negative quantities/financials (simplified)
    invalid_count = 0
    for col in headers:
        if "date" in col.lower() or "time" in col.lower():
            # count invalid date parse attempts (approximate)
            parsed = pd.to_datetime(df[col], errors="coerce")
            invalid_count += parsed.isna().sum()
        # negative checks for numeric columns
        is_num = numeric_flags[col] if numeric_flags is not None and col in numeric_flags else is_numeric_column(df[col])
        if is_num:
            s = numeric_data[col] if numeric_data is not None and col in numeric_data else numeric_series(df[col])
            if re.search(r"qty|quantity|units|count", col, re.I):
                invalid_count += (s < 0).sum()
            if re.search(r"price|sales|revenue|amount|cost|profit", col, re.I):
                invalid_count += (s < 0).sum()
    total_rows = df.shape[0]
    validity = 1.0 - invalid_count / (total_rows * 2) if total_rows > 0 else 1.0
    validity = max(0.0, min(1.0, validity))

    # consistency: whitespace issues (trimmed vs raw)
    whitespace = 0
    for col in headers:
        if df[col].dtype == object:
            whitespace += df[col].astype(str).str.strip().ne(df[col].astype(str)).sum()
    consistency = 1.0 - whitespace / total_cells if total_cells > 0 else 1.0

    # uniqueness: duplicate rows
    dup_rows = df.duplicated().sum()
    uniqueness = 1.0 - dup_rows / df.shape[0] if df.shape[0] > 0 else 1.0

    overall = (completeness + validity + consistency + uniqueness) / 4.0
    return {
        "score": round(overall * 100, 2),
        "completeness": round(completeness * 100, 2),
        "validity": round(validity * 100, 2),
        "consistency": round(consistency * 100, 2),
        "uniqueness": round(uniqueness * 100, 2),
        "missing_cells": missing,
        "duplicate_rows": int(dup_rows),
    }
# ----------------------------------------------------------------------

def build_graph_data(df: pd.DataFrame, specs: list[dict], max_points: int = 60, scatter_points: int = 1200) -> dict:
    """Pre-aggregate graph payloads in Python so the browser never scans 100k rows to build charts."""
    out = {}
    for idx, g in enumerate(specs[:12]):
        x, y, kind = g.get("x"), g.get("y"), g.get("type")
        if not x or not y or x not in df.columns or y not in df.columns or x == y:
            continue
        ys = numeric_series(df[y])
        if kind == "scatter":
            xs = numeric_series(df[x])
            valid = pd.DataFrame({"x": xs, "y": ys}).dropna()
            if len(valid) > scatter_points:
                step = max(1, len(valid) // scatter_points)
                valid = valid.iloc[::step].head(scatter_points)
            data = [{"x": float(a), "y": float(b)} for a, b in zip(valid["x"], valid["y"]) if np.isfinite(a) and np.isfinite(b)]
        else:
            temp = pd.DataFrame({"x": df[x].astype("string"), "y": ys})
            temp = temp.dropna(subset=["x", "y"])
            temp["x"] = temp["x"].str.strip()
            temp = temp[temp["x"] != ""]
            if kind == "line" and x == g.get("x"):
                parsed = pd.to_datetime(df.loc[temp.index, x], errors="coerce")
                temp = temp.assign(_date=parsed).dropna(subset=["_date"])
                grouped = temp.groupby("x", sort=False)["y"].sum().reset_index()
                order = temp[["x", "_date"]].drop_duplicates("x").sort_values("_date")["x"].tolist()
                order_map = {k: i for i, k in enumerate(order)}
                grouped["_ord"] = grouped["x"].map(order_map)
                grouped = grouped.sort_values("_ord")
            else:
                grouped = temp.groupby("x", sort=False)["y"].sum().reset_index()
            if kind not in {"line"}:
                grouped = grouped.sort_values("y", ascending=False)
            grouped = grouped.head(max_points)
            data = [{"x": str(a), "y": float(b)} for a, b in zip(grouped["x"], grouped["y"]) if np.isfinite(b)]
        out[str(idx)] = data
    return out

def analyze(df: pd.DataFrame, filename: str, include_rows: bool = True):
    df = df.copy()
    original_rows, original_cols = df.shape
    headers = [str(c) for c in df.columns]
    df.columns = headers

    # Detect/convert each column exactly once and reuse the result throughout analysis.
    numeric_data_all = {c: numeric_series(df[c]) for c in headers}
    numeric_ratio = {
        c: float(numeric_data_all[c].notna().mean()) if len(df) else 0.0
        for c in headers
    }
    numeric_flags = {
        c: bool(pd.api.types.is_numeric_dtype(df[c]) or numeric_ratio[c] >= 0.80)
        for c in headers
    }
    numeric = [c for c in headers if numeric_flags[c]]
    categorical = [c for c in headers if not numeric_flags[c]]
    numeric_data = {c: numeric_data_all[c] for c in numeric}
    roles = {c: role(c, df[c], numeric_flags[c]) for c in headers}
    time_cols = [
        c for c in headers
        if roles.get(c) != "identifier" and is_time_col(c, df[c])
    ]

    # Prefer a semantically named date/time column over generic period-like fields.
    time_cols.sort(key=lambda c: (
        0 if re.search(r"\b(date|timestamp|datetime)\b", str(c), re.I) else
        1 if re.search(r"\b(quarter|month|week|year|period)\b", str(c), re.I) else 2,
        headers.index(c)
    ))
    time_col = time_cols[0] if time_cols else None
    prepared_time = prepare_time_periods(df[time_col]) if time_col else None

    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())

    stats = {}
    for c in numeric:
        s = numeric_data[c].dropna()
        if len(s):
            stats[c] = {"label": humanize(c), "role": roles[c], "unit": detect_unit(c), "count": int(s.size),
                        "mean": float(s.mean()), "median": float(s.median()),
                        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0, "min": float(s.min()), "max": float(s.max())}

    correlations=[]
    if len(numeric) >= 2:
        corr_input = pd.DataFrame(numeric_data, index=df.index)
        corr_matrix = corr_input.corr()
        for i, a in enumerate(numeric):
            for b in numeric[i + 1:]:
                r = corr_matrix.at[a, b]
                if pd.notna(r):
                    n_pair = int(corr_input[[a, b]].notna().all(axis=1).sum())
                    if n_pair >= 3:
                        correlations.append({
                            "a": a, "a_label": humanize(a),
                            "b": b, "b_label": humanize(b),
                            "n": n_pair, "r": float(r)
                        })

    business_roles = {
        "revenue": next((c for c in headers if re.search(r"sales|revenue|amount|turnover|gmv|order.?value|total.?amount", c, re.I) and numeric_flags[c]), None),
        "profit": next((c for c in headers if re.search(r"profit|gross.?profit|net.?profit|margin|ebitda", c, re.I) and numeric_flags[c]), None),
        "cost": next((c for c in headers if re.search(r"cogs|cost|expense|spend", c, re.I) and numeric_flags[c]), None),
        "quantity": next((c for c in headers if re.search(r"qty|quantity|units|volume|count|production|produced", c, re.I) and numeric_flags[c]), None),
        "product": next((c for c in headers if re.search(r"product|item|model|sku|description|article", c, re.I) and not numeric_flags[c]), None),
        "category": next((c for c in headers if re.search(r"category|department|segment|class|type|channel|campaign|platform|subject|treatment|diagnosis|designation|route|carrier|supplier|store|property.?type", c, re.I) and not numeric_flags[c]), None),
        "city": next((c for c in headers if re.search(r"city|region|branch|area|location|territory|state", c, re.I) and not numeric_flags[c]), None),
        "salesperson": next((c for c in headers if re.search(r"sales.?person|sales.?rep|representative|agent|seller", c, re.I) and not numeric_flags[c]), None),
        "payment": next((c for c in headers if re.search(r"payment|pay.?method|tender|method", c, re.I) and not numeric_flags[c]), None),
        "customer": next((c for c in headers if re.search(r"customer|client|buyer|member|account", c, re.I) and not numeric_flags[c]), None),
        "orderId": next((c for c in headers if re.search(r"order.?id|order.?number|invoice|transaction.?id|receipt", c, re.I)), None),
    }

    dataset=detect_dataset_type(df,headers,roles,business_roles)
    analytic_numeric=[c for c in numeric if roles.get(c)!="identifier"]
    trend_specs=[]
    # Prefer semantic business measures, then remaining non-identifier numeric measures.
    priority=[business_roles.get(k) for k in ("revenue","profit","quantity","cost")]
    priority += [c for c in analytic_numeric if c not in priority]
    for metric in [c for c in priority if c]:
        t=trend_forecast(df,time_col,metric,prepared_time,numeric_data_all.get(metric)) if time_col else {"available":False,"reason":"No valid time/period column detected."}
        if t.get("available"):
            trend_specs.append(t)
    primary_trend=trend_specs[0] if trend_specs else {"available":False,"reason":"No valid time-series metric is available."}

    graph_specs=build_graph_specs(df,headers,business_roles,dataset["type"],time_col,analytic_numeric)
    graph_data=build_graph_data(df, graph_specs)

    # Rows are serialized directly by the API with pandas' native JSON writer.
    # This avoids building and recursively sanitizing 100k+ Python row dictionaries.
    safe_rows = df[headers].to_dict(orient="records") if include_rows else None
    capabilities=build_capabilities(df,headers,business_roles,time_col,numeric,numeric_data_all,prepared_time)
    capabilities["trend_forecast"]=bool(trend_specs)
    capabilities["forecast"]=any(t.get("can_forecast") for t in trend_specs)
    capabilities["graphs"]=bool(graph_specs)

    # Compute data quality score using backend
    quality = compute_quality_score(df, headers, numeric_flags, numeric_data_all)

    return {
        "file":filename,"records":original_rows,"fields":original_cols,
        "columns":[{"name":c} for c in headers],"rows":safe_rows,"time_column":time_col,
        "dataset_type":dataset,"business_roles":business_roles,
        "quality":quality,  # now includes score and breakdown
        "statistics":stats,
        "correlations":correlations[:20],"trend_analysis":primary_trend,"trend_specs":trend_specs,
        "graph_specs":graph_specs,"graph_data":graph_data,"capabilities":capabilities,
    }

@app.post("/api/analyze")
async def api_analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename supplied.")

    # Read once; reject oversized uploads before doing expensive analysis.
    data = await file.read(MAX_FILE_MB * 1024 * 1024 + 1)
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File is too large. Maximum allowed size is {MAX_FILE_MB} MB.")

    # Fast content-addressed cache: identical files skip parsing, analysis,
    # forecasting, graph-data generation, and JSON preparation.
    file_hash = hashlib.md5(data, usedforsecurity=False).hexdigest()
    cached_body = ANALYZE_CACHE.get(file_hash)
    if cached_body is not None:
        return Response(content=cached_body, media_type="application/json",
                        headers={"X-Analysis-Cache": "HIT"})

    # CPU-heavy work runs in FastAPI/Starlette's worker threadpool so the
    # asyncio event loop remains responsive for other requests.
    df = await run_in_threadpool(load_dataframe, data, file.filename)

    # Do not construct 100k+ Python row dictionaries. Sanitize the analysis
    # result after the CPU-heavy analyze() call completes.
    result = sanitize_json(await run_in_threadpool(analyze, df, file.filename, False))

    # Never send the complete 100k+ row dataset to the browser. Python performs the
    # full analysis; the browser only needs a deterministic preview for UI interactions.
    preview_limit = 500
    if len(df) > preview_limit:
        idx = np.linspace(0, len(df) - 1, preview_limit, dtype=int)
        preview_df = df.iloc[idx]
        result["rows_are_preview"] = True
        result["preview_rows"] = int(len(preview_df))
    else:
        preview_df = df
        result["rows_are_preview"] = False
        result["preview_rows"] = int(len(preview_df))
    result["capabilities"]["full_dataset_analyzed"] = True
    result["capabilities"]["browser_preview"] = int(len(preview_df))
    rows_json = preview_df.to_json(orient="records", date_format="iso", date_unit="ms", force_ascii=False)

    if ORJSON_AVAILABLE:
        meta = orjson.dumps(result)
    else:
        meta = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")

    body = meta[:-1] + b',"rows":' + rows_json.encode("utf-8") + b"}"

    # Keep only a small number of recent results to avoid unbounded memory use.
    if len(ANALYZE_CACHE) >= ANALYZE_CACHE_MAX_ENTRIES:
        oldest_key = next(iter(ANALYZE_CACHE))
        ANALYZE_CACHE.pop(oldest_key, None)
    ANALYZE_CACHE[file_hash] = body

    return Response(content=body, media_type="application/json",
                    headers={"X-Analysis-Cache": "MISS"})


# Production example (Linux/server):
# gunicorn -k uvicorn.workers.UvicornWorker -w 4 app:app --bind 0.0.0.0:8000
#
# Local Windows development:
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
