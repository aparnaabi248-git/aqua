"""Genuine CPCB water-quality observations from India's National Water Data Portal.

Source: National Water Informatics Centre (NWIC), Ministry of Jal Shakti.
Portal: https://nwdp.nwic.gov.in/ (CKAN-based; public CKAN API, no auth key needed)
Catalogue: https://nwdp.nwic.gov.in/dataset/?groups=water-quality&organization=cpcb

Three CPCB datasets (per-state CSV resources, direct download links):
- physical:   monthly readings (conductivity, colour, odour, turbidity, temperature, solids)
- chemical:   annual readings (pH, DO, TDS, hardness, metals incl. zinc, ...)
- biological: monthly readings (BOD, COD, fecal/total coliform)

These are SURFACE-WATER observations. They are displayed independently and are
NEVER passed into the UCI wastewater-treatment models (incompatible features,
units and target meaning). See Step 6 of the integration brief.
"""
import io
import json
import os
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone

import pandas as pd

PORTAL = "https://nwdp.nwic.gov.in"
CKAN_API = PORTAL + "/api/3/action"
CATALOGUE_URL = PORTAL + "/dataset/?groups=water-quality&organization=cpcb"
TIME_COL = "Data Acquisition Time"
UA = {"User-Agent": "AquaGuardAI-college-project/1.0 (contact: local demo)"}

DATASETS = {
    "physical": {
        "slug": "surface-water-quality-manual-physical-parameters-cpcb",
        "title": "Surface Water Quality (Manual - Physical Parameters), CPCB",
        "frequency": "monthly",
    },
    "chemical": {
        "slug": "surface-water-quality-manual-chemical-parameters-cpcb",
        "title": "Surface Water Quality (Manual - Chemical Parameters), CPCB",
        "frequency": "annual",
    },
    "biological": {
        "slug": "surface-water-quality-manual-biological-parameters-cpcb",
        "title": "Surface Water Quality (Manual - Biological Parameters), CPCB",
        "frequency": "monthly",
    },
}
for _k, _v in DATASETS.items():
    _v["page"] = f"{PORTAL}/dataset/{_v['slug']}"

CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "app_data", "govt_cache")

# Verified numeric parameter columns (units preserved in names). Anything else
# numeric is picked up by dtype fallback in numeric_params().
PHYSICAL_PARAMS = ["Electric Conductivity (\u03bcS/cm)", "Turbidity (NTU)",
                   "Temperature (\u00baC)", "Total Solids (mg/L)"]
BIOLOGICAL_PARAMS = ["Biochemical Oxygen Demand (mg/L)", "Chemical Oxygen Demand (mg/L)",
                     "Fecal Coliform (MPN/100mL)", "Total Coliform (MPN/100mL)"]
CHEMICAL_PARAMS = ["Amonia N (mgN/L)", "Boron (mg/L)", "Carbonate (mg/L)", "Calcium (mg/L)",
                   "Chloride (mg/L)", "Dissolved oxygen (mg/L)", "Total Dissolved Solids (mg/L)",
                   "Fluoride (mg/L)", "Bicarbonate (mg/L)", "Potassium (mg/L)", "Magnesium (mg/L)",
                   "Sodium (mg/L)", "Nitrite N+Nitrate N (mgN/L)", "Potential of Hydrogen (pH)",
                   "Sulphate (mg/L)", "Total Phosphorus (mgP/L)", "Arsenic (mg/L)",
                   "Total Alkalinity (mg/L as CaCO3)", "Cadmium (mg/L)", "Chromium (mg/L)",
                   "Iron(mg/L)", "Hardness Calcium (mgCaCO3/L)", "Mercury(mg/L)",
                   "Hardness_Magnesium (mg/L as CaCO3)", "Total Hardness (mgCaCO3/L)",
                   "Manganese (mg/L)", "Nitrate N (mgN/L)", "Nickel (mg/L)", "Lead (mg/L)",
                   "Zinc (mg/L)", "Sodium Adsorption Ratio (%)", "Nitrate (mg/L)", "Copper (mg/L)"]
PARAMS_BY_KIND = {"physical": PHYSICAL_PARAMS, "chemical": CHEMICAL_PARAMS,
                  "biological": BIOLOGICAL_PARAMS}
NON_PARAM_COLS = {"SlNo", "Station", "Agency", "State LGD Code", "State", "District LGD Code",
                  "District", "Tehsil", "Block", "Village", "River", "Basin", "Tributary",
                  "Subtributary", "SubSubtributary", "Local River", "Latitude", "Longitude",
                  TIME_COL}

_RES_RE = re.compile(r"CPCB\s+(.+?)\s*\((\d{4}\s*-\s*\d{4})\)\s*(Manual)?\s*$", re.IGNORECASE)


def _http_get_bytes(url, timeout):
    """GET bytes with a clear error on failure. Monkeypatch point for tests."""
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Portal returned HTTP {e.code} for {url}. "
                           f"{'Check access/permissions.' if e.code in (401, 403) else 'Retry later.'}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach {url}: {e.reason}. Check internet/DNS.")
    except TimeoutError:
        raise RuntimeError(f"Request to {url} timed out after {timeout}s.")
    except Exception as e:
        raise RuntimeError(f"Download failed for {url}: {e}")


def package_resources(kind, timeout=30):
    """List CKAN resources for a dataset kind. Returns [{name, url, state, period}]."""
    if kind not in DATASETS:
        raise ValueError(f"Unknown dataset kind {kind!r}. Choose from {sorted(DATASETS)}.")
    raw = _http_get_bytes(f"{CKAN_API}/package_show?id={DATASETS[kind]['slug']}", timeout)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise RuntimeError(f"Portal API returned non-JSON for {kind}: {e}")
    if not payload.get("success"):
        raise RuntimeError(f"Portal API error for {kind}: {payload.get('error')}")
    out = []
    for res in payload["result"].get("resources", []):
        url = (res.get("url") or "").strip()
        name = (res.get("name") or "").strip()
        if not url or res.get("format", "").upper() != "CSV":
            continue
        m = _RES_RE.search(name)
        state = m.group(1).strip() if m else name
        period = m.group(2).strip() if m else ""
        out.append({"name": name, "url": url, "state": state, "period": period})
    if not out:
        raise RuntimeError(f"No CSV resources listed for {kind} on the portal.")
    return out


def list_states(kind, timeout=30):
    """Return {state: sorted periods} for a dataset kind."""
    states = {}
    for r in package_resources(kind, timeout=timeout):
        states.setdefault(r["state"], set()).add(r["period"])
    return {s: sorted(p) for s, p in states.items()}


def resource_url(kind, state, period, timeout=30):
    """Resolve the exact verified download URL (no guessing)."""
    cands = [r for r in package_resources(kind, timeout=timeout)
             if r["state"].lower() == state.lower() and r["period"] == period]
    if not cands:
        raise RuntimeError(f"No {kind} resource for state={state!r} period={period!r} on the portal.")
    return cands[0]


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _cache_paths(kind, state, period):
    os.makedirs(CACHE_DIR, exist_ok=True)
    base = os.path.join(CACHE_DIR, f"govt_{kind}_{_slug(state)}_{_slug(period)}")
    return base + ".csv", base + ".meta.json"


def normalize(df, kind, dataset_title, source_url):
    """Validate + normalize a downloaded CSV. Keeps original columns/units.

    Returns (df, notes). Raises RuntimeError on empty/unusable data.
    """
    if df is None or len(df) == 0:
        raise RuntimeError("Downloaded CSV contains 0 data rows.")
    df = df.drop_duplicates().copy()
    notes = []
    if TIME_COL not in df.columns:
        raise RuntimeError(f"Required column {TIME_COL!r} missing. Got: {list(df.columns)[:8]}...")
    df["_observation_dt"] = pd.to_datetime(df[TIME_COL], dayfirst=True, errors="coerce")
    bad_dates = int(df["_observation_dt"].isna().sum())
    if bad_dates:
        notes.append(f"{bad_dates} row(s) had unparseable dates and are excluded from trends.")
    df["_dataset"] = dataset_title
    df["_source_url"] = source_url
    df["_retrieved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return df, notes


def refresh(kind, state, period, timeout=90):
    """Download + normalize + cache. Falls back to last good cache on failure.

    Returns dict(status, df|None, meta, message, notes, fresh:bool).
    status in {'fresh', 'cache', 'error'}. Never returns fabricated rows.
    """
    csv_path, meta_path = _cache_paths(kind, state, period)
    try:
        res = resource_url(kind, state, period)
        raw = _http_get_bytes(res["url"], timeout)
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise RuntimeError(
                f"Portal response for {state} {kind} ({period}) is not a parseable CSV "
                f"(HTTP 200, {len(raw)} bytes): {e}. The resource may be an HTML/JSON page. "
                f"Manual download: {DATASETS[kind]['page']}")
        if df is None or len(df) == 0:
            raise RuntimeError(
                f"Portal file for {state} {kind} ({period}) downloaded fine (HTTP 200, "
                f"{len(raw)} bytes) but contains 0 data rows — no observations published for "
                f"this period yet. Pick an earlier period such as 2021 - 2025 and refresh. "
                f"Resource page: {DATASETS[kind]['page']}")
        df, notes = normalize(df, kind, DATASETS[kind]["title"], res["url"])
        df.to_csv(csv_path, index=False)
        valid = df["_observation_dt"].dropna()
        meta = {"dataset": DATASETS[kind]["title"], "kind": kind, "state": state,
                "period": period, "source_url": res["url"],
                "portal_page": DATASETS[kind]["page"], "catalogue": CATALOGUE_URL,
                "frequency": DATASETS[kind]["frequency"],
                "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "latest_observation": (valid.max().isoformat() if len(valid) else None),
                "rows": len(df)}
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        return {"status": "fresh", "fresh": True, "df": df, "meta": meta,
                "message": f"Retrieved {len(df)} official records.", "notes": notes}
    except Exception as e:
        cached = load_cached(kind, state, period)
        if cached is not None:
            cdf, cmeta = cached
            return {"status": "cache", "fresh": False, "df": cdf, "meta": cmeta,
                    "message": f"Refresh failed ({e}). Showing last successful retrieval "
                               f"({cmeta.get('retrieved_at', 'unknown time')}).",
                    "notes": []}
        return {"status": "error", "fresh": False, "df": None, "meta": None,
                "message": f"Refresh failed and no cached data exists: {e}", "notes": []}


def load_cached(kind, state, period):
    """Return (df, meta) from last successful retrieval, or None."""
    csv_path, meta_path = _cache_paths(kind, state, period)
    if not (os.path.exists(csv_path) and os.path.exists(meta_path)):
        return None
    try:
        df = pd.read_csv(csv_path)
        if "_observation_dt" in df.columns:
            df["_observation_dt"] = pd.to_datetime(df["_observation_dt"], errors="coerce")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if len(df) == 0:
            return None
        return df, meta
    except Exception:
        return None


def numeric_params(df, kind):
    """Numeric measurement columns with units preserved; dtype fallback included."""
    known = [c for c in PARAMS_BY_KIND.get(kind, []) if c in df.columns]
    extra = [c for c in df.columns
             if c not in NON_PARAM_COLS and c not in known
             and not c.startswith("_") and pd.api.types.is_numeric_dtype(df[c])]
    return known + extra


# ------------------------------------------------- explorer / map / alerts ---
def station_summary(df):
    """One row per station: district, records, latest observation, coordinates.

    Returns DataFrame sorted by Station. Empty input -> empty frame.
    """
    cols = ["Station", "District", "records", "latest_observation", "Latitude", "Longitude"]
    if df is None or len(df) == 0 or "Station" not in df.columns:
        return pd.DataFrame(columns=cols)
    d = df.copy()
    if "_observation_dt" in d.columns:
        d["_observation_dt"] = pd.to_datetime(d["_observation_dt"], errors="coerce")
    grp = d.groupby(d["Station"].astype(str), sort=True)
    rows = []
    for station, g in grp:
        lat = g["Latitude"].dropna().astype(float).iloc[0] if "Latitude" in g and g["Latitude"].notna().any() else None
        lon = g["Longitude"].dropna().astype(float).iloc[0] if "Longitude" in g and g["Longitude"].notna().any() else None
        # Reject impossible coordinates rather than mapping them.
        if lat is not None and not (-90.0 <= lat <= 90.0):
            lat = None
        if lon is not None and not (-180.0 <= lon <= 180.0):
            lon = None
        latest = g["_observation_dt"].max() if "_observation_dt" in g else pd.NaT
        rows.append({
            "Station": station,
            "District": str(g["District"].dropna().astype(str).iloc[0]) if "District" in g and g["District"].notna().any() else "",
            "records": int(len(g)),
            "latest_observation": latest if pd.notna(latest) else None,
            "Latitude": lat, "Longitude": lon,
        })
    return pd.DataFrame(rows, columns=cols)


def map_points(df):
    """Split stations into mappable points vs stations lacking valid coordinates.

    Returns (points_df, missing_count). Never invents locations.
    """
    summary = station_summary(df)
    if len(summary) == 0:
        return summary, 0
    ok = summary.dropna(subset=["Latitude", "Longitude"])
    return ok.reset_index(drop=True), int(len(summary) - len(ok))


def compare_station_stats(df, station_a, station_b, param):
    """Summary stats for two stations on one parameter. Pure/testable.

    Returns dict{a: {...}|None, b: {...}|None, comparable: bool}.
    """
    def _stats(station):
        if df is None or len(df) == 0 or param not in df.columns:
            return None
        g = df[df["Station"].astype(str) == str(station)]
        vals = pd.to_numeric(g[param], errors="coerce").dropna()
        if len(vals) == 0:
            return None
        latest = None
        if "_observation_dt" in g:
            gl = g.loc[vals.index]
            if gl["_observation_dt"].notna().any():
                latest = gl.sort_values("_observation_dt").iloc[-1]
        return {"station": str(station), "n": int(len(vals)),
                "mean": float(vals.mean()), "min": float(vals.min()),
                "max": float(vals.max()),
                "latest_value": (float(pd.to_numeric(latest[param], errors="coerce"))
                                 if latest is not None else None)}
    a, b = _stats(station_a), _stats(station_b)
    return {"a": a, "b": b, "comparable": a is not None and b is not None}


def evaluate_user_thresholds(df, thresholds):
    """Flag filtered records against USER-configured thresholds.

    thresholds: {param: {"warn": float, "critical": float}} with warn < critical
    (upper-limit semantics: value > warn -> Warning, value > critical -> Critical).
    Returns (alerts_df, errors). alerts columns: station, district, date,
    parameter, value, severity, threshold, reason. Rows with missing values are
    skipped silently (counted in notes via errors list info, not fabricated).
    """
    errors, alerts = [], []
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["station", "district", "date", "parameter",
                                     "value", "severity", "threshold", "reason"]), errors
    for param, lim in (thresholds or {}).items():
        try:
            w, c = float(lim.get("warn")), float(lim.get("critical"))
        except (TypeError, ValueError, AttributeError):
            errors.append(f"{param}: thresholds must be numbers.")
            continue
        if not (w < c):
            errors.append(f"{param}: warning ({w}) must be below critical ({c}). Skipped.")
            continue
        if param not in df.columns:
            errors.append(f"{param}: column not present in retrieved data. Skipped.")
            continue
        vals = pd.to_numeric(df[param], errors="coerce")
        for idx in vals[vals.notna()].index:
            v = float(vals.loc[idx])
            sev, thr = (None, None)
            if v > c:
                sev, thr = "Critical", c
            elif v > w:
                sev, thr = "Warning", w
            if sev is None:
                continue
            row = df.loc[idx]
            dt = row.get("_observation_dt", None)
            alerts.append({
                "station": str(row.get("Station", "")),
                "district": str(row.get("District", "")),
                "date": (pd.Timestamp(dt).isoformat() if pd.notna(dt) else ""),
                "parameter": param, "value": v, "severity": sev, "threshold": thr,
                "reason": f"{param} = {v} exceeds your {sev.lower()} limit of {thr} "
                          f"(user-configured, not a CPCB standard).",
            })
    cols = ["station", "district", "date", "parameter", "value", "severity", "threshold", "reason"]
    out = pd.DataFrame(alerts, columns=cols)
    if len(out):
        out = out.sort_values(["severity", "date"], ascending=[True, True]).reset_index(drop=True)
    return out, errors
