"""Persistent sample-history store (CSV, survives restarts)."""
import os
import json
import pandas as pd
from datetime import datetime, timezone

from .aquaguard_config import APP_DATA_DIR

SAMPLE_PATH = os.path.join(APP_DATA_DIR, 'samples.csv')
_COLS = ['record_id', 'sample_name', 'location', 'collected_at', 'created_at',
         'source_mode', 'inputs_json', 'quality_class', 'quality_conf',
         'risk_class', 'risk_conf']


def _ensure():
    if not os.path.exists(SAMPLE_PATH):
        pd.DataFrame(columns=_COLS).to_csv(SAMPLE_PATH, index=False)


def load_samples():
    _ensure()
    try:
        df = pd.read_csv(SAMPLE_PATH)
        for c in _COLS:
            if c not in df.columns:
                df[c] = pd.NA
        return df
    except Exception:
        return pd.DataFrame(columns=_COLS)


def save_sample(sample_name, location, collected_at, source_mode, inputs_dict,
                quality_class, quality_conf, risk_class, risk_conf):
    _ensure()
    df = load_samples()
    rid = f"S{(len(df) + 1):05d}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    row = {
        'record_id': rid, 'sample_name': sample_name or rid, 'location': location or '',
        'collected_at': collected_at or '', 'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'source_mode': source_mode, 'inputs_json': json.dumps(inputs_dict),
        'quality_class': quality_class, 'quality_conf': quality_conf,
        'risk_class': risk_class, 'risk_conf': risk_conf,
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(SAMPLE_PATH, index=False)
    return rid


def _utc_timestamp(value, end_of_day=False):
    text = str(value)
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')
    if end_of_day and 'T' not in text and len(text) <= 10:
        return ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return ts


def _series_to_utc(series):
    ts = pd.to_datetime(series, errors='coerce')
    if getattr(ts.dt, 'tz', None) is not None:
        return ts.dt.tz_convert('UTC')
    return ts.dt.tz_localize('UTC')


def history_summary(df=None):
    """Return saved predictions normalized for dashboard/analytics/report views."""
    if df is None:
        df = load_samples()
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=['record_id', 'sample_name', 'location', 'collected_at', 'created_at',
                                    'source_mode', 'quality_class', 'quality_conf',
                                    'risk_class', 'risk_conf'])
    out = df.copy()
    out['created_at'] = _series_to_utc(out['created_at'])
    out['quality_conf'] = pd.to_numeric(out['quality_conf'], errors='coerce')
    out['risk_conf'] = pd.to_numeric(out['risk_conf'], errors='coerce')
    out = out.sort_values('created_at', ascending=False, kind='mergesort').reset_index(drop=True)
    return out


def filter_samples(df, search='', quality='All', risk='All', source='All', date_from=None, date_to=None):
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if search:
        s = search.lower()
        mask = out['sample_name'].astype(str).str.lower().str.contains(s, na=False) | \
               out['location'].astype(str).str.lower().str.contains(s, na=False) | \
               out['record_id'].astype(str).str.lower().str.contains(s, na=False)
        out = out[mask]
    if quality != 'All':
        out = out[out['quality_class'] == quality]
    if risk != 'All':
        out = out[out['risk_class'] == risk]
    if source != 'All':
        out = out[out['source_mode'] == source]
    if date_from:
        from_ts = _utc_timestamp(date_from)
        out = out[_series_to_utc(out['created_at']) >= from_ts]
    if date_to:
        to_ts = _utc_timestamp(date_to, end_of_day=True)
        out = out[_series_to_utc(out['created_at']) <= to_ts]
    return out
