"""Data modes: HISTORICAL / SIMULATED / LIVE with clear separation."""
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone

LAST_UPDATE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'app_data', 'last_update.json')


def set_last_update(mode, detail=''):
    try:
        os.makedirs(os.path.dirname(LAST_UPDATE_PATH), exist_ok=True)
        with open(LAST_UPDATE_PATH, 'w') as f:
            json.dump({'mode': mode, 'detail': detail,
                       'at': datetime.now(timezone.utc).isoformat(timespec='seconds')}, f)
    except Exception:
        pass


def live_source_available():
    status, _, _ = read_live_source()
    return status == 'ok'


def get_last_update():
    try:
        with open(LAST_UPDATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {'mode': '–', 'detail': 'No updates yet this session.', 'at': '–'}


def simulate_reading(df, seed=42):
    """Reproducible simulated treated-output reading from historical stats. Labelled simulated."""
    rng = np.random.default_rng(seed)
    stats = df[['PH_S', 'DBO_S', 'DQO_S', 'SS_S', 'COND_S']].describe()
    sample = {}
    for col in ['PH_S', 'DBO_S', 'DQO_S', 'SS_S', 'COND_S']:
        mu = float(stats.loc['mean', col])
        sd = float(stats.loc['std', col]) or mu * 0.1 or 1.0
        v = float(rng.normal(mu, sd))
        if col == 'PH_S':
            v = float(np.clip(v, 4.0, 10.0))
        else:
            v = max(0.0, v)
        sample[col] = round(v, 2)
    sample['_simulated'] = True
    sample['_seed'] = seed
    sample['_at'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    return sample


def read_live_source():
    """Clean LIVE integration point.

    Configure ONE of:
      AQUAGUARD_LIVE_CSV  -> path to a CSV whose last row has PH_S,DBO_S,DQO_S,SS_S,COND_S (+ optional timestamp/sample_id)
      AQUAGUARD_LIVE_JSON -> path to a JSON file with those keys
      AQUAGUARD_LIVE_URL  -> HTTP(S) URL returning JSON with those keys
    Returns (status, payload, message): status in ok/stale/error/not_configured.
    NEVER fabricates readings on failure.
    """
    csv_path = os.environ.get('AQUAGUARD_LIVE_CSV', '').strip()
    json_path = os.environ.get('AQUAGUARD_LIVE_JSON', '').strip()
    url = os.environ.get('AQUAGUARD_LIVE_URL', '').strip()
    need = ['PH_S', 'DBO_S', 'DQO_S', 'SS_S', 'COND_S']

    def _validate(d):
        for k in need:
            if k not in d or d[k] is None:
                return False, f"Live payload missing required field '{k}'."
            try:
                v = float(d[k])
            except (TypeError, ValueError):
                return False, f"Live field '{k}' is not numeric: {d[k]!r}."
            if not np.isfinite(v) or v < 0 or (k == 'PH_S' and (v < 0 or v > 14)):
                return False, f"Live field '{k}' out of valid range: {v}."
        return True, ''

    def _from_frame(ldf, origin):
        missing = [k for k in need if k not in ldf.columns]
        if missing:
            return 'error', None, (f'Live CSV {origin} missing columns: {missing}. '
                                   f'Needs exactly: {need}.')
        if len(ldf) == 0:
            return 'error', None, f'Live CSV {origin} has 0 data rows.'
        row = ldf.iloc[-1].to_dict()
        ok, msg = _validate(row)
        if not ok:
            return 'error', None, f'Live CSV last row invalid: {msg}'
        return 'ok', {k: float(row[k]) for k in need}, f'Last row of {origin}'

    if url:
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=10) as r:
                payload = json.loads(r.read().decode('utf-8'))
            ok, msg = _validate(payload)
            if not ok:
                return 'error', None, f'Live URL responded but payload invalid: {msg}'
            return 'ok', payload, f'Fetched from {url}'
        except Exception as e:
            return 'error', None, f'Live URL fetch failed: {e}. Check AQUAGUARD_LIVE_URL, network, and JSON format.'
    if csv_path:
        try:
            return _from_frame(pd.read_csv(csv_path), csv_path)
        except Exception as e:
            return 'error', None, f'Live CSV read failed: {e}.'
    if json_path:
        try:
            with open(json_path) as f:
                payload = json.load(f)
            ok, msg = _validate(payload)
            if not ok:
                return 'error', None, f'Live JSON invalid: {msg}'
            return 'ok', payload, f'Read from {json_path}'
        except Exception as e:
            return 'error', None, f'Live JSON read failed: {e}.'
    return ('not_configured', None,
            'No live source configured. Set AQUAGUARD_LIVE_CSV, AQUAGUARD_LIVE_JSON, or AQUAGUARD_LIVE_URL '
            '(via a .env file next to the project — see .env.example), or upload a CSV below. '
            'HISTORICAL and SIMULATED modes keep working.')


def read_live_upload(file_obj):
    """Validate an uploaded LIVE CSV (in-memory). Same rules as the CSV path.

    Expects columns PH_S,DBO_S,DQO_S,SS_S,COND_S; uses the last row.
    Returns (status, payload, message) with status in ok/error. Never writes to disk.
    """
    need = ['PH_S', 'DBO_S', 'DQO_S', 'SS_S', 'COND_S']
    try:
        raw = file_obj.read()
    except Exception as e:
        return 'error', None, f'Could not read uploaded file: {e}.'
    try:
        import io as _io
        ldf = pd.read_csv(_io.BytesIO(raw))
    except Exception as e:
        return 'error', None, (f'Uploaded file is not a valid CSV: {e}. '
                               f'Needs columns: {need}.')
    missing = [k for k in need if k not in ldf.columns]
    if missing:
        return 'error', None, (f'Uploaded CSV missing columns: {missing}. '
                               f'Needs exactly: {need}.')
    if len(ldf) == 0:
        return 'error', None, 'Uploaded CSV has 0 data rows.'
    row = ldf.iloc[-1].to_dict()
    for k in need:
        try:
            v = float(row[k])
        except (TypeError, ValueError):
            return 'error', None, f"Uploaded field '{k}' is not numeric: {row[k]!r}."
        if not np.isfinite(v) or v < 0 or (k == 'PH_S' and (v < 0 or v > 14)):
            return 'error', None, f"Uploaded field '{k}' out of valid range: {v}."
    return 'ok', {k: float(row[k]) for k in need}, 'Last row of uploaded CSV'


LIVE_CSV_TEMPLATE = ("PH_S,DBO_S,DQO_S,SS_S,COND_S,timestamp,sample_id\n"
                     "7.4,18.5,62.0,45.0,820.0,2026-10-11T12:00:00,field-001\n")
