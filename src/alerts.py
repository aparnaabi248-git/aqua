"""Alert engine: threshold evaluation, dedup, persistent history."""
import os
import pandas as pd
from datetime import datetime, timezone

from .aquaguard_config import APP_DATA_DIR, OPERATIONAL_LIMITS

ALERT_PATH = os.path.join(APP_DATA_DIR, 'alert_history.csv')
_ALERT_COLS = ['alert_id', 'timestamp', 'sample_id', 'parameter', 'label',
               'reading', 'unit', 'threshold', 'severity', 'message',
               'source_mode', 'status']


def _ensure_store():
    if not os.path.exists(ALERT_PATH):
        pd.DataFrame(columns=_ALERT_COLS).to_csv(ALERT_PATH, index=False)


def load_alerts():
    _ensure_store()
    try:
        df = pd.read_csv(ALERT_PATH)
        for c in _ALERT_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        return df
    except Exception:
        return pd.DataFrame(columns=_ALERT_COLS)


def _save(df):
    df.to_csv(ALERT_PATH, index=False)


def evaluate_output_sample(sample, sample_id='-', source_mode='HISTORICAL', timestamp=None):
    """Evaluate treated-output readings (DBO_S, DQO_S, SS_S, COND_S, PH_S).

    sample: dict param -> float. Returns list of alert dicts (without status).
    Severity: Normal (no alert), Warning (crosses warn limit), Critical (crosses critical limit).
    """
    ts = timestamp or datetime.now(timezone.utc).isoformat(timespec='seconds')
    alerts = []
    for param, cfg in OPERATIONAL_LIMITS.items():
        if param not in sample or sample[param] is None:
            continue
        try:
            v = float(sample[param])
        except (TypeError, ValueError):
            continue
        sev = None
        thr = None
        if param == 'PH_S':
            if v < cfg['critical_low'] or v > cfg['critical_high']:
                sev, thr = 'Critical', f"< {cfg['critical_low']} or > {cfg['critical_high']}"
            elif v < cfg['warn_low'] or v > cfg['warn_high']:
                sev, thr = 'Warning', f"{cfg['warn_low']}–{cfg['warn_high']}"
        else:
            if v > cfg['critical']:
                sev, thr = 'Critical', f"> {cfg['critical']}"
            elif v > cfg['warn']:
                sev, thr = 'Warning', f"> {cfg['warn']}"
        if sev:
            alerts.append({
                'alert_id': f"{sample_id}|{param}|{sev}|{round(v, 3)}",
                'timestamp': ts, 'sample_id': str(sample_id),
                'parameter': param, 'label': cfg['label'],
                'reading': v, 'unit': cfg['unit'],
                'threshold': f"{cfg['label']} {thr} {cfg['unit']}",
                'severity': sev,
                'message': f"{cfg['label']} = {v} {cfg['unit']} ({sev}: operational limit {thr}). Source: {cfg['source']}.",
                'source_mode': source_mode, 'status': 'open',
            })
    return alerts


def record_alerts(alerts):
    """Append alerts, skipping duplicates (same alert_id still open). Returns # added."""
    if not alerts:
        return 0
    df = load_alerts()
    existing = set(df.loc[df['status'] != 'resolved', 'alert_id'].astype(str)) if len(df) else set()
    new_rows = [a for a in alerts if str(a['alert_id']) not in existing]
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        _save(df)
    return len(new_rows)


def set_alert_status(alert_id, status):
    df = load_alerts()
    mask = df['alert_id'].astype(str) == str(alert_id)
    if not mask.any():
        return False
    df.loc[mask, 'status'] = status
    _save(df)
    return True
