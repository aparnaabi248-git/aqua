"""Shared AquaGuard AI utilities: loading, validation, feature building, prediction."""
import os
import json
import joblib
import pandas as pd
import numpy as np

from .aquaguard_config import (
    BASE_DIR, DATA_PROCESSED, MODELS_DIR, FEATURE_ORDER, INPUT_SPEC,
)

_PROCESSED_PATH = os.path.join(DATA_PROCESSED, 'processed_data.csv')
_TRAIN_RANGES_CACHE = None


def load_data():
    if not os.path.exists(_PROCESSED_PATH):
        return None
    df = pd.read_csv(_PROCESSED_PATH)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df


def load_model(target='quality'):
    model_path = os.path.join(MODELS_DIR, f'best_model_{target}.pkl')
    scaler_path = os.path.join(MODELS_DIR, f'scaler_{target}.pkl')
    le_path = os.path.join(MODELS_DIR, f'label_encoder_{target}.pkl')
    metadata_path = os.path.join(MODELS_DIR, f'model_metadata_{target}.json')
    if not os.path.exists(model_path):
        return None
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)
    except Exception:
        return None
    metadata = None
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception:
            metadata = None
    return {'model': model, 'scaler': scaler, 'label_encoder': le, 'metadata': metadata}


def get_training_ranges():
    """Return {feature: (min, max)} from processed data for out-of-range warnings."""
    global _TRAIN_RANGES_CACHE
    if _TRAIN_RANGES_CACHE is not None:
        return _TRAIN_RANGES_CACHE
    df = load_data()
    ranges = {}
    if df is not None:
        for feat in FEATURE_ORDER:
            if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
                s = df[feat].dropna()
                if len(s):
                    ranges[feat] = (float(s.min()), float(s.max()))
    # Month/DayOfYear valid calendar ranges even if data is 1990-only
    ranges.setdefault('Month', (1.0, 12.0))
    ranges.setdefault('DayOfYear', (1.0, 365.0))
    _TRAIN_RANGES_CACHE = ranges
    return ranges


def validate_inputs(values):
    """Validate user inputs against INPUT_SPEC. Returns (errors, warnings, cleaned)."""
    errors = []
    cleaned = {}
    spec = {s['key']: s for s in INPUT_SPEC}
    for key, s in spec.items():
        if key not in values or values[key] is None:
            errors.append(f"{s['label']} is required.")
            continue
        try:
            v = float(values[key])
        except (TypeError, ValueError):
            errors.append(f"{s['label']} must be a number.")
            continue
        if not np.isfinite(v):
            errors.append(f"{s['label']} must be a finite number.")
            continue
        if v < s['min'] or v > s['max']:
            errors.append(f"{s['label']} must be between {s['min']} and {s['max']} {s['unit']}. Got {v}.")
            continue
        cleaned[key] = v
    # month / day
    for key, lo, hi, label in [('Month', 1, 12, 'Month'), ('DayOfYear', 1, 365, 'Day of year')]:
        if key not in values or values[key] is None:
            errors.append(f"{label} is required.")
            continue
        try:
            v = int(values[key])
        except (TypeError, ValueError):
            errors.append(f"{label} must be a whole number.")
            continue
        if v < lo or v > hi:
            errors.append(f"{label} must be between {lo} and {hi}.")
            continue
        cleaned[key] = v
    return errors, cleaned


def build_feature_vector(cleaned):
    """Build the 35-feature vector in FEATURE_ORDER.

    Engineered features: removal efficiencies require treated-output values which
    are unknown at prediction time, so they are set to 0.0 (same convention as
    the original dashboard). Averages use the E/P/D stages available at input.
    """
    c = cleaned
    bod_rem = 0.0
    cod_rem = 0.0
    ss_rem = 0.0
    cond_rem = 0.0
    bod_cod = c['DBO_E'] / (c['DQO_E'] + 1.0)
    ss_ssv = c['SS_E'] / (c['SSV_E'] + 1.0)
    avg_ph = float(np.mean([c['PH_E'], c['PH_P'], c['PH_D']]))
    avg_dbo = float(np.mean([c['DBO_E'], c['DBO_P'], c['DBO_D']]))
    avg_dqo = float(np.mean([c['DQO_E'], c['DQO_P'], c['DQO_D']]))
    avg_cond = float(np.mean([c['COND_E'], c['COND_P'], c['COND_D']]))
    vec_map = dict(c)
    vec_map.update({
        'BOD_removal_eff': bod_rem, 'COD_removal_eff': cod_rem,
        'SS_removal_eff': ss_rem, 'COND_removal_eff': cond_rem,
        'BOD_COD_ratio': bod_cod, 'SS_SSV_ratio': ss_ssv,
        'avg_pH': avg_ph, 'avg_DBO': avg_dbo, 'avg_DQO': avg_dqo,
        'avg_COND': avg_cond,
    })
    return [float(vec_map[f]) for f in FEATURE_ORDER]


def check_out_of_range(feature_vector):
    """Return list of (feature, value, train_min, train_max) outside training range."""
    ranges = get_training_ranges()
    out = []
    for feat, val in zip(FEATURE_ORDER, feature_vector):
        if feat in ranges:
            lo, hi = ranges[feat]
            if val < lo or val > hi:
                out.append((feat, val, lo, hi))
    return out


def safe_predict(model_dict, feature_vector):
    """Predict with validation. Returns dict with ok/error info. Never raises."""
    if model_dict is None:
        return {'ok': False, 'error': 'Model is not loaded. Run python src/pipeline.py to train models.'}
    try:
        arr = np.asarray(feature_vector, dtype=float).reshape(1, -1)
    except Exception as e:
        return {'ok': False, 'error': f'Invalid input values: {e}'}
    if arr.shape[1] != len(FEATURE_ORDER):
        return {'ok': False, 'error': f'Expected {len(FEATURE_ORDER)} features, got {arr.shape[1]}.'}
    if not np.all(np.isfinite(arr)):
        return {'ok': False, 'error': 'Inputs must be finite numbers (no NaN/infinity).'}
    try:
        scaled = model_dict['scaler'].transform(arr)
        pred = model_dict['model'].predict(scaled)[0]
        classes = list(model_dict['label_encoder'].classes_)
        name = model_dict['label_encoder'].inverse_transform([pred])[0]
        probs = None
        if hasattr(model_dict['model'], 'predict_proba'):
            try:
                p = model_dict['model'].predict_proba(scaled)[0]
                probs = {str(cls): float(v) for cls, v in zip(classes, p)}
            except Exception:
                probs = None
        return {'ok': True, 'class': str(name), 'probabilities': probs, 'classes': classes}
    except Exception as e:
        return {'ok': False, 'error': f'Prediction failed: {e}'}


def top_contributing_features(model_dict, feature_vector, top_n=5):
    """Return [(feature, importance, value)] for tree models, else []."""
    try:
        if model_dict is None or not hasattr(model_dict['model'], 'feature_importances_'):
            return []
        imps = model_dict['model'].feature_importances_
        trips = list(zip(FEATURE_ORDER, [float(x) for x in imps], [float(x) for x in feature_vector]))
        trips.sort(key=lambda t: t[1], reverse=True)
        return trips[:top_n]
    except Exception:
        return []
