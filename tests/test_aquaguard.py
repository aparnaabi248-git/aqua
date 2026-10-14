"""AquaGuard AI tests: validation, features, alerts, history, modes."""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src import aquaguard_utils as U
from src.aquaguard_config import FEATURE_ORDER, INPUT_SPEC
from src import alerts as AL
from src import history as HI
from src import monitoring as MO


def _sample_inputs():
    return {s['key']: s['default'] for s in INPUT_SPEC} | {'Month': 6, 'DayOfYear': 180}


def test_feature_vector_length_and_order():
    errs, cleaned = U.validate_inputs(_sample_inputs())
    assert not errs, errs
    vec = U.build_feature_vector(cleaned)
    assert len(vec) == 35 == len(FEATURE_ORDER)
    assert all(isinstance(v, float) for v in vec)


def test_validation_rejects_bad():
    bad = _sample_inputs()
    bad['PH_E'] = 99  # outside 0-14
    errs, _ = U.validate_inputs(bad)
    assert errs
    bad2 = _sample_inputs()
    bad2['DBO_E'] = float('nan')
    errs2, _ = U.validate_inputs(bad2)
    assert errs2


def test_prediction_pipeline():
    qm = U.load_model('quality')
    rm = U.load_model('risk')
    assert qm is not None and rm is not None
    _, cleaned = U.validate_inputs(_sample_inputs())
    vec = U.build_feature_vector(cleaned)
    q = U.safe_predict(qm, vec)
    r = U.safe_predict(rm, vec)
    assert q['ok'] and q['class'] in ('GOOD', 'MODERATE', 'POOR')
    assert r['ok'] and r['class'] in ('LOW', 'MEDIUM', 'HIGH')
    assert q['probabilities'] and abs(sum(q['probabilities'].values()) - 1.0) < 1e-6


def test_safe_predict_bad_length():
    qm = U.load_model('quality')
    res = U.safe_predict(qm, [1.0, 2.0])
    assert not res['ok'] and 'Expected 35' in res['error']


def test_alerts_and_dedup(tmp_path, monkeypatch):
    monkeypatch.setattr(AL, 'ALERT_PATH', str(tmp_path / 'alerts.csv'))
    AL._ensure_store()
    a = AL.evaluate_output_sample({'DBO_S': 50, 'DQO_S': 10, 'SS_S': 10, 'COND_S': 100, 'PH_S': 7.0},
                                  sample_id='t1', source_mode='HISTORICAL', timestamp='2026-01-01T00:00:00+00:00')
    assert len(a) == 1 and a[0]['severity'] == 'Critical'  # DBO_S 50 > 30
    assert AL.record_alerts(a) == 1
    assert AL.record_alerts(a) == 0  # duplicate skipped
    assert AL.set_alert_status(a[0]['alert_id'], 'resolved') is True


def test_history_persists(tmp_path, monkeypatch):
    monkeypatch.setattr(HI, 'SAMPLE_PATH', str(tmp_path / 'samples.csv'))
    rid = HI.save_sample('n1', 'locA', '2026-01-01', 'MANUAL', {'PH_E': 7.0},
                         'GOOD', 0.9, 'LOW', 0.8)
    df = HI.load_samples()
    assert len(df) == 1 and df.iloc[0]['record_id'] == rid
    f = HI.filter_samples(df, search='locA', quality='GOOD')
    assert len(f) == 1


def test_saved_predictions_drive_shared_history_view(tmp_path, monkeypatch):
    monkeypatch.setattr(HI, 'SAMPLE_PATH', str(tmp_path / 'samples.csv'))
    HI.save_sample('a', 'Plant A', '2026-01-01', 'MANUAL', {'PH_E': 7.1}, 'GOOD', 0.91, 'LOW', 0.80)
    HI.save_sample('b', 'Plant B', '2026-02-01', 'SIMULATED', {'PH_E': 7.5}, 'POOR', 0.62, 'HIGH', 0.71)
    df = HI.load_samples()
    summary = HI.history_summary(df)
    assert len(df) == 2
    assert summary['quality_class'].tolist() == ['GOOD', 'POOR']
    min_day = summary['created_at'].min().date().isoformat()
    max_day = summary['created_at'].max().date().isoformat()
    assert len(HI.filter_samples(summary, date_from=min_day, date_to=max_day)) == 2


def test_simulation_reproducible():
    df = U.load_data()
    assert df is not None
    s1 = MO.simulate_reading(df, seed=7)
    s2 = MO.simulate_reading(df, seed=7)
    assert s1 == s2
    assert s1.get('_simulated') is True


def test_live_not_configured(monkeypatch):
    for k in ('AQUAGUARD_LIVE_CSV', 'AQUAGUARD_LIVE_JSON', 'AQUAGUARD_LIVE_URL'):
        monkeypatch.delenv(k, raising=False)
    status, payload, msg = MO.read_live_source()
    assert status == 'not_configured' and payload is None


def test_live_csv_env_path(tmp_path, monkeypatch):
    p = tmp_path / 'live.csv'
    p.write_text('PH_S,DBO_S,DQO_S,SS_S,COND_S\n7.4,18.5,62.0,45.0,820.0\n')
    monkeypatch.setenv('AQUAGUARD_LIVE_CSV', str(p))
    monkeypatch.delenv('AQUAGUARD_LIVE_URL', raising=False)
    monkeypatch.delenv('AQUAGUARD_LIVE_JSON', raising=False)
    status, payload, msg = MO.read_live_source()
    assert status == 'ok' and payload['PH_S'] == 7.4


def test_live_csv_missing_columns(tmp_path, monkeypatch):
    p = tmp_path / 'live.csv'
    p.write_text('PH_S,DBO_S\n7.4,18.5\n')
    monkeypatch.setenv('AQUAGUARD_LIVE_CSV', str(p))
    monkeypatch.delenv('AQUAGUARD_LIVE_URL', raising=False)
    monkeypatch.delenv('AQUAGUARD_LIVE_JSON', raising=False)
    status, payload, msg = MO.read_live_source()
    assert status == 'error' and 'missing columns' in msg


def test_live_upload_ok_and_bad():
    import io
    good = io.BytesIO(b'PH_S,DBO_S,DQO_S,SS_S,COND_S\n7.1,9.0,40.0,30.0,400.0\n')
    s, payload, _ = MO.read_live_upload(good)
    assert s == 'ok' and payload['COND_S'] == 400.0
    bad = io.BytesIO(b'PH_S,DBO_S\n7.1,9.0\n')
    s2, _, msg2 = MO.read_live_upload(bad)
    assert s2 == 'error' and 'missing columns' in msg2
    empty = io.BytesIO(b'PH_S,DBO_S,DQO_S,SS_S,COND_S\n')
    s3, _, msg3 = MO.read_live_upload(empty)
    assert s3 == 'error' and '0 data rows' in msg3
