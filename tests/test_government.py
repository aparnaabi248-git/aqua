"""Tests for the NWDP/CPCB government-data integration (offline, mocked HTTP)."""
import io
import json
import os
import sys
import urllib.error

import pandas as pd
import pytest

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from src import government_water_data as GV

SAMPLE_CSV = """SlNo,Station,Agency,State,District,Latitude,Longitude,Data Acquisition Time,Electric Conductivity (\u03bcS/cm),Colour,Turbidity (NTU)
1,ST-A,CPCB,Tamil Nadu,Thoothukudi,8.60,77.94,21-04-2021 08:30,348.0,Clear,2.5
2,ST-A,CPCB,Tamil Nadu,Thoothukudi,8.60,77.94,21-10-2021 08:30,135.0,Clear,1.1
3,ST-B,CPCB,Tamil Nadu,Chennai,13.08,80.27,not-a-date,500.0,Clear,3.0
"""

PKG_JSON = json.dumps({
    "success": True,
    "result": {"resources": [
        {"name": "Surface Water Quality Physical Parameters CPCB Tamil Nadu (2021 - 2025) Manual",
         "format": "CSV", "url": "https://nwdp.nwic.gov.in/dataset/x/resource/y/download/tn.csv"},
        {"name": "Some metadata PDF", "format": "PDF", "url": "https://example.com/a.pdf"},
    ]},
}).encode()


def _patch(monkeypatch, tmp_path, mapping):
    """Route _http_get_bytes through mapping {url_substring: bytes}; fail otherwise."""
    monkeypatch.setattr(GV, "CACHE_DIR", str(tmp_path))

    def fake(url, timeout=30):
        for key, val in mapping.items():
            if key in url:
                if isinstance(val, Exception):
                    raise val
                return val
        raise AssertionError(f"unexpected URL {url}")
    monkeypatch.setattr(GV, "_http_get_bytes", fake)


def test_fresh_fetch_and_normalize(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, {"package_show": PKG_JSON, "download/tn.csv": SAMPLE_CSV.encode()})
    res = GV.refresh("physical", "Tamil Nadu", "2021 - 2025")
    assert res["status"] == "fresh" and res["fresh"] is True
    df = res["df"]
    assert len(df) == 3 and "_observation_dt" in df.columns
    assert df["_observation_dt"].notna().sum() == 2  # one bad date
    assert any("unparseable" in n for n in res["notes"])
    assert res["meta"]["latest_observation"].startswith("2021-10-21")
    assert os.path.exists(os.path.join(str(tmp_path), [f for f in os.listdir(str(tmp_path)) if f.endswith(".csv")][0]))


def test_empty_csv_is_error_without_cache(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, {"package_show": PKG_JSON,
                                   "download/tn.csv": b"SlNo,Station\n"})
    res = GV.refresh("physical", "Tamil Nadu", "2021 - 2025")
    assert res["status"] == "error" and res["df"] is None
    assert "0 data rows" in res["message"]


def test_missing_time_column_is_error(monkeypatch, tmp_path):
    bad = b"SlNo,Station,Value\n1,ST-A,5\n"
    _patch(monkeypatch, tmp_path, {"package_show": PKG_JSON, "download/tn.csv": bad})
    res = GV.refresh("physical", "Tamil Nadu", "2021 - 2025")
    assert res["status"] == "error"
    assert "Data Acquisition Time" in res["message"]


def test_timeout_falls_back_to_cache(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, {"package_show": PKG_JSON, "download/tn.csv": SAMPLE_CSV.encode()})
    first = GV.refresh("physical", "Tamil Nadu", "2021 - 2025")
    assert first["status"] == "fresh"
    _patch(monkeypatch, tmp_path, {"package_show": TimeoutError("timed out")})
    second = GV.refresh("physical", "Tamil Nadu", "2021 - 2025")
    assert second["status"] == "cache" and second["fresh"] is False
    assert len(second["df"]) == 3
    assert "timed out" in second["message"]


def test_http_error_without_cache(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, {"package_show": urllib.error.HTTPError(
        "http://x", 403, "Forbidden", {}, None)})
    res = GV.refresh("physical", "Tamil Nadu", "2021 - 2025")
    assert res["status"] == "error" and "403" in res["message"]


def test_numeric_params_uses_known_and_fallback(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, {"package_show": PKG_JSON, "download/tn.csv": SAMPLE_CSV.encode()})
    res = GV.refresh("physical", "Tamil Nadu", "2021 - 2025")
    params = GV.numeric_params(res["df"], "physical")
    assert "Electric Conductivity (\u03bcS/cm)" in params
    assert "Turbidity (NTU)" in params
    assert "Station" not in params and "SlNo" not in params


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        GV.package_resources("nonsense")


HEADER_ONLY_CSV = ("SlNo,Station,Agency,State,District,Latitude,Longitude,Data Acquisition Time,"
                   "Electric Conductivity (\u03bcS/cm),Colour,Turbidity (NTU)\n").encode()


def test_header_only_future_period_reports_zero_rows(monkeypatch, tmp_path):
    """The portal's 2026-2030 bucket is a real CSV with headers but 0 rows."""
    pkg_2026 = json.dumps({
        "success": True,
        "result": {"resources": [
            {"name": "Surface Water Quality Physical Parameters CPCB Tamil Nadu (2026 - 2030) Manual",
             "format": "CSV", "url": "https://nwdp.nwic.gov.in/dataset/x/resource/y/download/tn2630.csv"},
        ]},
    }).encode()
    _patch(monkeypatch, tmp_path, {"package_show": pkg_2026, "download/tn2630.csv": HEADER_ONLY_CSV})
    res = GV.refresh("physical", "Tamil Nadu", "2026 - 2030")
    assert res["status"] == "error" and res["df"] is None
    assert "0 data rows" in res["message"] and "2026 - 2030" in res["message"]


def test_html_instead_of_csv_is_error(monkeypatch, tmp_path):
    html = b"<html><head><title>Error</title></head><body>Access denied</body></html>"
    _patch(monkeypatch, tmp_path, {"package_show": PKG_JSON, "download/tn.csv": html})
    res = GV.refresh("physical", "Tamil Nadu", "2021 - 2025")
    assert res["status"] == "error" and res["df"] is None


def test_failed_http_request_without_cache(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, {"package_show": PKG_JSON,
                                   "download/tn.csv": urllib.error.URLError("no route to host")})
    res = GV.refresh("physical", "Tamil Nadu", "2021 - 2025")
    assert res["status"] == "error" and res["df"] is None
    assert "no route" in res["message"]


def test_valid_csv_reports_records(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, {"package_show": PKG_JSON, "download/tn.csv": SAMPLE_CSV.encode()})
    res = GV.refresh("physical", "Tamil Nadu", "2021 - 2025")
    assert res["status"] == "fresh"
    assert res["meta"]["rows"] == 3
    assert "Data Acquisition Time" in res["df"].columns


def _explorer_frame():
    return pd.DataFrame({
        "Station": ["ST-A", "ST-A", "ST-B", "ST-C"],
        "District": ["D1", "D1", "D2", "D3"],
        "Latitude": [8.6, 8.6, 13.08, None],
        "Longitude": [77.94, 77.94, 80.27, None],
        "Data Acquisition Time": ["21-04-2021 08:30", "21-10-2021 08:30", "05-05-2021 09:00", "bad-date"],
        "_observation_dt": pd.to_datetime(["2021-04-21", "2021-10-21", "2021-05-05", None]),
        "Turbidity (NTU)": [2.5, 1.1, 9.9, None],
    })


def test_station_summary_counts_and_coords():
    s = GV.station_summary(_explorer_frame())
    assert len(s) == 3
    row_a = s[s["Station"] == "ST-A"].iloc[0]
    assert row_a["records"] == 2 and row_a["Latitude"] == 8.6
    assert GV.station_summary(pd.DataFrame()).empty


def test_map_points_never_invents_locations():
    pts, missing = GV.map_points(_explorer_frame())
    assert len(pts) == 2 and missing == 1
    assert pts["Latitude"].between(-90, 90).all()
    bad = _explorer_frame()
    bad["Latitude"] = 999.0
    pts2, missing2 = GV.map_points(bad)
    assert len(pts2) == 0 and missing2 == 3


def test_compare_station_stats():
    cmp = GV.compare_station_stats(_explorer_frame(), "ST-A", "ST-B", "Turbidity (NTU)")
    assert cmp["comparable"] is True
    assert cmp["a"]["n"] == 2 and abs(cmp["a"]["mean"] - 1.8) < 1e-9
    assert cmp["b"]["max"] == 9.9
    cmp2 = GV.compare_station_stats(_explorer_frame(), "ST-A", "ST-C", "Turbidity (NTU)")
    assert cmp2["comparable"] is False  # ST-C has no valid values
    cmp3 = GV.compare_station_stats(_explorer_frame(), "ST-A", "ST-B", "Nope (x)")
    assert cmp3["comparable"] is False


def test_user_thresholds_flag_and_validate():
    df = _explorer_frame()
    alerts, errs = GV.evaluate_user_thresholds(df, {"Turbidity (NTU)": {"warn": 2.0, "critical": 5.0}})
    assert not errs
    assert set(alerts["severity"]) == {"Warning", "Critical"}
    crit = alerts[alerts["severity"] == "Critical"].iloc[0]
    assert crit["station"] == "ST-B" and crit["threshold"] == 5.0 and "user-configured" in crit["reason"]
    # invalid config reported, not applied
    alerts2, errs2 = GV.evaluate_user_thresholds(df, {"Turbidity (NTU)": {"warn": 9.0, "critical": 5.0},
                                                      "Missing (x)": {"warn": 1, "critical": 2}})
    assert len(alerts2) == 0 and len(errs2) == 2
    # empty input -> empty alerts, no crash
    alerts3, _ = GV.evaluate_user_thresholds(pd.DataFrame(), {"Turbidity (NTU)": {"warn": 1, "critical": 2}})
    assert len(alerts3) == 0
