from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from energy_arbitrage import original_logic as ol

ROOT = Path(__file__).resolve().parents[1]


def test_original_notebook_is_preserved():
    notebook = ROOT / "notebooks" / "original" / "Analytathon1.ipynb"
    assert notebook.exists()
    assert hashlib.sha256(notebook.read_bytes()).hexdigest() == "a8fb534d22c3a167b257ab545f69609bbd2e7f67d64591a6d8a6a510f1c705be"


def test_original_model_constants_are_preserved():
    assert ol.LAG_BURNIN == 48 * 14
    assert ol.BEST_THRESH == 0.20
    clf = ol.make_clf()
    reg = ol.make_reg()
    if ol.HAS_LGB:
        assert clf.__class__.__name__ == "LGBMClassifier"
        assert clf.get_params()["n_estimators"] == 800
        assert clf.get_params()["learning_rate"] == 0.03
        assert clf.get_params()["max_depth"] == 6
        assert clf.get_params()["num_leaves"] == 63
        assert clf.get_params()["min_child_samples"] == 30
        assert clf.get_params()["subsample"] == 0.8
        assert clf.get_params()["colsample_bytree"] == 0.7
        assert clf.get_params()["reg_alpha"] == 0.1
        assert clf.get_params()["reg_lambda"] == 1.0
        assert reg.__class__.__name__ == "LGBMRegressor"
        assert reg.get_params()["n_estimators"] == 800


def test_transition_rules_are_preserved():
    assert ol.TRANSITIONS["IDA1→IDA2"][3] == "hod >= 11"
    assert ol.TRANSITIONS["IDA2→IDA3"][3] == "hod >= 17"
    assert ol.TRANSITIONS["DAM→BM"][0] == "spr_BM_DAM"
    assert ol.TRANSITIONS["DAM→IDA1"][0] == "spr_IDA1_DAM"


def test_clean_and_feature_engineer_sample_data():
    sample = ROOT / "data" / "sample" / "isem_synthetic_sample.csv"
    raw = ol.load_market_data(sample)
    cleaned = ol.clean_market_data(raw)
    df = ol.add_spreads_and_fundamentals(cleaned)
    df = ol.engineer_features(df)
    for col in ["spr_IDA1_DAM", "spr_BM_DAM", "NetDemand_DAM", "hour_sin", "wind_uncertainty_DAM", "ren_pct_DAM"]:
        assert col in df.columns
    xcols = ol.build_xcols(df, [a.upper() for a in ol.TRANSITIONS["DAM→IDA1"][1]], ol.TRANSITIONS["DAM→IDA1"][2])
    d, yc, yr, orig = ol.build_ds(df, "spr_IDA1_DAM", xcols)
    assert len(d) > 0
    assert min(orig) >= ol.LAG_BURNIN
    assert set(yc.unique()).issubset({-1, 1})


def test_no_local_junk_files_committed():
    forbidden = {".DS_Store", ".Rhistory", ".RData", ".Ruserdata", "__MACOSX"}
    bad = [p for p in ROOT.rglob("*") if p.name in forbidden]
    assert bad == []


def test_settings_file_is_present_and_non_empty():
    settings = ROOT / "config" / "settings.yaml"
    assert settings.exists()
    text = settings.read_text()
    assert "prefer_lightgbm: true" in text
    assert "n_estimators: 800" in text
    assert "lag_burnin_periods: 672" in text


def test_code_is_modularised_not_monolithic():
    expected = ["config.py", "data.py", "features.py", "modeling.py", "backtesting.py", "visualization.py", "pipeline.py"]
    src_dir = ROOT / "src" / "energy_arbitrage"
    for name in expected:
        path = src_dir / name
        assert path.exists(), f"Missing {name}"
        assert len(path.read_text().splitlines()) > 10


def test_pre_generated_outputs_and_figures_exist():
    expected_outputs = [
        ROOT / "outputs" / "original_model_performance.csv",
        ROOT / "outputs" / "original_trading_performance.csv",
        ROOT / "outputs" / "original_market_price_summary.csv",
    ]
    expected_figures = [
        ROOT / "reports" / "figures" / "original_model_accuracy.png",
        ROOT / "reports" / "figures" / "original_total_pnl.png",
        ROOT / "reports" / "figures" / "original_accuracy_vs_profitability.png",
    ]
    for path in expected_outputs + expected_figures:
        assert path.exists(), f"Missing {path}"
        assert path.stat().st_size > 100


def test_gitignore_blocks_cache_and_os_artifacts():
    text = (ROOT / ".gitignore").read_text()
    for token in [".DS_Store", "__pycache__/", ".pytest_cache/", ".Rhistory", "data/private/"]:
        assert token in text
