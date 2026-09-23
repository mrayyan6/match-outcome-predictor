"""The site has to give the same probabilities as the python models."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loader import FEATURES, PROCESSED  # noqa: E402
from model import MODELS, apply_calibration, seasons, slug  # noqa: E402

NODE = shutil.which("node")
WEB = ROOT / "web" / "data"


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.parametrize("league", ["Premier League", "La Liga"])
@pytest.mark.parametrize("kind", ["rf", "xgb"])
@pytest.mark.parametrize("method", ["raw", "platt", "isotonic"])
def test_browser_matches_python(league, kind, method, tmp_path):
    matches = pd.read_csv(PROCESSED / "matches.csv")
    test = matches[(matches["league"] == league) & (matches["season"] == seasons()["test"])]

    model = joblib.load(MODELS / f"{slug(league)}_{kind}.joblib")
    cal = json.loads((MODELS / f"{slug(league)}_calibration.json").read_text())[kind]
    expected = apply_calibration(model.predict_proba(test[FEATURES]), cal, method)

    rows = tmp_path / "rows.json"
    rows.write_text(test[FEATURES].to_json(orient="records"), encoding="utf-8")
    key = slug(league)
    out = subprocess.run(
        [NODE, str(ROOT / "tests" / "js_predictions.mjs"), str(WEB / key / f"{kind}.json"),
         str(WEB / key / "calibration.json"), method, str(rows)],
        capture_output=True, text=True, check=True,
    )
    got = np.array(json.loads(out.stdout))
    np.testing.assert_allclose(got, expected, atol=2e-4)
