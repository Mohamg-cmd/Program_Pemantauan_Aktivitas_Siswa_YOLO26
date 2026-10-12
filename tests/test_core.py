import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analytics import fuzzy_ahp_weights, wp_score, compare_rankings  # noqa: E402
from src.identity import IdentityConfig, load_seat_map  # noqa: E402


def test_fuzzy_ahp_weights_and_consistency():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    r = fuzzy_ahp_weights(cfg["ahp"]["matrix"], cfg["criteria"])
    assert r["consistent"]
    assert math.isclose(r["weights"]["Angkat Tangan"], 0.4284662962, rel_tol=0, abs_tol=1e-5)
    assert math.isclose(r["CR"], 0.0263792837, rel_tol=0, abs_tol=1e-5)
    assert math.isclose(sum(r["weights"].values()), 1.0, abs_tol=1e-9)


def test_seat_map_is_21_and_unique():
    seats = load_seat_map(str(ROOT / "data" / "student_seats.csv"), expected_students=21)
    assert len(seats) == 21
    assert seats["student_id"].is_unique
    assert seats["seat_id"].is_unique


def test_wp_returns_21_ranked_entities():
    idx = [f"S{i:02d}" for i in range(1, 22)]
    X = pd.DataFrame({
        "Angkat Tangan": np.arange(21),
        "Berdiri": np.arange(21)[::-1],
        "Diam": np.ones(21) * 5,
        "Tidur": np.arange(21) + 1,
    }, index=idx)
    w = {"Angkat Tangan": 0.4284662962, "Berdiri": 0.2897828791,
         "Diam": 0.1708868636, "Tidur": 0.1108639611}
    r, _ = wp_score(X, w, {"Angkat Tangan": "benefit", "Berdiri": "benefit", "Diam": "cost", "Tidur": "cost"})
    assert len(r) == 21
    assert set(r["rank"]) == set(range(1, 22))


def test_rank_metric_identity():
    s = pd.Series([1, 2, 3], index=["a", "b", "c"])
    m = compare_rankings(s, s)
    assert m["mean_rank_shift"] == 0.0
    assert m["n_rank_changes"] == 0
