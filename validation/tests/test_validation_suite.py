from pathlib import Path
import importlib.util
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_local(name: str, filename: str):
    path = ROOT / "src" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


identity_reconcile = _load_local("validation_identity_reconcile", "identity_reconcile.py")
seat_purity = _load_local("validation_seat_purity", "seat_purity.py")


def test_identity_and_purity():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        out = root / "out"
        out.mkdir()
        pd.DataFrame({
            "track_id": [1, 1, 2, 3],
            "student_id": ["S01", "S01", None, "S02"],
            "seat_id": ["seat_01", "seat_01", None, "seat_02"],
            "assignment_status": ["assigned", "duplicate_track", "unassigned", "assigned"],
            "track_mapping_status": ["assigned", "assigned", "unassigned", "assigned"],
            "track_seat_purity": [1.0, 1.0, float("nan"), 1.0],
        }).to_csv(root / "identity.csv", index=False)
        pd.DataFrame({
            "student_id": ["S01", "S02"], "seat_id": ["seat_01", "seat_02"],
            "seat_purity": [1.0, 1.0], "frame_coverage": [0.9, 0.95]
        }).to_csv(root / "student_audit.csv", index=False)
        pd.DataFrame({"student_id": ["S01", "S02"], "seat_id": ["seat_01", "seat_02"]}).to_csv(root / "seats.csv", index=False)
        rep = identity_reconcile.reconcile(root, {
            "identity_assignments_csv": "identity.csv",
            "student_audit_csv": "student_audit.csv",
            "track_to_student_csv": None,
            "expected_students": 2,
            "legacy_claims": {}
        }, out)
        assert rep["total_assignment_rows"] == 4
        assert rep["row_status_counts"]["duplicate_track"] == 1
        assert rep["row_status_counts"]["unassigned"] == 1
        assert rep["unique_assigned_students"] == 2
        pur = seat_purity.calculate(root, {
            "identity_assignments_csv": "identity.csv",
            "student_audit_csv": "student_audit.csv",
            "seats_csv": "seats.csv",
        }, out)
        assert abs(pur["conditional_seat_purity"] - 1.0) < 1e-9
        assert abs(pur["primary_assignment_coverage"] - 0.5) < 1e-9
