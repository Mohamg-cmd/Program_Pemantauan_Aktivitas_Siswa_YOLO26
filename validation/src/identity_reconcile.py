from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

STATUS_ORDER = ["assigned", "duplicate_track", "unassigned", "ambiguous", "other_missing"]


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else root / p


def _read_csv(path: Path, usecols=None, chunksize: int | None = None):
    if chunksize:
        return pd.read_csv(path, usecols=usecols, chunksize=chunksize)
    return pd.read_csv(path, usecols=usecols)


def reconcile(root: Path, cfg: dict[str, Any], out_dir: Path) -> dict:
    assignments = _resolve(root, cfg["identity_assignments_csv"])
    student_audit = _resolve(root, cfg.get("student_audit_csv"))
    track_summary = _resolve(root, cfg.get("track_to_student_csv"))
    expected_students = int(cfg.get("expected_students", 21))
    if assignments is None or not assignments.exists():
        raise FileNotFoundError(f"identity_assignments.csv not found: {assignments}")

    row_counts = Counter()
    mapping_counts = Counter()
    rows = 0
    assigned_students = set()
    track_ids = set()
    track_status = {}
    columns = ["assignment_status", "track_mapping_status", "student_id", "track_id"]
    for chunk in _read_csv(assignments, usecols=columns, chunksize=100_000):
        rows += len(chunk)
        status = chunk["assignment_status"].fillna("other_missing").astype(str)
        status = status.where(status.isin(STATUS_ORDER), "other_missing")
        row_counts.update(status.tolist())
        mapstatus = chunk["track_mapping_status"].fillna("other_missing").astype(str)
        mapping_counts.update(mapstatus.tolist())
        assigned_students.update(chunk.loc[status == "assigned", "student_id"].dropna().astype(str).tolist())
        track_ids.update(chunk["track_id"].dropna().astype(int).tolist())
        for tid, st in zip(chunk["track_id"], mapstatus):
            if pd.isna(tid):
                continue
            track_status[int(tid)] = str(st)

    identity_rows = [{"category": k, "rows": row_counts.get(k, 0), "share": row_counts.get(k, 0) / rows if rows else 0.0} for k in STATUS_ORDER]
    pd.DataFrame(identity_rows).to_csv(out_dir / "identity_status_reconciliation.csv", index=False)
    map_rows = [{"track_mapping_status": k, "rows": v} for k, v in sorted(mapping_counts.items())]
    pd.DataFrame(map_rows).to_csv(out_dir / "identity_track_level_reconciliation.csv", index=False)

    conflicts = []
    legacy = cfg.get("legacy_claims") or {}
    checks = [
        ("duplicate_track_rows", row_counts.get("duplicate_track", 0), legacy.get("duplicate_track_rows")),
        ("unassigned_rows", row_counts.get("unassigned", 0), legacy.get("unassigned_rows")),
    ]
    for metric, actual, claimed in checks:
        if claimed is not None and actual != int(claimed):
            conflicts.append({"metric": metric, "actual": actual, "legacy_claim": int(claimed), "status": "CONFLICT"})
        elif claimed is not None:
            conflicts.append({"metric": metric, "actual": actual, "legacy_claim": int(claimed), "status": "MATCH"})
    pd.DataFrame(conflicts).to_csv(out_dir / "identity_status_conflicts.csv", index=False)

    student_audit_info = {}
    if student_audit and student_audit.exists():
        sa = pd.read_csv(student_audit)
        student_audit_info = {
            "rows": len(sa),
            "unique_students": int(sa["student_id"].nunique()) if "student_id" in sa else None,
            "seat_purity_mean": float(sa["seat_purity"].mean()) if "seat_purity" in sa else None,
            "seat_purity_median": float(sa["seat_purity"].median()) if "seat_purity" in sa else None,
            "frame_coverage_mean": float(sa["frame_coverage"].mean()) if "frame_coverage" in sa else None,
            "frame_coverage_min": float(sa["frame_coverage"].min()) if "frame_coverage" in sa else None,
            "frame_coverage_max": float(sa["frame_coverage"].max()) if "frame_coverage" in sa else None,
        }

    track_info = {}
    if track_summary and track_summary.exists():
        ts = pd.read_csv(track_summary)
        if "track_id" in ts:
            track_info["rows"] = len(ts)
            track_info["unique_track_ids"] = int(ts["track_id"].nunique())
        if "mapping_status" in ts:
            track_info["mapping_status_counts"] = ts["mapping_status"].value_counts(dropna=False).to_dict()

    report = {
        "total_assignment_rows": rows,
        "row_status_counts": dict(row_counts),
        "track_mapping_status_counts": dict(mapping_counts),
        "unique_assigned_students": len(assigned_students),
        "expected_students": expected_students,
        "expected_students_match": len(assigned_students) == expected_students,
        "unique_track_ids": len(track_ids),
        "student_audit": student_audit_info,
        "track_summary": track_info,
        "legacy_conflicts": conflicts,
        "scientific_status": "reconciled_full_file_audit",
    }
    (out_dir / "identity_reconciliation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
