from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else root / p


def load_seat_map(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    required = {"student_id", "seat_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"student_seats.csv missing columns: {sorted(missing)}")
    return dict(zip(df["student_id"].astype(str), df["seat_id"].astype(str)))


def calculate(root: Path, cfg: dict[str, Any], out_dir: Path) -> dict:
    assignments = _resolve(root, cfg["identity_assignments_csv"])
    student_audit = _resolve(root, cfg["student_audit_csv"])
    seats_csv = _resolve(root, cfg["seats_csv"])
    if not assignments or not assignments.exists():
        raise FileNotFoundError(assignments)
    if not seats_csv or not seats_csv.exists():
        raise FileNotFoundError(seats_csv)
    seat_map = load_seat_map(seats_csv)

    total = 0
    primary_assigned = 0
    seat_match = 0
    seat_mismatch = 0
    student_missing = 0
    status_counts = {}
    track_purity_values = []

    usecols = [
        "student_id", "seat_id", "assignment_status", "track_seat_purity",
    ]
    for chunk in pd.read_csv(assignments, usecols=usecols, chunksize=100_000):
        total += len(chunk)
        statuses = chunk["assignment_status"].fillna("other_missing").astype(str)
        status_counts.update(statuses.value_counts().to_dict())
        mask = statuses == "assigned"
        primary_assigned += int(mask.sum())
        assigned = chunk.loc[mask, ["student_id", "seat_id"]].copy()
        assigned["expected_seat"] = assigned["student_id"].astype(str).map(seat_map)
        student_missing += int(assigned["expected_seat"].isna().sum())
        valid = assigned["expected_seat"].notna()
        seat_match += int((assigned.loc[valid, "seat_id"].astype(str) == assigned.loc[valid, "expected_seat"].astype(str)).sum())
        seat_mismatch += int(valid.sum() - (assigned.loc[valid, "seat_id"].astype(str) == assigned.loc[valid, "expected_seat"].astype(str)).sum())
        vals = pd.to_numeric(chunk.loc[mask, "track_seat_purity"], errors="coerce").dropna().tolist()
        track_purity_values.extend(vals)

    conditional = seat_match / primary_assigned if primary_assigned else np.nan
    coverage = primary_assigned / total if total else np.nan
    coverage_adjusted = seat_match / total if total else np.nan

    student_detail = pd.DataFrame()
    student_mean = student_median = np.nan
    frame_cov_mean = frame_cov_min = frame_cov_max = np.nan
    if student_audit and student_audit.exists():
        sa = pd.read_csv(student_audit)
        cols = [c for c in ["student_id", "seat_id", "seat_purity", "frame_coverage", "unassigned_frames", "duplicate_track_frames"] if c in sa.columns]
        student_detail = sa[cols].copy()
        student_detail.to_csv(out_dir / "seat_purity_student_detail.csv", index=False)
        if "seat_purity" in sa:
            student_mean = float(sa["seat_purity"].mean())
            student_median = float(sa["seat_purity"].median())
        if "frame_coverage" in sa:
            frame_cov_mean = float(sa["frame_coverage"].mean())
            frame_cov_min = float(sa["frame_coverage"].min())
            frame_cov_max = float(sa["frame_coverage"].max())

    report_rows = [
        {"metric": "total_assignment_rows", "value": total, "definition": "all rows in identity_assignments.csv"},
        {"metric": "primary_assigned_rows", "value": primary_assigned, "definition": "rows with assignment_status == assigned"},
        {"metric": "primary_assignment_coverage", "value": coverage, "definition": "primary_assigned_rows / all assignment rows"},
        {"metric": "conditional_seat_purity", "value": conditional, "definition": "seat-matching primary assigned rows / primary assigned rows"},
        {"metric": "coverage_adjusted_seat_consistency", "value": coverage_adjusted, "definition": "seat-matching primary assigned rows / all assignment rows"},
        {"metric": "seat_mismatch_rows_among_primary_assigned", "value": seat_mismatch, "definition": "assigned rows whose seat_id differs from expected seat for student_id"},
        {"metric": "missing_student_seat_map_rows", "value": student_missing, "definition": "assigned rows with no expected seat in student_seats.csv"},
        {"metric": "student_audit_mean_seat_purity", "value": student_mean, "definition": "mean of seat_purity column in student_identity_audit.csv"},
        {"metric": "student_audit_median_seat_purity", "value": student_median, "definition": "median of seat_purity column in student_identity_audit.csv"},
        {"metric": "student_audit_mean_frame_coverage", "value": frame_cov_mean, "definition": "mean of frame_coverage column in student_identity_audit.csv"},
        {"metric": "student_audit_min_frame_coverage", "value": frame_cov_min, "definition": "minimum frame_coverage across students"},
        {"metric": "student_audit_max_frame_coverage", "value": frame_cov_max, "definition": "maximum frame_coverage across students"},
        {"metric": "track_seat_purity_mean", "value": float(np.mean(track_purity_values)) if track_purity_values else np.nan, "definition": "mean track_seat_purity among assigned rows with a numeric value"},
        {"metric": "track_seat_purity_median", "value": float(np.median(track_purity_values)) if track_purity_values else np.nan, "definition": "median track_seat_purity among assigned rows with a numeric value"},
    ]
    pd.DataFrame(report_rows).to_csv(out_dir / "seat_purity_report.csv", index=False)
    definition = """# Seat Purity Definitions\n\n- **Conditional seat purity**: among rows that the identity layer labels `assignment_status=assigned`, the share whose `seat_id` matches the seat expected for the associated `student_id` in `student_seats.csv`.\n- **Primary assignment coverage**: share of all identity-assignment rows that are labeled `assigned`.\n- **Coverage-adjusted seat consistency**: seat-matching primary assigned rows divided by all identity-assignment rows.\n- **Student-audit mean/median seat purity**: summary statistics copied from `student_identity_audit.csv`; these are track/student-audit statistics and must not be described as ground-truth MOT accuracy.\n\nThese quantities answer different questions and should not be conflated. A conditional purity of 100% does not mean 100% of all observations were assigned.\n"""
    (out_dir / "seat_purity_definition.md").write_text(definition, encoding="utf-8")
    return {
        "total": total,
        "primary_assigned": primary_assigned,
        "primary_assignment_coverage": coverage,
        "conditional_seat_purity": conditional,
        "coverage_adjusted_seat_consistency": coverage_adjusted,
        "seat_mismatch": seat_mismatch,
        "student_mean_seat_purity": student_mean,
        "student_median_seat_purity": student_median,
    }
