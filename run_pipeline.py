"""run_pipeline.py — student-level YOLO26-M classroom research pipeline.

End-to-end structure:
    YOLO26-M
        -> BoT-SORT + ReID (optional re-run from original video)
        -> trajectory audit
        -> fixed-seat student identity
        -> merge track fragments at student level
        -> temporal activity profiles
        -> Fuzzy AHP (Buckley) weighting
        -> Weighted Product ranking
        -> OAT + normalization + Monte Carlo sensitivity
        -> reproducibility Excel report

The large source video is NOT required for downstream analysis when a valid
trajectory_botsort.jsonl is already available.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analytics import (
    CRITERIA, ORIENTATION, activity_matrix, activity_transitions,
    ablation_summary, error_propagation, environment_provenance,
    fuzzy_ahp_weights, load_trajectory_jsonl, monte_carlo_wp,
    normalization_sensitivity, sensitivity_oat, summarize_tracks,
    temporal_comparison, wp_score, evaluate_detector, write_excel_report,
    mot_status,
)
from src.detect_track import run_detection_tracking
from src.identity import (
    IdentityConfig, IdentityConfigError, assign_track_to_seat,
    build_student_level_detections, format_identity_audit, fragment_warnings,
    load_seat_map, long_gap_audit, merge_track_fragments,
    validate_seat_assignment,
)


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else root / p


def _cfg(cfg: dict, dotted: str, default=None):
    cur = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur if cur is not None else default


def main(config_path: str, args: argparse.Namespace) -> int:
    cfg_file = Path(config_path).resolve()
    root = cfg_file.parent
    with cfg_file.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = _resolve(root, _cfg(cfg, "output_dir", "output"))
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    criteria = list(_cfg(cfg, "criteria", CRITERIA))
    orientation = dict(_cfg(cfg, "orientation", ORIENTATION))
    fps = float(_cfg(cfg, "temporal.fps", 25.0))
    min_track_frames = int(_cfg(cfg, "temporal.min_track_frames", 15))
    strict_classes = bool(_cfg(cfg, "temporal.strict_classes", True))
    representation = str(_cfg(cfg, "temporal.representation", "duration"))
    expected_students = int(_cfg(cfg, "identity.expected_students", 21))

    if fps <= 0:
        raise ValueError("temporal.fps harus > 0")
    if criteria != CRITERIA:
        print(f"WARNING: criteria config = {criteria}; expected = {CRITERIA}")

    # ------------------------------------------------------------
    # Stage 1: source detections / trajectories
    # ------------------------------------------------------------
    trajectory_path = _resolve(root, _cfg(cfg, "source.trajectory_jsonl"))
    detections_path = _resolve(root, _cfg(cfg, "source.detections_csv"))
    source_model = _resolve(root, _cfg(cfg, "source.model_path"))
    video_path = _resolve(root, _cfg(cfg, "source.video_path"))

    if args.track_level:
        print("DEBUG MODE: ranking will be performed at track level only.")

    if args.force_track:
        if source_model is None or video_path is None or trajectory_path is None:
            raise ValueError("force-track memerlukan source.model_path, source.video_path, dan source.trajectory_jsonl")
        print("[1] Re-running YOLO26-M + BoT-SORT/ReID from the original video...")
        tracker_cfg = _resolve(root, _cfg(cfg, "tracking.tracker", "botsort_reid.yaml"))
        detections = run_detection_tracking(
            model_path=str(source_model),
            video_path=str(video_path),
            trajectory_jsonl=str(trajectory_path),
            tracker=str(tracker_cfg),
            conf=float(_cfg(cfg, "detection.conf", 0.25)),
            iou=float(_cfg(cfg, "detection.iou", 0.50)),
            imgsz=int(_cfg(cfg, "detection.imgsz", 640)),
            device=str(_cfg(cfg, "detection.device", "0")),
            with_reid=bool(_cfg(cfg, "tracking.with_reid", True)),
            output_dir=str(out_dir),
            reid_model=_cfg(cfg, "tracking.reid_model"),
            save_video=bool(_cfg(cfg, "detection.save_video", False)),
        )
        if detections_path is not None:
            detections.to_csv(detections_path, index=False)
    elif detections_path is not None and detections_path.exists() and not args.prefer_trajectory:
        print(f"[1] Using existing detections.csv: {detections_path}")
        detections = pd.read_csv(detections_path)
    elif trajectory_path is not None and trajectory_path.exists():
        print(f"[1] Loading existing trajectory JSONL: {trajectory_path}")
        detections = load_trajectory_jsonl(str(trajectory_path))
        if detections_path is not None:
            detections.to_csv(detections_path, index=False)
    else:
        raise FileNotFoundError(
            "Tidak ada sumber deteksi. Letakkan trajectory_botsort.jsonl atau "
            "detections.csv di lokasi yang ditentukan config; video+model hanya "
            "diperlukan bila ingin re-run tracking."
        )

    detections["frame"] = pd.to_numeric(detections["frame"], errors="raise").astype(int)
    detections["track_id"] = pd.to_numeric(detections["track_id"], errors="raise").astype(int)
    detections["class_name"] = detections["class_name"].astype(str)
    if strict_classes:
        from src.analytics import validate_classes
        validate_classes(detections, strict=True)

    total_video_frames = int(detections["frame"].max()) + 1
    track_summary = summarize_tracks(detections, fps)
    track_summary.to_csv(out_dir / "track_summary.csv", index=False)
    raw_track_ids = int(detections["track_id"].nunique())
    print(f"    frames={total_video_frames:,}; raw detections={len(detections):,}; raw track IDs={raw_track_ids}")
    print(f"    track IDs are trajectories, NOT student identities.")

    # ------------------------------------------------------------
    # Stage 2: fixed-seat identity
    # ------------------------------------------------------------
    identity_assignments = pd.DataFrame()
    track_to_student = pd.DataFrame()
    student_audit = pd.DataFrame()
    student_detections = pd.DataFrame()
    identity_report = {"passed": False}

    mode = "track_level" if args.track_level else str(_cfg(cfg, "identity.mode", "student_level"))
    if mode == "student_level":
        seats_csv = _resolve(root, _cfg(cfg, "identity.seats_csv"))
        if seats_csv is None:
            raise IdentityConfigError("identity.seats_csv wajib diisi untuk student_level")
        id_raw = dict(_cfg(cfg, "identity", {}) or {})
        id_raw.pop("seats_csv", None)
        id_raw.pop("mode", None)
        icfg = IdentityConfig.from_dict({**id_raw, "mode": "student_level"})
        seats = load_seat_map(str(seats_csv), expected_students)

        print("[2] Fixed-seat identity association...")
        identity_assignments = assign_track_to_seat(detections, seats, icfg)
        identity_assignments.to_csv(out_dir / "identity_assignments.csv", index=False)

        track_to_student, student_audit = merge_track_fragments(
            identity_assignments, fps, icfg,
            total_video_frames=total_video_frames, seats=seats,
        )
        track_to_student.to_csv(out_dir / "track_to_student_summary.csv", index=False)
        student_audit.to_csv(out_dir / "student_identity_audit.csv", index=False)
        long_gap = long_gap_audit(track_to_student, icfg)
        long_gap.to_csv(out_dir / "long_gap_audit.csv", index=False)
        identity_report = validate_seat_assignment(
            identity_assignments, seats, icfg, total_video_frames=total_video_frames,
        )
        print(format_identity_audit(identity_report))
        for msg in fragment_warnings(track_to_student, icfg)[:25]:
            print(msg)

        if args.identity_audit_only:
            print(f"\nAudit outputs written under: {out_dir}")
            return 0 if identity_report["passed"] else 1

        # CRITICAL: student-level ranking is blocked if fixed-seat validation fails.
        if not identity_report["passed"]:
            print("\nStudent-level ranking BLOCKED because identity validation failed.", file=sys.stderr)
            print("Inspect student_identity_audit.csv and long_gap_audit.csv.", file=sys.stderr)
            return 1
        student_detections = build_student_level_detections(identity_assignments)
        student_ids = [f"S{i:02d}" for i in range(1, expected_students + 1)]
    else:
        # Explicit debug mode only. Never silently fall back here.
        print("[2] Track-level debug mode: fixed-seat identity is bypassed by explicit user flag.")
        student_detections = detections.copy()
        student_detections["student_id"] = "track_" + student_detections["track_id"].astype(str)
        student_detections["seat_id"] = None
        student_ids = sorted(student_detections["student_id"].unique().tolist())

    # ------------------------------------------------------------
    # Stage 3: temporal student profiles
    # ------------------------------------------------------------
    print("[3] Temporal aggregation...")
    duration = activity_matrix(student_detections, fps, "duration", all_students=student_ids)
    frequency = activity_matrix(student_detections, fps, "frequency", all_students=student_ids)
    duration.to_csv(out_dir / "student_activity_matrix_duration.csv")
    frequency.to_csv(out_dir / "student_activity_matrix_frequency.csv")
    transitions = activity_transitions(student_detections)
    transitions.to_csv(out_dir / "student_activity_transitions.csv", index=False)
    matrix = duration if representation == "duration" else frequency

    # ------------------------------------------------------------
    # Stage 4: Fuzzy AHP
    # ------------------------------------------------------------
    print("[4] Fuzzy AHP (Buckley)...")
    ahp = fuzzy_ahp_weights(_cfg(cfg, "ahp.matrix"), criteria)
    ahp["source"] = _cfg(cfg, "ahp.source", "predefined_model")
    ahp["expert_count"] = int(_cfg(cfg, "ahp.expert_count", 0))
    print("    weights:", ", ".join(f"{k}={v:.6f}" for k, v in ahp["weights"].items()))
    print(f"    lambda_max={ahp['lambda_max']:.6f}; CI={ahp['CI']:.6f}; RI={ahp['RI']:.2f}; CR={ahp['CR']:.6f}")
    if ahp["source"] == "researcher_constructed_supervisor_reviewed" and ahp["expert_count"] == 0:
        print("    NOTE: pairwise matrix is researcher-constructed and supervisor-reviewed; not multi-expert elicitation.")
    if not ahp["consistent"] and _cfg(cfg, "ahp.block_on_inconsistent", True) and not args.force_inconsistent:
        print("CR > 0.10. Final ranking blocked.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------
    # Stage 5: Weighted Product
    # ------------------------------------------------------------
    print("[5] Weighted Product ranking...")
    norm_method = str(_cfg(cfg, "wp.normalization", "minmax_floor"))
    floor = float(_cfg(cfg, "wp.floor", 0.01))
    zero_eps = float(_cfg(cfg, "wp.zero_epsilon", 1e-6))
    ranking, zero_df = wp_score(matrix, ahp["weights"], orientation, norm_method, floor, zero_eps)
    ranking.to_csv(out_dir / "ranking_wp_final.csv")
    # Keep a compact student-only ranking table for the manuscript.
    compact_cols = [c for c in criteria if c in ranking.columns] + ["S", "V", "rank"]
    ranking[compact_cols].to_csv(out_dir / "student_wp_ranking.csv")
    zero_df.to_csv(out_dir / "zero_handling.csv", index=False)
    print(ranking[["V", "rank"]].head(21).round(6).to_string())

    # ------------------------------------------------------------
    # Stage 6: robustness
    # ------------------------------------------------------------
    perturb = _cfg(cfg, "sensitivity.perturbations", [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20])
    sens = sensitivity_oat(matrix, ahp["weights"], orientation, perturb, norm_method, floor, zero_eps)
    sens.to_csv(out_dir / "sensitivity_oat.csv", index=False)
    print("[6] OAT sensitivity written.")

    norm_sens = normalization_sensitivity(
        matrix, ahp["weights"], orientation,
        _cfg(cfg, "sensitivity.normalization_methods", ["minmax_floor", "ratio_floor"]),
        floor, zero_eps,
    )
    norm_sens.to_csv(out_dir / "sensitivity_normalization.csv", index=False)

    mc = pd.DataFrame()
    if args.monte_carlo or bool(_cfg(cfg, "monte_carlo.enabled", True)):
        mc = monte_carlo_wp(
            matrix, ahp["weights"], orientation,
            int(_cfg(cfg, "monte_carlo.iterations", 5000)),
            int(_cfg(cfg, "monte_carlo.seed", 42)),
            float(_cfg(cfg, "monte_carlo.concentration", 50.0)),
            norm_method, floor, zero_eps,
        )
        mc.to_csv(out_dir / "sensitivity_montecarlo.csv", index=False)
        print("[7] Monte Carlo sensitivity written.")

    temporal_cmp = temporal_comparison(
        duration, frequency, ahp["weights"], orientation, norm_method, floor, zero_eps
    ) if bool(_cfg(cfg, "experiments.temporal_comparison", True)) else pd.DataFrame()
    temporal_cmp.to_csv(out_dir / "temporal_comparison.csv", index=False)

    ablation = ablation_summary(
        detections, student_detections, identity_report, min_track_frames
    ) if bool(_cfg(cfg, "experiments.ablation", True)) else pd.DataFrame()
    ablation.to_csv(out_dir / "ablation_results.csv", index=False)

    err_df = error_propagation(student_audit, matrix, ranking) if mode == "student_level" else pd.DataFrame()
    err_df.to_csv(out_dir / "error_propagation.csv", index=False)

    detector_metrics = None
    if args.detector_eval or bool(_cfg(cfg, "experiments.detector_evaluation", False)):
        data_yaml = _resolve(root, _cfg(cfg, "source.data_yaml"))
        detector_metrics, _ = evaluate_detector(
            str(source_model) if source_model else "",
            str(data_yaml) if data_yaml else "",
            str(_cfg(cfg, "detector_evaluation.split", "test")),
            int(_cfg(cfg, "detection.imgsz", 640)),
            str(_cfg(cfg, "detection.device", "0")),
            int(_cfg(cfg, "detector_evaluation.batch", 16)),
        )
        detector_metrics.to_csv(out_dir / "detection_metrics.csv", index=False)

    # ------------------------------------------------------------
    # Stage 7: reproducibility workbook
    # ------------------------------------------------------------
    provenance = environment_provenance(cfg, {
        "n_frames": total_video_frames,
        "n_raw_detections": len(detections),
        "n_raw_track_ids": raw_track_ids,
        "n_ranked_entities": len(matrix),
        "identity_validation_passed": identity_report.get("passed", "NOT_APPLICABLE"),
        "ranking_entity_level": mode,
    })
    mot_df = mot_status(_cfg(cfg, "mot_evaluation.ground_truth_csv"))
    report_path = out_dir / "yolo26_classroom_research_report.xlsx"
    write_excel_report(
        str(report_path), cfg, provenance, track_summary,
        identity_assignments, track_to_student, student_audit,
        duration, frequency, transitions, ahp, ranking,
        ranking[criteria], zero_df, sens, norm_sens, mc,
        temporal_cmp, ablation, err_df, detector_metrics, mot_df,
    )
    print(f"\nOK — report: {report_path}")
    print(f"OK — outputs: {out_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO26 student-level classroom analytics")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force-track", action="store_true", help="Re-run YOLO26-M + BoT-SORT/ReID from original video")
    parser.add_argument("--prefer-trajectory", action="store_true", help="Prefer trajectory JSONL over an existing detections.csv")
    parser.add_argument("--identity-audit-only", action="store_true")
    parser.add_argument("--track-level", action="store_true", help="DEBUG ONLY: aggregate raw track IDs instead of students")
    parser.add_argument("--monte-carlo", action="store_true", help="Force Monte Carlo sensitivity")
    parser.add_argument("--detector-eval", action="store_true", help="Run YOLO model.val on the configured test split")
    parser.add_argument("--force-inconsistent", action="store_true", help="Continue despite CR > 0.10")
    args = parser.parse_args()
    try:
        sys.exit(main(args.config, args))
    except (IdentityConfigError, FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(2)
