"""
Fixed-seat student identity resolution for classroom video.

Core rule
---------
BoT-SORT ``track_id`` is a trajectory identifier, NOT a student identity.
Because the classroom recording has 21 fixed seats and students do not change
seats, identity is resolved at TRACK level from spatial evidence and then
propagated to every observation belonging to that track fragment.

This is deliberately different from a strict per-frame ROI gate. A valid
track may temporarily move its bounding-box anchor outside a seat ROI when a
student stands, raises a hand, leans, or is partially occluded. Once a track's
seat is established with sufficient evidence, those temporary excursions do
not erase the identity.

Primary cue:
    fixed-seat spatial consistency over the whole track.
Secondary cue:
    temporal continuity within the track.
Optional cue:
    externally supplied ReID similarity.

Conflict rule:
    at most one primary track may occupy a student/seat in a single frame;
    other simultaneous claims are flagged ``duplicate_track`` and excluded
    from activity aggregation, never silently counted twice.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

SEAT_REQUIRED_COLUMNS = [
    "student_id", "seat_id", "seat_x", "seat_y",
    "roi_x1", "roi_y1", "roi_x2", "roi_y2",
]

STATUS_ASSIGNED = "assigned"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_UNASSIGNED = "unassigned"
STATUS_DUPLICATE = "duplicate_track"
VALID_STATUSES = {STATUS_ASSIGNED, STATUS_AMBIGUOUS, STATUS_UNASSIGNED, STATUS_DUPLICATE}


class IdentityConfigError(ValueError):
    pass


class IdentityValidationError(RuntimeError):
    pass


@dataclass
class IdentityConfig:
    mode: str = "student_level"
    seats_csv: Optional[str] = None
    assignment_method: str = "fixed_seat"
    expected_students: int = 21

    # Geometry / track-level identity
    max_seat_distance: float = 150.0
    roi_bonus: float = 0.25
    min_track_seat_score: float = 0.45
    min_track_seat_purity: float = 0.60
    min_track_positive_fraction: float = 0.25
    track_ambiguity_margin: float = 0.08
    short_track_frames: int = 10
    short_track_min_seat_score: float = 0.65

    # Kept for compatibility; no longer used as a hard per-frame identity gate.
    min_identity_confidence: float = 0.35
    temporal_smoothing_window: int = 15
    hysteresis_switch_margin: float = 0.20
    reassociation_gap_frames: int = 90

    # Cue weights are used for track-quality scoring.
    allow_reid_secondary: bool = True
    weight_spatial: float = 0.70
    weight_temporal: float = 0.20
    weight_reid: float = 0.10

    # Validation / quality gates
    max_ambiguous_ratio: float = 0.05
    max_duplicate_ratio: float = 0.30
    min_student_frame_coverage: float = 0.50
    min_mean_student_frame_coverage: float = 0.70
    min_seat_purity: float = 0.70
    max_long_gap_ratio: float = 0.25

    # Long-gap links are diagnostics, not automatic identity failures. A link
    # is accepted as strong fixed-seat evidence only when both adjacent track
    # fragments have strong seat evidence. Otherwise it remains flagged.
    long_gap_auto_verify_purity: float = 0.90
    long_gap_auto_verify_score: float = 0.70

    def __post_init__(self) -> None:
        if self.mode not in {"student_level", "track_level"}:
            raise IdentityConfigError("identity.mode harus 'student_level' atau 'track_level'")
        if self.assignment_method != "fixed_seat":
            raise IdentityConfigError("assignment_method yang didukung hanya 'fixed_seat'")
        if self.max_seat_distance <= 0:
            raise IdentityConfigError("max_seat_distance harus > 0")
        for name, val in [
            ("min_track_seat_score", self.min_track_seat_score),
            ("min_track_seat_purity", self.min_track_seat_purity),
            ("min_track_positive_fraction", self.min_track_positive_fraction),
            ("track_ambiguity_margin", self.track_ambiguity_margin),
            ("min_identity_confidence", self.min_identity_confidence),
            ("max_ambiguous_ratio", self.max_ambiguous_ratio),
            ("max_duplicate_ratio", self.max_duplicate_ratio),
            ("min_student_frame_coverage", self.min_student_frame_coverage),
            ("min_mean_student_frame_coverage", self.min_mean_student_frame_coverage),
            ("min_seat_purity", self.min_seat_purity),
            ("max_long_gap_ratio", self.max_long_gap_ratio),
            ("long_gap_auto_verify_purity", self.long_gap_auto_verify_purity),
            ("long_gap_auto_verify_score", self.long_gap_auto_verify_score),
        ]:
            if not (0.0 <= float(val) <= 1.0):
                raise IdentityConfigError(f"{name} harus di [0,1]")
        if self.short_track_frames < 1:
            raise IdentityConfigError("short_track_frames harus >= 1")
        if self.temporal_smoothing_window < 1:
            raise IdentityConfigError("temporal_smoothing_window harus >= 1")
        if self.temporal_smoothing_window % 2 == 0:
            self.temporal_smoothing_window += 1
        if self.weight_spatial <= 0:
            raise IdentityConfigError("weight_spatial harus > 0")
        if min(self.weight_spatial, self.weight_temporal, self.weight_reid) < 0:
            raise IdentityConfigError("bobot cue tidak boleh negatif")
        if self.expected_students < 1:
            raise IdentityConfigError("expected_students harus >= 1")

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "IdentityConfig":
        d = dict(d or {})
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise IdentityConfigError(
                f"Kunci config identity tidak dikenal: {sorted(unknown)}. "
                f"Kunci valid: {sorted(known)}"
            )
        return cls(**d)

    def active_weights(self, reid_available: bool) -> tuple[float, float, float]:
        ws, wt = float(self.weight_spatial), float(self.weight_temporal)
        wr = float(self.weight_reid) if (reid_available and self.allow_reid_secondary) else 0.0
        total = ws + wt + wr
        if total <= 0:
            raise IdentityConfigError("total cue weight harus > 0")
        return ws / total, wt / total, wr / total


def load_seat_map(seats_csv: str, expected_students: Optional[int] = 21) -> pd.DataFrame:
    path = Path(seats_csv)
    if not path.exists():
        raise IdentityConfigError(
            f"Student-level mode requires {seats_csv}. File tidak ditemukan."
        )
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise IdentityConfigError(f"Gagal membaca {seats_csv}: {exc}") from exc

    missing = [c for c in SEAT_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise IdentityConfigError(f"{seats_csv} kehilangan kolom wajib: {missing}")
    if df.empty:
        raise IdentityConfigError(f"{seats_csv} kosong")

    df = df.copy()
    df["student_id"] = df["student_id"].astype(str).str.strip()
    df["seat_id"] = df["seat_id"].astype(str).str.strip()
    num_cols = ["seat_x", "seat_y", "roi_x1", "roi_y1", "roi_x2", "roi_y2"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[num_cols].isna().any(axis=1).any():
        bad_ids = df.loc[df[num_cols].isna().any(axis=1), "student_id"].tolist()
        raise IdentityConfigError(f"Koordinat non-numerik/kosong pada: {bad_ids}")

    dup_students = df.loc[df["student_id"].duplicated(keep=False), "student_id"].unique().tolist()
    if dup_students:
        raise IdentityConfigError(f"Duplicate student_id: {sorted(dup_students)}")
    dup_seats = df.loc[df["seat_id"].duplicated(keep=False), "seat_id"].unique().tolist()
    if dup_seats:
        raise IdentityConfigError(f"Duplicate seat_id: {sorted(dup_seats)}")

    bad_roi = df[(df["roi_x2"] <= df["roi_x1"]) | (df["roi_y2"] <= df["roi_y1"])]
    if not bad_roi.empty:
        raise IdentityConfigError(
            f"ROI tidak valid pada student_id: {bad_roi['student_id'].tolist()}"
        )
    if expected_students is not None and len(df) != int(expected_students):
        raise IdentityConfigError(
            f"Expected {expected_students} student identities but found {len(df)}"
        )
    return df.sort_values("student_id").reset_index(drop=True)


def bottom_center(x1, y1, x2, y2):
    return (np.asarray(x1, dtype=float) + np.asarray(x2, dtype=float)) / 2.0, np.asarray(y2, dtype=float)


def _spatial_scores(ax, ay, seats, max_seat_distance, roi_bonus):
    sx = seats["seat_x"].to_numpy(float)[None, :]
    sy = seats["seat_y"].to_numpy(float)[None, :]
    x1 = seats["roi_x1"].to_numpy(float)[None, :]
    y1 = seats["roi_y1"].to_numpy(float)[None, :]
    x2 = seats["roi_x2"].to_numpy(float)[None, :]
    y2 = seats["roi_y2"].to_numpy(float)[None, :]
    ax = np.asarray(ax, float)[:, None]
    ay = np.asarray(ay, float)[:, None]
    dist = np.hypot(ax - sx, ay - sy)
    score = np.clip(1.0 - dist / float(max_seat_distance), 0.0, 1.0)
    inside = (ax >= x1) & (ax <= x2) & (ay >= y1) & (ay <= y2)
    return np.clip(score + inside.astype(float) * float(roi_bonus), 0.0, 1.0)


def _track_quality_from_scores(score_matrix, seats, track_values, cfg: IdentityConfig):
    """Return per-seat aggregate evidence for one track."""
    if score_matrix.size == 0:
        return None
    positive = score_matrix > 0
    best_idx = np.argmax(score_matrix, axis=1)
    best_score = score_matrix[np.arange(len(score_matrix)), best_idx]
    valid_best = best_score > 0
    n_total = len(score_matrix)
    n_valid = int(valid_best.sum())
    if n_valid == 0:
        return None

    mean_scores = score_matrix[valid_best].mean(axis=0)
    median_scores = np.median(score_matrix[valid_best], axis=0)
    inside_fraction = np.zeros(score_matrix.shape[1], dtype=float)
    # Infer ROI containment again from score > raw distance score is not possible;
    # caller supplies it via track_values if desired. Use valid support as robust
    # baseline and track-level dominance as primary purity metric.
    support_fraction = (score_matrix > 0).mean(axis=0)
    best_counts = np.bincount(best_idx[valid_best], minlength=score_matrix.shape[1]).astype(float)
    purity = best_counts / max(n_valid, 1)

    # A seat with stronger mean and median evidence plus dominant majority wins.
    aggregate = 0.45 * purity + 0.35 * mean_scores + 0.20 * median_scores
    order = np.argsort(aggregate)[::-1]
    best = int(order[0])
    second = float(aggregate[order[1]]) if len(order) > 1 else 0.0
    margin = float(aggregate[best] - second)

    return {
        "aggregate": aggregate,
        "best_idx": best,
        "best_score": float(aggregate[best]),
        "second_score": second,
        "margin": margin,
        "purity": float(purity[best]),
        "positive_fraction": float((best_score > 0).mean()),
        "mean_score": float(mean_scores[best]),
        "median_score": float(median_scores[best]),
        "support_fraction": float(support_fraction[best]),
        "n_frames": int(n_total),
        "n_valid": n_valid,
    }


def _track_reid_scores(track_id, student_ids, reid_similarity):
    if reid_similarity is None or reid_similarity.empty:
        return None
    cols = {"track_id", "student_id", "similarity"}
    if not cols.issubset(reid_similarity.columns):
        return None
    rs = reid_similarity
    out = {}
    for sid in student_ids:
        q = rs[(rs["track_id"].astype(int) == int(track_id)) & (rs["student_id"].astype(str) == str(sid))]
        if not q.empty:
            out[str(sid)] = float(pd.to_numeric(q["similarity"], errors="coerce").max())
    return out or None


def assign_track_to_seat(
    detections: pd.DataFrame,
    seats: pd.DataFrame,
    config: IdentityConfig,
    reid_similarity: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Resolve identity at TRACK level, then propagate it to all observations.

    Unlike the old implementation, a frame does NOT become UNASSIGNED merely
    because its bottom-center temporarily leaves the ROI/distance gate.
    A track receives one dominant seat from its whole trajectory. Its identity
    is then inherited by each observation in that track fragment.
    """
    if detections is None or detections.empty:
        return pd.DataFrame(columns=[
            "frame", "track_id", "student_id", "seat_id", "identity_confidence",
            "identity_source", "assignment_status"
        ])
    required = {"frame", "track_id", "x1", "y1", "x2", "y2"}
    missing = required - set(detections.columns)
    if missing:
        raise IdentityConfigError(f"detections kehilangan kolom: {sorted(missing)}")

    df = detections.copy()
    df["frame"] = pd.to_numeric(df["frame"], errors="raise").astype(int)
    df["track_id"] = pd.to_numeric(df["track_id"], errors="raise").astype(int)
    ax, ay = bottom_center(df["x1"], df["y1"], df["x2"], df["y2"])
    df["anchor_x"], df["anchor_y"] = ax, ay

    # Precompute spatial scores once.
    S = _spatial_scores(ax, ay, seats, config.max_seat_distance, config.roi_bonus)
    seat_student = seats["student_id"].astype(str).to_numpy()
    seat_ids = seats["seat_id"].astype(str).to_numpy()

    ws, wt, wr = config.active_weights(
        bool(reid_similarity is not None and not reid_similarity.empty)
    )

    # Track-level seat inference. This is the key fix: identity is inferred
    # from the whole trajectory rather than from each frame independently.
    track_meta = {}
    for tid, idx in df.groupby("track_id", sort=False).groups.items():
        idx = np.asarray(idx)
        sm = _track_quality_from_scores(S[idx], seats, df.loc[idx], config)
        if sm is None:
            track_meta[int(tid)] = {
                "seat_idx": -1,
                "track_conf": 0.0,
                "track_purity": 0.0,
                "track_score": 0.0,
                "status": STATUS_UNASSIGNED,
                "source": "none",
                "n_frames": len(idx),
                "second_seat_idx": -1,
                "margin": 0.0,
            }
            continue

        best = sm["best_idx"]
        n_frames = sm["n_frames"]
        r = _track_reid_scores(int(tid), seat_student, reid_similarity)
        reid_best = float(r.get(seat_student[best], 0.0)) if r else 0.0
        spatial_component = 0.55 * sm["mean_score"] + 0.45 * sm["median_score"]
        temporal_component = sm["purity"]
        # For track-level assignment, use spatial evidence + temporal purity.
        track_conf = ws * spatial_component + wt * temporal_component + wr * reid_best
        # ReID is a secondary tiebreaker, never sufficient by itself.
        short_track = n_frames < config.short_track_frames
        valid = (
            spatial_component >= (config.short_track_min_seat_score if short_track else config.min_track_seat_score)
            and sm["purity"] >= config.min_track_seat_purity
            and sm["positive_fraction"] >= config.min_track_positive_fraction
        )
        ambiguous = (
            len(sm["aggregate"]) > 1
            and sm["margin"] < config.track_ambiguity_margin
            and sm["purity"] < min(0.95, config.min_track_seat_purity + 0.10)
        )

        # Strong fixed-seat evidence wins over a long temporal gap. A fragment
        # can be linked to the same seat even after many seconds because seats
        # are physically fixed in this experiment.
        status = STATUS_ASSIGNED if valid and not ambiguous else (
            STATUS_AMBIGUOUS if valid else STATUS_UNASSIGNED
        )
        best_order = np.argsort(sm["aggregate"])[::-1]
        second_idx = int(best_order[1]) if len(best_order) > 1 else -1
        track_meta[int(tid)] = {
            "seat_idx": int(best),
            "track_conf": float(track_conf),
            "track_purity": float(sm["purity"]),
            "track_score": float(spatial_component),
            "status": status,
            "source": "fixed_seat_track+temporal" + ("+reid" if r else ""),
            "n_frames": int(n_frames),
            "second_seat_idx": second_idx,
            "margin": float(sm["margin"]),
        }

    # Propagate each track identity to every row. Do NOT apply frame-level hard
    # spatial gating here; that was the source of the massive unassigned rate.
    out = df.copy()
    out["student_id"] = None
    out["seat_id"] = None
    out["identity_confidence"] = 0.0
    out["identity_source"] = "none"
    out["assignment_status"] = STATUS_UNASSIGNED
    out["track_identity_confidence"] = 0.0
    out["track_seat_purity"] = 0.0
    out["track_seat_score"] = 0.0
    out["track_mapping_status"] = STATUS_UNASSIGNED
    out["track_n_frames"] = 0

    for tid, meta in track_meta.items():
        idx = out.index[out["track_id"].eq(tid)]
        if meta["seat_idx"] < 0:
            continue
        sidx = meta["seat_idx"]
        out.loc[idx, "student_id"] = seat_student[sidx]
        out.loc[idx, "seat_id"] = seat_ids[sidx]
        out.loc[idx, "identity_confidence"] = meta["track_conf"]
        out.loc[idx, "identity_source"] = meta["source"]
        out.loc[idx, "track_identity_confidence"] = meta["track_conf"]
        out.loc[idx, "track_seat_purity"] = meta["track_purity"]
        out.loc[idx, "track_seat_score"] = meta["track_score"]
        out.loc[idx, "track_mapping_status"] = meta["status"]
        out.loc[idx, "track_n_frames"] = meta["n_frames"]
        out.loc[idx, "assignment_status"] = (
            STATUS_ASSIGNED if meta["status"] == STATUS_ASSIGNED else meta["status"]
        )

    # Frame-level one-to-one conflict resolution. This only removes true
    # simultaneous claims to the same physical seat; it never invalidates a
    # trajectory merely because an individual frame is outside its ROI.
    out = _resolve_frame_conflicts(out)
    keep_first = [c for c in detections.columns]
    extra = [
        c for c in out.columns
        if c not in keep_first
    ]
    return out.sort_values(["frame", "track_id"], kind="mergesort").reset_index(drop=True)


def _resolve_frame_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    assigned = df["assignment_status"].eq(STATUS_ASSIGNED)
    if not assigned.any():
        return df
    sub = df.loc[assigned].copy()
    if sub.empty:
        return df

    # One primary observation per (frame, student). Prefer high-quality tracks,
    # then longer tracks, then per-frame detector confidence.
    sub["_priority_track"] = pd.to_numeric(sub.get("track_identity_confidence", 0.0), errors="coerce").fillna(0.0)
    sub["_priority_purity"] = pd.to_numeric(sub.get("track_seat_purity", 0.0), errors="coerce").fillna(0.0)
    sub["_priority_length"] = pd.to_numeric(sub.get("track_n_frames", 0), errors="coerce").fillna(0)
    sub["_priority_det"] = pd.to_numeric(sub.get("confidence", 0.0), errors="coerce").fillna(0.0)
    ordered = sub.sort_values(
        ["frame", "student_id", "_priority_track", "_priority_purity", "_priority_length", "_priority_det", "track_id"],
        ascending=[True, True, False, False, False, False, True],
        kind="mergesort",
    )
    winner_idx = ordered.drop_duplicates(["frame", "student_id"], keep="first").index
    losers = ordered.index.difference(winner_idx)
    df.loc[losers, "assignment_status"] = STATUS_DUPLICATE
    return df


def validate_seat_assignment(
    assignments: pd.DataFrame,
    seats: pd.DataFrame,
    config: IdentityConfig,
    total_video_frames: Optional[int] = None,
) -> dict:
    report = {
        "expected_students": int(config.expected_students),
        "n_assigned_students": 0,
        "assigned_student_ids": [],
        "n_unassigned_detections": 0,
        "n_ambiguous_detections": 0,
        "n_duplicate_detections": 0,
        "n_fragmented_students": 0,
        "n_unique_track_ids": 0,
        "n_track_fragments_merged": 0,
        "n_raw_detections": 0,
        "n_primary_assigned": 0,
        "primary_assignment_rate": 0.0,
        "duplicate_rate": 0.0,
        "unassigned_rate": 0.0,
        "mean_student_frame_coverage": 0.0,
        "min_student_frame_coverage": 0.0,
        "max_student_frame_coverage": 0.0,
        "mean_seat_purity": 0.0,
        "median_seat_purity": 0.0,
        "n_long_gap_fragments": 0,
        "n_long_gap_auto_verified": 0,
        "n_long_gap_needs_review": 0,
        "long_gap_ratio": 0.0,
        "errors": [],
        "warnings": [],
        "passed": False,
    }
    if assignments is None or assignments.empty:
        report["errors"].append("Tidak ada assignment identitas yang dihasilkan.")
        return report

    a = assignments.copy()
    total = len(a)
    status = a["assignment_status"].astype(str)
    report["n_raw_detections"] = int(total)
    report["n_primary_assigned"] = int(status.eq(STATUS_ASSIGNED).sum())
    report["n_unique_track_ids"] = int(a["track_id"].nunique())
    report["n_unassigned_detections"] = int(status.eq(STATUS_UNASSIGNED).sum())
    report["n_ambiguous_detections"] = int(status.eq(STATUS_AMBIGUOUS).sum())
    report["n_duplicate_detections"] = int(status.eq(STATUS_DUPLICATE).sum())
    report["primary_assignment_rate"] = float(status.eq(STATUS_ASSIGNED).sum() / total)
    report["duplicate_rate"] = float(status.eq(STATUS_DUPLICATE).sum() / total)
    report["unassigned_rate"] = float(status.eq(STATUS_UNASSIGNED).sum() / total)

    ok = a.loc[status.eq(STATUS_ASSIGNED)].copy()
    students = sorted(ok["student_id"].dropna().astype(str).unique().tolist())
    report["n_assigned_students"] = len(students)
    report["assigned_student_ids"] = students
    known = set(seats["student_id"].astype(str))
    unknown = sorted(set(students) - known)
    if unknown:
        report["errors"].append(f"Student ID di luar peta kursi: {unknown}")

    if not ok.empty:
        frags = ok.groupby("student_id")["track_id"].nunique()
        report["n_fragmented_students"] = int((frags > 1).sum())
        report["n_track_fragments_merged"] = int((frags - 1).clip(lower=0).sum())

        purity_series = pd.to_numeric(ok.get("track_seat_purity", 0.0), errors="coerce").dropna()
        if not purity_series.empty:
            report["mean_seat_purity"] = float(purity_series.mean())
            report["median_seat_purity"] = float(purity_series.median())

    total_frames = int(total_video_frames) if total_video_frames else int(a["frame"].max()) + 1
    total_frames = max(total_frames, 1)
    student_coverages = []
    for sid in [str(x) for x in seats["student_id"]]:
        g = a[a["student_id"].astype(str).eq(sid) & status.eq(STATUS_ASSIGNED)]
        assigned_frames = int(g["frame"].nunique())
        student_coverages.append(assigned_frames / total_frames)
    if student_coverages:
        report["mean_student_frame_coverage"] = float(np.mean(student_coverages))
        report["min_student_frame_coverage"] = float(np.min(student_coverages))
        report["max_student_frame_coverage"] = float(np.max(student_coverages))

    # Long-gap diagnostic comes from track summary generated below.
    t2s, _ = merge_track_fragments(a, fps=1.0, config=config, total_video_frames=total_frames, seats=seats)
    if not t2s.empty and "long_gap" in t2s.columns:
        long_mask = t2s["long_gap"].fillna(False).astype(bool)
        report["n_long_gap_fragments"] = int(long_mask.sum())
        report["n_long_gap_auto_verified"] = int(
            (long_mask & t2s["long_gap_status"].eq("strong_fixed_seat_evidence")).sum()
        )
        report["n_long_gap_needs_review"] = int(
            (long_mask & t2s["long_gap_status"].eq("needs_visual_review")).sum()
        )
        report["long_gap_ratio"] = float(report["n_long_gap_fragments"] / max(len(t2s), 1))

    # Hard validation gates.
    if len(students) != config.expected_students:
        report["errors"].append(
            f"Expected {config.expected_students} assigned students but found {len(students)}."
        )
    if len(students) < config.expected_students:
        report["warnings"].append(
            f"Only {len(students)} of {config.expected_students} expected student IDs are resolved; "
            "students without valid identity are not assigned zero activity automatically."
        )
    if report["unassigned_rate"] > 0.05:
        report["warnings"].append(
            f"Unassigned detection rate {report['unassigned_rate']:.3f}; inspect detector/tracking coverage."
        )
    if report["duplicate_rate"] > config.max_duplicate_ratio:
        report["errors"].append(
            f"Duplicate ratio {report['duplicate_rate']:.3f} > max_duplicate_ratio {config.max_duplicate_ratio:.3f}."
        )
    if report["n_ambiguous_detections"] / total > config.max_ambiguous_ratio:
        report["errors"].append(
            f"Ambiguous ratio {(report['n_ambiguous_detections']/total):.3f} > max_ambiguous_ratio {config.max_ambiguous_ratio:.3f}."
        )
    if report["min_student_frame_coverage"] < config.min_student_frame_coverage:
        report["errors"].append(
            f"Minimum student frame coverage {report['min_student_frame_coverage']:.3f} < minimum {config.min_student_frame_coverage:.3f}."
        )
    if report["mean_student_frame_coverage"] < config.min_mean_student_frame_coverage:
        report["errors"].append(
            f"Mean student frame coverage {report['mean_student_frame_coverage']:.3f} < minimum {config.min_mean_student_frame_coverage:.3f}."
        )
    if report["mean_seat_purity"] < config.min_seat_purity:
        report["errors"].append(
            f"Mean seat purity {report['mean_seat_purity']:.3f} < minimum {config.min_seat_purity:.3f}."
        )
    if report["n_long_gap_needs_review"] > 0:
        report["warnings"].append(
            f"{report['n_long_gap_needs_review']} long-gap fragment links remain for visual review; "
            f"{report['n_long_gap_auto_verified']} have strong fixed-seat evidence."
        )
    elif report["n_long_gap_auto_verified"] > 0:
        report["warnings"].append(
            f"{report['n_long_gap_auto_verified']} long-gap fragment links were accepted "
            "using strong fixed-seat evidence; this is not independent ground-truth identity verification."
        )
    if report["n_duplicate_detections"]:
        report["warnings"].append(
            f"{report['n_duplicate_detections']} observations are duplicate-track conflicts and are excluded from activity aggregation."
        )

    report["passed"] = len(report["errors"]) == 0
    return report


def merge_track_fragments(
    assignments: pd.DataFrame,
    fps: float,
    config: IdentityConfig,
    total_video_frames: Optional[int] = None,
    seats: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize track fragments and merge them at student level."""
    if assignments is None or assignments.empty:
        return pd.DataFrame(), pd.DataFrame()
    if fps <= 0:
        raise ValueError("fps harus > 0")
    a = assignments.copy()
    status = a["assignment_status"].astype(str)
    ok = a.loc[status.eq(STATUS_ASSIGNED)].copy()
    total_frames = int(total_video_frames) if total_video_frames else int(a["frame"].max()) + 1
    total_frames = max(total_frames, 1)

    trecs = []
    for tid, g in a.groupby("track_id", sort=True):
        g_ok = g[g["assignment_status"].eq(STATUS_ASSIGNED)].copy()
        if g_ok.empty:
            trecs.append({
                "track_id": int(tid), "student_id": None, "seat_id": None,
                "first_frame": int(g["frame"].min()), "last_frame": int(g["frame"].max()),
                "n_frames": int(g["frame"].nunique()), "duration_sec": 0.0,
                "mean_identity_confidence": 0.0, "seat_purity": 0.0,
                "track_identity_confidence": 0.0, "track_seat_score": 0.0,
                "track_coverage_ratio": 0.0,
                "max_gap_frames": 0, "gap_count": 0, "intertrack_gap_frames": 0,
                "prev_track_id": None, "prev_track_seat_purity": 0.0,
                "prev_track_seat_score": 0.0, "long_gap": False,
                "long_gap_status": "not_long_gap",
                "long_gap_evidence_score": 0.0,
                "mapping_status": STATUS_UNASSIGNED,
            })
            continue
        counts = g_ok["student_id"].astype(str).value_counts()
        dominant = str(counts.index[0])
        purity = float(counts.iloc[0] / counts.sum())
        dom = g_ok[g_ok["student_id"].astype(str).eq(dominant)]
        frames = np.sort(dom["frame"].astype(int).unique())
        gaps = np.diff(frames) - 1 if len(frames) > 1 else np.array([], dtype=int)
        max_gap = int(gaps.max()) if len(gaps) else 0
        gap_count = int((gaps > 0).sum()) if len(gaps) else 0
        nf = int(len(frames))
        span = int(frames[-1] - frames[0] + 1) if len(frames) else 0
        trecs.append({
            "track_id": int(tid),
            "student_id": dominant,
            "seat_id": str(dom["seat_id"].iloc[0]),
            "first_frame": int(frames[0]),
            "last_frame": int(frames[-1]),
            "n_frames": nf,
            "duration_sec": nf / fps,
            "mean_identity_confidence": float(dom["identity_confidence"].mean()),
            "seat_purity": purity,
            "track_identity_confidence": float(dom["track_identity_confidence"].iloc[0]) if "track_identity_confidence" in dom.columns else 0.0,
            "track_seat_score": float(dom["track_seat_score"].iloc[0]) if "track_seat_score" in dom.columns else 0.0,
            "track_coverage_ratio": float(nf / span) if span else 0.0,
            "max_gap_frames": max_gap,
            "gap_count": gap_count,
            "intertrack_gap_frames": 0,
            "prev_track_id": None,
            "prev_track_seat_purity": 0.0,
            "prev_track_seat_score": 0.0,
            "long_gap": bool(max_gap > config.reassociation_gap_frames),
            "long_gap_status": "not_long_gap",
            "long_gap_evidence_score": 0.0,
            "mapping_status": STATUS_ASSIGNED if purity >= 0.90 else STATUS_AMBIGUOUS,
        })
    track_to_student = pd.DataFrame(trecs)

    # Compute GAPS BETWEEN TRACK FRAGMENTS belonging to the same fixed seat.
    # A long gap is not automatically an identity error in this dataset: if
    # both fragments strongly resolve to the same fixed seat, they may still
    # belong to the same physical student. We report the gap for audit rather
    # than rejecting it blindly.
    if not track_to_student.empty:
        for sid, idxs in track_to_student[track_to_student["student_id"].notna()].groupby("student_id").groups.items():
            idxs = list(idxs)
            idxs.sort(key=lambda i: int(track_to_student.loc[i, "first_frame"]))
            prev_idx = None
            prev_last = None
            for i in idxs:
                first = int(track_to_student.loc[i, "first_frame"])
                gap = 0 if prev_last is None else max(first - int(prev_last) - 1, 0)
                track_to_student.loc[i, "intertrack_gap_frames"] = int(gap)
                is_long = bool(gap > config.reassociation_gap_frames)
                track_to_student.loc[i, "long_gap"] = is_long

                if prev_idx is not None:
                    track_to_student.loc[i, "prev_track_id"] = int(track_to_student.loc[prev_idx, "track_id"])
                    track_to_student.loc[i, "prev_track_seat_purity"] = float(track_to_student.loc[prev_idx, "seat_purity"])
                    track_to_student.loc[i, "prev_track_seat_score"] = float(track_to_student.loc[prev_idx, "track_seat_score"])

                if is_long:
                    cur_purity = float(track_to_student.loc[i, "seat_purity"])
                    cur_score = float(track_to_student.loc[i, "track_seat_score"])
                    if prev_idx is not None:
                        prev_purity = float(track_to_student.loc[prev_idx, "seat_purity"])
                        prev_score = float(track_to_student.loc[prev_idx, "track_seat_score"])
                        same_student = str(track_to_student.loc[prev_idx, "student_id"]) == str(track_to_student.loc[i, "student_id"])
                        same_seat = str(track_to_student.loc[prev_idx, "seat_id"]) == str(track_to_student.loc[i, "seat_id"])
                        evidence = min(
                            0.5 * (cur_purity + cur_score),
                            0.5 * (prev_purity + prev_score),
                        )
                        track_to_student.loc[i, "long_gap_evidence_score"] = float(evidence)
                        strong = (
                            same_student
                            and same_seat
                            and cur_purity >= config.long_gap_auto_verify_purity
                            and prev_purity >= config.long_gap_auto_verify_purity
                            and cur_score >= config.long_gap_auto_verify_score
                            and prev_score >= config.long_gap_auto_verify_score
                        )
                        track_to_student.loc[i, "long_gap_status"] = (
                            "strong_fixed_seat_evidence" if strong else "needs_visual_review"
                        )
                    else:
                        track_to_student.loc[i, "long_gap_status"] = "needs_visual_review"
                else:
                    track_to_student.loc[i, "long_gap_status"] = "not_long_gap"
                    track_to_student.loc[i, "long_gap_evidence_score"] = 0.0

                prev_idx = i
                prev_last = (
                    max(int(prev_last), int(track_to_student.loc[i, "last_frame"]))
                    if prev_last is not None
                    else int(track_to_student.loc[i, "last_frame"])
                )

    # Student-level summary. IMPORTANT: unassigned is defined relative to the
    # full video frame count, not by filtering rows after student_id becomes NaN.
    # Include ALL expected seats, even if a student currently has zero assigned
    # frames. This is required for a truthful coverage audit.
    srecs = []
    if seats is not None and not seats.empty:
        seat_by_student = seats.set_index(seats["student_id"].astype(str))
        expected_ids = [str(x) for x in seats["student_id"]]
    else:
        seat_by_student = None
        expected_ids = sorted(ok["student_id"].dropna().astype(str).unique().tolist())
    for sid in expected_ids:
        seat_row = seat_by_student.loc[sid] if seat_by_student is not None else None
        g_ok = ok[ok["student_id"].astype(str).eq(sid)].copy()
        seat_id = str(seat_row["seat_id"]) if seat_row is not None else (str(g_ok["seat_id"].iloc[0]) if not g_ok.empty else "")
        tids = sorted(int(x) for x in g_ok["track_id"].unique()) if not g_ok.empty else []
        assigned_frames = int(g_ok["frame"].nunique()) if not g_ok.empty else 0
        seat_rows = a[a["seat_id"].astype(str).eq(seat_id)]
        dup_frames = int(seat_rows.loc[seat_rows["assignment_status"].eq(STATUS_DUPLICATE), "frame"].nunique())
        ambiguous_frames = int(seat_rows.loc[seat_rows["assignment_status"].eq(STATUS_AMBIGUOUS), "frame"].nunique())
        cover = assigned_frames / total_frames
        trec = track_to_student[track_to_student["student_id"].astype(str).eq(sid)]
        purity = (
            float(np.average(trec["seat_purity"], weights=trec["n_frames"]))
            if not trec.empty and trec["n_frames"].sum() > 0 else 0.0
        )
        mean_conf = float(g_ok["identity_confidence"].mean()) if not g_ok.empty else 0.0
        srecs.append({
            "student_id": sid,
            "seat_id": seat_id,
            "assigned_track_ids": ";".join(map(str, tids)),
            "n_track_fragments": len(tids),
            "total_frames": assigned_frames,
            "total_duration_sec": assigned_frames / fps,
            "mean_identity_confidence": mean_conf,
            "frame_coverage": cover,
            "unassigned_frames": max(total_frames - assigned_frames, 0),
            "ambiguous_frames": ambiguous_frames,
            "duplicate_track_frames": dup_frames,
            "seat_purity": purity,
            "mapping_status": STATUS_ASSIGNED if assigned_frames > 0 else STATUS_UNASSIGNED,
        })
    student_audit = pd.DataFrame(srecs)
    return track_to_student, student_audit


def fragment_warnings(track_to_student: pd.DataFrame, config: IdentityConfig) -> list[str]:
    """Report only long-gap links whose fixed-seat evidence is not strong enough for automatic acceptance."""
    msgs: list[str] = []
    if track_to_student is None or track_to_student.empty:
        return msgs
    long_df = track_to_student[
        track_to_student["long_gap"].fillna(False).astype(bool)
        & track_to_student["student_id"].notna()
    ].copy()
    if long_df.empty:
        return msgs

    review_df = long_df[long_df["long_gap_status"].eq("needs_visual_review")].copy()
    auto_df = long_df[long_df["long_gap_status"].eq("strong_fixed_seat_evidence")].copy()

    if review_df.empty:
        msgs.append(
            f"LONG-GAP AUDIT: {len(auto_df)} long-gap fragment links exceed "
            f"{config.reassociation_gap_frames} frames and meet the strong "
            "fixed-seat evidence rule; no visual-review flag remains."
        )
        return msgs

    msgs.append(
        f"WARNING: {len(review_df)} of {len(long_df)} long-gap fragment links "
        f"require visual review; {len(auto_df)} were accepted from strong fixed-seat evidence."
    )
    for r in review_df.sort_values("intertrack_gap_frames", ascending=False).head(20).itertuples():
        msgs.append(
            f"REVIEW: track_id={r.track_id} -> {r.student_id} follows an inter-track "
            f"gap of {int(r.intertrack_gap_frames)} frames; "
            f"evidence_score={float(r.long_gap_evidence_score):.3f}. Verify visually before final reporting."
        )
    return msgs


def long_gap_audit(track_to_student: pd.DataFrame, config: IdentityConfig) -> pd.DataFrame:
    """Return a compact audit table for long-gap fragment links."""
    cols = [
        "student_id", "seat_id", "prev_track_id", "track_id",
        "intertrack_gap_frames", "prev_track_seat_purity",
        "seat_purity", "prev_track_seat_score", "track_seat_score",
        "long_gap_evidence_score", "long_gap_status",
    ]
    if track_to_student is None or track_to_student.empty:
        return pd.DataFrame(columns=cols)
    out = track_to_student[
        track_to_student["long_gap"].fillna(False).astype(bool)
        & track_to_student["student_id"].notna()
    ].copy()
    if out.empty:
        return pd.DataFrame(columns=cols)
    out["long_gap_threshold_frames"] = int(config.reassociation_gap_frames)
    out = out.sort_values("intertrack_gap_frames", ascending=False, kind="mergesort")
    cols_out = [c for c in cols if c in out.columns] + ["long_gap_threshold_frames"]
    return out[cols_out].reset_index(drop=True)


def build_student_level_detections(assignments: pd.DataFrame) -> pd.DataFrame:
    """Keep one primary observation per (frame, student), never duplicate-track rows."""
    cols = ["frame", "student_id", "seat_id", "class_name", "confidence", "track_id"]
    if assignments is None or assignments.empty:
        return pd.DataFrame(columns=cols)
    ok = assignments[assignments["assignment_status"].eq(STATUS_ASSIGNED)].copy()
    if ok.empty:
        return ok.reindex(columns=cols)
    sort_cols = ["frame", "student_id", "track_identity_confidence", "track_seat_purity", "confidence", "track_id"]
    asc = [True, True, False, False, False, True]
    for c in sort_cols:
        if c not in ok.columns:
            ok[c] = 0.0
    ok = ok.sort_values(sort_cols, ascending=asc, kind="mergesort")
    dedup = ok.drop_duplicates(["frame", "student_id"], keep="first")
    return dedup.sort_values(["frame", "student_id"]).reset_index(drop=True)[
        [c for c in cols if c in dedup.columns]
    ]


def format_identity_audit(report: dict) -> str:
    lines = [
        "=" * 64,
        "IDENTITY AUDIT",
        "=" * 64,
        f"Expected students                = {report['expected_students']}",
        f"Assigned student IDs             = {report['n_assigned_students']}",
        f"Raw detections                   = {report.get('n_raw_detections', 'N/A')}",
        f"Primary assigned detections      = {report.get('n_primary_assigned', report['n_assigned_students'] and 'N/A')}",
        f"Primary assignment rate          = {report.get('primary_assignment_rate', 0.0):.2%}",
        f"Unassigned detections            = {report['n_unassigned_detections']} ({report['unassigned_rate']:.2%})",
        f"Ambiguous detections             = {report['n_ambiguous_detections']}",
        f"Duplicate-track rows             = {report['n_duplicate_detections']} ({report['duplicate_rate']:.2%})",
        f"Mean student frame coverage     = {report['mean_student_frame_coverage']:.2%}",
        f"Minimum student frame coverage  = {report['min_student_frame_coverage']:.2%}",
        f"Maximum student frame coverage  = {report['max_student_frame_coverage']:.2%}",
        f"Mean seat purity                = {report['mean_seat_purity']:.2%}",
        f"Median seat purity              = {report['median_seat_purity']:.2%}",
        f"Unique BoT-SORT track IDs        = {report['n_unique_track_ids']}",
        f"Fragmented students             = {report['n_fragmented_students']}",
        f"Track fragments merged          = {report['n_track_fragments_merged']}",
        "-" * 64,
        "CATATAN: track_id adalah trajectory identifier; student_id ditentukan dari fixed-seat evidence.",
    ]
    for w in report.get("warnings", []):
        lines.append(f"WARNING: {w}")
    for e in report.get("errors", []):
        lines.append(f"ERROR:   {e}")
    lines.append(f"RESULT: {'PASSED' if report['passed'] else 'FAILED'}")
    lines.append("=" * 64)
    return "\n".join(lines)
