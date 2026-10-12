"""Core analysis for the student-level YOLO26 pipeline.

Stages covered here:
    trajectory JSONL -> student-level activity matrices -> Fuzzy AHP -> WP
    -> sensitivity -> Monte Carlo -> temporal comparison -> reporting.

The implementation deliberately keeps these semantics explicit:
    * track_id is a trajectory ID, never a student identity;
    * Fuzzy AHP weights come from the configured pairwise matrix;
    * the matrix is labelled researcher-constructed/supervisor-reviewed unless
      real expert elicitation data are supplied;
    * MOT metrics are not fabricated when no trajectory-level GT exists;
    * ranking is a decision-support output over observable activities only.
"""
from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

CRITERIA = ["Angkat Tangan", "Berdiri", "Diam", "Tidur"]
ORIENTATION = {
    "Angkat Tangan": "benefit",
    "Berdiri": "benefit",
    "Diam": "cost",
    "Tidur": "cost",
}
RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
            6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
SAATY_TFN = {
    1: (1, 1, 1), 2: (1, 2, 3), 3: (2, 3, 4), 4: (3, 4, 5),
    5: (4, 5, 6), 6: (5, 6, 7), 7: (6, 7, 8), 8: (7, 8, 9), 9: (9, 9, 9)
}

DETECTION_COLUMNS = [
    "frame", "track_id", "class_id", "class_name",
    "confidence", "x1", "y1", "x2", "y2",
]


def load_trajectory_jsonl(path: str) -> pd.DataFrame:
    """Convert trajectory_botsort.jsonl into the common detection table."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Trajectory JSONL tidak ditemukan: {path}")
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL rusak pada baris {line_no}: {exc}") from exc
            frame = int(obj.get("frame"))
            for d in obj.get("detections", []):
                bbox = d.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                rows.append({
                    "frame": frame,
                    "track_id": int(d.get("track_id", -1)),
                    "class_id": int(d.get("class_id", -1)),
                    "class_name": str(d.get("class_name", "")),
                    "confidence": float(d.get("confidence", 0.0)),
                    "x1": float(bbox[0]), "y1": float(bbox[1]),
                    "x2": float(bbox[2]), "y2": float(bbox[3]),
                })
    df = pd.DataFrame(rows, columns=DETECTION_COLUMNS)
    if df.empty:
        raise ValueError("Trajectory JSONL tidak menghasilkan deteksi apa pun.")
    return df


def validate_classes(df: pd.DataFrame, strict: bool = True) -> None:
    found = set(df["class_name"].astype(str).dropna().unique())
    allowed = set(CRITERIA)
    extra = sorted(found - allowed)
    missing = sorted(allowed - found)
    if extra and strict:
        raise ValueError(f"Class di luar kriteria: {extra}")
    if missing:
        print(f"WARNING: class tidak muncul di trajectory: {missing}")


def summarize_tracks(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    if fps <= 0:
        raise ValueError("fps harus > 0")
    recs = []
    for tid, g in df.groupby("track_id", sort=True):
        g = g.sort_values("frame")
        frames = np.sort(g["frame"].astype(int).unique())
        first, last = int(frames[0]), int(frames[-1])
        span = last - first + 1
        seq = g["class_name"].astype(str).tolist()
        recs.append({
            "track_id": int(tid),
            "n_frames": int(len(frames)),
            "first_frame": first,
            "last_frame": last,
            "span_frames": span,
            "coverage_ratio": float(len(frames) / span) if span else 0.0,
            "duration_sec": float(len(frames) / fps),
            "mean_confidence": float(g["confidence"].mean()),
            "median_confidence": float(g["confidence"].median()),
            "dominant_activity": g["class_name"].value_counts().idxmax(),
            "class_switch_count": int(sum(a != b for a, b in zip(seq, seq[1:]))),
        })
    return pd.DataFrame(recs)


def activity_matrix(
    student_detections: pd.DataFrame,
    fps: float,
    representation: str = "duration",
    all_students: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build an exact student x activity matrix from primary observations."""
    if representation not in {"duration", "frequency"}:
        raise ValueError("representation harus duration atau frequency")
    if fps <= 0:
        raise ValueError("fps harus > 0")
    if student_detections is None or student_detections.empty:
        idx = list(all_students or [])
        return pd.DataFrame(0.0, index=idx, columns=CRITERIA)

    d = student_detections.copy()
    d = d[d["student_id"].notna()].copy()
    d["student_id"] = d["student_id"].astype(str)
    d["class_name"] = d["class_name"].astype(str)
    counts = d.groupby(["student_id", "class_name"])["frame"].nunique().unstack(fill_value=0)
    counts = counts.reindex(columns=CRITERIA, fill_value=0)
    if all_students is not None:
        counts = counts.reindex([str(s) for s in all_students], fill_value=0)
    counts = counts.sort_index()
    if representation == "frequency":
        return counts.astype(int)
    return counts.astype(float) / float(fps)


def activity_transitions(student_detections: pd.DataFrame) -> pd.DataFrame:
    if student_detections is None or student_detections.empty:
        return pd.DataFrame(columns=["student_id", "from_class", "to_class", "count"])
    d = student_detections[["student_id", "frame", "class_name"]].copy()
    d["student_id"] = d["student_id"].astype(str)
    d = d.sort_values(["student_id", "frame"])
    out = []
    for sid, g in d.groupby("student_id"):
        seq = g["class_name"].astype(str).tolist()
        pairs = [(a, b) for a, b in zip(seq, seq[1:]) if a != b]
        out.extend((sid, a, b) for a, b in pairs)
    if not out:
        return pd.DataFrame(columns=["student_id", "from_class", "to_class", "count"])
    return (pd.DataFrame(out, columns=["student_id", "from_class", "to_class"])
            .value_counts()
            .rename("count")
            .reset_index())


def _tfn(value: float) -> tuple[float, float, float]:
    value = float(value)
    if value >= 1:
        k = int(round(value))
        if k not in SAATY_TFN:
            raise ValueError(f"Nilai Saaty di luar 1..9: {value}")
        return SAATY_TFN[k]
    inv = 1.0 / value
    k = int(round(inv))
    if k not in SAATY_TFN:
        raise ValueError(f"Nilai kebalikan tidak cocok dengan skala Saaty: {value}")
    a = SAATY_TFN[k]
    return (1.0 / a[2], 1.0 / a[1], 1.0 / a[0])


def consistency_ratio(A: np.ndarray) -> tuple[float, float, float, float]:
    n = A.shape[0]
    col_sum = A.sum(axis=0)
    norm = A / col_sum
    pv = norm.mean(axis=1)
    weighted = A @ pv
    lam = float(np.mean(weighted / pv))
    ci = float((lam - n) / (n - 1)) if n > 1 else 0.0
    ri = float(RI_TABLE.get(n, 1.49))
    cr = float(ci / ri) if ri > 0 else 0.0
    return lam, ci, ri, cr


def fuzzy_ahp_weights(matrix: Sequence[Sequence[float]], criteria: Sequence[str]) -> dict:
    """Buckley-style fuzzy geometric-mean weighting plus crisp CR."""
    A = np.asarray(matrix, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Matriks AHP harus persegi, dapat {A.shape}")
    n = A.shape[0]
    if n != len(criteria):
        raise ValueError("Jumlah kriteria dan ukuran matriks tidak sama")
    if (A <= 0).any():
        raise ValueError("Semua nilai matriks AHP harus > 0")
    reciprocal_err = np.abs(A * A.T - 1.0)
    np.fill_diagonal(reciprocal_err, 0.0)
    if float(reciprocal_err.max()) > 1e-6:
        raise ValueError("Matriks AHP tidak resiprokal")

    fuzzy = [[_tfn(A[i, j]) for j in range(n)] for i in range(n)]
    geom = []
    for i in range(n):
        l = math.prod(fuzzy[i][j][0] for j in range(n)) ** (1.0 / n)
        m = math.prod(fuzzy[i][j][1] for j in range(n)) ** (1.0 / n)
        u = math.prod(fuzzy[i][j][2] for j in range(n)) ** (1.0 / n)
        geom.append((l, m, u))
    sum_l = sum(x[0] for x in geom)
    sum_m = sum(x[1] for x in geom)
    sum_u = sum(x[2] for x in geom)
    crisp = []
    for l, m, u in geom:
        wl, wm, wu = l / sum_u, m / sum_m, u / sum_l
        crisp.append((wl + wm + wu) / 3.0)
    w = np.asarray(crisp, dtype=float)
    w /= w.sum()
    lam, ci, ri, cr = consistency_ratio(A)
    return {
        "weights": {c: float(x) for c, x in zip(criteria, w)},
        "weight_vector": w,
        "matrix": A.tolist(),
        "lambda_max": lam,
        "CI": ci,
        "RI": ri,
        "CR": cr,
        "consistent": bool(cr <= 0.10),
    }


def _normalize_wp(X: pd.DataFrame, orientation: dict, method: str, floor: float) -> pd.DataFrame:
    X = X.astype(float).copy()
    if method not in {"minmax_floor", "ratio_floor"}:
        raise ValueError("normalization method harus minmax_floor atau ratio_floor")
    if not (0.0 < floor < 1.0):
        raise ValueError("floor harus di antara 0 dan 1")
    out = pd.DataFrame(index=X.index, columns=X.columns, dtype=float)
    for c in X.columns:
        s = X[c].to_numpy(dtype=float)
        if method == "minmax_floor":
            lo, hi = float(np.min(s)), float(np.max(s))
            if hi > lo:
                z = floor + (1.0 - floor) * (s - lo) / (hi - lo)
            else:
                z = np.ones_like(s)
        else:
            # ratio normalization; for cost use min positive / x,
            # for benefit use x / max. Zero handling happens before transform.
            eps = 1e-6
            safe = np.maximum(s, eps)
            if orientation[c] == "benefit":
                mx = float(np.max(safe))
                z = safe / mx if mx > 0 else np.ones_like(safe)
            else:
                mn = float(np.min(safe))
                z = mn / safe
            z = floor + (1.0 - floor) * z
        out[c] = np.clip(z, floor, 1.0)
    return out


def wp_score(
    matrix: pd.DataFrame,
    weights: dict,
    orientation: dict,
    normalization: str = "minmax_floor",
    floor: float = 0.01,
    zero_epsilon: float = 1e-6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return normalized matrix and WP ranking."""
    if matrix is None or matrix.empty:
        return pd.DataFrame(), pd.DataFrame()
    X = matrix.astype(float).copy()
    zero_counts = []
    for c in X.columns:
        zc = int((X[c] == 0).sum())
        zero_counts.append({"criterion": c, "zero_count": zc, "orientation": orientation[c]})
    # WP needs strictly positive values.
    X = X.mask(X <= 0, zero_epsilon)
    N = _normalize_wp(X, orientation, normalization, floor)
    w = np.asarray([weights[c] for c in X.columns], dtype=float)
    w /= w.sum()
    signs = np.asarray([1.0 if orientation[c] == "benefit" else -1.0 for c in X.columns])
    exponents = w * signs
    log_s = np.log(N.to_numpy()) @ exponents
    S = np.exp(log_s - np.max(log_s))
    V = S / S.sum() if S.sum() > 0 else np.zeros_like(S)
    rank = pd.Series(V, index=N.index).rank(method="min", ascending=False).astype(int)
    ranking = N.copy()
    ranking["S"] = S
    ranking["V"] = V
    ranking["rank"] = rank
    ranking = ranking.sort_values(["rank", "V"], ascending=[True, False])
    return ranking, pd.DataFrame(zero_counts)


def compare_rankings(base_rank: pd.Series, new_rank: pd.Series, top_k=(1, 5, 10)) -> dict:
    idx = base_rank.index.intersection(new_rank.index)
    b = base_rank.reindex(idx).astype(float)
    n = new_rank.reindex(idx).astype(float)
    if not len(idx):
        return {"n_entities": 0}
    delta = (b - n).abs()
    if b.nunique() > 1 and n.nunique() > 1 and len(idx) >= 2:
        rho, _ = spearmanr(b, n)
        tau, _ = kendalltau(b, n)
    else:
        rho = tau = np.nan
    bs = b.sort_values().index.tolist()
    ns = n.sort_values().index.tolist()
    out = {
        "n_entities": len(idx),
        "mean_rank_shift": float(delta.mean()),
        "max_rank_shift": int(delta.max()),
        "n_rank_changes": int((b != n).sum()),
        "spearman_rho": float(rho),
        "kendall_tau": float(tau),
        "top1_changed": bool(bs[0] != ns[0]),
    }
    for k in top_k:
        kk = min(k, len(idx))
        out[f"top{k}_overlap"] = len(set(bs[:kk]) & set(ns[:kk])) / kk
    return out


def sensitivity_oat(
    matrix: pd.DataFrame,
    weights: dict,
    orientation: dict,
    perturbations: Sequence[float],
    normalization: str,
    floor: float,
    zero_epsilon: float,
) -> pd.DataFrame:
    base, _ = wp_score(matrix, weights, orientation, normalization, floor, zero_epsilon)
    base_rank = base["rank"].reindex(matrix.index)
    rows = []
    criteria = list(matrix.columns)
    for c in criteria:
        for delta in perturbations:
            w = dict(weights)
            w[c] = max(1e-9, w[c] * (1.0 + float(delta)))
            total = sum(w.values())
            w = {k: v / total for k, v in w.items()}
            sc, _ = wp_score(matrix, w, orientation, normalization, floor, zero_epsilon)
            metrics = compare_rankings(base_rank, sc["rank"].reindex(matrix.index))
            rows.append({"criterion": c, "perturbation": float(delta), **metrics})
    return pd.DataFrame(rows)


def normalization_sensitivity(
    matrix: pd.DataFrame,
    weights: dict,
    orientation: dict,
    methods: Sequence[str],
    floor: float,
    zero_epsilon: float,
) -> pd.DataFrame:
    base, _ = wp_score(matrix, weights, orientation, "minmax_floor", floor, zero_epsilon)
    base_rank = base["rank"].reindex(matrix.index)
    rows = []
    for method in methods:
        sc, _ = wp_score(matrix, weights, orientation, method, floor, zero_epsilon)
        m = compare_rankings(base_rank, sc["rank"].reindex(matrix.index))
        rows.append({"baseline_normalization": "minmax_floor", "scenario_normalization": method, **m})
    return pd.DataFrame(rows)


def monte_carlo_wp(
    matrix: pd.DataFrame,
    base_weights: dict,
    orientation: dict,
    iterations: int,
    seed: int,
    concentration: float,
    normalization: str,
    floor: float,
    zero_epsilon: float,
) -> pd.DataFrame:
    if iterations <= 0:
        raise ValueError("iterations harus > 0")
    criteria = list(matrix.columns)
    base = wp_score(matrix, base_weights, orientation, normalization, floor, zero_epsilon)[0]
    baseline_rank = base["rank"].reindex(matrix.index).to_numpy()
    n = len(matrix)
    ranks = np.zeros((iterations, n), dtype=float)
    rng = np.random.default_rng(seed)
    alpha = np.clip(np.asarray([base_weights[c] for c in criteria]) * concentration, 1e-6, None)
    for i in range(iterations):
        wv = rng.dirichlet(alpha)
        w = {c: float(v) for c, v in zip(criteria, wv)}
        sc = wp_score(matrix, w, orientation, normalization, floor, zero_epsilon)[0]
        ranks[i] = sc["rank"].reindex(matrix.index).to_numpy()
    k5, k10 = min(5, n), min(10, n)
    out = pd.DataFrame({
        "student_id": matrix.index,
        "baseline_rank": baseline_rank,
        "mean_rank": ranks.mean(axis=0),
        "std_rank": ranks.std(axis=0),
        "min_rank": ranks.min(axis=0),
        "max_rank": ranks.max(axis=0),
        "top1_frequency": (ranks == 1).mean(axis=0),
        "top5_frequency": (ranks <= k5).mean(axis=0),
        "top10_frequency": (ranks <= k10).mean(axis=0),
        "n_iterations": iterations,
        "seed": seed,
        "method": "dirichlet_around_base",
    })
    return out.sort_values("mean_rank").reset_index(drop=True)


def temporal_comparison(
    duration: pd.DataFrame,
    frequency: pd.DataFrame,
    weights: dict,
    orientation: dict,
    normalization: str,
    floor: float,
    zero_epsilon: float,
) -> pd.DataFrame:
    d = wp_score(duration, weights, orientation, normalization, floor, zero_epsilon)[0]["rank"]
    f = wp_score(frequency, weights, orientation, normalization, floor, zero_epsilon)[0]["rank"]
    return pd.DataFrame([compare_rankings(d, f)])


def ablation_summary(raw: pd.DataFrame, student: pd.DataFrame, identity_report: dict, min_frames: int) -> pd.DataFrame:
    raw_tracks = int(raw["track_id"].nunique())
    track_summary = summarize_tracks(raw, fps=25.0)
    temporal_entities = int((track_summary["n_frames"] >= min_frames).sum())
    return pd.DataFrame([
        {"stage": "A_detection_only", "entity_definition": "frame-level detections", "entity_count": int(len(raw)), "note": "Raw YOLO26 detections"},
        {"stage": "B_detection_plus_tracking", "entity_definition": "raw BoT-SORT track IDs", "entity_count": raw_tracks, "note": "Track IDs are not students"},
        {"stage": "C_temporal_filter", "entity_definition": f"track IDs with >= {min_frames} frames", "entity_count": temporal_entities, "note": "Persistence filter only"},
        {"stage": "D_full_student_level", "entity_definition": "fixed-seat student profiles", "entity_count": int(student["student_id"].nunique()) if not student.empty else 0, "note": f"Identity audit: {'PASSED' if identity_report.get('passed') else 'FAILED'}"},
    ])


def error_propagation(student_audit: pd.DataFrame, matrix: pd.DataFrame, ranking: pd.DataFrame) -> pd.DataFrame:
    rows = []
    measured = {
        "identity_coverage": float(student_audit["frame_coverage"].mean()) if not student_audit.empty else np.nan,
        "duplicate_frame_fraction": float(student_audit["duplicate_track_frames"].sum() / max(student_audit["total_frames"].sum(), 1)) if not student_audit.empty else np.nan,
        "ambiguous_frame_fraction": float(student_audit["ambiguous_frames"].sum() / max(student_audit["total_frames"].sum(), 1)) if not student_audit.empty else np.nan,
    }
    rows += [
        {"source": "identity", "quantity": k, "value": v, "status": "MEASURED"}
        for k, v in measured.items()
    ]
    rows += [
        {"source": "reasoning", "quantity": "identity_errors_can_propagate_to_activity_counts", "value": "NOT_QUANTIFIED", "status": "INFERRED"},
        {"source": "reasoning", "quantity": "ranking_depends_on_AHP_weights_and_WP_normalization", "value": "MODEL_DEPENDENT", "status": "INFERRED"},
    ]
    return pd.DataFrame(rows)


def environment_provenance(cfg: dict, extra: Optional[dict] = None) -> pd.DataFrame:
    rec = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "model_path": cfg.get("source", {}).get("model_path", "NOT_AVAILABLE"),
        "trajectory_jsonl": cfg.get("source", {}).get("trajectory_jsonl", "NOT_AVAILABLE"),
        "configured_fps": cfg.get("temporal", {}).get("fps", "NOT_AVAILABLE"),
        "tracker": cfg.get("tracking", {}).get("tracker", "NOT_AVAILABLE"),
        "tracker_with_reid": cfg.get("tracking", {}).get("with_reid", "NOT_AVAILABLE"),
        "ahp_source": cfg.get("ahp", {}).get("source", "NOT_AVAILABLE"),
        "ahp_expert_count": cfg.get("ahp", {}).get("expert_count", "NOT_AVAILABLE"),
        "wp_normalization": cfg.get("wp", {}).get("normalization", "NOT_AVAILABLE"),
        "wp_floor": cfg.get("wp", {}).get("floor", "NOT_AVAILABLE"),
        "sensitivity_perturbations": str(cfg.get("sensitivity", {}).get("perturbations", "NOT_AVAILABLE")),
    }
    try:
        import torch
        rec["torch_version"] = torch.__version__
        rec["cuda_available"] = torch.cuda.is_available()
        rec["cuda_version"] = getattr(torch.version, "cuda", "NOT_AVAILABLE")
        rec["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT_AVAILABLE"
    except Exception:
        rec.update({"torch_version": "NOT_AVAILABLE", "cuda_available": "NOT_AVAILABLE", "cuda_version": "NOT_AVAILABLE", "gpu": "NOT_AVAILABLE"})
    try:
        import ultralytics
        rec["ultralytics_version"] = ultralytics.__version__
    except Exception:
        rec["ultralytics_version"] = "NOT_AVAILABLE"
    if extra:
        rec.update(extra)
    return pd.DataFrame({"key": list(rec.keys()), "value": [str(v) for v in rec.values()]})


def mot_status(gt_csv: Optional[str]) -> pd.DataFrame:
    available = bool(gt_csv) and Path(str(gt_csv)).exists()
    row = {
        "ground_truth_available": available,
        "gt_path": gt_csv or "NOT_AVAILABLE",
        "HOTA": "NOT_COMPUTED" if not available else "READY_FOR_TRACKEVAL",
        "IDF1": "NOT_COMPUTED" if not available else "READY_FOR_TRACKEVAL",
        "MOTA": "NOT_COMPUTED" if not available else "READY_FOR_TRACKEVAL",
        "identity_switches": "NOT_COMPUTED",
        "status": "NOT_AVAILABLE" if not available else "READY_TO_COMPUTE",
        "note": "No trajectory-level ground truth was supplied; formal MOT metrics are not fabricated.",
    }
    return pd.DataFrame([row])


def evaluate_detector(model_path: str, data_yaml: str, split: str, imgsz: int, device: str, batch: int) -> tuple[pd.DataFrame, dict]:
    """Run Ultralytics model.val only when explicitly enabled and sources exist."""
    if not Path(model_path).exists():
        return pd.DataFrame([{"status": "NOT_AVAILABLE", "reason": f"model not found: {model_path}"}]), {}
    if not Path(data_yaml).exists():
        return pd.DataFrame([{"status": "NOT_AVAILABLE", "reason": f"data.yaml not found: {data_yaml}"}]), {}
    from ultralytics import YOLO
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, split=split, imgsz=imgsz, device=device, batch=batch, verbose=False)
    rows = []
    names = metrics.names if hasattr(metrics, "names") else model.names
    # Ultralytics object stores per-class metrics in class-specific arrays.
    box = metrics.box
    p = np.asarray(getattr(box, "p", []), dtype=float)
    r = np.asarray(getattr(box, "r", []), dtype=float)
    ap50 = np.asarray(getattr(box, "ap50", []), dtype=float)
    ap = np.asarray(getattr(box, "ap", []), dtype=float)
    for i in range(len(p)):
        rows.append({
            "scope": "per_class", "class_id": i, "class": names.get(i, str(i)) if isinstance(names, dict) else str(names[i]),
            "precision": float(p[i]), "recall": float(r[i]), "mAP50": float(ap50[i]), "mAP50_95": float(ap[i]),
        })
    overall = {
        "scope": "overall", "class_id": -1, "class": "ALL",
        "precision": float(np.mean(p)) if len(p) else float(metrics.results_dict.get("metrics/precision(B)", np.nan)),
        "recall": float(np.mean(r)) if len(r) else float(metrics.results_dict.get("metrics/recall(B)", np.nan)),
        "mAP50": float(np.mean(ap50)) if len(ap50) else float(metrics.results_dict.get("metrics/mAP50(B)", np.nan)),
        "mAP50_95": float(np.mean(ap)) if len(ap) else float(metrics.results_dict.get("metrics/mAP50-95(B)", np.nan)),
    }
    rows.insert(0, overall)
    return pd.DataFrame(rows), overall


def write_excel_report(
    path: str,
    cfg: dict,
    provenance: pd.DataFrame,
    track_summary: pd.DataFrame,
    assignments: pd.DataFrame,
    track_to_student: pd.DataFrame,
    identity_audit: pd.DataFrame,
    duration: pd.DataFrame,
    frequency: pd.DataFrame,
    transitions: pd.DataFrame,
    ahp: dict,
    ranking: pd.DataFrame,
    normalized: pd.DataFrame,
    zero_df: pd.DataFrame,
    sensitivity: pd.DataFrame,
    normalization_df: pd.DataFrame,
    monte_carlo: pd.DataFrame,
    temporal_cmp: pd.DataFrame,
    ablation: pd.DataFrame,
    error_df: pd.DataFrame,
    detector_metrics: Optional[pd.DataFrame],
    mot_df: pd.DataFrame,
) -> str:
    """Create the reproducibility workbook; no dummy research values."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def flat_config(d, prefix=""):
        rows = []
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                rows.extend(flat_config(v, key))
            else:
                rows.append((key, str(v)))
        return rows

    assignment_export = assignments.head(5000).copy() if assignments is not None else pd.DataFrame()
    if assignments is not None and len(assignments) > 5000:
        assignment_export = pd.concat([
            assignment_export,
            pd.DataFrame([{"NOTE": f"Full Identity_Assignments has {len(assignments):,} rows; complete CSV is kept at output/identity_assignments.csv."}])
        ], ignore_index=True)
    sheets = {
        "Config": pd.DataFrame(flat_config(cfg), columns=["key", "value"]),
        "Provenance": provenance,
        "Detection_Metrics": detector_metrics if detector_metrics is not None and not detector_metrics.empty else pd.DataFrame([{"status": "NOT_RUN", "reason": "Detector evaluation disabled or unavailable."}]),
        "Track_Summary": track_summary,
        "Identity_Assignments": assignment_export,
        "Track_to_Student": track_to_student,
        "Student_Identity_Audit": identity_audit,
        "Activity_Duration": duration,
        "Activity_Frequency": frequency,
        "Activity_Transitions": transitions,
        "Fuzzy_AHP_Weights": pd.DataFrame({"criterion": list(ahp["weights"].keys()), "weight": list(ahp["weights"].values())}),
        "Fuzzy_AHP_Consistency": pd.DataFrame({"metric": ["lambda_max", "CI", "RI", "CR", "consistent", "source", "expert_count"], "value": [ahp["lambda_max"], ahp["CI"], ahp["RI"], ahp["CR"], ahp["consistent"], cfg["ahp"]["source"], cfg["ahp"].get("expert_count", 0)]}),
        "WP_Normalized": normalized,
        "WP_Ranking": ranking,
        "Zero_Handling": zero_df,
        "Sensitivity_OAT": sensitivity,
        "Sensitivity_Normalization": normalization_df,
        "Sensitivity_MonteCarlo": monte_carlo,
        "Temporal_Comparison": temporal_cmp,
        "Ablation": ablation,
        "Error_Propagation": error_df,
        "MOT_Evaluation": mot_df,
    }

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe = name[:31]
            out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
            if out.index.name or out.index.nlevels > 1:
                out.to_excel(writer, sheet_name=safe, index=True)
            else:
                out.to_excel(writer, sheet_name=safe, index=False)
            ws = writer.book[safe]
            ws.freeze_panes = "A2"
            if ws.max_row > 1:
                ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = cell.font.copy(bold=True)
                cell.alignment = cell.alignment.copy(horizontal="center", vertical="center", wrap_text=True)
            for col_cells in ws.columns:
                max_len = 0
                for cell in list(col_cells)[:250]:
                    max_len = max(max_len, len(str(cell.value)) if cell.value is not None else 0)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(10, max_len + 2), 52)
    return path
