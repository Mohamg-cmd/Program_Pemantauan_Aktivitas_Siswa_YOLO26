from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.class_imbalance import analyze as analyze_imbalance
from src.detector_eval import audit_existing, run_eval
from src.identity_reconcile import reconcile as reconcile_identity
from src.seat_purity import calculate as calculate_purity


def resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else root / p


def load_cfg(path: Path) -> tuple[Path, dict]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return path.parent.resolve(), cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO26 validation checks: configured quality checks")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--run-detector", action="store_true", help="actually run model.val() on configured split; no training")
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    root, cfg = load_cfg(cfg_path)
    project_root = resolve(root, cfg.get("project_root", ".")) or root
    out_dir = resolve(root, cfg.get("output_dir", "output/validation"))
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    # 1) class imbalance
    ci = cfg["class_imbalance"]
    data_yaml = resolve(root, ci["data_yaml"])
    print("[1/4] Class imbalance audit")
    summary["class_imbalance"] = analyze_imbalance(
        data_yaml,
        root,
        [int(x) for x in ci.get("minority_classes", [0, 1])],
        int(ci.get("oversample_factor", 4)),
        out_dir,
    )

    # 2) detector evaluation / provenance
    de = cfg["detector_evaluation"]
    print("[2/4] Detector evaluation audit")
    summary["detector_existing_artifacts"] = audit_existing(
        project_root,
        out_dir,
        [project_root / "output", project_root / "output" / "validation", project_root / "focal", project_root / "baseline", project_root / "balanced", project_root / "test_eval", project_root],
    )
    should_run = bool(args.run_detector or de.get("run_validation", False))
    if should_run:
        summary["detector_evaluation"] = run_eval(
            root,
            resolve(root, de["model_path"]),
            resolve(root, de["data_yaml"]),
            str(de.get("split", "test")),
            int(de.get("imgsz", 640)),
            int(de.get("batch", 16)),
            str(de.get("device", "0")),
            bool(de.get("plots", True)),
            out_dir,
        )
    else:
        # Prefer an existing validated detector artifact rather than relabeling it
        # as NOT_RUN. This preserves frozen experiment results without rerunning.
        candidates = [
            project_root / "output" / "detector_eval_metadata.json",
            project_root / "output" / "validation" / "detector_eval_metadata.json",
        ]
        existing = next((p for p in candidates if p.exists()), None)
        if existing:
            try:
                meta = json.loads(existing.read_text(encoding="utf-8"))
                meta["status"] = "EXISTING_VALIDATED_ARTIFACT"
                meta["source"] = str(existing)
                summary["detector_evaluation"] = meta
            except Exception as exc:
                summary["detector_evaluation"] = {"status": "ARTIFACT_FOUND_BUT_UNREADABLE", "source": str(existing), "reason": str(exc)}
        else:
            summary["detector_evaluation"] = {"status": "NOT_RUN", "reason": "No existing detector artifact found and --run-detector was not supplied"}

    # 3) identity reconciliation
    print("[3/4] Full-file identity reconciliation")
    summary["identity_reconciliation"] = reconcile_identity(root, cfg["identity_reconciliation"], out_dir)

    # 4) seat purity
    print("[4/4] Seat purity definition/check")
    summary["seat_purity"] = calculate_purity(root, cfg["seat_purity"], out_dir)

    (out_dir / "VALIDATION_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nDONE")
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
