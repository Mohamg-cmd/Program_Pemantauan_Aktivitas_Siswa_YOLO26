from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else root / p


def _names_from_model(model: Any) -> dict[int, str]:
    names = getattr(model, "names", {}) or {}
    return {int(k): str(v) for k, v in names.items()} if isinstance(names, dict) else {i: str(v) for i, v in enumerate(names)}


def _extract_metrics(metrics: Any, names: dict[int, str]) -> tuple[pd.DataFrame, dict]:
    box = metrics.box
    idx = list(getattr(box, "ap_class_index", range(len(getattr(box, "p", [])))))
    per_class = []
    for j, class_id in enumerate(idx):
        per_class.append({
            "class_id": int(class_id),
            "class_name": names.get(int(class_id), f"class_{class_id}"),
            "precision": float(box.p[j]),
            "recall": float(box.r[j]),
            "mAP50": float(box.ap50[j]),
            "mAP50_95": float(box.ap[j]),
        })
    overall = {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
    }
    return pd.DataFrame(per_class), overall


def run_eval(root: Path, model_path: Path, data_yaml: Path, split: str, imgsz: int, batch: int, device: str, plots: bool, out_dir: Path) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("ultralytics is required for detector evaluation. Install requirements.txt") from exc
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        plots=plots,
        project=str(out_dir.parent),
        name="detector_eval",
    )
    names = _names_from_model(model)
    per_class, overall = _extract_metrics(metrics, names)
    per_class.to_csv(out_dir / "detector_eval_per_class.csv", index=False)
    summary = pd.DataFrame([overall])
    summary.to_csv(out_dir / "detector_eval_summary.csv", index=False)
    metadata = {
        "model_path": str(model_path),
        "data_yaml": str(data_yaml),
        "split": split,
        "evaluation_type": "held_out_test_evaluation" if split.lower() == "test" else f"{split}_evaluation",
        "training_performed": False,
        "metrics": overall,
        "names": names,
    }
    (out_dir / "detector_eval_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def audit_existing(root: Path, out_dir: Path, search_roots: list[Path]) -> dict:
    files = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for p in search_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".csv", ".json", ".txt", ".yaml", ".yml"}:
                lower = p.name.lower()
                if any(k in lower for k in ["result", "metric", "eval", "test", "confusion", "f1", "pr"]):
                    files.append({"path": str(p), "name": p.name, "suffix": p.suffix.lower(), "size_bytes": p.stat().st_size})
    df = pd.DataFrame(files).drop_duplicates(subset=["path"]).sort_values("path") if files else pd.DataFrame(columns=["path", "name", "suffix", "size_bytes"])
    df.to_csv(out_dir / "detector_existing_artifacts_inventory.csv", index=False)
    return {"found_files": len(df), "inventory": str(out_dir / "detector_existing_artifacts_inventory.csv")}
