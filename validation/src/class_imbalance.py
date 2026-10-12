from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

DEFAULT_NAMES = {0: "Hand Raising", 1: "Standing", 2: "Quiet Sitting", 3: "Sleeping"}


def _resolve(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    p = Path(value)
    return p if p.is_absolute() else root / p


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_dataset_path(data_yaml: Path, cfg_value: str | None, split: str) -> Path | list[Path]:
    data = _load_yaml(data_yaml)
    base = Path(str(data.get("path", ".")))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    raw = cfg_value if cfg_value is not None else data.get(split)
    if raw is None:
        raise FileNotFoundError(f"No '{split}' entry in {data_yaml}")
    if isinstance(raw, list):
        return [_resolve_dataset_entry(base, data_yaml.parent, str(x)) for x in raw]
    return _resolve_dataset_entry(base, data_yaml.parent, str(raw))


def _resolve_dataset_entry(base: Path, yaml_dir: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    candidate = base / p
    if candidate.exists():
        return candidate
    return yaml_dir / p


def _label_dir_for_images(images_dir: Path) -> Path:
    parts = list(images_dir.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts)
    except ValueError:
        # Common alternative: .../train/images -> .../train/labels
        if images_dir.name == "images":
            return images_dir.parent / "labels"
        return images_dir.parent / "labels"


def _iter_image_label_pairs(split_path: Path) -> Iterable[tuple[Path, Path]]:
    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        for line in split_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s:
                continue
            img = Path(s)
            label = img.parent.parent / "labels" / f"{img.stem}.txt" if img.parent.name == "images" else img.with_suffix(".txt")
            yield img, label
        return
    if not split_path.is_dir():
        raise FileNotFoundError(f"Training split path not found: {split_path}")
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    label_root = _label_dir_for_images(split_path) if split_path.name == "images" else split_path / "labels"
    for img in sorted(p for p in split_path.rglob("*") if p.suffix.lower() in image_exts):
        label = label_root / f"{img.stem}.txt"
        if not label.exists() and img.parent.name != "images":
            label = img.with_suffix(".txt")
        yield img, label


def analyze(data_yaml: Path, root: Path, minority_classes: list[int], oversample_factor: int, out_dir: Path) -> dict:
    train_path = _resolve_dataset_path(data_yaml, None, "train")
    if isinstance(train_path, list):
        pairs = [pair for p in train_path for pair in _iter_image_label_pairs(p)]
    else:
        pairs = list(_iter_image_label_pairs(train_path))

    instance_counts = Counter()
    image_counts = Counter()
    total_images = 0
    images_with_labels = 0
    manifest_lines: list[str] = []
    per_image_rows = []

    for img, label in pairs:
        total_images += 1
        present = set()
        if label.exists():
            images_with_labels += 1
            for line in label.read_text(encoding="utf-8", errors="ignore").splitlines():
                s = line.strip()
                if not s:
                    continue
                toks = s.split()
                try:
                    c = int(toks[0])
                except (ValueError, IndexError):
                    continue
                instance_counts[c] += 1
                present.add(c)
        for c in present:
            image_counts[c] += 1
        repeat = oversample_factor if present.intersection(minority_classes) else 1
        manifest_lines.extend([str(img.resolve())] * repeat)
        per_image_rows.append({"image": str(img), "classes": ";".join(map(str, sorted(present))), "repeat": repeat})

    class_ids = sorted(set(instance_counts) | set(image_counts))
    if not class_ids:
        raise RuntimeError("No YOLO labels were found in the training split.")
    n_instances = sum(instance_counts.values())
    n_classes = len(class_ids)
    weights = {}
    for c in class_ids:
        count = instance_counts[c]
        weights[c] = (n_instances / (n_classes * count)) if count else 0.0
    s = sum(weights.values()) or 1.0
    weights_norm = {c: w / s for c, w in weights.items()}

    names = _load_yaml(data_yaml).get("names", DEFAULT_NAMES)
    if isinstance(names, dict):
        names = {int(k): str(v) for k, v in names.items()}
    else:
        names = {i: str(v) for i, v in enumerate(names)}

    rows = []
    for c in class_ids:
        rows.append({
            "class_id": c,
            "class_name": names.get(c, DEFAULT_NAMES.get(c, f"class_{c}")),
            "instance_count": instance_counts[c],
            "instance_share": instance_counts[c] / n_instances if n_instances else 0.0,
            "image_count": image_counts[c],
            "image_share": image_counts[c] / total_images if total_images else 0.0,
            "inverse_frequency_weight": weights[c],
            "normalized_weight": weights_norm[c],
            "is_minority": c in minority_classes,
        })
    pd.DataFrame(rows).to_csv(out_dir / "class_distribution.csv", index=False)
    pd.DataFrame(per_image_rows).to_csv(out_dir / "train_image_distribution.csv", index=False)
    pd.DataFrame({"class_id": list(weights_norm), "class_name": [names.get(c, f"class_{c}") for c in weights_norm], "normalized_weight": list(weights_norm.values())}).to_csv(
        out_dir / "recommended_class_weights.csv", index=False
    )
    (out_dir / "oversampling_manifest.txt").write_text("\n".join(manifest_lines), encoding="utf-8")
    report = {
        "data_yaml": str(data_yaml),
        "train_path": str(train_path),
        "total_images": total_images,
        "images_with_labels": images_with_labels,
        "total_instances": n_instances,
        "minority_classes": minority_classes,
        "oversample_factor": oversample_factor,
        "oversampled_manifest_entries": len(manifest_lines),
        "strategy_status": "artifact_generated_only_no_training",
        "class_distribution": rows,
    }
    (out_dir / "class_imbalance_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
