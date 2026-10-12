"""Optional YOLO26-M + BoT-SORT/ReID stage.

The downstream research pipeline does NOT require this stage when a valid
trajectory_botsort.jsonl already exists. This is important because the large
source CCTV video is not required merely to redo identity association, temporal
aggregation, or decision-support analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .analytics import DETECTION_COLUMNS


def resolve_tracker_config(tracker: str, with_reid: bool, output_dir: str, reid_model: Optional[str] = None) -> str:
    if not with_reid:
        return tracker
    p = Path(tracker)
    if p.exists():
        return str(p)
    # Fallback: create an explicit BoT-SORT config from the bundled values.
    out = Path(output_dir) / "botsort_reid.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    model_line = f"\nmodel: {reid_model}" if reid_model else "\nmodel: auto"
    out.write_text(
        "tracker_type: botsort\n"
        "track_high_thresh: 0.50\ntrack_low_thresh: 0.10\n"
        "new_track_thresh: 0.60\ntrack_buffer: 120\nmatch_thresh: 0.80\n"
        "fuse_score: true\ngmc_method: sparseOptFlow\n"
        "proximity_thresh: 0.50\nappearance_thresh: 0.25\n"
        "with_reid: true" + model_line + "\n",
        encoding="utf-8",
    )
    return str(out)


def run_detection_tracking(
    model_path: str,
    video_path: str,
    trajectory_jsonl: str,
    tracker: str,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    with_reid: bool,
    output_dir: str,
    reid_model: Optional[str] = None,
    save_video: bool = False,
) -> pd.DataFrame:
    from ultralytics import YOLO
    if not Path(model_path).exists():
        raise FileNotFoundError(f"YOLO26 checkpoint tidak ditemukan: {model_path}")
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")

    tracker_cfg = resolve_tracker_config(tracker, with_reid, output_dir, reid_model)
    model = YOLO(model_path)
    stream = model.track(
        source=video_path,
        tracker=tracker_cfg,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        persist=True,
        stream=True,
        save=save_video,
        verbose=False,
    )
    rows = []
    with open(trajectory_jsonl, "w", encoding="utf-8") as fout:
        import json
        for frame_idx, result in enumerate(stream):
            dets = []
            if result.boxes is not None and len(result.boxes):
                ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [-1] * len(result.boxes)
                clses = result.boxes.cls.int().cpu().tolist()
                confs = result.boxes.conf.cpu().tolist()
                boxes = result.boxes.xyxy.cpu().tolist()
                names = result.names if isinstance(result.names, dict) else model.names
                for tid, cid, cf, box in zip(ids, clses, confs, boxes):
                    cname = names[int(cid)] if isinstance(names, dict) else names[int(cid)]
                    dets.append({"track_id": int(tid), "bbox": [float(v) for v in box],
                                 "class_id": int(cid), "class_name": str(cname), "confidence": float(cf)})
                    if int(tid) >= 0:
                        rows.append({"frame": frame_idx, "track_id": int(tid), "class_id": int(cid),
                                     "class_name": str(cname), "confidence": float(cf),
                                     "x1": float(box[0]), "y1": float(box[1]),
                                     "x2": float(box[2]), "y2": float(box[3])})
            fout.write(json.dumps({"frame": frame_idx, "detections": dets}, ensure_ascii=False) + "\n")
    df = pd.DataFrame(rows, columns=DETECTION_COLUMNS)
    if df.empty:
        raise RuntimeError("Tracking selesai tetapi tidak ada tracked detections.")
    return df
