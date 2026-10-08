"""YOLO inference and basic PPE/hazard analysis helpers.

This module contains prototype utilities. It does not provide a validated
worker-safety system; suitable model weights and project data are external.
"""

import logging
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def _load_model(model_path: str) -> YOLO:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"YOLO model weights not found: {path}")
    return YOLO(str(path))


def _validate_confidence(conf_threshold: float) -> None:
    if not 0.0 <= conf_threshold <= 1.0:
        raise ValueError("conf_threshold must be between 0 and 1.")


def _serialize_result(result, image_path: str) -> Dict:
    detections = []
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            nx1, ny1, nx2, ny2 = box.xyxyn[0].tolist()
            class_id = int(box.cls[0].item())
            width, height = x2 - x1, y2 - y1
            detections.append({
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": round(float(box.conf[0].item()), 3),
                "bbox_xyxy": {
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                },
                "bbox_normalized": {
                    "x1": round(float(nx1), 4), "y1": round(float(ny1), 4),
                    "x2": round(float(nx2), 4), "y2": round(float(ny2), 4),
                },
                "dimensions": {
                    "width": int(width), "height": int(height),
                    "area": int(width * height),
                },
            })
    return {
        "image_path": str(image_path),
        "image_shape": tuple(int(value) for value in result.orig_shape),
        "detections": detections,
    }


def detect_single_image(
    image_path: str, model_path: str, conf_threshold: float = 0.5
) -> Dict:
    """Run YOLO on one image and return detections as plain Python values."""
    _validate_confidence(conf_threshold)
    image = Path(image_path)
    if not image.is_file():
        raise FileNotFoundError(f"Image not found: {image}")
    model = _load_model(model_path)
    result = model.predict(source=str(image), conf=conf_threshold, verbose=False)[0]
    return _serialize_result(result, str(image))


def draw_detections(
    image_path: str, model_path: str, output_path: str,
    conf_threshold: float = 0.5,
) -> None:
    """Save an annotated copy of an image."""
    _validate_confidence(conf_threshold)
    image = Path(image_path)
    if not image.is_file():
        raise FileNotFoundError(f"Image not found: {image}")
    model = _load_model(model_path)
    result = model.predict(source=str(image), conf=conf_threshold, verbose=False)[0]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), result.plot()):
        raise OSError(f"Could not write annotated image to {output}")


def detect_batch_images(
    image_folder: str, model_path: str, conf_threshold: float = 0.5
) -> List[Dict]:
    """Run inference over supported image files beneath a folder."""
    _validate_confidence(conf_threshold)
    folder = Path(image_folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Image folder not found: {folder}")
    paths = sorted(
        path for path in folder.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    )
    model = _load_model(model_path)
    return [
        _serialize_result(
            model.predict(source=str(path), conf=conf_threshold, verbose=False)[0],
            str(path),
        )
        for path in paths
    ]


def detect_video(
    video_path: str, model_path: str, output_path: str,
    conf_threshold: float = 0.5,
) -> None:
    """Write an annotated video, releasing both OpenCV handles on errors."""
    _validate_confidence(conf_threshold)
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    model = _load_model(model_path)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video: {video}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Video has invalid FPS or frame dimensions.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise OSError(f"Could not create output video: {output}")

    try:
        frame_count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(source=frame, conf=conf_threshold, verbose=False)[0]
            writer.write(result.plot())
            frame_count += 1
        if frame_count == 0:
            raise ValueError(f"Video contains no readable frames: {video}")
    finally:
        capture.release()
        writer.release()
    logger.info("Annotated video saved to %s", output)


def filter_detections_by_class(detections: Dict, target_classes: List[str]) -> Dict:
    """Return a copy containing detections whose class names are requested."""
    filtered = dict(detections)
    filtered["detections"] = [
        detection for detection in detections.get("detections", [])
        if detection.get("class_name") in target_classes
    ]
    return filtered


def check_ppe_compliance(
    detections: Dict, ppe_classes: Dict[str, List[str]], padding: int = 50
) -> Dict:
    """Apply a simple bounding-box proximity heuristic to worker/PPE detections.

    This is a prototype heuristic; PPE association should be validated against
    labeled images before any operational use.
    """
    workers = [
        detection for detection in detections.get("detections", [])
        if detection.get("class_name") == "worker"
    ]
    expected_items = ppe_classes.get("worker", [])
    ppe_items = [
        detection for detection in detections.get("detections", [])
        if detection.get("class_name") in expected_items
    ]
    non_compliant = []
    for worker in workers:
        box = worker["bbox_xyxy"]
        nearby = set()
        for item in ppe_items:
            pbox = item["bbox_xyxy"]
            inside_expanded_worker = (
                pbox["x1"] >= box["x1"] - padding
                and pbox["x2"] <= box["x2"] + padding
                and pbox["y1"] >= box["y1"] - padding
                and pbox["y2"] <= box["y2"] + padding
            )
            if inside_expanded_worker:
                nearby.add(item["class_name"])
        missing = [item for item in expected_items if item not in nearby]
        if missing:
            non_compliant.append({
                "worker_bbox": box,
                "confidence": worker.get("confidence"),
                "missing_ppe": missing,
            })
    return {
        "image_path": detections.get("image_path"),
        "total_workers": len(workers),
        "workers_with_ppe": len(workers) - len(non_compliant),
        "workers_without_ppe": len(non_compliant),
        "non_compliant_workers": non_compliant,
    }


def check_hazard_zone_violation(detections: Dict, hazard_zones: List[Dict]) -> Dict:
    """Flag worker centers that fall inside configured rectangular zones."""
    violations = []
    workers = [
        detection for detection in detections.get("detections", [])
        if detection.get("class_name") == "worker"
    ]
    for worker in workers:
        box = worker["bbox_xyxy"]
        center_x = (box["x1"] + box["x2"]) / 2
        center_y = (box["y1"] + box["y2"]) / 2
        for zone in hazard_zones:
            if zone["x1"] <= center_x <= zone["x2"] and zone["y1"] <= center_y <= zone["y2"]:
                violations.append({
                    "worker_bbox": box,
                    "worker_confidence": worker.get("confidence"),
                    "hazard_zone": zone.get("name", "unnamed"),
                    "severity": zone.get("severity", "medium"),
                    "worker_position": {"center_x": int(center_x), "center_y": int(center_y)},
                })
    return {"image_path": detections.get("image_path"), "violations": violations}


class WorkerTracker:
    """Greedy nearest-centroid tracker for prototype video analysis."""

    def __init__(self, max_distance: float = 50.0, max_missed_frames: int = 30):
        if max_distance <= 0 or max_missed_frames < 0:
            raise ValueError("Tracking distance must be positive and missed frames non-negative.")
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.tracked_workers = {}
        self.next_id = 0

    def update(self, detections: Dict, frame_id: int) -> Dict:
        workers = [
            detection for detection in detections.get("detections", [])
            if detection.get("class_name") == "worker"
        ]
        active = {
            worker_id: track for worker_id, track in self.tracked_workers.items()
            if frame_id - track["last_frame"] <= self.max_missed_frames
        }
        updated = dict(detections)
        output_detections = [
            detection for detection in detections.get("detections", [])
            if detection.get("class_name") != "worker"
        ]
        matched_ids = set()

        for worker in workers:
            box = worker["bbox_xyxy"]
            center = {
                "x": (box["x1"] + box["x2"]) / 2,
                "y": (box["y1"] + box["y2"]) / 2,
            }
            candidates = []
            for worker_id, track in active.items():
                if worker_id in matched_ids:
                    continue
                previous = track["positions"][-1]
                distance = float(np.hypot(center["x"] - previous["x"], center["y"] - previous["y"]))
                if distance <= self.max_distance:
                    candidates.append((distance, worker_id))
            if candidates:
                _, track_id = min(candidates)
            else:
                track_id = self.next_id
                self.next_id += 1
                active[track_id] = {"positions": [], "frames": [], "last_frame": frame_id}
            matched_ids.add(track_id)
            track = active[track_id]
            track["positions"].append({**center, "frame": frame_id})
            track["frames"].append(frame_id)
            track["last_frame"] = frame_id
            output_detections.append({**worker, "track_id": track_id})

        self.tracked_workers = active
        updated["detections"] = output_detections
        return updated

    def get_dwell_time(self, worker_id: int, fps: float = 30.0) -> float:
        """Estimate observed track duration in seconds from first to last frame."""
        if fps <= 0:
            raise ValueError("fps must be greater than zero.")
        track = self.tracked_workers.get(worker_id)
        if not track or not track["frames"]:
            return 0.0
        return (track["frames"][-1] - track["frames"][0] + 1) / fps
