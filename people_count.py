import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / "Ultralytics"))
from ultralytics import YOLO  # noqa: E402


PERSON_CLASS_ID = 0
MAX_COUNT_ZONES = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="People counting with YOLO26x, full-frame detection, and editable count-zone pairs."
    )
    parser.add_argument("--source", default="Dataset/entrance.mov", help="Input video path.")
    parser.add_argument("--detector", default="yolo26x.pt", help="YOLO detector weights.")
    parser.add_argument("--output", default="outputs/entrance_counted.mp4", help="Output video path.")
    parser.add_argument("--summary", default="outputs/entrance_count.json", help="Output JSON summary path.")
    parser.add_argument("--count-zones-file", default="count_zones.json", help="JSON file created by select_count_zones.py.")
    parser.add_argument(
        "--zone",
        action="append",
        default=[],
        metavar="X1,Y1;X2,Y2;X3,Y3;X4,Y4",
        help="Count zone polygon. Repeat for area pairs. Pixels or 0-1 ratios are supported.",
    )
    parser.add_argument("--conf", type=float, default=0.30, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.50, help="Detection NMS IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO detector image size.")
    parser.add_argument("--device", default="", help="Device for inference, e.g. cpu, 0, cuda:0.")
    parser.add_argument("--half", action="store_true", help="Use FP16 detector inference when supported.")
    parser.add_argument("--anchor", choices=("bottom", "center"), default="bottom", help="Point used for zone matching.")
    parser.add_argument("--min-track-frames", type=int, default=5, help="Frames before a track can contribute to a count.")
    parser.add_argument(
        "--first-count-min-track-frames",
        type=int,
        default=1,
        help="Minimum seen frames before an uncounted active track is counted on the first frame.",
    )
    parser.add_argument(
        "--final-count-min-track-frames",
        type=int,
        default=1,
        help="Minimum seen frames before an uncounted active track is counted on the final frame.",
    )
    parser.add_argument(
        "--final-ignore-box",
        action="append",
        default=[],
        metavar="X1,Y1,X2,Y2",
        help="Box ignored only during final-frame remaining counts. Repeat for more boxes. Pixels or 0-1 ratios are supported.",
    )
    parser.add_argument("--max-age", type=int, default=75, help="Frames to keep unmatched tracks alive.")
    parser.add_argument("--iou-weight", type=float, default=0.42, help="Predicted-box IoU weight in association cost.")
    parser.add_argument("--motion-weight", type=float, default=0.43, help="Predicted bottom-point distance weight.")
    parser.add_argument("--direction-weight", type=float, default=0.15, help="Velocity direction consistency weight.")
    parser.add_argument("--match-threshold", type=float, default=0.70, help="Maximum association cost for a match.")
    parser.add_argument("--min-iou", type=float, default=0.01, help="Minimum predicted IoU unless motion is strong.")
    parser.add_argument("--motion-gate", type=float, default=1.25, help="Maximum normalized bottom-point distance for matching.")
    parser.add_argument("--smooth-alpha", type=float, default=0.72, help="Bottom-boundary smoothing strength.")
    parser.add_argument("--drift-threshold", type=float, default=0.95, help="Normalized drift where correction snaps harder.")
    parser.add_argument("--max-match-distance", type=float, default=90.0, help="Hard pixel distance limit for matching predicted and observed foot points.")
    parser.add_argument("--border-margin-ratio", type=float, default=0.05, help="Frame-edge margin ratio used to shorten lost-track memory.")
    parser.add_argument("--border-max-age", type=int, default=5, help="Frames to keep an unmatched track alive when it is near a frame border.")
    parser.add_argument("--count-highlight-frames", type=int, default=30, help="Frames to highlight a track after it increases the count.")
    parser.add_argument("--max-frames", type=int, default=0, help="Debug limit; 0 processes the full video.")
    parser.add_argument("--progress-every", type=int, default=50, help="Print progress every N frames.")
    return parser.parse_args()


def parse_zone(raw_zone):
    points = []
    for raw_point in raw_zone.split(";"):
        parts = [part.strip() for part in raw_point.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Zone point must be X,Y: {raw_point}")
        points.append((float(parts[0]), float(parts[1])))
    if len(points) < 3:
        raise ValueError(f"Zone needs at least 3 points: {raw_zone}")
    return points


def parse_box(raw_box):
    parts = [part.strip() for part in raw_box.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Box must be X1,Y1,X2,Y2: {raw_box}")
    return tuple(float(part) for part in parts)


def resolve_zone(raw_points, width, height):
    flat_values = [value for point in raw_points for value in point]
    use_ratio = all(0.0 <= value <= 1.0 for value in flat_values)
    resolved = []
    for x, y in raw_points:
        if use_ratio:
            x *= width
            y *= height
        x = max(0, min(int(round(x)), width - 1))
        y = max(0, min(int(round(y)), height - 1))
        resolved.append((x, y))

    contour = np.asarray(resolved, dtype=np.float32)
    if abs(cv2.contourArea(contour)) < 50:
        raise ValueError(f"Zone has too little area after clipping: {raw_points}")
    return resolved


def resolve_box(raw_box, width, height):
    x1, y1, x2, y2 = raw_box
    use_ratio = all(0.0 <= value <= 1.0 for value in raw_box)
    if use_ratio:
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height

    left = max(0, min(int(round(min(x1, x2))), width - 1))
    right = max(0, min(int(round(max(x1, x2))), width - 1))
    top = max(0, min(int(round(min(y1, y2))), height - 1))
    bottom = max(0, min(int(round(max(y1, y2))), height - 1))
    if right <= left or bottom <= top:
        raise ValueError(f"Ignore box has too little area after clipping: {raw_box}")
    return (left, top, right, bottom)


def load_count_zones(args, width, height):
    raw_zones = []
    zones_file = Path(args.count_zones_file)
    if zones_file.exists():
        data = json.loads(zones_file.read_text(encoding="utf-8"))
        for item in data.get("count_zones", []):
            points = [(point["x"], point["y"]) for point in item.get("points", [])]
            raw_zones.append(points)
    raw_zones.extend(parse_zone(raw_zone) for raw_zone in args.zone)
    return [resolve_zone(points, width, height) for points in raw_zones][:MAX_COUNT_ZONES]


def load_final_ignore_boxes(args, width, height):
    raw_boxes = []
    zones_file = Path(args.count_zones_file)
    if zones_file.exists():
        data = json.loads(zones_file.read_text(encoding="utf-8"))
        for item in data.get("final_ignore_boxes", []):
            if all(key in item for key in ("x1", "y1", "x2", "y2")):
                raw_boxes.append((item["x1"], item["y1"], item["x2"], item["y2"]))
    raw_boxes.extend(parse_box(raw_box) for raw_box in args.final_ignore_box)
    return [resolve_box(raw_box, width, height) for raw_box in raw_boxes]


def box_anchor(box_xyxy, anchor):
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0
    if anchor == "center":
        return np.asarray([cx, (y1 + y2) / 2.0], dtype=np.float32)
    return np.asarray([cx, y2], dtype=np.float32)


def point_in_zone(point, zone):
    contour = np.asarray(zone, dtype=np.int32)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def locate_zone(point, zones):
    for zone_index, zone in enumerate(zones):
        if point_in_zone(point, zone):
            return zone_index
    return None


def locate_box(point, boxes):
    x, y = point
    for box_index, (x1, y1, x2, y2) in enumerate(boxes):
        if x1 <= x <= x2 and y1 <= y <= y2:
            return box_index
    return None


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_size(box):
    x1, y1, x2, y2 = box
    return np.asarray([max(1.0, x2 - x1), max(1.0, y2 - y1)], dtype=np.float32)


def box_diagonal(box):
    width, height = box_size(box)
    return float(np.hypot(width, height))


def bottom_center(box):
    x1, _, x2, y2 = box
    return np.asarray([(x1 + x2) / 2.0, y2], dtype=np.float32)


def make_box_from_bottom(bottom, size):
    width, height = size
    x_center, y_bottom = bottom
    return np.asarray(
        [x_center - width / 2.0, y_bottom - height, x_center + width / 2.0, y_bottom],
        dtype=np.float32,
    )


def normalized_bottom_distance(track_box, det_box):
    distance = float(np.linalg.norm(bottom_center(track_box) - bottom_center(det_box)))
    scale = 0.5 * (box_diagonal(track_box) + box_diagonal(det_box))
    return distance / max(1.0, scale)


def bottom_distance(box_a, box_b):
    return float(np.linalg.norm(bottom_center(box_a) - bottom_center(box_b)))


def is_near_frame_border(box, frame_size, margin_ratio):
    if frame_size is None or margin_ratio <= 0:
        return False
    frame_width, frame_height = frame_size
    margin_x = frame_width * margin_ratio
    margin_y = frame_height * margin_ratio
    x1, y1, x2, y2 = box
    return x1 <= margin_x or x2 >= frame_width - margin_x or y1 <= margin_y or y2 >= frame_height - margin_y


def direction_cost(track_velocity, previous_bottom, detection_bottom):
    if np.linalg.norm(track_velocity) < 1e-3:
        return 0.5
    det_delta = detection_bottom - previous_bottom
    det_norm = np.linalg.norm(det_delta)
    vel_norm = np.linalg.norm(track_velocity)
    if det_norm < 1e-3 or vel_norm < 1e-3:
        return 0.5
    cosine = float(np.dot(track_velocity, det_delta) / (vel_norm * det_norm))
    return (1.0 - max(-1.0, min(1.0, cosine))) / 2.0


def collect_result_boxes(result):
    boxes = []
    if result.boxes is None:
        return boxes
    for box in result.boxes.xyxy.cpu().numpy():
        boxes.append(np.asarray(box, dtype=np.float32))
    return boxes


def detect_people(frame, detector, detector_kwargs):
    result = detector.predict(frame, **detector_kwargs)[0]
    return collect_result_boxes(result)


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    smoothed_bottom: np.ndarray
    smoothed_size: np.ndarray
    velocity: np.ndarray
    first_seen: int
    last_seen: int
    seen_frames: int = 1
    missed_frames: int = 0

    @classmethod
    def create(cls, track_id, box, frame_index):
        box = np.asarray(box, dtype=np.float32)
        return cls(
            track_id=track_id,
            box=box,
            smoothed_bottom=bottom_center(box),
            smoothed_size=box_size(box),
            velocity=np.zeros(2, dtype=np.float32),
            first_seen=frame_index,
            last_seen=frame_index,
        )

    def predicted_box(self, frame_index):
        frame_delta = max(1, frame_index - self.last_seen)
        predicted_bottom = self.smoothed_bottom + self.velocity * frame_delta
        return make_box_from_bottom(predicted_bottom, self.smoothed_size)

    def update(self, detection_box, frame_index, smooth_alpha, drift_threshold):
        detection_box = np.asarray(detection_box, dtype=np.float32)
        predicted_box = self.predicted_box(frame_index)
        observed_bottom = bottom_center(detection_box)
        frame_delta = max(1, frame_index - self.last_seen)
        measured_velocity = (observed_bottom - self.smoothed_bottom) / frame_delta
        self.velocity = 0.65 * self.velocity + 0.35 * measured_velocity

        drift = normalized_bottom_distance(predicted_box, detection_box)
        alpha = smooth_alpha
        if drift > drift_threshold:
            alpha = min(alpha, 0.35)

        self.smoothed_bottom = alpha * (self.smoothed_bottom + self.velocity * frame_delta) + (1.0 - alpha) * observed_bottom
        self.smoothed_size = alpha * self.smoothed_size + (1.0 - alpha) * box_size(detection_box)
        self.box = make_box_from_bottom(self.smoothed_bottom, self.smoothed_size)
        self.last_seen = frame_index
        self.seen_frames += 1
        self.missed_frames = 0


class MotionTracker:
    def __init__(
        self,
        max_age,
        iou_weight,
        motion_weight,
        direction_weight,
        match_threshold,
        min_iou,
        motion_gate,
        smooth_alpha,
        drift_threshold,
        max_match_distance,
        border_margin_ratio,
        border_max_age,
    ):
        self.max_age = max_age
        total_weight = iou_weight + motion_weight + direction_weight
        if total_weight <= 0:
            raise ValueError("At least one association weight must be positive.")
        self.iou_weight = iou_weight / total_weight
        self.motion_weight = motion_weight / total_weight
        self.direction_weight = direction_weight / total_weight
        self.match_threshold = match_threshold
        self.min_iou = min_iou
        self.motion_gate = motion_gate
        self.smooth_alpha = smooth_alpha
        self.drift_threshold = drift_threshold
        self.max_match_distance = max_match_distance
        self.border_margin_ratio = border_margin_ratio
        self.border_max_age = border_max_age
        self.next_id = 1
        self.tracks = []

    def _association_cost(self, track, detection_box, frame_index, frame_size):
        if track.missed_frames > 0 and is_near_frame_border(track.box, frame_size, self.border_margin_ratio):
            return None

        predicted_box = track.predicted_box(frame_index)
        iou = box_iou(predicted_box, detection_box)
        pixel_distance = bottom_distance(predicted_box, detection_box)
        if self.max_match_distance > 0 and pixel_distance > self.max_match_distance:
            return None

        motion_distance = normalized_bottom_distance(predicted_box, detection_box)
        if iou < self.min_iou and motion_distance > self.motion_gate:
            return None

        direction = direction_cost(track.velocity, track.smoothed_bottom, bottom_center(detection_box))
        motion_cost = min(motion_distance, self.motion_gate) / self.motion_gate
        iou_cost = 1.0 - iou
        return self.iou_weight * iou_cost + self.motion_weight * motion_cost + self.direction_weight * direction

    def update(self, detection_boxes, frame_index, frame_size):
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detection_boxes)))
        candidates = []

        for track_index, track in enumerate(self.tracks):
            for det_index, detection_box in enumerate(detection_boxes):
                cost = self._association_cost(track, detection_box, frame_index, frame_size)
                if cost is not None and cost <= self.match_threshold:
                    candidates.append((cost, track_index, det_index))

        for _, track_index, det_index in sorted(candidates, key=lambda item: item[0]):
            if track_index not in unmatched_tracks or det_index not in unmatched_detections:
                continue
            self.tracks[track_index].update(
                detection_boxes[det_index],
                frame_index,
                self.smooth_alpha,
                self.drift_threshold,
            )
            unmatched_tracks.remove(track_index)
            unmatched_detections.remove(det_index)

        for track_index in unmatched_tracks:
            self.tracks[track_index].missed_frames += 1

        for det_index in unmatched_detections:
            self.tracks.append(Track.create(self.next_id, detection_boxes[det_index], frame_index))
            self.next_id += 1

        self.tracks = [
            track
            for track in self.tracks
            if track.missed_frames
            <= (
                self.border_max_age
                if is_near_frame_border(track.box, frame_size, self.border_margin_ratio)
                else self.max_age
            )
        ]
        return [track for track in self.tracks if track.last_seen == frame_index]


class ZonePairCounter:
    def __init__(self, zones, min_track_frames):
        self.zones = zones[:MAX_COUNT_ZONES]
        self.min_track_frames = min_track_frames
        self.first_zone_by_track = {}
        self.counted_track_ids = set()
        self.total_count = 0
        self.zone_counts = {}
        for pair_start in range(0, len(self.zones) - 1, 2):
            self.zone_counts[f"area{pair_start + 1}_to_area{pair_start + 2}"] = 0
            self.zone_counts[f"area{pair_start + 2}_to_area{pair_start + 1}"] = 0
        self.events = []
        self.first_ignored_tracks = []
        self.final_ignored_tracks = []

    @property
    def enabled(self):
        return len(self.zones) >= 2

    def pair_index_for_zone(self, zone_index):
        pair_index = zone_index // 2
        pair_start = pair_index * 2
        if pair_start + 1 >= len(self.zones):
            return None
        return pair_index

    def update(self, track, frame_index, anchor):
        if not self.enabled:
            return []

        point = box_anchor(track.box, anchor)
        zone_index = locate_zone(point, self.zones)
        if zone_index is None:
            return []

        pair_index = self.pair_index_for_zone(zone_index)
        if pair_index is None:
            return []

        first_zones = self.first_zone_by_track.setdefault(track.track_id, {})
        if pair_index not in first_zones:
            first_zones[pair_index] = zone_index
            return []

        first_zone = first_zones[pair_index]
        can_count = (
            track.track_id not in self.counted_track_ids
            and zone_index != first_zone
            and track.seen_frames >= self.min_track_frames
        )
        if not can_count:
            return []

        direction = f"area{first_zone + 1}_to_area{zone_index + 1}"
        self.zone_counts[direction] += 1
        self.total_count += 1
        self.counted_track_ids.add(track.track_id)
        event = {
            "frame": frame_index,
            "track_id": int(track.track_id),
            "pair": int(pair_index + 1),
            "from_area": first_zone + 1,
            "to_area": zone_index + 1,
            "direction": direction,
            "first_count_for_track": True,
        }
        self.events.append(event)
        return [event]

    def count_first_present(self, tracks, frame_index, anchor, min_track_frames, ignore_boxes):
        if not self.enabled:
            return []

        events = []
        for track in tracks:
            if track.track_id in self.counted_track_ids:
                continue
            if track.seen_frames < min_track_frames:
                continue

            point = box_anchor(track.box, anchor)
            ignore_box_index = locate_box(point, ignore_boxes)
            if ignore_box_index is not None:
                self.first_ignored_tracks.append(
                    {
                        "frame": frame_index,
                        "track_id": int(track.track_id),
                        "ignore_box": int(ignore_box_index + 1),
                    }
                )
                continue

            zone_index = locate_zone(point, self.zones)
            self.total_count += 1
            self.counted_track_ids.add(track.track_id)
            event = {
                "frame": frame_index,
                "track_id": int(track.track_id),
                "event_type": "first_frame_present",
                "area": int(zone_index + 1) if zone_index is not None else None,
                "first_count_for_track": True,
            }
            self.events.append(event)
            events.append(event)
        return events

    def count_final_remaining(self, tracks, frame_index, anchor, min_track_frames, final_ignore_boxes):
        if not self.enabled:
            return []

        events = []
        for track in tracks:
            if track.track_id in self.counted_track_ids:
                continue
            if track.seen_frames < min_track_frames:
                continue

            point = box_anchor(track.box, anchor)
            ignore_box_index = locate_box(point, final_ignore_boxes)
            if ignore_box_index is not None:
                self.final_ignored_tracks.append(
                    {
                        "frame": frame_index,
                        "track_id": int(track.track_id),
                        "ignore_box": int(ignore_box_index + 1),
                    }
                )
                continue

            zone_index = locate_zone(point, self.zones)
            self.total_count += 1
            self.counted_track_ids.add(track.track_id)
            event = {
                "frame": frame_index,
                "track_id": int(track.track_id),
                "event_type": "final_frame_remaining",
                "area": int(zone_index + 1) if zone_index is not None else None,
                "first_count_for_track": True,
            }
            self.events.append(event)
            events.append(event)
        return events


def color_for_track(track_id):
    value = int(track_id)
    return (
        80 + (value * 37) % 176,
        80 + (value * 17) % 176,
        80 + (value * 29) % 176,
    )


def muted_color(color):
    return tuple(int(channel * 0.45 + 40) for channel in color)


def zone_color(zone_index):
    colors = [
        (255, 0, 255),
        (0, 220, 255),
        (80, 255, 120),
        (255, 170, 40),
    ]
    return colors[zone_index % len(colors)]


def draw_label(frame, text, origin, color, font_scale, thickness):
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(0, min(x, frame.shape[1] - text_w - 4))
    y = max(text_h + baseline + 4, y)
    cv2.rectangle(frame, (x, y - text_h - baseline - 4), (x + text_w + 4, y + 2), color, -1)
    cv2.putText(frame, text, (x + 2, y - baseline), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_count_zones(frame, zones, font_scale, thickness):
    if not zones:
        return
    overlay = frame.copy()
    for zone_index, zone in enumerate(zones):
        contour = np.asarray(zone, dtype=np.int32)
        cv2.fillPoly(overlay, [contour], zone_color(zone_index))
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    for zone_index, zone in enumerate(zones):
        contour = np.asarray(zone, dtype=np.int32)
        color = zone_color(zone_index)
        cv2.polylines(frame, [contour], True, color, max(2, thickness + 1), cv2.LINE_AA)
        label_point = tuple(int(value) for value in contour[0])
        draw_label(frame, f"Area {zone_index + 1}", (label_point[0] + 8, label_point[1] + 28), color, font_scale * 0.62, max(1, thickness - 1))


def draw_final_ignore_boxes(frame, boxes, font_scale, thickness):
    for box_index, (x1, y1, x2, y2) in enumerate(boxes):
        color = (80, 80, 255)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.16, frame, 0.84, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(2, thickness + 1), cv2.LINE_AA)
        draw_label(
            frame,
            f"Final ignore {box_index + 1}",
            (x1 + 8, y1 + 28),
            color,
            font_scale * 0.62,
            max(1, thickness - 1),
        )


def draw_overlay(frame, total_count, current_people, visible_track_boxes, counter, font_scale, thickness):
    width = frame.shape[1]
    panel_w = min(width - 20, 760)
    panel_h = 126 if counter.enabled else 96
    cv2.rectangle(frame, (12, 12), (12 + panel_w, 12 + panel_h), (20, 20, 20), -1)
    cv2.rectangle(frame, (12, 12), (12 + panel_w, 12 + panel_h), (0, 220, 255), 2)
    cv2.putText(
        frame,
        f"Total count: {total_count}",
        (28, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 1.15,
        (255, 255, 255),
        max(2, thickness),
        cv2.LINE_AA,
    )
    detail = f"Tracked now: {current_people} | Visible boxes: {visible_track_boxes}"
    cv2.putText(
        frame,
        detail,
        (28, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.72,
        (220, 220, 220),
        max(1, thickness - 1),
        cv2.LINE_AA,
    )
    if counter.enabled:
        count_parts = []
        for direction, value in counter.zone_counts.items():
            label = direction.replace("area", "A").replace("_to_", "->")
            count_parts.append(f"{label}: {value}")
        counts = " | ".join(count_parts)
        cv2.putText(
            frame,
            counts,
            (28, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.62,
            (210, 210, 210),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )


def open_video_writer(output_path, fps, frame_size):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    if writer.isOpened():
        return writer, output_path
    fallback = output_path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(fallback), cv2.VideoWriter_fourcc(*"XVID"), fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path} or {fallback}")
    return writer, fallback


def main():
    args = parse_args()
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Input video not found: {source_path}")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_limit = args.max_frames if args.max_frames > 0 else total_frames
    count_zones = load_count_zones(args, width, height)
    final_ignore_boxes = load_final_ignore_boxes(args, width, height)

    writer, actual_output_path = open_video_writer(Path(args.output), fps, (width, height))
    detector = YOLO(args.detector)
    tracker = MotionTracker(
        max_age=args.max_age,
        iou_weight=args.iou_weight,
        motion_weight=args.motion_weight,
        direction_weight=args.direction_weight,
        match_threshold=args.match_threshold,
        min_iou=args.min_iou,
        motion_gate=args.motion_gate,
        smooth_alpha=args.smooth_alpha,
        drift_threshold=args.drift_threshold,
        max_match_distance=args.max_match_distance,
        border_margin_ratio=args.border_margin_ratio,
        border_max_age=args.border_max_age,
    )
    counter = ZonePairCounter(count_zones, args.min_track_frames)

    detector_kwargs = {
        "classes": [PERSON_CLASS_ID],
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device:
        detector_kwargs["device"] = args.device
    if args.half:
        detector_kwargs["half"] = True

    font_scale = max(0.65, min(width, height) / 1080.0)
    thickness = max(2, int(round(min(width, height) / 540.0)))
    stable_track_ids = set()
    count_highlight_until = {}
    track_history = {}
    frame_counts = []
    max_people_in_frame = 0
    processed_frames = 0
    started = time.monotonic()
    ok, pending_frame = cap.read()

    while ok and processed_frames < frame_limit:
        frame = pending_frame
        if processed_frames + 1 < frame_limit:
            next_ok, next_frame = cap.read()
        else:
            next_ok, next_frame = False, None

        detection_boxes = detect_people(frame, detector, detector_kwargs)
        active_tracks = tracker.update(detection_boxes, processed_frames, (width, height))
        current_people = len(active_tracks)
        visible_track_boxes = 0
        is_first_frame = processed_frames == 0
        is_final_frame = processed_frames + 1 >= frame_limit or not next_ok
        draw_count_zones(frame, count_zones, font_scale, thickness)
        if is_first_frame or is_final_frame:
            draw_final_ignore_boxes(frame, final_ignore_boxes, font_scale, thickness)

        for track in active_tracks:
            if track.seen_frames >= args.min_track_frames:
                stable_track_ids.add(track.track_id)
            events = counter.update(track, processed_frames, args.anchor)
            if events:
                count_highlight_until[track.track_id] = processed_frames + args.count_highlight_frames
            track_history[track.track_id] = {
                "first_seen": track.first_seen,
                "last_seen": track.last_seen,
                "seen_frames": track.seen_frames,
            }

        if is_first_frame:
            first_events = counter.count_first_present(
                active_tracks,
                processed_frames,
                args.anchor,
                args.first_count_min_track_frames,
                final_ignore_boxes,
            )
            for event in first_events:
                count_highlight_until[event["track_id"]] = processed_frames + args.count_highlight_frames

        if is_final_frame:
            final_events = counter.count_final_remaining(
                active_tracks,
                processed_frames,
                args.anchor,
                args.final_count_min_track_frames,
                final_ignore_boxes,
            )
            for event in final_events:
                count_highlight_until[event["track_id"]] = processed_frames + args.count_highlight_frames

        for track in active_tracks:
            point = box_anchor(track.box, args.anchor)
            zone_index = locate_zone(point, count_zones)

            is_new_count = processed_frames <= count_highlight_until.get(track.track_id, -1)
            visible_track_boxes += 1
            is_counted = track.track_id in counter.counted_track_ids if counter.enabled else track.track_id in stable_track_ids
            base_color = color_for_track(track.track_id)
            if is_new_count:
                color = (0, 0, 255)
                label = f"NEW COUNT {track.track_id}"
                box_thickness = thickness + 2
            elif is_counted:
                color = base_color
                label = f"Counted {track.track_id}"
                box_thickness = thickness
            elif counter.enabled and zone_index is not None:
                color = base_color
                label = f"Area {zone_index + 1} ID {track.track_id}"
                box_thickness = thickness
            else:
                color = muted_color(base_color)
                label = f"ID {track.track_id}"
                box_thickness = max(1, thickness - 1)

            x1, y1, x2, y2 = [int(round(value)) for value in track.box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            cv2.circle(frame, tuple(int(round(value)) for value in point), max(4, box_thickness + 2), color, -1)
            draw_label(frame, label, (x1, y1 - 8), color, font_scale * 0.62, max(1, box_thickness - 1))

        frame_counts.append(current_people)
        max_people_in_frame = max(max_people_in_frame, current_people)

        total_count = counter.total_count if counter.enabled else len(stable_track_ids)
        draw_overlay(
            frame,
            total_count,
            current_people,
            visible_track_boxes,
            counter,
            font_scale,
            thickness,
        )
        writer.write(frame)
        processed_frames += 1
        ok, pending_frame = next_ok, next_frame

        if args.progress_every > 0 and processed_frames % args.progress_every == 0:
            elapsed = time.monotonic() - started
            fps_now = processed_frames / elapsed if elapsed else 0.0
            print(f"frame {processed_frames}/{frame_limit} | count={total_count} | {fps_now:.2f} FPS", flush=True)

    cap.release()
    writer.release()

    total_count = counter.total_count if counter.enabled else len(stable_track_ids)
    summary = {
        "source": str(source_path),
        "detector": args.detector,
        "uses_yolo_track": False,
        "tracker": "custom_ocsort_style_motion_tracker",
        "counting_mode": "zone_pair_sequence_unique_tracks" if counter.enabled else "stable_unique_tracks",
        "association": {
            "iou_weight": args.iou_weight,
            "motion_weight": args.motion_weight,
            "direction_weight": args.direction_weight,
            "match_threshold": args.match_threshold,
            "min_iou": args.min_iou,
            "motion_gate": args.motion_gate,
            "smooth_alpha": args.smooth_alpha,
            "drift_threshold": args.drift_threshold,
            "max_match_distance": args.max_match_distance,
            "border_margin_ratio": args.border_margin_ratio,
            "border_max_age": args.border_max_age,
            "max_age": args.max_age,
        },
        "counting": {
            "total_count": total_count,
            "count_highlight_frames": args.count_highlight_frames,
            "min_track_frames": args.min_track_frames,
            "first_count_min_track_frames": args.first_count_min_track_frames,
            "first_frame_present_count": sum(
                1 for event in counter.events if event.get("event_type") == "first_frame_present"
            ),
            "first_ignored_track_count": len(counter.first_ignored_tracks),
            "first_ignored_tracks": counter.first_ignored_tracks,
            "final_count_min_track_frames": args.final_count_min_track_frames,
            "final_frame_remaining_enabled": counter.enabled,
            "final_frame_remaining_count": sum(
                1 for event in counter.events if event.get("event_type") == "final_frame_remaining"
            ),
            "final_ignored_track_count": len(counter.final_ignored_tracks),
            "final_ignored_tracks": counter.final_ignored_tracks,
            "events": counter.events,
            "zone_counts": counter.zone_counts,
            "counted_track_ids": sorted(int(track_id) for track_id in counter.counted_track_ids),
            "first_zone_by_track": {
                str(track_id): {
                    str(pair_index + 1): int(zone_index + 1)
                    for pair_index, zone_index in sorted(first_zones.items())
                }
                for track_id, first_zones in sorted(counter.first_zone_by_track.items())
            },
            "box_display_mode": "all_active_tracks_full_frame_with_new_count_highlight",
            "stable_track_ids": sorted(int(track_id) for track_id in stable_track_ids),
        },
        "count_zones": [
            {
                "name": f"area{zone_index + 1}",
                "points": [{"x": int(x), "y": int(y)} for x, y in zone],
            }
            for zone_index, zone in enumerate(count_zones)
        ],
        "final_ignore_boxes": [
            {
                "name": f"final_ignore{box_index + 1}",
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
            }
            for box_index, (x1, y1, x2, y2) in enumerate(final_ignore_boxes)
        ],
        "output_video": str(actual_output_path),
        "video": {
            "width": width,
            "height": height,
            "fps": fps,
            "frames_in_source": total_frames,
            "frames_processed": processed_frames,
            "max_people_in_frame": max_people_in_frame,
            "average_people_in_frame": sum(frame_counts) / len(frame_counts) if frame_counts else 0.0,
        },
        "track_frames": {str(track_id): values for track_id, values in sorted(track_history.items())},
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Done. Total people counted: {total_count}", flush=True)
    print(f"Output video: {actual_output_path}", flush=True)
    print(f"Summary JSON: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
