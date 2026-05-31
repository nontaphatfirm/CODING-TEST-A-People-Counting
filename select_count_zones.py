import argparse
import json
from pathlib import Path

import cv2
import numpy as np


MAX_COUNT_ZONES = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Draw or edit count-zone polygons and final-frame ignore boxes.")
    parser.add_argument("--source", default="Dataset/entrance.mov", help="Input video path.")
    parser.add_argument("--output", default="count_zones.json", help="Output JSON path.")
    parser.add_argument("--frame", type=int, default=-1, help="Frame index to preview. Default uses saved frame or the middle frame.")
    parser.add_argument("--max-width", type=int, default=1280, help="Resize preview window to this width if needed.")
    return parser.parse_args()


def read_frame(source, frame_index):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index < 0:
        frame_index = max(0, total // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {source}")
    return frame, frame_index


def clamp_point(point, width, height):
    x, y = point
    return max(0, min(int(round(x)), width - 1)), max(0, min(int(round(y)), height - 1))


def valid_zone(points):
    if len(points) < 3:
        return False
    contour = np.asarray(points, dtype=np.float32)
    return abs(cv2.contourArea(contour)) >= 50


def normalize_box(box):
    x1, y1, x2, y2 = box
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def clamp_box(box, width, height):
    x1, y1, x2, y2 = normalize_box(box)
    return (
        max(0, min(int(round(x1)), width - 1)),
        max(0, min(int(round(y1)), height - 1)),
        max(0, min(int(round(x2)), width - 1)),
        max(0, min(int(round(y2)), height - 1)),
    )


def valid_box(box):
    x1, y1, x2, y2 = normalize_box(box)
    return x2 - x1 >= 8 and y2 - y1 >= 8


def load_saved_config(output_path, width, height):
    if not output_path.exists():
        return [], [], None
    data = json.loads(output_path.read_text(encoding="utf-8"))
    zones = []
    for item in data.get("count_zones", []):
        points = [
            clamp_point((point["x"], point["y"]), width, height)
            for point in item.get("points", [])
        ]
        if valid_zone(points):
            zones.append(points)
    ignore_boxes = []
    for item in data.get("final_ignore_boxes", []):
        if all(key in item for key in ("x1", "y1", "x2", "y2")):
            box = clamp_box((item["x1"], item["y1"], item["x2"], item["y2"]), width, height)
            if valid_box(box):
                ignore_boxes.append(box)
    return zones[:MAX_COUNT_ZONES], ignore_boxes, data.get("frame_index")


def point_in_zone(point, zone):
    contour = np.asarray(zone, dtype=np.int32)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def nearest_zone_handle(point, zones, hit_radius):
    best = None
    best_distance = float("inf")
    point_array = np.asarray(point, dtype=np.float32)
    for zone_index, zone in enumerate(zones):
        for point_index, vertex in enumerate(zone):
            distance = float(np.linalg.norm(point_array - np.asarray(vertex, dtype=np.float32)))
            if distance < best_distance and distance <= hit_radius:
                best = {"mode": "vertex", "zone_index": zone_index, "point_index": point_index}
                best_distance = distance

        if best is None and point_in_zone(point, zone):
            best = {"mode": "move", "zone_index": zone_index, "previous": point}
    return best


def nearest_box_handle(point, boxes, hit_radius):
    best = None
    best_distance = float("inf")
    point_array = np.asarray(point, dtype=np.float32)
    for box_index, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for corner_index, corner in enumerate(corners):
            distance = float(np.linalg.norm(point_array - np.asarray(corner, dtype=np.float32)))
            if distance < best_distance and distance <= hit_radius:
                best = {"mode": "corner", "box_index": box_index, "corner_index": corner_index}
                best_distance = distance

        if best is None and point_in_box(point, box):
            best = {"mode": "move_box", "box_index": box_index, "previous": point}
    return best


def save_zones(output_path, source, frame_index, width, height, zones, ignore_boxes):
    data = {
        "source": str(source),
        "frame_index": frame_index,
        "image_width": width,
        "image_height": height,
        "count_zones": [
            {
                "name": f"area{zone_index + 1}",
                "points": [{"x": x, "y": y} for x, y in zone],
            }
            for zone_index, zone in enumerate(zones)
        ],
        "final_ignore_boxes": [
            {
                "name": f"final_ignore{box_index + 1}",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
            for box_index, (x1, y1, x2, y2) in enumerate(ignore_boxes)
        ],
    }
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def draw_zone(canvas, zone, zone_index):
    colors = [
        (255, 0, 255),
        (0, 220, 255),
        (80, 255, 120),
        (255, 170, 40),
    ]
    color = colors[zone_index % len(colors)]
    contour = np.asarray(zone, dtype=np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [contour], color)
    cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
    cv2.polylines(canvas, [contour], True, color, 3, cv2.LINE_AA)
    for vertex in zone:
        cv2.circle(canvas, vertex, 8, color, -1)
    cv2.putText(canvas, f"Area {zone_index + 1}", (zone[0][0] + 8, zone[0][1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def draw_ignore_box(canvas, box, box_index):
    x1, y1, x2, y2 = box
    color = (80, 80, 255)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.16, canvas, 0.84, 0, canvas)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    for corner in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        cv2.circle(canvas, corner, 7, color, -1)
    cv2.putText(canvas, f"Final ignore {box_index + 1}", (x1 + 8, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main():
    args = parse_args()
    source = Path(args.source)
    output_path = Path(args.output)

    saved_frame_index = None
    if args.frame < 0 and output_path.exists():
        data = json.loads(output_path.read_text(encoding="utf-8"))
        saved_frame_index = data.get("frame_index")

    frame_index = args.frame if args.frame >= 0 else (saved_frame_index if saved_frame_index is not None else -1)
    frame, frame_index = read_frame(source, frame_index)
    height, width = frame.shape[:2]
    scale = min(1.0, args.max_width / width)
    display_size = (int(round(width * scale)), int(round(height * scale)))

    zones, ignore_boxes, _ = load_saved_config(output_path, width, height)
    draft_points = []
    draft_box = {"start": None, "current": None}
    edit_mode = "zone"
    action = {"mode": None, "edit": None}
    hit_radius = max(12, int(round(18 / max(scale, 1e-6))))
    window = "Edit count zones / final ignore boxes"

    def to_original(point):
        x, y = point
        return clamp_point((x / scale, y / scale), width, height)

    def mouse_callback(event, x, y, flags, param):
        nonlocal edit_mode
        point = to_original((x, y))
        if event == cv2.EVENT_LBUTTONDOWN:
            if edit_mode == "ignore":
                edit = nearest_box_handle(point, ignore_boxes, hit_radius)
                if edit:
                    action.update({"mode": "edit_box", "edit": edit})
                    return
                draft_box.update({"start": point, "current": point})
                action.update({"mode": "draw_box", "edit": None})
                return

            edit = nearest_zone_handle(point, zones, hit_radius)
            if edit:
                action.update({"mode": "edit_zone", "edit": edit})
                return

            draft_points.append(point)
            if len(draft_points) == 4:
                if valid_zone(draft_points):
                    if len(zones) >= MAX_COUNT_ZONES:
                        zones.pop(0)
                    zones.append(list(draft_points))
                draft_points.clear()
        elif event == cv2.EVENT_MOUSEMOVE and action["mode"] == "edit_zone":
            edit = action["edit"]
            zone_index = edit["zone_index"]
            if edit["mode"] == "vertex":
                zones[zone_index][edit["point_index"]] = point
            elif edit["mode"] == "move":
                previous = edit["previous"]
                dx = point[0] - previous[0]
                dy = point[1] - previous[1]
                zones[zone_index] = [
                    clamp_point((vx + dx, vy + dy), width, height)
                    for vx, vy in zones[zone_index]
                ]
                edit["previous"] = point
        elif event == cv2.EVENT_MOUSEMOVE and action["mode"] == "edit_box":
            edit = action["edit"]
            box_index = edit["box_index"]
            x1, y1, x2, y2 = ignore_boxes[box_index]
            if edit["mode"] == "corner":
                if edit["corner_index"] == 0:
                    x1, y1 = point
                elif edit["corner_index"] == 1:
                    x2, y1 = point
                elif edit["corner_index"] == 2:
                    x2, y2 = point
                else:
                    x1, y2 = point
                ignore_boxes[box_index] = clamp_box((x1, y1, x2, y2), width, height)
            elif edit["mode"] == "move_box":
                previous = edit["previous"]
                dx = point[0] - previous[0]
                dy = point[1] - previous[1]
                ignore_boxes[box_index] = clamp_box((x1 + dx, y1 + dy, x2 + dx, y2 + dy), width, height)
                edit["previous"] = point
        elif event == cv2.EVENT_MOUSEMOVE and action["mode"] == "draw_box":
            draft_box["current"] = point
        elif event == cv2.EVENT_LBUTTONUP and action["mode"]:
            if action["mode"] == "draw_box" and draft_box["start"] is not None:
                box = clamp_box((*draft_box["start"], *point), width, height)
                if valid_box(box):
                    ignore_boxes.append(box)
                draft_box.update({"start": None, "current": None})
            action.update({"mode": None, "edit": None})

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse_callback)

    while True:
        canvas = frame.copy()
        for zone_index, zone in enumerate(zones):
            draw_zone(canvas, zone, zone_index)
        for box_index, box in enumerate(ignore_boxes):
            draw_ignore_box(canvas, box, box_index)

        if draft_points:
            color = (255, 0, 255) if len(zones) == 0 else (0, 220, 255)
            for point in draft_points:
                cv2.circle(canvas, point, 7, color, -1)
            if len(draft_points) > 1:
                cv2.polylines(canvas, [np.asarray(draft_points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
            cv2.putText(canvas, f"Area {len(zones) + 1}: point {len(draft_points)}/4", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        if draft_box["start"] is not None and draft_box["current"] is not None:
            draw_ignore_box(canvas, clamp_box((*draft_box["start"], *draft_box["current"]), width, height), len(ignore_boxes))

        mode_text = "Mode: Count zones (z)" if edit_mode == "zone" else "Mode: Final ignore boxes (i)"
        help_text = "z/i switch | zone: click 4 points | ignore: drag box | drag corners/inside edit | s save | u undo | c clear mode | q quit"
        cv2.rectangle(canvas, (16, 16), (min(width - 16, 1160), 88), (20, 20, 20), -1)
        cv2.putText(canvas, mode_text, (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, help_text, (28, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 2)

        preview = cv2.resize(canvas, display_size, interpolation=cv2.INTER_AREA)
        cv2.imshow(window, preview)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("z"):
            edit_mode = "zone"
            draft_box.update({"start": None, "current": None})
            action.update({"mode": None, "edit": None})
        if key == ord("i"):
            edit_mode = "ignore"
            draft_points.clear()
            action.update({"mode": None, "edit": None})
        if key == ord("s"):
            if len(zones) < 2 or len(zones) % 2 != 0:
                print("Please create count zones as complete pairs, e.g. 2 or 4 zones, before saving.")
                continue
            save_zones(output_path, source, frame_index, width, height, zones, ignore_boxes)
            print(f"Saved {len(zones)} count zones and {len(ignore_boxes)} final ignore boxes to {output_path}")
            break
        if key == ord("u"):
            if draft_points:
                draft_points.pop()
            elif draft_box["start"] is not None:
                draft_box.update({"start": None, "current": None})
            elif edit_mode == "ignore" and ignore_boxes:
                ignore_boxes.pop()
            elif zones:
                zones.pop()
        if key == ord("c"):
            if edit_mode == "ignore":
                ignore_boxes.clear()
                draft_box.update({"start": None, "current": None})
            else:
                zones.clear()
                draft_points.clear()
        if key == ord("q") or key == 27:
            print("Closed without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
