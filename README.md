# People Counting with YOLO26x + Count-Zone Pairs

This project detects people across the full frame with YOLO26x, tracks them with a custom OC-SORT-style motion tracker, then counts each tracked person once when their bottom-center point moves across a count-zone pair such as `area1` to `area2`, or `area3` to `area4`.

It does not use `YOLO.track()`, and it does not split enter/exit counts. Both directions are combined into one total.

## 1. Draw Count Zones

Draw editable polygon pairs around entrance/exit floor areas. `area1/area2` is the first entrance, and `area3/area4` is the second entrance. You can also draw first/final-frame ignore boxes in the same tool.

```powershell
python select_count_zones.py
```

Controls:

- `z`: edit count zones
- `i`: edit final-frame ignore boxes
- Click 4 points: create one area polygon. Create zones in pairs: `area1/area2`, then `area3/area4`.
- Drag with the mouse in ignore mode: create one final ignore box
- Drag a corner: edit that corner
- Drag inside an area or ignore box: move it
- `s`: save to `count_zones.json`
- `u`: undo last point, last area, or last ignore box in the current mode
- `c`: clear the current mode
- `q` or `Esc`: quit without saving

You can choose another preview frame:

```powershell
python select_count_zones.py --frame 1278
```

## 2. Run Counting

```powershell
python people_count.py --device 0 --half
```

Outputs:

- `outputs/entrance_counted.mp4`
- `outputs/entrance_count.json`

The script automatically loads `count_zones.json`, including ignore boxes drawn in `select_count_zones.py`. YOLO always detects people across the full frame, and the output video shows boxes for every active track in the full frame. Count zones are used only for counting.

## Counting Logic

- Detection: YOLO26x detects people in the full frame.
- Tracking point: the bottom-center of the person box is treated as the foot point.
- First frame flush: on the first processed frame, any visible active `track_id` is added to the total once. By default this needs only 1 seen frame, controlled by `--first-count-min-track-frames`.
- First area touch: when a new `track_id` first steps into one area of a pair, that first area is stored for that pair.
- Count event: if the same `track_id` later steps into the other area of the same pair, total count increases by 1. Supported pairs are `area1/area2` and `area3/area4`.
- Final frame flush: on the last processed frame, any still-visible active `track_id` that has not been counted yet is added to the total once. By default this needs only 1 seen frame, controlled by `--final-count-min-track-frames`.
- First/final ignore boxes: `--final-ignore-box` skips selected areas only during the first-frame and final-frame flushes. It does not affect detection, tracking, or normal area crossing counts during the video.
- No double count: once a `track_id` has been counted, it will not be counted again even if it walks back and forth while still using the same ID.

In the output video, boxes and labels are drawn for every active track. A track that just increases the count is highlighted in red for `--count-highlight-frames` frames.

## Approach

The system separates detection, tracking, and counting. YOLO26x runs on the full video frame so people can be detected before they reach the entrance zones. A custom motion tracker then assigns each person a `track_id` using bounding-box overlap, bottom-point motion, and direction consistency. Counting is not based on detector boxes entering a cropped ROI; instead, the detector sees the full frame and the counting zones are used only as geometric tests.

Each person's bottom-center point is treated as the foot position. When that point first touches one area in a configured pair, the system stores the first area for that `track_id`. If the same `track_id` later touches the other area in the same pair, the total count increases once. The same ID is never counted again, even if it walks back and forth.

The first and last frame have extra handling. On the first frame, visible people can be counted immediately because they may already be in the scene before the video starts. On the final frame, remaining visible people that were never counted are added once because they may not have had enough time to cross a zone before the video ends. Ignore boxes can exclude selected areas from only these first/final-frame flushes.

## Limitations

- The tracker is motion-based and does not use ReID appearance features, so heavy occlusion or groups wearing similar clothing can still cause ID switches.
- If YOLO misses a person for too long, the tracker may lose the original ID and create a new one when the person reappears.
- Counting depends on the bottom-center point, so inaccurate boxes, reflections, partial bodies, or unusual camera angles can place the foot point in the wrong zone.
- First-frame and final-frame flushes are useful for incomplete clips, but they can overcount people who are visible in the scene but should not be part of the entrance count. Use ignore boxes to exclude those areas.
- Zone quality matters. Poorly drawn or overlapping areas can produce early, late, or missed counts.
- The system is designed to count entrance/exit door traffic for event flow analysis. It is not intended to count every person visible in the whole scene unless they are part of the configured door traffic.

## Tuning

```powershell
python people_count.py --device 0 --half --count-highlight-frames 45
python people_count.py --device 0 --half --motion-gate 1.10 --match-threshold 0.62
python people_count.py --device 0 --half --max-match-distance 80 --border-max-age 4
python people_count.py --device 0 --half --smooth-alpha 0.80 --drift-threshold 0.80
python people_count.py --device 0 --half --final-ignore-box "0.80,0.00,1.00,1.00"
```

To reduce ID swaps near frame edges, the tracker uses three guards:

- hard bottom-point distance limit with `--max-match-distance`
- shorter lost-track memory near frame borders with `--border-max-age`
- border gating that prevents already-lost edge tracks from matching a newly entering person

You can pass zones directly if needed:

```powershell
python people_count.py --device 0 --half --zone "100,700;450,700;450,850;100,850" --zone "500,700;850,700;850,850;500,850" --zone "900,700;1200,700;1200,850;900,850" --zone "1250,700;1550,700;1550,850;1250,850"
```

## Notes

- Detector: `yolo26x.pt`
- Tracking is implemented in `people_count.py`, not by Ultralytics tracking.
- `total_count` is the number of unique tracks that moved from one area to the other area within a count-zone pair.
- Zone membership uses `cv2.pointPolygonTest` against the selected foot point.
