# live_main.py
import threading
import time
from urllib.request import urlopen

import cv2
import numpy as np

from config import ReIDConfig
from detector import PersonDetector
from feature_extractor import FeatureManager
from tracker import PersonTracker
from visualizer import Visualizer

# ---- EDIT THESE: use the /shot.jpg link instead of /video for each phone ----
CAMERA_URLS = {
    1: "http://192.168.0.106:8080/shot.jpg",
    2: "http://192.168.0.102:8080/shot.jpg",
}
# -------------------------------------------------------------------------

PROCESS_WIDTH = 1020     # size used only for AI detection (speed)
DISPLAY_HEIGHT = 680    # size used only for what you see on screen (quality)

stop_event = threading.Event()
latest_frames = {}
frames_lock = threading.Lock()


def fetch_snapshot(url):
    with urlopen(url, timeout=2) as response:
        data = response.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def run_camera(camera_id, url, config, feature_manager):
    detector = PersonDetector(config)
    tracker = PersonTracker(config, feature_manager)

    frame_id = 0
    known_before = set()
    while not stop_event.is_set():
        try:
            frame = fetch_snapshot(url)
        except Exception:
            time.sleep(0.3)
            continue

        if frame is None:
            continue

        h, w = frame.shape[:2]
        scale = PROCESS_WIDTH / w
        small = cv2.resize(frame, (PROCESS_WIDTH, int(h * scale)))

        raw_detections = detector.detect(small)
        scale_back = 1 / scale
        detections = [
            ([x1 * scale_back, y1 * scale_back, x2 * scale_back, y2 * scale_back], feat)
            for (x1, y1, x2, y2), feat in raw_detections
        ]

        results = tracker.process_detections(detections, frame_id, camera_id)
        display_frame = Visualizer.draw_results(frame, results)

        with frames_lock:
            latest_frames[camera_id] = display_frame

        now_ids = {track_id for _, track_id in results if track_id is not None}
        newly_matched = now_ids - known_before
        for track_id in newly_matched:
            print(f"Camera {camera_id}: person ID {track_id} is visible here now")
        known_before = now_ids

        frame_id += 1


def display_loop():
    window_name = "Cameras"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    while not stop_event.is_set():
        with frames_lock:
            frames = [latest_frames[cid] for cid in sorted(latest_frames) if cid in latest_frames]

        if len(frames) >= 1:
            resized = [cv2.resize(f, (int(f.shape[1] * DISPLAY_HEIGHT / f.shape[0]), DISPLAY_HEIGHT)) for f in frames]
            combined = np.hstack(resized) if len(resized) > 1 else resized[0]
            cv2.imshow(window_name, combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()


def main():
    config = ReIDConfig(
        yolo_model='./models/yolov8n.pt',
        reid_model='osnet_ain_x1_0',
        device='cpu',
        waiting_frames=5,
        max_features_per_id=10,
        min_bbox_size=100,
        iou_threshold=0.4,
        feature_match_threshold=0.65,
        max_frames_missing=40
    )
    from pathlib import Path
    feature_manager = FeatureManager(config, db_path=Path('live_reid_database.json'))
    feature_manager.feature_db = {}
    feature_manager.next_id = 1

    threads = []
    for camera_id, url in CAMERA_URLS.items():
        t = threading.Thread(target=run_camera, args=(camera_id, url, config, feature_manager), daemon=True)
        t.start()
        threads.append(t)

    time.sleep(1)
    display_loop()

    for t in threads:
        t.join(timeout=2)

    feature_manager.save_database()
    print("\nStopped. Final known people:")
    for obj_id, entries in sorted(feature_manager.feature_db.items()):
        cameras = sorted({e['camera_id'] for e in entries})
        print(f"  ID {obj_id}: seen in camera(s) {cameras}")


if __name__ == "__main__":
    main()