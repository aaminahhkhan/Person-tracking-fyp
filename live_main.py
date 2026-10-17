# live_main.py
import threading
import time
from urllib.request import urlopen

import cv2
import numpy as np
import torch
from config import ReIDConfig
from detector import PersonDetector
from live_tracker import GlobalIdentities, LiveTracker
from visualizer import Visualizer

# ---- EDIT THESE: use the /shot.jpg link instead of /video for each phone ----
CAMERA_URLS = {
    1: "http://10.0.66.169:8080/shot.jpg",
    2: 0,
}
# -------------------------------------------------------------------------

PROCESS_WIDTH = 1020     # size used only for AI detection (speed)
DISPLAY_HEIGHT = 680    # size used only for what you see on screen (quality)

stop_event = threading.Event()
latest_frames = {}
frames_lock = threading.Lock()


# def fetch_snapshot(url):
#     with urlopen(url, timeout=2) as response:
#         data = response.read()
#     arr = np.frombuffer(data, dtype=np.uint8)
#     frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#     return frame
captures = {}

def fetch_snapshot(source):
    if isinstance(source, str) and source.startswith("http"):
        with urlopen(source, timeout=2) as response:
            data = response.read()
        return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    # laptop webcam (0) or a video file path
    cap = captures.get(source)
    if cap is None:
        cap = cv2.VideoCapture(source)        
        captures[source] = cap
    ok, frame = cap.read()
    if not ok:
        if isinstance(source, str):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the video
        return None
    return frame


def run_camera(camera_id, url, config, identities):
    detector = PersonDetector(config)
    tracker = LiveTracker(config, identities, camera_id)

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

        # track on the small frame (so size rules are consistent), show on the full frame
        results_small = tracker.update(detector.detect_full(small), frame_id, small.shape)
        scale_back = 1 / scale
        results = [((x1 * scale_back, y1 * scale_back, x2 * scale_back, y2 * scale_back), tid)
                   for (x1, y1, x2, y2), tid in results_small]
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
    # ALL thresholds live in config.py now. Only the model names are chosen here.
    torch.set_num_threads(4)
    config = ReIDConfig(yolo_model='yolo26n.pt', device='cpu')
    identities = GlobalIdentities(config)

    threads = []
    for camera_id, url in CAMERA_URLS.items():
        t = threading.Thread(target=run_camera, args=(camera_id, url, config, identities), daemon=True)
        t.start()
        threads.append(t)

    time.sleep(1)
    display_loop()

    for t in threads:
        t.join(timeout=2)

    print("\nStopped. Final known people:")
    for gid, cams in sorted(identities.protos.items()):
        if gid not in identities.redirect:
            print(f"  ID {gid}: seen in camera(s) {sorted(cams)}")


if __name__ == "__main__":
    main()