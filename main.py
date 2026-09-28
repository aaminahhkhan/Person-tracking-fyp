# main.py
import argparse
import re
from collections import defaultdict
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from config import ReIDConfig
from detector import PersonDetector
from feature_extractor import FeatureManager
from tracker import PersonTracker
from visualizer import Visualizer


def publish_live_frame(endpoint, camera_id, frame):
    encoded_successfully, encoded_frame = cv2.imencode(
        '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75]
    )
    if not encoded_successfully:
        return

    request = Request(
        f'{endpoint}?{urlencode({"camera_id": camera_id})}',
        data=encoded_frame.tobytes(),
        headers={'Content-Type': 'image/jpeg'},
        method='POST'
    )
    try:
        with urlopen(request, timeout=1):
            pass
    except (OSError, URLError):
        pass


def make_private_feature_manager(config, db_path):
    """A fresh, empty identity list used for ONE camera only."""
    manager = FeatureManager(config, db_path=db_path)
    manager.feature_db = {}
    manager.next_id = 1
    return manager


def prototype(entries):
    """One 'average look' vector for a person, built from their stored appearances."""
    vectors = np.stack([np.asarray(e['feature'], dtype=np.float32) for e in entries])
    proto = vectors.mean(axis=0)
    norm = np.linalg.norm(proto)
    return proto / norm if norm > 0 else proto


def track_one_camera(camera_id, video_path, config, detector, project_dir, live_url):
    """PASS 1: follow people inside a single camera. Identities here are local
    to this camera only, so nothing from another camera can confuse them."""
    local_manager = make_private_feature_manager(config, project_dir / 'reid_database.json')
    tracker = PersonTracker(config, local_manager)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for camera {camera_id}: {video_path}")

    frame_results = []
    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = detector.detect(frame)
            results = tracker.process_detections(detections, frame_id, camera_id)
            frame_results.append(results)
            if live_url:
                preview = Visualizer.draw_results(frame.copy(), results)
                publish_live_frame(live_url, camera_id, preview)
            frame_id += 1
    finally:
        cap.release()

    return frame_results, local_manager


def clean_camera_identities(frame_results, local_manager, floor):
    """Keep the people who were really tracked in this camera. Very short-lived
    IDs (brief glitches) are merged into a real person only if they never appear
    in the same frame as that person; otherwise they are dropped."""
    support = defaultdict(int)
    together = defaultdict(set)
    for results in frame_results:
        ids = {track_id for _, track_id in results if track_id is not None}
        for track_id in ids:
            support[track_id] += 1
            together[track_id].update(ids - {track_id})

    if not support:
        return {}, {}

    min_support = max(5, int(0.2 * max(support.values())))
    real_ids = [
        t for t in sorted(support)
        if support[t] >= min_support and local_manager.feature_db.get(t)
    ]
    protos = {t: prototype(local_manager.feature_db[t]) for t in real_ids}

    merged = {}
    for track_id in support:
        if track_id in protos or not local_manager.feature_db.get(track_id):
            continue
        glitch_proto = prototype(local_manager.feature_db[track_id])
        best_id, best_sim = None, floor
        for real_id in real_ids:
            if real_id in together[track_id]:
                continue
            sim = float(np.dot(glitch_proto, protos[real_id]))
            if sim >= best_sim:
                best_id, best_sim = real_id, sim
        if best_id is not None:
            merged[track_id] = best_id

    return protos, merged


def match_people_across_cameras(camera_protos, floor):
    """PASS 2: decide which people in different cameras are the same person.
    For each new camera we score every (person here) x (known person) pair and
    pick the best one-to-one pairing overall, so two people in one camera can
    never receive the same global ID."""
    known_looks = {}
    id_map = {}
    next_global_id = 1

    for camera_id in sorted(camera_protos):
        protos = camera_protos[camera_id]
        local_ids = sorted(protos)
        if not local_ids:
            continue

        known_ids = sorted(known_looks)
        chosen = {}
        if known_ids:
            sim = np.zeros((len(local_ids), len(known_ids)), dtype=np.float32)
            for i, local_id in enumerate(local_ids):
                for j, global_id in enumerate(known_ids):
                    sim[i, j] = max(
                        float(np.dot(protos[local_id], look))
                        for look in known_looks[global_id]
                    )
            print(f"\nCamera {camera_id}: similarity to people already known")
            print("  rows = people in this camera (local IDs " + str(local_ids) + ")")
            print("  cols = known global IDs " + str(known_ids))
            print(np.round(sim, 2))

            rows, cols = linear_sum_assignment(-sim)
            for r, c in zip(rows, cols):
                if sim[r, c] >= floor:
                    chosen[local_ids[r]] = known_ids[c]

        for local_id in local_ids:
            global_id = chosen.get(local_id)
            if global_id is None:
                global_id = next_global_id
                next_global_id += 1
            next_global_id = max(next_global_id, global_id + 1)
            id_map[(camera_id, local_id)] = global_id
            known_looks.setdefault(global_id, []).append(protos[local_id])

    return id_map


def render_camera(camera_id, video_path, frame_results, final_ids, output_dir, live_url, frames_seen):
    """PASS 3: draw the final identities onto the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video for camera {camera_id}: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    output_path = output_dir / f'output_camera_{camera_id}.mp4'
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not initialize output video writer: {output_path}")

    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id < len(frame_results):
                drawn = []
                for bbox, local_id in frame_results[frame_id]:
                    if local_id is None:
                        drawn.append((bbox, None))
                    elif local_id in final_ids:
                        drawn.append((bbox, final_ids[local_id]))
                        frames_seen[final_ids[local_id]][camera_id] += 1
                frame = Visualizer.draw_results(frame, drawn)
            out.write(frame)
            if live_url:
                publish_live_frame(live_url, camera_id, frame)
            frame_id += 1
    finally:
        cap.release()
        out.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, help='Project name under projects/')
    parser.add_argument('--live-frame-url', help='Local endpoint for publishing annotated preview frames')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.project):
        parser.error('project name must start with a letter or number and contain only letters, numbers, underscores, or hyphens')

    project_dir = Path(__file__).resolve().parent / 'projects' / args.project
    video_dir = project_dir / 'videos'
    output_dir = project_dir / 'outputs'
    video_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    # Two people count as the same person only if their similarity is at least this.
    cross_camera_floor = config.feature_match_threshold - 0.15

    detector = PersonDetector(config)
    video_paths = sorted(
        (
            path for path in video_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv'}
        ),
        key=lambda path: [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', path.name)
        ]
    )
    if not video_paths:
        raise FileNotFoundError(f"No video files found in: {video_dir}")

    print("Pass 1: following people inside each camera separately...")
    camera_results = {}
    camera_managers = {}
    camera_protos = {}
    camera_merged = {}
    for camera_id, video_path in enumerate(video_paths, start=1):
        print(f"  camera {camera_id}: {video_path.name}")
        frame_results, local_manager = track_one_camera(
            camera_id, video_path, config, detector, project_dir, args.live_frame_url
        )
        protos, merged = clean_camera_identities(frame_results, local_manager, cross_camera_floor)
        camera_results[camera_id] = frame_results
        camera_managers[camera_id] = local_manager
        camera_protos[camera_id] = protos
        camera_merged[camera_id] = merged

    print("\nPass 2: matching people across cameras...")
    id_map = match_people_across_cameras(camera_protos, cross_camera_floor)

    final_ids = {}
    for camera_id in camera_results:
        per_camera = {local: g for (cam, local), g in id_map.items() if cam == camera_id}
        for glitch_id, real_id in camera_merged[camera_id].items():
            if real_id in per_camera:
                per_camera[glitch_id] = per_camera[real_id]
        final_ids[camera_id] = per_camera

    print("\nPass 3: drawing final IDs onto the videos...")
    frames_seen = defaultdict(lambda: defaultdict(int))
    for camera_id, video_path in enumerate(video_paths, start=1):
        render_camera(
            camera_id, video_path, camera_results[camera_id],
            final_ids[camera_id], output_dir, args.live_frame_url, frames_seen
        )

    global_manager = make_private_feature_manager(config, project_dir / 'reid_database.json')
    for (camera_id, local_id), global_id in id_map.items():
        entries = camera_managers[camera_id].feature_db.get(local_id, [])
        global_manager.feature_db.setdefault(global_id, []).extend(entries)
    global_manager.next_id = max(global_manager.feature_db, default=0) + 1
    global_manager.save_database()

    print('\nGlobal ID | Cameras seen in | Frames visible per camera')
    print('----------+-----------------+--------------------------')
    for global_id in sorted(frames_seen):
        cameras = sorted(frames_seen[global_id])
        detail = ', '.join(f"cam{c}: {frames_seen[global_id][c]}" for c in cameras)
        print(f"{global_id:9} | {', '.join(map(str, cameras)):15} | {detail}")


if __name__ == "__main__":
    main()