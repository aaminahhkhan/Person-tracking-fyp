# main.py
import argparse
import cv2
import re
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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

    # Configuration
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

    # Initialize components
    feature_manager = FeatureManager(config, db_path=project_dir / 'reid_database.json')
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

    for camera_id, video_path in enumerate(video_paths, start=1):
        tracker = PersonTracker(config, feature_manager)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video for camera {camera_id}: {video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        output_path = output_dir / f'output_camera_{camera_id}.mp4'
        out = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        if not out.isOpened():
            cap.release()
            raise RuntimeError(f"Could not initialize output video writer: {output_path}")

        frame_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                valid_detections = detector.detect(frame)
                results = tracker.process_detections(valid_detections, frame_id, camera_id)
                frame = Visualizer.draw_results(frame, results)
                out.write(frame)
                if args.live_frame_url:
                    publish_live_frame(args.live_frame_url, camera_id, frame)

                frame_id += 1
        finally:
            cap.release()
            out.release()

    feature_manager.save_database()

    print('\nGlobal ID | Cameras seen in | Total embeddings')
    print('----------+-----------------+-----------------')
    for global_id, entries in sorted(feature_manager.feature_db.items()):
        cameras = sorted({entry['camera_id'] for entry in entries if entry['camera_id'] is not None})
        print(f"{global_id:9} | {', '.join(map(str, cameras)) or 'unknown':15} | {len(entries):16}")


if __name__ == "__main__":
    main()