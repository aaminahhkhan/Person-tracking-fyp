# main.py
import cv2
from pathlib import Path

from config import ReIDConfig
from detector import PersonDetector
from feature_extractor import FeatureManager
from tracker import PersonTracker
from visualizer import Visualizer


def main():
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
    feature_manager = FeatureManager(config)
    detector = PersonDetector(config)
    tracker = PersonTracker(config, feature_manager)
    display_id_map = {}
    next_display_id = [1]

    def get_display_id(real_id):
        if real_id not in display_id_map:
            display_id_map[real_id] = next_display_id[0]
            next_display_id[0] += 1
        return display_id_map[real_id]

    # Video setup
    video_path = Path('./videos/1.mp4')
    output_path = Path('output.mp4')

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    if not out.isOpened():
        raise RuntimeError(f"Could not initialize output video writer: {output_path}")

    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            valid_detections = detector.detect(frame)
            results = tracker.process_detections(valid_detections, frame_id)

            # Visualize results
            display_results = [
                (bbox, get_display_id(obj_id) if obj_id is not None else None)
                for bbox, obj_id in results
            ]
            frame = Visualizer.draw_results(frame, display_results)

            out.write(frame)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_id += 1
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        feature_manager.save_database()


if __name__ == "__main__":
    main()