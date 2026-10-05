# tracker.py
from collections import defaultdict
import numpy as np
from config import ReIDConfig
from feature_extractor import FeatureManager
from utils import calculate_iou

class PersonTracker:
    def __init__(self, config: ReIDConfig, feature_manager: FeatureManager):
        self.config = config
        self.feature_manager = feature_manager
        self.detection_history = {}
        self.waiting_detections = defaultdict(list)
        self.previous_detection_count = 0
        self.detection_change_frame = 0
        self.is_in_waiting_period = False
       

    def check_detection_count_change(self, current_count, frame_id):
        if current_count != self.previous_detection_count:
            self.is_in_waiting_period = True
            self.detection_change_frame = frame_id
            self.waiting_detections.clear()
            self.previous_detection_count = current_count
            return True
        return False

    def update_missing_tracks(self):
        for track_id in list(self.detection_history.keys()):
            self.detection_history[track_id]['frames_missing'] += 1
            if self.detection_history[track_id]['frames_missing'] > self.config.max_frames_missing:
                del self.detection_history[track_id]

    def process_detections(self, valid_detections, frame_id, camera_id):
        results = []
        current_detection_count = len(valid_detections)
        detection_count_changed = self.check_detection_count_change(current_detection_count, frame_id)

        self.update_missing_tracks()

        if self.is_in_waiting_period:
            results.extend(self._process_waiting_period(valid_detections, frame_id, camera_id))
        else:
            results.extend(self._process_normal_period(valid_detections, camera_id))

        return results

    def _process_waiting_period(self, valid_detections, frame_id, camera_id):
        results = []
        for bbox, features in valid_detections:
            detection_key = tuple(map(int, bbox))
            self.waiting_detections[detection_key].append({
                'frame_id': frame_id,
                'features': features,
                'bbox': bbox,
                'camera_id': camera_id
            })
            results.append((bbox, None))
            
        if frame_id - self.detection_change_frame >= self.config.waiting_frames:
            results.extend(self._process_waiting_detections(camera_id))
            self.is_in_waiting_period = False
            
        return results

    def _process_waiting_detections(self, camera_id):
        results = []
        used_ids_this_batch = set()
        for det_key, det_history in self.waiting_detections.items():
            if len(det_history) >= self.config.waiting_frames * 0.8:
                avg_features = np.mean([d['features'] for d in det_history], axis=0)
                avg_features = avg_features / np.linalg.norm(avg_features)

                matched_id, similarity = self.feature_manager.match_or_create(avg_features, exclude_ids=used_ids_this_batch)
                print(f"Camera {camera_id}: best similarity = {similarity:.2f}, assigned ID: {matched_id}")

                used_ids_this_batch.add(matched_id)
                self.feature_manager.update_feature_array(matched_id, avg_features, camera_id)
                latest_detection = det_history[-1]
                self.detection_history[matched_id] = {
                    'bbox': latest_detection['bbox'],
                    'features': latest_detection['features'],
                    'camera_id': camera_id,
                    'frames_missing': 0
                }
                results.append((latest_detection['bbox'], matched_id))

        self.waiting_detections.clear()
        return results

    def _process_normal_period(self, valid_detections, camera_id):
        results = []
        track_ids = list(self.detection_history.keys())

        pairs = []
        for det_idx, (bbox, features) in enumerate(valid_detections):
            for track_id in track_ids:
                track_info = self.detection_history[track_id]
                iou = calculate_iou(bbox, track_info['bbox'])
                appearance_similarity = float(np.dot(
                    np.asarray(features, dtype=np.float32),
                    np.asarray(track_info['features'], dtype=np.float32)
                ))
                combined_score = (0.5 * iou) + (0.5 * appearance_similarity)
                pairs.append((combined_score, det_idx, track_id))

        pairs.sort(key=lambda p: p[0], reverse=True)
        assigned_dets = set()
        assigned_tracks = set()
        det_to_track = {}
        for score, det_idx, track_id in pairs:
            if det_idx in assigned_dets or track_id in assigned_tracks:
                continue
            if score < 0.3:
                continue
            det_to_track[det_idx] = track_id
            assigned_dets.add(det_idx)
            assigned_tracks.add(track_id)

        used_ids_this_frame = set()
        for det_idx, (bbox, features) in enumerate(valid_detections):
            matched_id = det_to_track.get(det_idx)
            if matched_id is not None:
                self._update_track(matched_id, bbox, features, camera_id)
            else:
                matched_id = self._create_new_track(bbox, features, camera_id, used_ids_this_frame)
            used_ids_this_frame.add(matched_id)
            results.append((bbox, matched_id))
        return results

    def _update_track(self, track_id, bbox, features, camera_id):
        self.detection_history[track_id] = {
            'bbox': bbox,
            'features': features,
            'camera_id': camera_id,
            'frames_missing': 0
        }
        self.feature_manager.update_feature_array(track_id, features, camera_id)

    def _create_new_track(self, bbox, features, camera_id, used_ids=None):
        matched_id, similarity = self.feature_manager.match_or_create(features, exclude_ids=used_ids)
        print(f"Camera {camera_id}: best similarity = {similarity:.2f}, assigned ID: {matched_id}")
        self._update_track(matched_id, bbox, features, camera_id)
        return matched_id