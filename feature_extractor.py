# feature_manager.py
import json
import threading
from pathlib import Path

import numpy as np

from config import ReIDConfig


class FeatureManager:
    def __init__(self, config: ReIDConfig, db_path: Path = Path('reid_database.json')):
        self.config = config
        self.db_path = db_path
        self.feature_db = {}
        self.next_id = 1
        self.lock = threading.Lock()
        self.load_database()

    def load_database(self):
        self.feature_db = {}
        self.next_id = 1

        if not self.db_path.exists():
            try:
                with open(self.db_path, 'w') as f:
                    json.dump({}, f)
            except Exception as e:
                print(f"Warning: Could not create database file ({str(e)})")
            return

        try:
            with open(self.db_path, 'r') as f:
                loaded_db = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
            print(f"Warning: Could not read database file ({str(e)})")
            loaded_db = {}

        for key, entries in loaded_db.items():
            try:
                obj_id = int(key)
            except (TypeError, ValueError):
                continue

            normalized_entries = []
            for entry in entries:
                if isinstance(entry, dict):
                    vector = entry.get('feature')
                    camera_id = entry.get('camera_id')
                else:
                    vector = entry
                    camera_id = None
                if vector is not None:
                    normalized_entries.append({
                        'camera_id': camera_id,
                        'feature': np.asarray(vector, dtype=np.float32)
                    })

            self.feature_db[obj_id] = normalized_entries
            self.next_id = max(self.next_id, obj_id + 1)

    def save_database(self):
        with self.lock:
            try:
                serializable_db = {
                    str(k): [
                        {
                            'camera_id': entry['camera_id'],
                            'feature': entry['feature'].tolist()
                        }
                        for entry in entries
                    ]
                    for k, entries in self.feature_db.items()
                }
                with open(self.db_path, 'w') as f:
                    json.dump(serializable_db, f)
            except Exception as e:
                print(f"Warning: Could not save database ({str(e)})")

    def match_or_create(self, query_features, exclude_ids=None, min_similarity=None):
        """Check for a match AND (if needed) reserve a new ID, as ONE uninterruptible
        step. This is what stops two cameras from both deciding 'new person' at the
        same instant for the same real person."""
        with self.lock:
            query_vector = np.asarray(query_features, dtype=np.float32)
            if min_similarity is None:
                min_similarity = self.config.feature_match_threshold
            exclude_ids = exclude_ids or set()

            best_similarity = 0.0
            best_id = None
            for obj_id, entries in self.feature_db.items():
                if obj_id in exclude_ids:
                    continue
                obj_best = 0.0
                for entry in entries:
                    similarity = float(np.dot(query_vector, np.asarray(entry['feature'], dtype=np.float32)))
                    obj_best = max(obj_best, similarity)
                if obj_best > best_similarity:
                    best_similarity = obj_best
                    best_id = obj_id

            if best_similarity >= min_similarity:
                return best_id, best_similarity

            new_id = self.next_id
            self.next_id += 1
            self.feature_db[new_id] = []
            return new_id, best_similarity

    def match_features(self, query_features, min_similarity=None, exclude_ids=None, margin=0.0):
        with self.lock:
            query_vector = np.asarray(query_features, dtype=np.float32)
            if min_similarity is None:
                min_similarity = self.config.feature_match_threshold
            exclude_ids = exclude_ids or set()

            best_similarity = 0.0
            best_id = None
            second_best_similarity = 0.0

            for obj_id, entries in self.feature_db.items():
                if obj_id in exclude_ids:
                    continue
                obj_best = 0.0
                for entry in entries:
                    similarity = float(np.dot(query_vector, np.asarray(entry['feature'], dtype=np.float32)))
                    obj_best = max(obj_best, similarity)
                if obj_best > best_similarity:
                    second_best_similarity = best_similarity
                    best_similarity = obj_best
                    best_id = obj_id
                elif obj_best > second_best_similarity:
                    second_best_similarity = obj_best

            if best_similarity < min_similarity:
                return None, best_similarity

            if margin > 0 and best_similarity - second_best_similarity < margin:
                return None, best_similarity

            return best_id, best_similarity

    def update_feature_array(self, obj_id, new_features, camera_id):
        with self.lock:
            if obj_id not in self.feature_db:
                self.feature_db[obj_id] = []

            new_vector = np.asarray(new_features, dtype=np.float32)
            should_add = True
            max_similarity = 0.0

            for entry in self.feature_db[obj_id]:
                existing_features = entry['feature']
                similarity = float(np.dot(new_vector, np.asarray(existing_features, dtype=np.float32)))
                max_similarity = max(max_similarity, similarity)
                if similarity > 0.95:
                    should_add = False

            has_camera_entry = any(
                entry['camera_id'] == camera_id for entry in self.feature_db[obj_id]
            )
            if not has_camera_entry:
                should_add = True

            if not should_add and max_similarity < 0.99:
                should_add = True

            if should_add:
                if len(self.feature_db[obj_id]) >= self.config.max_features_per_id:
                    camera_entry_counts = {}
                    for entry in self.feature_db[obj_id]:
                        entry_camera_id = entry['camera_id']
                        camera_entry_counts[entry_camera_id] = camera_entry_counts.get(entry_camera_id, 0) + 1

                    removable_index = next(
                        (
                            index for index, entry in enumerate(self.feature_db[obj_id])
                            if camera_entry_counts[entry['camera_id']] > 1
                        ),
                        None
                    )
                    if removable_index is not None:
                        self.feature_db[obj_id].pop(removable_index)
                self.feature_db[obj_id].append({
                    'camera_id': camera_id,
                    'feature': new_vector
                })

    def get_next_id(self):
        with self.lock:
            current_id = self.next_id
            self.next_id += 1
            return current_id