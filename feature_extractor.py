# feature_manager.py
import json
from pathlib import Path

import numpy as np

from config import ReIDConfig


class FeatureManager:
    def __init__(self, config: ReIDConfig, db_path: Path = Path('reid_database.json')):
        self.config = config
        self.db_path = db_path
        self.feature_db = {}
        self.next_id = 1
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

        for key, vectors in loaded_db.items():
            try:
                obj_id = int(key)
            except (TypeError, ValueError):
                continue

            self.feature_db[obj_id] = [
                np.asarray(vector, dtype=np.float32) for vector in vectors
            ]
            self.next_id = max(self.next_id, obj_id + 1)

    def save_database(self):
        try:
            serializable_db = {
                str(k): [v.tolist() for v in vectors]
                for k, vectors in self.feature_db.items()
            }
            with open(self.db_path, 'w') as f:
                json.dump(serializable_db, f)
        except Exception as e:
            print(f"Warning: Could not save database ({str(e)})")

    def match_features(self, query_features, min_similarity=None):
        query_vector = np.asarray(query_features, dtype=np.float32)
        if min_similarity is None:
            min_similarity = self.config.feature_match_threshold

        max_similarity = 0.0
        matched_id = None

        for obj_id, stored_features in self.feature_db.items():
            for features in stored_features:
                similarity = float(np.dot(query_vector, np.asarray(features, dtype=np.float32)))
                if similarity > max_similarity:
                    max_similarity = similarity
                    matched_id = obj_id

        if max_similarity < min_similarity:
            return None, max_similarity

        return matched_id, max_similarity

    def update_feature_array(self, obj_id, new_features):
        if obj_id not in self.feature_db:
            self.feature_db[obj_id] = []

        new_vector = np.asarray(new_features, dtype=np.float32)
        should_add = True
        max_similarity = 0.0

        for existing_features in self.feature_db[obj_id]:
            similarity = float(np.dot(new_vector, np.asarray(existing_features, dtype=np.float32)))
            max_similarity = max(max_similarity, similarity)
            if similarity > 0.95:
                should_add = False

        if not should_add and max_similarity < 0.99:
            should_add = True

        if should_add:
            if len(self.feature_db[obj_id]) >= self.config.max_features_per_id:
                self.feature_db[obj_id].pop(0)
            self.feature_db[obj_id].append(new_vector)

    def get_next_id(self):
        current_id = self.next_id
        self.next_id += 1
        return current_id