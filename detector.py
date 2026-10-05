# detector.py
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
import numpy as np
from config import ReIDConfig
from utils import calculate_iou

class PersonDetector:
    def __init__(self, config: ReIDConfig):
        self.config = config
        self.yolo = YOLO(config.yolo_model)
        self.extractor = FeatureExtractor(
            model_name=config.reid_model,
            device=config.device
        )

    def extract_features(self, frame, bbox):
        try:
            x1, y1, x2, y2 = map(int, bbox)
            height, width = y2 - y1, x2 - x1
            area = height * width
            
            if area < 4000 or height < 30 or width < 15:
                print(f"REJECTED: box too small — height={height}, width={width}, area={area}")
                return None
                
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
                
            features = self.extractor(crop)
            features_np = features.cpu().numpy().flatten()
            features_normalized = features_np / np.linalg.norm(features_np)
            return features_normalized
        except Exception as e:
            print(f"Feature extraction failed: {str(e)}")
            return None

    def detect(self, frame):
        detections = self.yolo(frame, classes=0, verbose=False)[0]
        
        raw_boxes = []
        for det in detections.boxes.data:
            x1, y1, x2, y2, conf, _ = det
            if conf < self.config.detection_conf_threshold:
                continue
            raw_boxes.append(([float(x1), float(y1), float(x2), float(y2)], float(conf)))

        # Remove duplicate boxes on the same real person (heavily overlapping boxes)
        raw_boxes.sort(key=lambda item: item[1], reverse=True)
        deduped_boxes = []
        for bbox, conf in raw_boxes:
            is_duplicate = False
            for kept_bbox, _ in deduped_boxes:
                if calculate_iou(bbox, kept_bbox) > 0.5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduped_boxes.append((bbox, conf))

        valid_detections = []
        for bbox, conf in deduped_boxes:
            features = self.extract_features(frame, bbox)
            if features is not None:
                valid_detections.append((bbox, features))
                
        return valid_detections