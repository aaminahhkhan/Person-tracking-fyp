# detector.py
import cv2
import numpy as np
from ultralytics import YOLO
from torchreid.utils import FeatureExtractor
from config import ReIDConfig
from utils import calculate_iou


def brighten_if_dark(frame, dark_mean):
    """Dark scene? Lift the shadows (gamma + CLAHE) so YOLO and the re-ID model can see people."""
    mean = float(frame.mean())
    if mean >= dark_mean:
        return frame
    gamma = min(3.0, max(1.0, np.log(max(mean, 1.0) / 255) / np.log(110 / 255)))
    table = (255 * (np.arange(256) / 255) ** (1 / gamma)).astype(np.uint8)
    out = cv2.LUT(frame, table)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class PersonDetector:
    def __init__(self, config: ReIDConfig):
        self.config = config
        self.yolo = YOLO(config.yolo_model)
        self.extractor = FeatureExtractor(model_name=config.reid_model, device=config.device)

    def extract_features(self, frame, bbox):
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(x1, 0), max(y1, 0)
            height, width = y2 - y1, x2 - x1
            if height < 30 or width < 15:
                return None
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            f = self.extractor(crop).cpu().numpy().flatten()
            return f / np.linalg.norm(f)
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            return None

    def detect(self, frame):
        """Older API (recorded-video main.py): list of (bbox, features)."""
        return [(d['bbox'], d['feat']) for d in self.detect_full(frame)
                if d['feat'] is not None and d['conf'] >= self.config.detection_conf_threshold]

    def detect_full(self, frame):
        """Live API: list of dicts with bbox, feat, conf. Weak detections are kept (low conf);
        the tracker decides whether they may start a NEW person."""
        work = brighten_if_dark(frame, self.config.dark_mean_brightness)
        res = self.yolo(work, classes=0, conf=self.config.detect_low_conf, verbose=False)[0]
        boxes = sorted(
            [([float(b[0]), float(b[1]), float(b[2]), float(b[3])], float(b[4])) for b in res.boxes.data],
            key=lambda x: x[1], reverse=True)
        kept = []
        for bbox, conf in boxes:
            if all(calculate_iou(bbox, k[0]) <= 0.5 for k in kept):
                kept.append((bbox, conf))
        return [{'bbox': b, 'conf': c, 'feat': self.extract_features(work, b)} for b, c in kept]