from dataclasses import dataclass

@dataclass
class ReIDConfig:
    # ---- models ----
    yolo_model: str = 'yolo26s.pt'
    reid_model: str = 'osnet_ain_x1_0'
    device: str = 'cpu'

    # ---- detection ----
    detect_low_conf: float = 0.15      # weak detections: allowed ONLY to continue an existing person
    new_track_conf: float = 0.45       # a NEW person must be detected at least this confidently
    min_height_frac: float = 0.20      # a new person's box must be >= 20% of the frame height
    min_aspect: float = 1.2            # new person's box height/width (rejects close-ups, door/cloth pieces)
    dark_mean_brightness: float = 100  # frame darker than this (0-255) gets brightened before detection

    # ---- tracking ----
    confirm_hits: int = 5              # frames a new person must be seen before getting an ID
    lost_keep_frames: int = 150        # remember a person who disappeared (occlusion) for this many frames
    relink_similarity: float = 0.60    # a reappearing person is re-linked to a lost track at this look-alike score
    max_features_per_id: int = 10

    # ---- identity matching ----
    feature_match_threshold: float = 0.65   # same camera: "this is the same person" score
    cross_camera_drop: float = 0.10         # other camera is allowed to be this much lower (lighting differs)
    merge_extra: float = 0.10               # two IDs are merged only if this much ABOVE the threshold
    # kept for the older recorded-video main.py
    detection_conf_threshold: float = 0.15
    waiting_frames: int = 2
    min_bbox_size: int = 50
    iou_threshold: float = 0.9
    max_frames_missing: int = 40