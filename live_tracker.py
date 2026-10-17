# live_tracker.py  -- tracking + identity logic for live cameras
import threading
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils import calculate_iou


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class GlobalIdentities:
    """One shared 'address book' for all cameras: one look-alike vector per person per camera."""

    def __init__(self, config):
        self.cfg = config
        self.lock = threading.RLock()
        self.protos = {}        # gid -> {camera_id: unit vector}
        self.redirect = {}      # merged gid -> surviving gid
        self.together = set()   # pairs of IDs seen at the same moment in the same camera
        self.next_id = 1

    def resolve(self, gid):
        while gid in self.redirect:
            gid = self.redirect[gid]
        return gid

    def match_or_create(self, vec, camera_id, exclude=()):
        """Find the best known person (or reserve a new ID) in ONE uninterruptible step."""
        with self.lock:
            exclude = {self.resolve(g) for g in exclude}
            best_gid, best_score, best_sim = None, -1.0, 0.0
            for gid, cams in self.protos.items():
                if gid in exclude:
                    continue
                for cam, p in cams.items():
                    sim = float(np.dot(vec, p))
                    need = self.cfg.feature_match_threshold - (0 if cam == camera_id else self.cfg.cross_camera_drop)
                    if sim - need > best_score:
                        best_gid, best_score, best_sim = gid, sim - need, sim
            if best_gid is not None and best_score >= 0:
                return best_gid, best_sim
            gid = self.next_id
            self.next_id += 1
            self.protos[gid] = {}
            return gid, best_sim

    def update(self, gid, camera_id, vec):
        with self.lock:
            gid = self.resolve(gid)
            cams = self.protos.setdefault(gid, {})
            cams[camera_id] = _unit(0.85 * cams[camera_id] + 0.15 * vec) if camera_id in cams else vec.copy()

    def note_together(self, gids):
        with self.lock:
            gids = sorted({self.resolve(g) for g in gids})
            for i in range(len(gids)):
                for j in range(i + 1, len(gids)):
                    self.together.add((gids[i], gids[j]))

    def try_merge(self):
        """Two IDs that were NEVER seen together but look almost identical = one fragmented person."""
        with self.lock:
            gids = sorted(g for g in self.protos if g not in self.redirect and self.protos[g])
            for i in range(len(gids)):
                for j in range(i + 1, len(gids)):
                    a, b = gids[i], gids[j]
                    if a in self.redirect or b in self.redirect:
                        continue
                    pair = tuple(sorted((a, b)))
                    if pair in self.together:
                        continue
                    best = max(
                        float(np.dot(pa, pb)) - (0 if ca == cb else self.cfg.cross_camera_drop)
                        for ca, pa in self.protos[a].items() for cb, pb in self.protos[b].items()
                    )
                    if best >= self.cfg.feature_match_threshold + self.cfg.merge_extra:
                        for cam, p in self.protos[b].items():
                            self.protos[a][cam] = _unit(self.protos[a][cam] + p) if cam in self.protos[a] else p
                        self.redirect[b] = a
                        self.together = {
                            tuple(sorted((a if x == b else x, a if y == b else y))) for x, y in self.together
                        }
                        print(f"[merge] ID {b} was the same person as ID {a}")


class Track:
    def __init__(self, tid, bbox, feat, frame_id):
        self.tid = tid
        self.bbox = np.array(bbox, dtype=np.float32)
        self.vel = np.zeros(4, dtype=np.float32)
        self.feat = feat.copy() if feat is not None else None
        self.gid = None
        self.hits = 1
        self.missed = 0
        self.seen_frame = frame_id


class LiveTracker:
    def __init__(self, config, identities, camera_id):
        self.cfg = config
        self.ids = identities
        self.cam = camera_id
        self.tracks = []
        self.next_tid = 1
        self.frame_count = 0

    # ---------- helpers ----------
    def _predicted(self, t):
        return t.bbox + t.vel * min(t.missed + 1, 8)

    @staticmethod
    def _overlap_frac(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
        return inter / area

    def _good_new_person(self, d, frame_h):
        x1, y1, x2, y2 = d['bbox']
        h, w = y2 - y1, x2 - x1
        return (d['conf'] >= self.cfg.new_track_conf
                and h >= self.cfg.min_height_frac * frame_h
                and h / max(w, 1) >= self.cfg.min_aspect)

    # ---------- main step ----------
    def update(self, dets, frame_id, frame_shape):
        cfg = self.cfg
        frame_h, frame_w = frame_shape[:2]
        self.frame_count += 1

        # 1) which detections are partly hidden behind / in front of another person?
        for i, d in enumerate(dets):
            d['occluded'] = any(self._overlap_frac(d['bbox'], o['bbox']) > 0.15
                                for j, o in enumerate(dets) if j != i)
            x1, y1, x2, y2 = d['bbox']
            d['edge'] = x1 < 3 or y1 < 3 or x2 > frame_w - 3 or y2 > frame_h - 3
            d['clean'] = (not d['occluded'] and not d['edge'] and d['conf'] >= cfg.new_track_conf
                          and d['feat'] is not None)

        # 2) match existing tracks (even ones that disappeared a while ago) to detections
        T, D = len(self.tracks), len(dets)
        matches = {}
        if T and D:
            score = np.full((T, D), -1.0)
            for ti, t in enumerate(self.tracks):
                pred = self._predicted(t)
                for di, d in enumerate(dets):
                    iou = calculate_iou(list(pred), d['bbox'])
                    have_app = t.feat is not None and d['feat'] is not None
                    sim = float(np.dot(t.feat, d['feat'])) if have_app else 0.0
                    if d['occluded'] or not have_app:
                        s = 0.8 * iou + 0.2 * sim if have_app else iou
                        ok = iou >= 0.1
                    else:
                        s = 0.5 * iou + 0.5 * sim
                        ok = iou >= 0.1 or (t.missed > 0 and sim >= cfg.relink_similarity)
                    if ok and s >= 0.25:
                        score[ti, di] = s
            rows, cols = linear_sum_assignment(-score)
            for r, c in zip(rows, cols):
                if score[r, c] > 0:
                    matches[c] = r

        # 3) update matched tracks
        matched_tracks = set()
        for di, ti in matches.items():
            t, d = self.tracks[ti], dets[di]
            new_box = np.array(d['bbox'], dtype=np.float32)
            t.vel = 0.7 * t.vel + 0.3 * (new_box - t.bbox) / max(1, t.missed + 1)
            t.bbox = new_box
            t.hits += 1
            t.missed = 0
            t.seen_frame = frame_id
            matched_tracks.add(ti)
            if d['clean']:     # only learn the person's look from clean, unobstructed views
                t.feat = d['feat'].copy() if t.feat is None else _unit(0.8 * t.feat + 0.2 * d['feat'])
                if t.gid is not None and t.hits % 3 == 0:
                    self.ids.update(t.gid, self.cam, t.feat)

        # 4) new tentative tracks (strict quality gate)
        for di, d in enumerate(dets):
            if di in matches:
                continue
            if self._good_new_person(d, frame_h) and d['feat'] is not None:
                t = Track(self.next_tid, d['bbox'], d['feat'] if d['clean'] else None, frame_id)
                self.next_tid += 1
                self.tracks.append(t)
                matches[di] = len(self.tracks) - 1
                matched_tracks.add(len(self.tracks) - 1)

        # 5) unmatched tracks age; forget very old ones
        survivors = []
        for ti, t in enumerate(self.tracks):
            if ti not in matched_tracks:
                t.missed += 1
            limit = cfg.lost_keep_frames if t.gid is not None else 8
            if t.missed <= limit:
                survivors.append(t)
        self.tracks = survivors
        # 6) confirm tentative tracks -> give them an ID (one-to-one inside this camera)
        for t in self.tracks:
            if t.gid is None and t.hits >= cfg.confirm_hits and t.feat is not None and t.missed == 0:
                in_use = {x.gid for x in self.tracks if x.gid is not None and x.missed == 0}
                t.gid, sim = self.ids.match_or_create(t.feat, self.cam, exclude=in_use)
                self.ids.update(t.gid, self.cam, t.feat)
                print(f"Camera {self.cam}: person gets ID {t.gid} (best similarity {sim:.2f})")

        # 7) housekeeping + output
        visible = [self.ids.resolve(t.gid) for t in self.tracks if t.gid is not None and t.missed == 0]
        if len(visible) > 1:
            self.ids.note_together(visible)
        if self.frame_count % 30 == 0:
            self.ids.try_merge()

        out = []
        for t in self.tracks:
            if t.missed == 0:
                out.append((tuple(t.bbox), self.ids.resolve(t.gid) if t.gid is not None else None))
        return out