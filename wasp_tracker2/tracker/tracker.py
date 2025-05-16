from supervision import ByteTrack, Detections
import numpy as np
import time

class WaspTracker:
    def __init__(self, track_thresh=0.5, track_buffer=30):
        self.tracker = ByteTrack(track_activation_threshold=track_thresh, lost_track_buffer=track_buffer)

    def update(self, detections):
        if not detections:
            print("[TRACKER] No detections passed to tracker.")
            return Detections.empty()

        boxes = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, *_ in detections], dtype=np.float32)
        scores = np.array([conf for *_, conf, _ in detections], dtype=np.float32)
        class_ids = np.array([cls for *_, cls in detections], dtype=int)

        tracked = self.tracker.update_with_detections(
            Detections(xyxy=boxes, confidence=scores, class_id=class_ids)
        )
        print(f"[TRACKER] {len(tracked.xyxy)} objects tracked.")\

        return tracked
