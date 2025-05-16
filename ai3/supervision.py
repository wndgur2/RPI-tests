import numpy as np

class Detections:
    def __init__(self, xyxy: np.ndarray, confidence: np.ndarray, class_id: np.ndarray):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = None

class ByteTrack:
    def __init__(self, track_activation_threshold, lost_track_buffer, minimum_matching_threshold, frame_rate):
        # 간이 추적기 초기화
        self.next_id = 1
        self.tracks = []
        self.track_thresh = track_activation_threshold
        self.buffer = lost_track_buffer

    def update_with_detections(self, detections: Detections) -> Detections:
        # 단순 ID 할당 추적기 (가장 가까운 바운딩박스와 매칭)
        detections.tracker_id = np.arange(self.next_id, self.next_id + len(detections.xyxy))
        self.next_id += len(detections.xyxy)
        return detections
