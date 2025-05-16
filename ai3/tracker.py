from supervision import ByteTrack, Detections

class WaspTracker:
    def __init__(self, track_thresh=0.25, track_buffer=30):
        self.tracker = ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )

    def update(self, detections: Detections) -> Detections:
        return self.tracker.update_with_detections(detections)
