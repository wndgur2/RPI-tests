from supervision import ByteTrack, Detections

class WaspTracker:
    def __init__(self, track_thresh=0.25, track_buffer=30):
        """
        track_thresh: 새 트랙을 시작할 신뢰도 임계값.
        track_buffer: 객체가 사라진 후 ID를 유지할 프레임 수.
        """
        self.tracker = ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=0.8,  # 필요시 조정
            frame_rate=30                     # 사용 FPS
        )
        
    def update(self, detections: Detections) -> Detections:
        """
        최신 말벌 검출 결과를 받아 트래커 업데이트.
        반환된 Detections 객체에 tracker_id 속성이 포함됩니다.
        """
        return self.tracker.update_with_detections(detections)
