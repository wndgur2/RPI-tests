import os
os.environ["HAILORT_CONSOLE_LOGGER_LEVEL"] = "error"
os.environ["HAILORT_LOGGER_PATH"] = "NONE"


import cv2
import numpy as np
import threading
from flask import Flask, Response

from oak_camera import OakCamera
from hailo_yolo import HailoYoloDetector
from tracker import WaspTracker
from supervision import Detections

# 클래스 목록 및 모델 경로
CLASSES = ["Bee", "Wasp"]
MODEL_PATH = "/home/ssafy/project/RPI-tests/ai/models/yolov8n_level0.hef"

# 객체 초기화
oak = OakCamera(resolution=(640, 480), fps=30)
detector = HailoYoloDetector(model_path=MODEL_PATH, classes=CLASSES,
                             conf_threshold=0.25, iou_threshold=0.45)
tracker = WaspTracker(track_thresh=0.25, track_buffer=30)

# Flask 스트리밍 준비
app = Flask(__name__)
last_frame = None
frame_lock = threading.Lock()

def detection_loop():
    global last_frame
    try:
        frame_idx = 0
        while True:
            # 프레임 캡처
            frame, depth = oak.get_frames()
            # 검출
            dets = detector.detect(frame)
            print(f"[DEBUG] frame {frame_idx}: {len(dets)} detections")
            frame_idx += 1
            wasp_dets, bee_dets = [], []
            for x1, y1, x2, y2, score, cls_id in dets:
                if score < 0.4: continue
                if CLASSES[cls_id] == "Wasp":
                    wasp_dets.append((x1, y1, x2, y2, score, cls_id))
                else:
                    bee_dets.append((x1, y1, x2, y2, score, cls_id))

            # 말벌 트래킹
            tracked_wasps = None
            if wasp_dets:
                boxes = np.array([[d[0], d[1], d[2], d[3]] for d in wasp_dets], dtype=np.float32)
                scores = np.array([d[4] for d in wasp_dets], dtype=np.float32)
                class_ids = np.zeros(len(wasp_dets), dtype=int)
                det_obj = Detections(xyxy=boxes, confidence=scores, class_id=class_ids)
                tracked_wasps = tracker.update(det_obj)

            # 결과 그리기
            frame_draw = frame.copy()
            # 꿀벌 (파란색)
            for x1, y1, x2, y2, score, _ in bee_dets:
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame_draw, f"Bee ({score*100:.1f}%)",
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            # 말벌 (빨간색) 및 3D 좌표
            if tracked_wasps is not None:
                for i, box in enumerate(tracked_wasps.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    track_id = int(tracked_wasps.tracker_id[i])
                    cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"Wasp #{track_id}"
                    cv2.putText(frame_draw, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # 3D 좌표 계산
                    cx = x1 + (x2-x1)//2
                    cy = y1 + (y2-y1)//2
                    Z = float(depth[cy, cx])  # mm 단위
                    if Z > 0:
                        X = (cx - oak.c_x) * Z / oak.f_x
                        Y = (cy - oak.c_y) * Z / oak.f_y
                        X_m, Y_m, Z_m = X/1000.0, Y/1000.0, Z/1000.0
                        coord_text = f"ID {track_id}: X={X_m:.2f}m, Y={Y_m:.2f}m, Z={Z_m:.2f}m"
                        cv2.putText(frame_draw, coord_text, (x1, y2+15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                        print(f"Wasp ID {track_id}: X={X_m:.2f} m, Y={Y_m:.2f} m, Z={Z_m:.2f} m")

            # 전역 프레임 업데이트
            with frame_lock:
                last_frame = frame_draw

    finally:
        oak.close()
        detector.close()

# Flask 라우트
@app.route('/')
def index():
    return '<h3>Wasp/Bee Detection Stream</h3><img src="/video_feed" width="640" height="480" />'

@app.route('/video_feed')
def video_feed():
    return Response(frame_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def frame_generator():
    global last_frame
    while True:
        with frame_lock:
            if last_frame is None: continue
            frame = last_frame.copy()
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buffer.tobytes() + b'\r\n')

# 백그라운드에서 검출 루프 시작
threading.Thread(target=detection_loop, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
