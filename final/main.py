import os
import cv2
import threading
import numpy as np
from flask import Flask, Response
from oak_camera import OakCamera
from hailo_yolo import HailoYoloDetector
from tracker import WaspTracker
from supervision import Detections

CLASSES = ["Bee", "Wasp"]
MODEL_PATH = "/home/ssafy/project/RPI-tests/ai/models/yolov8n_level0.hef"

oak = OakCamera(resolution=(640, 480), fps=30)
detector = HailoYoloDetector(MODEL_PATH, CLASSES, conf_threshold=0.25, iou_threshold=0.45)
tracker = WaspTracker(track_thresh=0.25, track_buffer=30)

app = Flask(__name__)
frame_lock = threading.Lock()
last_frame = None

def detection_loop():
    global last_frame
    frame_idx = 0
    while True:
        print("[INFO] Getting frames from OAK-D...", flush=True)
        frame, depth = oak.get_frames()
        if frame is None:
            print("[WARN] No frame received from OAK-D", flush=True)
            continue

        # 2️⃣ 대신 테스트 이미지 사용
        frame = cv2.imread("test.jpg")
        if frame is None:
            print("[ERROR] test.jpg not found or failed to load", flush=True)
            break
        
        # 3️⃣ 임의의 depth 맵도 생성 (그냥 1로 채움)
        depth = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint16) * 1000


        print(f"[INFO] Frame {frame_idx}: Running detection...", flush=True)
        dets = detector.detect(frame)
        print(f"[INFO] Detected {len(dets)} objects", flush=True)
        wasp_dets = [d for d in dets if CLASSES[d[5]] == "Wasp"]
        bee_dets  = [d for d in dets if CLASSES[d[5]] == "Bee"]

        frame_draw = frame.copy()

        # Bee (파란색)
        for x1, y1, x2, y2, score, _ in bee_dets:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Wasp 추적
        if wasp_dets:
            boxes = np.array([[d[0], d[1], d[2], d[3]] for d in wasp_dets], dtype=np.float32)
            scores = np.array([d[4] for d in wasp_dets], dtype=np.float32)
            class_ids = np.zeros(len(wasp_dets), dtype=int)

            det_obj = Detections(xyxy=boxes, confidence=scores, class_id=class_ids)
            tracked = tracker.update(det_obj)

            for i, box in enumerate(tracked.xyxy):
                x1, y1, x2, y2 = map(int, box)
                tid = int(tracked.tracker_id[i])
                cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cx = min(int((x1 + x2) // 2), depth.shape[1] - 1)
                cy = min(int((y1 + y2) // 2), depth.shape[0] - 1)
                Z = float(depth[cy, cx])
                if Z > 0:
                    X = (cx - oak.c_x) * Z / oak.f_x
                    Y = (cy - oak.c_y) * Z / oak.f_y
                    label = f"Wasp #{tid}: ({X/1000:.2f}, {Y/1000:.2f}, {Z/1000:.2f})m"
                    cv2.putText(frame_draw, label, (x1, y2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.putText(frame_draw, f"Wasp count: {len(wasp_dets)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        with frame_lock:
            last_frame = frame_draw.copy()
        frame_idx += 1

@app.route('/')
def index():
    return '<h3>Live Wasp Detection</h3><img src="/video_feed" width="640" height="480" />'

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            with frame_lock:
                if last_frame is None:
                    continue
                ret, buffer = cv2.imencode('.jpg', last_frame)
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    threading.Thread(target=detection_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
