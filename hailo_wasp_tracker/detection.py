import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import threading
import numpy as np
import cv2
import time
import hailo
from flask import Flask, Response

from oak_camera import OakCamera
from tracker import WaspTracker
from supervision import Detections
from hailo_apps_infra.hailo_rpi_common import app_callback_class
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp

# 클래스 및 전역 설정
CLASSES = ["Bee", "Wasp"]
oak = OakCamera(resolution=(640, 480), fps=30)
tracker = WaspTracker(track_thresh=0.25, track_buffer=30)

# Flask 앱 정의
app = Flask(__name__)
last_frame = None
frame_lock = threading.Lock()
frame_count = 0


class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.frame = None
        self.depth = None

    def update_frames(self, frame, depth):
        self.frame = frame
        self.depth = depth

    def get_latest(self):
        return self.frame, self.depth


def app_callback(pad, info, user_data):
    global last_frame
    global frame_count
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    frame, depth = user_data.get_latest()
    if frame is None or depth is None:
        return Gst.PadProbeReturn.OK

    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    frame_count += 1
    print("==========================")
    print("frame_count:", frame_count)
    bee_dets = []
    wasp_dets = []
    for det in detections:
        bbox = det.get_bbox()
        x1 = int(bbox.xmin() * frame.shape[1])
        y1 = int(bbox.ymin() * frame.shape[0])
        x2 = int(bbox.xmax() * frame.shape[1])
        y2 = int(bbox.ymax() * frame.shape[0])
        score = det.get_confidence()
        label = det.get_label()

        if label == "Wasp":
            wasp_dets.append((x1, y1, x2, y2, score))
        elif label == "Bee":
            bee_dets.append((x1, y1, x2, y2, score))
        print('label', 'x1', 'y1', 'x2', 'y2', 'score') 
        print(label, x1, y1, x2, y2, score)
    print("==========================")
    frame_draw = frame.copy()

    for x1, y1, x2, y2, _ in bee_dets:
        cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if wasp_dets:
        boxes = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, _ in wasp_dets], dtype=np.float32)
        scores = np.array([score for *_, score in wasp_dets], dtype=np.float32)
        class_ids = np.zeros(len(wasp_dets), dtype=int)
        tracked = tracker.update(Detections(xyxy=boxes, confidence=scores, class_id=class_ids))

        for i, box in enumerate(tracked.xyxy):
            x1, y1, x2, y2 = map(int, box)
            tid = int(tracked.tracker_id[i])
            cx = min((x1 + x2) // 2, depth.shape[1] - 1)
            cy = min((y1 + y2) // 2, depth.shape[0] - 1)
            Z = float(depth[cy, cx])
            if Z > 0:
                X = (cx - oak.c_x) * Z / oak.f_x
                Y = (cy - oak.c_y) * Z / oak.f_y
                label = f"Wasp #{tid}: ({X/1000:.2f}, {Y/1000:.2f}, {Z/1000:.2f})m"
                cv2.putText(frame_draw, label, (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(frame_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.putText(frame_draw, f"Wasp count: {len(wasp_dets)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # JPEG 인코딩 후 Flask 스트리밍 버퍼에 저장
    with frame_lock:
        last_frame = cv2.imencode('.jpg', cv2.cvtColor(frame_draw, cv2.COLOR_RGB2BGR))[1].tobytes()

    return Gst.PadProbeReturn.OK


def oak_loop(user_data):
    while True:
        frame, depth = oak.get_frames()
        if frame is not None:
            user_data.update_frames(frame, depth)
        time.sleep(0.01)


@app.route("/")
def index():
    return "<h3>Wasp Detection Stream</h3><img src='/video_feed' width='640' height='480'>"


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if last_frame is None:
                    continue
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + last_frame + b"\r\n")
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


def start_gstreamer():
    Gst.init(None)
    user_data = user_app_callback_class()
    threading.Thread(target=oak_loop, args=(user_data,), daemon=True).start()
    app_instance = GStreamerDetectionApp(app_callback, user_data)
    threading.Thread(target=app_instance.run, daemon=True).start()


if __name__ == "__main__":
    start_gstreamer()
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
