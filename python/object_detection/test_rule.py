import depthai as dai
import cv2
import numpy as np
import queue
import threading
from object_detection_utils import ObjectDetectionUtils
from utils import HailoAsyncInference
from supervision import ByteTrack, Detections
import time

MODEL_PATH     = "/home/ssafy/project/RPI-tests/python/object_detection/models/yolov8n.hef"
LABELS_PATH    = "/home/ssafy/project/RPI-tests/python/object_detection/wasp.txt"
BATCH_SIZE     = 1
TRACK_THRESH   = 0.5
TRACK_BUFFER   = 30
FPS            = 30
MIN_SCORE      = 0.1
INPUT_SIZE     = 640
FRAME_WIDTH    = 1920
FRAME_HEIGHT   = 1080
FOCAL_LENGTH   = 875.0  # OAK-D Lite default fx, fy
CENTER_X       = FRAME_WIDTH  / 2
CENTER_Y       = FRAME_HEIGHT / 2


def on_wasp_tracked(track_id: int, x: float, y: float, z: float, start_time: float):
    latency = time.perf_counter() - start_time
    print(f"[Callback] ID={track_id:2d} → X={x:.3f} m, Y={y:.3f} m, Z={z:.3f} m | ⏱ {latency*1000:.1f} ms")

def build_pipeline():
    pipeline = dai.Pipeline()

    cam = pipeline.createColorCamera()
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(FRAME_WIDTH, FRAME_HEIGHT)
    cam.setInterleaved(False)
    cam.setFps(FPS)

    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam.video.link(xout_rgb.input)

    monoL = pipeline.createMonoCamera()
    monoR = pipeline.createMonoCamera()
    monoL.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoR.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

    stereo = pipeline.createStereoDepth()
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setSubpixel(True)
    monoL.out.link(stereo.left)
    monoR.out.link(stereo.right)

    xout_depth = pipeline.createXLinkOut()
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline

def main():
    pipeline = build_pipeline()
    with dai.Device(pipeline) as dev:
        q_rgb   = dev.getOutputQueue("rgb", maxSize=1, blocking=False)
        q_depth = dev.getOutputQueue("depth", maxSize=1, blocking=False)

        det_utils = ObjectDetectionUtils(LABELS_PATH)
        in_q, out_q = queue.Queue(), queue.Queue()
        hailo_inf = HailoAsyncInference(MODEL_PATH, in_q, out_q, BATCH_SIZE, send_original_frame=True)
        threading.Thread(target=hailo_inf.run, daemon=True).start()

        tracker = ByteTrack(TRACK_THRESH, TRACK_BUFFER, 0.65, FPS)

        frame_total, dropped = 0, 0
        fps_counter = 0
        fps_timer_start = time.perf_counter()

        while True:
            start_time = time.perf_counter()
            in_rgb = q_rgb.tryGet()
            in_depth = q_depth.tryGet()
            frame_total += 1
            if in_rgb is None or in_depth is None:
                dropped += 1
                continue

            if frame_total % 100 == 0:
                print(f"Dropped frames: {dropped}/{frame_total} ({(dropped/frame_total)*100:.1f}%)")

            frame_bgr = in_rgb.getCvFrame()
            depth_frame = in_depth.getFrame()
            rgb_for_model = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            proc = det_utils.preprocess(rgb_for_model, INPUT_SIZE, INPUT_SIZE)
            in_q.put(([frame_bgr], [proc]))

            infer_start = time.perf_counter()
            raw = out_q.get()
            infer_end = time.perf_counter()
            print(f"[Profiler] Inference took {(infer_end - infer_start)*1000:.1f} ms")
            _, inference_results = raw

            dets = det_utils.extract_detections(inference_results, threshold=MIN_SCORE)

            wt_dets = []
            for box, cls, score in zip(dets['detection_boxes'], dets['detection_classes'], dets['detection_scores']):
                if cls == 1:
                    ymin, xmin, ymax, xmax = box
                    x1 = int(xmin * FRAME_WIDTH)
                    y1 = int(ymin * FRAME_HEIGHT)
                    x2 = int(xmax * FRAME_WIDTH)
                    y2 = int(ymax * FRAME_HEIGHT)
                    wt_dets.append([x1, y1, x2, y2, score])

            if wt_dets:
                xyxy = np.array([d[:4] for d in wt_dets], dtype=float)
                confidence = np.array([d[4] for d in wt_dets], dtype=float)
                class_id = np.ones(len(wt_dets), dtype=int)
            else:
                xyxy = np.zeros((0, 4), dtype=float)
                confidence = np.zeros((0,), dtype=float)
                class_id = np.zeros((0,), dtype=int)

            detections = Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            tracked = tracker.update_with_detections(detections=detections)

            fps_counter += 1
            if time.perf_counter() - fps_timer_start >= 1.0:
                print(f"[FPS] Processed frames per second: {fps_counter}")
                fps_counter = 0
                fps_timer_start = time.perf_counter()

            for (x1, y1, x2, y2), tid in zip(tracked.xyxy, tracked.tracker_id):
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if 0 <= cx < FRAME_WIDTH and 0 <= cy < FRAME_HEIGHT:
                    z = depth_frame[cy, cx] / 1000.0
                    if z > 0:
                        x = (cx - CENTER_X) * z / FOCAL_LENGTH
                        y = (cy - CENTER_Y) * z / FOCAL_LENGTH
                        print(f"[Frame] ID={tid:2d} → (m) X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
                        on_wasp_tracked(tid, x, y, z, start_time)

if __name__ == "__main__":
    main()
