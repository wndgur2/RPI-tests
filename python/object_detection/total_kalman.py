import depthai as dai
import cv2
import numpy as np
import queue
import threading
import time
import argparse

from object_detection_utils import ObjectDetectionUtils
from utils import HailoAsyncInference
from supervision import ByteTrack, Detections
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

MODEL_PATH = "/home/ssafy/project/RPI-tests/python/object_detection/models/yolov8n2.hef"
LABELS_PATH = "/home/ssafy/project/RPI-tests/python/object_detection/wasp.txt"
BATCH_SIZE = 1
TRACK_THRESH = 0.1
TRACK_BUFFER = 30
FPS = 30
PREDICT_FRAMES_AHEAD = 5
MIN_SCORE = 0.1
INPUT_SIZE = 640
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

def on_wasp_tracked(track_id, x, y, z, turret=None):
    print(f"[Callback] ID={track_id:2d} → X={x:.3f} m, Y={y:.3f} m, Z={z:.3f} m")
    if turret and z != 0:
        turret.look_at(x * 1000, y * 1000, z * 1000)

def build_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.createColorCamera()
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    cam.setVideoSize(FRAME_WIDTH, FRAME_HEIGHT)
    cam.setInterleaved(False)
    cam.setFps(FPS)
    # cam.initialControl.setManualExposure(500, 800)

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
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    monoL.out.link(stereo.left)
    monoR.out.link(stereo.right)

    xout_depth = pipeline.createXLinkOut()
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline

def main(turret_enabled=False, use_ukf=False):
    turret = None
    if turret_enabled:
        from turret.turret import Turret
        turret = Turret()
        turret.laser.on()

    pipeline = build_pipeline()
    with dai.Device(pipeline) as dev:
        q_rgb = dev.getOutputQueue("rgb", maxSize=1, blocking=True)
        q_depth = dev.getOutputQueue("depth", maxSize=1, blocking=True)

        calib = dev.readCalibration()
        intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, FRAME_WIDTH, FRAME_HEIGHT)
        fx_val, fy_val = float(intrinsics[0][0]), float(intrinsics[1][1])
        cx0, cy0 = float(intrinsics[0][2]), float(intrinsics[1][2])

        print(f"[Intrinsics] fx={fx_val:.1f}, fy={fy_val:.1f}, cx={cx0:.1f}, cy={cy0:.1f}")

        det_utils = ObjectDetectionUtils(LABELS_PATH)
        frame_q = queue.Queue(maxsize=1)
        result_q = queue.Queue()

        hailo_inf = HailoAsyncInference(MODEL_PATH, frame_q, result_q, BATCH_SIZE, send_original_frame=True)
        threading.Thread(target=hailo_inf.run, daemon=True).start()

        def frame_reader():
            while True:
                in_rgb = q_rgb.get()
                device_ts = in_rgb.getTimestamp().total_seconds()
                host_ts = time.monotonic()
                delay_ms = (host_ts - device_ts) * 1000
                # print(f"[USB Delay Estimate] {delay_ms:.2f} ms")

                frame_bgr = in_rgb.getCvFrame()
                rgb_for_model = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                proc = det_utils.preprocess(rgb_for_model, INPUT_SIZE, INPUT_SIZE)
                frame_q.queue.clear()
                frame_q.put(([frame_bgr], [proc]))

        threading.Thread(target=frame_reader, daemon=True).start()

        tracker = ByteTrack(TRACK_THRESH, TRACK_BUFFER, 0.65, FPS)
        target_track_id = None
        ukf = None

        while True:
            if result_q.empty():
                time.sleep(0.001)
                continue

            frame_bgr, results = result_q.get()
            dets = det_utils.extract_detections(results, threshold=MIN_SCORE)

            wt_dets = [
                [int(xmin * FRAME_WIDTH), int(ymin * FRAME_HEIGHT), int(xmax * FRAME_WIDTH), int(ymax * FRAME_HEIGHT), score]
                for (ymin, xmin, ymax, xmax), cls, score in zip(dets['detection_boxes'], dets['detection_classes'], dets['detection_scores'])
                if cls == 1
            ]

            xyxy = np.array([d[:4] for d in wt_dets], dtype=float) if wt_dets else np.zeros((0, 4))
            confidence = np.array([d[4] for d in wt_dets], dtype=float) if wt_dets else np.zeros((0,))
            class_id = np.ones(len(wt_dets), dtype=int) if wt_dets else np.zeros((0,), dtype=int)

            detections = Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            tracked = tracker.update_with_detections(detections=detections)

            tracked_ids = [int(tid) for tid in tracked.tracker_id]
            if target_track_id not in tracked_ids:
                if len(tracked.xyxy) > 0:
                    best_idx = np.argmax(tracked.confidence)
                    target_track_id = int(tracked.tracker_id[best_idx])
                    if use_ukf:
                        def fx_kf(x, dt): x[:3] += x[3:] * dt; return x
                        def hx_kf(x): return x[:3]
                        points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2., kappa=0)
                        ukf = UKF(dim_x=6, dim_z=3, fx=fx_kf, hx=hx_kf, dt=PREDICT_FRAMES_AHEAD * (1.0/FPS), points=points)
                        ukf.x = np.zeros(6); ukf.P *= 10.0; ukf.Q *= 0.01; ukf.R *= 0.1
                else:
                    target_track_id = None
                    ukf = None

            if target_track_id is not None:
                for bbox, tid in zip(tracked.xyxy, tracked.tracker_id):
                    if int(tid) == target_track_id:
                        x1, y1, x2, y2 = bbox
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        depth_msg = q_depth.tryGet()
                        if depth_msg is None:
                            continue

                        depth_frame = depth_msg.getFrame()
                        h, w = depth_frame.shape
                        if 1 <= cy < h - 1 and 1 <= cx < w - 1:
                            roi = depth_frame[cy - 1:cy + 2, cx - 1:cx + 2]
                            valid = roi[roi > 0]
                            if valid.size > 0:
                                depth = np.median(valid)
                                x = (cx - cx0) * depth / fx_val / 1000.0
                                y = (cy - cy0) * depth / fy_val / 1000.0
                                z = depth / 1000.0
                                if use_ukf and ukf:
                                    ukf.predict()
                                    ukf.update([x, y, z])
                                    x, y, z = ukf.x[:3]
                                on_wasp_tracked(track_id=target_track_id, x=x, y=y, z=z, turret=turret)
                        break

    if turret:
        turret.off()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-turret", type=str, default="false")
    parser.add_argument("-ukf", type=str, default="true")
    args = parser.parse_args()

    turret_flag = args.turret.lower() == "true"
    ukf_flag = args.ukf.lower() == "true"

    main(turret_enabled=turret_flag, use_ukf=ukf_flag)