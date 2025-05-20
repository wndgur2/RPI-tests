import depthai as dai
import cv2
import numpy as np
import queue
import threading
import time
import argparse
import os

from object_detection_utils2 import ObjectDetectionUtils
from utils import HailoAsyncInference
from supervision import ByteTrack, Detections
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

MODEL_PATH     = "/home/ssafy/project/RPI-tests/python/object_detection/models/yolov8n2.hef"
LABELS_PATH    = "/home/ssafy/project/RPI-tests/python/object_detection/wasp.txt"
BATCH_SIZE     = 1
TRACK_THRESH   = 0.15
TRACK_BUFFER   = 30
FPS            = 30
PREDICT_FRAMES_AHEAD = 3
MIN_SCORE      = 0.15
INPUT_SIZE     = 640
FRAME_WIDTH    = 1280
FRAME_HEIGHT   = 720

os.makedirs("debug_frames", exist_ok=True)

def on_wasp_tracked(track_id, x, y, z, turret=None):
    print(f"[Callback] ID={track_id:2d} → X={x:.3f} m, Y={y:.3f} m, Z={z:.3f} m")
    if turret and z > 0:
        turret.look_at(x * 1000, y * 1000, z * 1000)

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
    stereo.setSubpixel(True)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
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
        from turret_module import Turret
        turret = Turret()
        turret.laser.on()

    pipeline = build_pipeline()
    with dai.Device(pipeline) as dev:
        q_rgb   = dev.getOutputQueue("rgb",   maxSize=1, blocking=True)
        q_depth = dev.getOutputQueue("depth", maxSize=1, blocking=True)

        # Calibration (color-aligned)
        calib = dev.readCalibration()
        intrinsics = np.array(
            calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, FRAME_WIDTH, FRAME_HEIGHT),
            dtype=np.float64
        )
        dist_coeffs = np.array(
            calib.getDistortionCoefficients(dai.CameraBoardSocket.CAM_A),
            dtype=np.float64
        )
        fx_val, fy_val = intrinsics[0,0], intrinsics[1,1]
        cx0, cy0 = intrinsics[0,2], intrinsics[1,2]
        print(f"[Intrinsics] fx={fx_val:.1f}, fy={fy_val:.1f}, cx={cx0:.1f}, cy={cy0:.1f}")

        det_utils = ObjectDetectionUtils(LABELS_PATH)
        frame_q = queue.Queue(maxsize=1)
        result_q = queue.Queue()
        meta_q = queue.Queue(maxsize=1)

        hailo_inf = HailoAsyncInference(MODEL_PATH, frame_q, result_q, BATCH_SIZE, send_original_frame=True)
        threading.Thread(target=hailo_inf.run, daemon=True).start()

        def frame_reader():
            while True:
                if not result_q.empty():
                    time.sleep(0.001)
                    continue
                in_rgb = q_rgb.get()
                frame_bgr = in_rgb.getCvFrame()
                rgb_for_model = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                proc, scale, x_offset, y_offset = det_utils.preprocess(
                    rgb_for_model, INPUT_SIZE, INPUT_SIZE)
                try:
                    frame_q.put(([frame_bgr], [proc]), block=False)
                    meta_q.queue.clear()
                    meta_q.put((scale, x_offset, y_offset))
                except queue.Full:
                    pass
        threading.Thread(target=frame_reader, daemon=True).start()

        tracker = ByteTrack(TRACK_THRESH, TRACK_BUFFER, 0.5, FPS)
        target_track_id = None
        ukf = None

        while True:
            if result_q.empty() or meta_q.empty():
                time.sleep(0.001)
                continue

            frame_bgr, results = result_q.get()
            scale, x_offset, y_offset = meta_q.get()
            dets = det_utils.extract_detections(results, threshold=MIN_SCORE)

            xyxy, confidence, class_id = [], [], []
            for (ymin, xmin, ymax, xmax), cls, score in zip(
                    dets['detection_boxes'], dets['detection_classes'], dets['detection_scores']):
                if cls == 1:
                    box = [xmin * INPUT_SIZE, ymin * INPUT_SIZE,
                           xmax * INPUT_SIZE, ymax * INPUT_SIZE]
                    xmin_r = (box[0] - x_offset) / scale
                    ymin_r = (box[1] - y_offset) / scale
                    xmax_r = (box[2] - x_offset) / scale
                    ymax_r = (box[3] - y_offset) / scale
                    xyxy.append([xmin_r, ymin_r, xmax_r, ymax_r])
                    confidence.append(score)
                    class_id.append(cls)

            if not xyxy:
                continue

            detections = Detections(xyxy=np.array(xyxy), confidence=np.array(confidence), class_id=np.array(class_id))
            tracked = tracker.update_with_detections(detections=detections)

            tracked_ids = [int(tid) for tid in tracked.tracker_id]
            if target_track_id not in tracked_ids:
                if tracked.xyxy.size > 0:
                    best_idx = np.argmax(tracked.confidence)
                    target_track_id = int(tracked.tracker_id[best_idx])
                    if use_ukf:
                        def fx_kf(x, dt): x[:3] += x[3:] * dt; return x
                        def hx_kf(x): return x[:3]
                        points = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2., kappa=0)
                        ukf = UKF(dim_x=6, dim_z=3, fx=fx_kf, hx=hx_kf,
                                  dt=PREDICT_FRAMES_AHEAD * (1.0/FPS), points=points)
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
                                depth_mm = np.median(valid)
                                # Radial distance (meters)
                                R = depth_mm / 1000.0
                                # Undistort pixel for angle calculation
                                pt = np.array([[[cx, cy]]], dtype=np.float32)
                                und = cv2.undistortPoints(pt, intrinsics, dist_coeffs, P=intrinsics)
                                u_ud, v_ud = und[0,0]
                                theta_x = (u_ud - cx0) / fx_val
                                theta_y = (v_ud - cy0) / fy_val
                                # Convert radial R to axial Z
                                Z = R / np.sqrt(1 + theta_x**2 + theta_y**2)
                                X = theta_x * Z
                                Y = theta_y * Z
                                if use_ukf and ukf:
                                    ukf.predict()
                                    ukf.update([X, Y, Z])
                                    X, Y, Z = ukf.x[:3]
                                on_wasp_tracked(track_id=target_track_id, x=X, y=Y, z=Z, turret=turret)
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
