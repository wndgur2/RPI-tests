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
    # Invert Y axis for turret coordinate system
    if turret and z > 0:
        turret.look_at(x * 1000, -y * 1000, z * 1000)

def build_pipeline():
    pipeline = dai.Pipeline()
    # Color camera
    cam = pipeline.createColorCamera()
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setVideoSize(FRAME_WIDTH, FRAME_HEIGHT)
    cam.setInterleaved(False)
    cam.setFps(FPS)
    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam.video.link(xout_rgb.input)

    # Mono cameras for stereo
    monoL = pipeline.createMonoCamera()
    monoR = pipeline.createMonoCamera()
    monoL.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    monoR.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

    # Stereo depth
    stereo = pipeline.createStereoDepth()
    stereo.setSubpixel(True)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    monoL.out.link(stereo.left)
    monoR.out.link(stereo.right)

    # Spatial calculator
    spatials = pipeline.createSpatialLocationCalculator()
    # Ensure ROI config is received before computing spatial locations
    spatials.inputConfig.setWaitForMessage(True)
    spatials.inputDepth.setBlocking(False)
    stereo.depth.link(spatials.inputDepth)

    xout_spatial = pipeline.createXLinkOut()
    xout_spatial.setStreamName("spatial")
    spatials.out.link(xout_spatial.input)

    xin_config = pipeline.createXLinkIn()
    xin_config.setStreamName("spatialConfig")
    xin_config.out.link(spatials.inputConfig)

    return pipeline


def main(turret_enabled=False, use_ukf=False):
    turret = None
    if turret_enabled:
        from turret_module import Turret
        turret = Turret()
        turret.laser.on()

    pipeline = build_pipeline()
    with dai.Device(pipeline) as dev:
        q_rgb      = dev.getOutputQueue("rgb", maxSize=1, blocking=True)
        q_spatial  = dev.getOutputQueue("spatial", maxSize=1, blocking=True)
        q_config   = dev.getInputQueue("spatialConfig")

        det_utils = ObjectDetectionUtils(LABELS_PATH)
        frame_q   = queue.Queue(maxsize=1)
        result_q  = queue.Queue()
        meta_q    = queue.Queue(maxsize=1)

        hailo_inf = HailoAsyncInference(MODEL_PATH, frame_q, result_q, BATCH_SIZE, send_original_frame=True)
        threading.Thread(target=hailo_inf.run, daemon=True).start()

        def frame_reader():
            while True:
                if not result_q.empty(): time.sleep(0.001); continue
                in_rgb = q_rgb.get()
                frame = in_rgb.getCvFrame()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                proc, scale, xo, yo = det_utils.preprocess(rgb, INPUT_SIZE, INPUT_SIZE)
                try:
                    frame_q.put(([frame], [proc]), block=False)
                    meta_q.queue.clear()
                    meta_q.put((scale, xo, yo))
                except queue.Full:
                    pass
        threading.Thread(target=frame_reader, daemon=True).start()

        tracker = ByteTrack(TRACK_THRESH, TRACK_BUFFER, 0.5, FPS)
        target_id = None
        ukf = None

        while True:
            if result_q.empty() or meta_q.empty(): time.sleep(0.001); continue
            frame_bgr, results = result_q.get()
            scale, x_offset, y_offset = meta_q.get()
            dets = det_utils.extract_detections(results, threshold=MIN_SCORE)

            # Prepare detections
            boxes, confs, clsids = [], [], []
            for (ymin,xmin,ymax,xmax), cls, sc in zip(dets['detection_boxes'], dets['detection_classes'], dets['detection_scores']):
                if cls==1:
                    x1 = (xmin*INPUT_SIZE - x_offset)/scale
                    y1 = (ymin*INPUT_SIZE - y_offset)/scale
                    x2 = (xmax*INPUT_SIZE - x_offset)/scale
                    y2 = (ymax*INPUT_SIZE - y_offset)/scale
                    boxes.append([x1,y1,x2,y2]); confs.append(sc); clsids.append(cls)
            if not boxes: continue

            detections = Detections(xyxy=np.array(boxes), confidence=np.array(confs), class_id=np.array(clsids))
            tracked    = tracker.update_with_detections(detections=detections)

            ids = [int(i) for i in tracked.tracker_id]
            if target_id not in ids:
                if tracked.xyxy.size>0:
                    best = np.argmax(tracked.confidence)
                    target_id = int(tracked.tracker_id[best])
                    if use_ukf:
                        def fx_kf(x,dt): x[:3]+=x[3:]*dt; return x
                        def hx_kf(x): return x[:3]
                        ps = MerweScaledSigmaPoints(n=6,alpha=0.1,beta=2.,kappa=0)
                        ukf = UKF(dim_x=6,dim_z=3,fx=fx_kf,hx=hx_kf,dt=PREDICT_FRAMES_AHEAD*(1./FPS),points=ps)
                        ukf.x=np.zeros(6); ukf.P*=10; ukf.Q*=0.01; ukf.R*=0.1
                else:
                    target_id=None; ukf=None

            # For the target, request spatial location
            if target_id is not None:
                for bb, tid in zip(tracked.xyxy, tracked.tracker_id):
                    if int(tid)!=target_id: continue
                    x1,y1,x2,y2 = bb
                    cx = (x1+x2)/2; cy = (y1+y2)/2
                    # ROI config
                    cfg = dai.SpatialLocationCalculatorConfig()
                    data = dai.SpatialLocationCalculatorConfigData()
                    data.depthThresholds.lowerThreshold = 100
                    data.depthThresholds.upperThreshold = 10000
                                        # Normalize pixel ROI to [0,1] coordinates relative to color frame
                    nx1 = max((cx - 5) / FRAME_WIDTH, 0.0)
                    ny1 = max((cy - 5) / FRAME_HEIGHT, 0.0)
                    nx2 = min((cx + 5) / FRAME_WIDTH, 1.0)
                    ny2 = min((cy + 5) / FRAME_HEIGHT, 1.0)
                    data.roi = dai.Rect(dai.Point2f(nx1, ny1), dai.Point2f(nx2, ny2))
                    cfg.addROI(data)
                    q_config.send(cfg)
                    spatial = q_spatial.get().getSpatialLocations()
                    if spatial:
                        sc = spatial[0].spatialCoordinates
                        X = sc.x/1000.0; Y = sc.y/1000.0; Z = sc.z/1000.0
                        if use_ukf and ukf:
                            ukf.predict(); ukf.update([X,Y,Z]); X,Y,Z = ukf.x[:3]
                        on_wasp_tracked(target_id, X, Y, Z, turret)
                    break

    if turret: turret.off()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-turret", type=str, default="false")
    parser.add_argument("-ukf", type=str, default="true")
    args = parser.parse_args()
    main(turret_enabled=(args.turret.lower()=="true"), use_ukf=(args.ukf.lower()=="true"))
