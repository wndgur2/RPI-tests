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

def on_wasp_tracked(track_id: int, x: float, y: float, z: float, start_time: float, turret=None):
    latency = time.perf_counter() - start_time
    print(f"[Callback] ID={track_id:2d} → X={x:.3f} m, Y={y:.3f} m, Z={z:.3f} m | ⏱ {latency*1000:.1f} ms")
    if turret and z != 0:
        turret.look_at(x * 1000, y * -1000, z * 1000)

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
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    monoL.out.link(stereo.left)
    monoR.out.link(stereo.right)

    xout_depth = pipeline.createXLinkOut()
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    spatials = pipeline.createSpatialLocationCalculator()
    spatials.inputConfig.setWaitForMessage(True)
    stereo.depth.link(spatials.inputDepth)

    xout_spatial = pipeline.createXLinkOut()
    xout_spatial.setStreamName("spatial")
    spatials.out.link(xout_spatial.input)

    xin_spatial = pipeline.createXLinkIn()
    xin_spatial.setStreamName("spatialConfig")
    xin_spatial.out.link(spatials.inputConfig)

    return pipeline

def main(turret_enabled: bool = False):
    turret = None
    if turret_enabled:
        from turret.turret import Turret
        turret = Turret()
        turret.laser.on()

    pipeline = build_pipeline()
    with dai.Device(pipeline) as dev:
        q_rgb     = dev.getOutputQueue("rgb", maxSize=1, blocking=False)
        q_depth   = dev.getOutputQueue("depth", maxSize=1, blocking=False)
        q_spatial = dev.getOutputQueue("spatial", maxSize=1, blocking=False)
        q_config  = dev.getInputQueue("spatialConfig")

        det_utils = ObjectDetectionUtils(LABELS_PATH)
        in_q, out_q = queue.Queue(), queue.Queue()
        hailo_inf = HailoAsyncInference(MODEL_PATH, in_q, out_q, BATCH_SIZE, send_original_frame=True)
        threading.Thread(target=hailo_inf.run, daemon=True).start()

        tracker = ByteTrack(TRACK_THRESH, TRACK_BUFFER, 0.65, FPS)
        target_track_id = None

        while True:
            start_time = time.perf_counter()
            in_rgb = q_rgb.tryGet()
            if in_rgb is None:
                continue

            frame_bgr = in_rgb.getCvFrame()
            rgb_for_model = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            proc = det_utils.preprocess(rgb_for_model, INPUT_SIZE, INPUT_SIZE)
            in_q.put(([frame_bgr], [proc]))

            _, inference_results = out_q.get()
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
            tracked_ids = [int(tid) for tid in tracked.tracker_id]

            if target_track_id not in tracked_ids:
                if len(tracked.xyxy) > 0:
                    best_idx = np.argmax(tracked.confidence)
                    target_track_id = int(tracked.tracker_id[best_idx])
                else:
                    target_track_id = None

            if target_track_id is not None:
                for i, (bbox, tid) in enumerate(zip(tracked.xyxy, tracked.tracker_id)):
                    if int(tid) == target_track_id:
                        x1, y1, x2, y2 = bbox
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        config = dai.SpatialLocationCalculatorConfig()
                        roi = dai.SpatialLocationCalculatorConfigData()
                        roi.roi = dai.Rect(dai.Point2f(cx - 8, cy - 8), dai.Point2f(cx + 8, cy + 8))
                        roi.depthThresholds.lowerThreshold = 100
                        roi.depthThresholds.upperThreshold = 10000
                        config.addROI(roi)

                        q_config.send(config)
                        spatial_data = q_spatial.get().getSpatialLocations()

                        if spatial_data:
                            coords = spatial_data[0].spatialCoordinates
                            x, y, z = coords.x / 1000.0, coords.y / 1000.0, coords.z / 1000.0
                            on_wasp_tracked(target_track_id, x, y, z, start_time, turret)
                        break

    if turret:
        turret.off()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-turret", type=str, default="false", help="Enable turret control (true/false)")
    args = parser.parse_args()
    turret_flag = args.turret.lower() == "true"
    main(turret_enabled=turret_flag)