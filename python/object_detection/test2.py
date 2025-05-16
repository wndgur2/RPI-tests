import depthai as dai
import cv2
import numpy as np
import queue
import threading
from object_detection_utils import ObjectDetectionUtils
from utils import HailoAsyncInference
from supervision import ByteTrack, Detections

MODEL_PATH     = "/home/ssafy/project/RPI-tests/python/object_detection/models/yolov8n2.hef"
LABELS_PATH    = "/home/ssafy/project/RPI-tests/python/object_detection/wasp.txt"
BATCH_SIZE     = 1
TRACK_THRESH   = 0.5
TRACK_BUFFER   = 30
FPS            = 30
MIN_SCORE      = 0.25
INPUT_SIZE     = 640


def on_wasp_tracked(track_id: int, x: float, y: float, z: float, conf: float):
    print(f"[Callback] ID={track_id:2d} → X={x:.3f} m, Y={y:.3f} m, Z={z:.3f} m, conf={conf:.2f}")

def build_pipeline():
    p = dai.Pipeline()

    cam = p.createColorCamera()
    cam.setPreviewSize(1280, 720)
    cam.setInterleaved(False)
    cam.setFps(FPS)

    xout_rgb = p.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam.preview.link(xout_rgb.input)

    monoL = p.createMonoCamera()
    monoR = p.createMonoCamera()
    monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

    stereo = p.createStereoDepth()
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    monoL.out.link(stereo.left)
    monoR.out.link(stereo.right)

    spatialCalc = p.createSpatialLocationCalculator()
    spatialCalc.setWaitForConfigInput(True)
    stereo.depth.link(spatialCalc.inputDepth)

    xin_spatial_cfg = p.createXLinkIn()
    xin_spatial_cfg.setStreamName("spatial_cfg")
    xin_spatial_cfg.out.link(spatialCalc.inputConfig)

    xout_spatial = p.createXLinkOut()
    xout_spatial.setStreamName("spatial_data")
    spatialCalc.out.link(xout_spatial.input)

    return p

def main():
    pipeline = build_pipeline()
    with dai.Device(pipeline) as dev:
        q_rgb = dev.getOutputQueue("rgb", maxSize=4, blocking=False)
        q_spatial_cfg = dev.getInputQueue("spatial_cfg")
        q_spatial_data = dev.getOutputQueue("spatial_data", maxSize=4, blocking=False)

        det_utils = ObjectDetectionUtils(LABELS_PATH)
        in_q, out_q = queue.Queue(), queue.Queue()
        hailo_inf = HailoAsyncInference(MODEL_PATH, in_q, out_q, BATCH_SIZE, send_original_frame=True)
        threading.Thread(target=hailo_inf.run, daemon=True).start()

        tracker = ByteTrack(TRACK_THRESH, TRACK_BUFFER, 0.65, FPS)

        while True:
            in_rgb = q_rgb.get()
            frame_bgr = in_rgb.getCvFrame()

            rgb_for_model = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            proc = det_utils.preprocess(rgb_for_model, INPUT_SIZE, INPUT_SIZE)
            in_q.put(([frame_bgr], [proc]))

            raw = out_q.get()
            original_frames, inference_results = raw
            infer_res = inference_results[0] if isinstance(inference_results, list) else inference_results
            dets = det_utils.extract_detections(infer_res, threshold=MIN_SCORE)

            wt_dets = []
            for box, cls, score in zip(
                dets['detection_boxes'],
                dets['detection_classes'],
                dets['detection_scores']):
                if cls == 1:
                    ymin, xmin, ymax, xmax = box
                    x1 = int(xmin * 1280); y1 = int(ymin * 720)
                    x2 = int(xmax * 1280); y2 = int(ymax * 720)
                    wt_dets.append([x1, y1, x2, y2, score])

            if len(wt_dets) > 0:
                xyxy = np.array([d[:4] for d in wt_dets], dtype=float)
                confidence = np.array([d[4] for d in wt_dets], dtype=float)
                class_id = np.ones(len(wt_dets), dtype=int)
            else:
                xyxy = np.zeros((0, 4), dtype=float)
                confidence = np.zeros((0,), dtype=float)
                class_id = np.zeros((0,), dtype=int)

            detections = Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            tracked = tracker.update_with_detections(detections=detections)

            spatial_config_list = []
            for (x1, y1, x2, y2) in tracked.xyxy:
                cfg = dai.SpatialLocationCalculatorConfigData()
                roi = dai.Rect(
                    dai.Point2f(x1 / 1280, y1 / 720),
                    dai.Point2f(x2 / 1280, y2 / 720)
                )
                cfg.roi = roi
                cfg.depthThresholds.lowerThreshold = 100
                cfg.depthThresholds.upperThreshold = 10000
                spatial_config_list.append(cfg)

            if spatial_config_list:
                config = dai.SpatialLocationCalculatorConfig()
                config.setROIs(spatial_config_list)
                q_spatial_cfg.send(config)
                spatial_data = q_spatial_data.get()
                spatial_locations = spatial_data.getSpatialLocations()

                for (roi, tid, conf) in zip(spatial_locations, tracked.tracker_id, tracked.confidence):
                    coords = roi.spatialCoordinates
                    x_m = coords.x / 1000.0
                    y_m = coords.y / 1000.0
                    z_m = coords.z / 1000.0
                    print(f"[Spatial] ID={tid:2d} → (m) X={x_m:.3f}, Y={y_m:.3f}, Z={z_m:.3f}, conf={conf:.2f}")
                    on_wasp_tracked(tid, x_m, y_m, z_m, conf)
            else:
                # 트랙된 객체가 없으면 spatialConfig 생략
                continue


            spatial_data = q_spatial_data.get()
            spatial_locations = spatial_data.getSpatialLocations()

            for (roi, tid, conf) in zip(spatial_locations, tracked.tracker_id, tracked.confidence):
                coords = roi.spatialCoordinates
                x_m = coords.x / 1000.0
                y_m = coords.y / 1000.0
                z_m = coords.z / 1000.0
                print(f"[Spatial] ID={tid:2d} → (m) X={x_m:.3f}, Y={y_m:.3f}, Z={z_m:.3f}, conf={conf:.2f}")
                on_wasp_tracked(tid, x_m, y_m, z_m, conf)

            for (x1, y1, x2, y2), tid in zip(tracked.xyxy, tracked.tracker_id):
                cv2.rectangle(frame_bgr, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0),2)
                cv2.putText(frame_bgr, f"ID:{tid}", (int(x1),int(y1)-8), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

            cv2.imwrite("./output.jpg", frame_bgr)

if __name__ == "__main__":
    main()