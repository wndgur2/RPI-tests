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
    if turret:
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
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    monoL.out.link(stereo.left)
    monoR.out.link(stereo.right)

    xout_depth = pipeline.createXLinkOut()
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return pipeline

def main(turret_enabled: bool = False):
    turret = None
    if turret_enabled:
        from turret.turret import Turret
        turret = Turret()
        turret.laser.on()

    pipeline = build_pipeline()
    with dai.Device(pipeline) as dev:
        # Output Queues
        q_rgb     = dev.getOutputQueue("rgb", maxSize=1, blocking=False)
        q_depth   = dev.getOutputQueue("depth", maxSize=1, blocking=False)

        # Intrinsics
        calib = dev.readCalibration()
        intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, FRAME_WIDTH, FRAME_HEIGHT)
        fx = intrinsics[0][0]
        fy = intrinsics[1][1]
        cx0 = intrinsics[0][2]
        cy0 = intrinsics[1][2]

        print(f"[Intrinsics] fx={fx:.1f}, fy={fy:.1f}, cx={cx0:.1f}, cy={cy0:.1f}")

        # Inference + utils
        det_utils = ObjectDetectionUtils(LABELS_PATH)
        in_q, out_q = queue.Queue(), queue.Queue()
        hailo_inf = HailoAsyncInference(MODEL_PATH, in_q, out_q, BATCH_SIZE, send_original_frame=True)
        threading.Thread(target=hailo_inf.run, daemon=True).start()

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

            best_wasp = None
            best_score = -1

            for box, cls, score in zip(dets['detection_boxes'], dets['detection_classes'], dets['detection_scores']):
                if cls == 1 and score > best_score:
                    best_score = score
                    ymin, xmin, ymax, xmax = box
                    best_wasp = [xmin, ymin, xmax, ymax]

            if best_wasp:
                x1 = int(best_wasp[0] * FRAME_WIDTH)
                y1 = int(best_wasp[1] * FRAME_HEIGHT)
                x2 = int(best_wasp[2] * FRAME_WIDTH)
                y2 = int(best_wasp[3] * FRAME_HEIGHT)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                depth_frame = q_depth.get().getFrame()
                h, w = depth_frame.shape

                if 1 <= cy < h-1 and 1 <= cx < w-1:
                    roi = depth_frame[cy-1:cy+2, cx-1:cx+2]  # 3x3 주변 영역
                    valid = roi[roi > 0]
                    if valid.size > 0:
                        depth = np.median(valid)
                        x = (cx - cx0) * depth / fx / 1000.0
                        y = (cy - cy0) * depth / fy / 1000.0
                        z = depth / 1000.0
                        on_wasp_tracked(track_id=0, x=x, y=y, z=z, start_time=start_time, turret=turret)


    if turret:
        turret.off()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-turret", type=str, default="false", help="Enable turret control (true/false)")
    args = parser.parse_args()
    turret_flag = args.turret.lower() == "true"
    try:
        main(turret_enabled=turret_flag)
    finally:
        turret.off()
