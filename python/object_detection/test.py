import depthai as dai
import cv2
import numpy as np
import queue
import threading
from object_detection_utils import ObjectDetectionUtils
from utils import HailoAsyncInference
from supervision import ByteTrack, Detections
import numpy as np

MODEL_PATH     = "/home/ssafy/project/RPI-tests/python/object_detection/models/yolov8n.hef"
LABELS_PATH    = "/home/ssafy/project/RPI-tests/python/object_detection/wasp.txt"
BATCH_SIZE     = 1
TRACK_THRESH   = 0.5   # ByteTrack의 confidence threshold
TRACK_BUFFER   = 30    # ByteTrack의 max_lost
FPS            = 30
MIN_SCORE      = 0.1   # Wasp 검출 최소 score


INPUT_SIZE     = 640   # 모델 입력 크기

def on_wasp_tracked(track_id: int, x: float, y: float, z: float):
    """
    트랙이 업데이트될 때마다 호출되는 콜백
    """
    print(f"[Callback] ID={track_id:2d} → X={x:.3f} m, Y={y:.3f} m, Z={z:.3f} m")

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
    xout_depth = p.createXLinkOut()
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    return p

def main():
    pipeline = build_pipeline()
    with dai.Device(pipeline) as dev:
        q_rgb   = dev.getOutputQueue("rgb",   maxSize=4, blocking=False)
        q_depth = dev.getOutputQueue("depth", maxSize=4, blocking=False)

        calib      = dev.readCalibration()
        intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, (1280, 720))
        fx, fy = intrinsics[0][0], intrinsics[1][1]
        cx, cy = intrinsics[0][2], intrinsics[1][2]

        det_utils   = ObjectDetectionUtils(LABELS_PATH)
        in_q, out_q = queue.Queue(), queue.Queue()
        hailo_inf   = HailoAsyncInference(
            MODEL_PATH, in_q, out_q, BATCH_SIZE, send_original_frame=True
        )
        th_inf = threading.Thread(target=hailo_inf.run, daemon=True)
        th_inf.start()

        tracker = ByteTrack(TRACK_THRESH, TRACK_BUFFER, 0.65, FPS)

        while True:
            import time
            time.sleep(3)
            print("===============================================")
            in_rgb   = q_rgb.get()
            in_depth = q_depth.get()
            frame_bgr   = in_rgb.getCvFrame()
            depth_frame = in_depth.getFrame()

            rgb_for_model = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            proc = det_utils.preprocess(rgb_for_model, INPUT_SIZE, INPUT_SIZE)

            # 전처리된 프레임 → 원본 프레임 순으로 전달
            in_q.put(([frame_bgr], [proc]))

            # 추론 결과 받아오기
            raw = out_q.get()
            original_frames, inference_results = raw

            # 배치 크기 1인 경우에 리스트 안에 결과가 하나 들어있을 때 꺼내기
            infer_res = (
                inference_results[0]
                if isinstance(inference_results, list) and len(inference_results) == 1
                else inference_results
            )

            # 추론 결과만 넘겨서 스칼라 score 비교가 가능하게
            dets = det_utils.extract_detections(infer_res, threshold=MIN_SCORE)

            # Wasp 클래스만 필터링
            wt_dets = []
            for box, cls, score in zip(
                dets['detection_boxes'],
                dets['detection_classes'],
                dets['detection_scores']
            ):
                if cls == 1:  # Wasp 클래스
                    ymin, xmin, ymax, xmax = box
                    x1 = int(xmin * 1280); y1 = int(ymin * 720)
                    x2 = int(xmax * 1280); y2 = int(ymax * 720)
                    wt_dets.append([x1, y1, x2, y2, score])

            # wt_dets -> xyxy, confidence, class_id 배열로 변환
            if len(wt_dets) > 0:
                xyxy = np.array([d[:4] for d in wt_dets], dtype=float)
                confidence = np.array([d[4] for d in wt_dets], dtype=float)
                class_id = np.ones(len(wt_dets), dtype=int)
            else:
                # 빈 검출일 때도 2D/(0,4), 1D/(0,) 배열로
                xyxy = np.zeros((0, 4), dtype=float)
                confidence = np.zeros((0,), dtype=float)
                class_id = np.zeros((0,), dtype=int)

            # supervision.Detections 생성
            detections = Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id
            )

            # 트래커 업데이트
            if wt_dets:
                xyxy = np.array([d[:4] for d in wt_dets], dtype=float)
                confidence = np.array([d[4] for d in wt_dets], dtype=float)
                class_id = np.ones(len(wt_dets), dtype=int)
            else:
                xyxy = np.zeros((0,4), dtype=float)
                confidence = np.zeros((0,), dtype=float)
                class_id = np.zeros((0,), dtype=int)

            detections = Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=class_id
            )

            # update_with_detections 호출: tracker_id 필드가 들어간 Detections 반환
            tracked = tracker.update_with_detections(detections=detections)

            # tracked.xyxy, tracked.tracker_id 순회
            for (x1, y1, x2, y2), tid in zip(tracked.xyxy, tracked.tracker_id):
                cx_px = int((x1 + x2) / 2)
                cy_px = int((y1 + y2) / 2)
                z_m  = depth_frame[cy_px, cx_px] / 1000.0
                x_m = (cx_px - cx) * z_m / fx
                y_m = (cy_px - cy) * z_m / fy
                print(f"[Frame] ID={tid:2d} → (m) X={x_m:.3f}, Y={y_m:.3f}, Z={z_m:.3f}")
                on_wasp_tracked(tid, x_m, y_m, z_m)

            # 화면에 표시
            for (x1, y1, x2, y2), tid in zip(tracked.xyxy, tracked.tracker_id):
                cv2.rectangle(frame_bgr, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0),2)
                cv2.putText(frame_bgr, f"ID:{tid}", (int(x1),int(y1)-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
            cv2.imwrite("./output.jpg",frame_bgr)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
