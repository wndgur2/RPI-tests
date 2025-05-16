import depthai as dai
import numpy as np
import cv2
from hailo_platform import HEF, VDevice, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams

# 클래스 설정
CLASSES = ['Wasp', 'Bee']
TARGET_CLASS = 'Wasp'
MODEL_PATH = '/home/ssafy/project/RPI-tests/ai/models/yolov8n.hef'

# Hailo Device 초기화 및 설정
hef = HEF(MODEL_PATH)

# VDevice 생성
device = VDevice()

# ConfigureParams 생성 (기본 설정 사용)
configure_params = ConfigureParams.create_from_hef(hef, interface=None)

# Network Group 설정
network_group = device.configure(hef, configure_params)[0]

# Input/Output VStreamParams 생성
input_vstreams_params = InputVStreamParams.make(network_group)
output_vstreams_params = OutputVStreamParams.make(network_group)

# InferVStreams 초기화
with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
    # OAK-D 파이프라인 설정
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(640, 640)
    cam_rgb.setInterleaved(False)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.preview.link(xout_rgb.input)

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    stereo = pipeline.create(dai.node.StereoDepth)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    with dai.Device(pipeline) as oak_device:
        q_rgb = oak_device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_depth = oak_device.getOutputQueue(name="depth", maxSize=4, blocking=False)

        while True:
            in_rgb = q_rgb.get()
            in_depth = q_depth.get()

            frame = in_rgb.getCvFrame()
            depth_frame = in_depth.getFrame()

            # Hailo 입력 데이터 형식 맞추기 (NCHW 형식)
            input_frame = np.expand_dims(frame.transpose(2, 0, 1), axis=0)
            input_data = {list(input_vstreams_params.keys())[0]: input_frame}

            # 추론 수행
            results = infer_pipeline.infer(input_data)

            # 결과 해석 (첫 번째 output layer 사용)
            output_layer_name = list(results.keys())[0]
            detections = results[output_layer_name][0]

            for det in detections:
                x1, y1, x2, y2, conf, class_id = det[:6]
                class_id = int(class_id)
                conf = float(conf)
                label = CLASSES[class_id]

                if label == TARGET_CLASS and conf > 0.4:
                    x1, y1, x2, y2 = int(x1*640), int(y1*640), int(x2*640), int(y2*640)
                    cx, cy = (x1+x2)//2, (y1+y2)//2

                    # 깊이 정보 획득 (mm 단위)
                    z = depth_frame[cy, cx]
                    X, Y, Z = cx, cy, z

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                    cv2.putText(frame, f"{label}: ({X}, {Y}, {Z} mm)", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                    print(f"Detected {label} at X:{X}, Y:{Y}, Z:{Z}mm")

            # 영상 출력
            cv2.imshow('Hailo-OAKD Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cv2.destroyAllWindows()
