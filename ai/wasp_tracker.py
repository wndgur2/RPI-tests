#!/usr/bin/env python3
# Usage:
#   cd /home/ssafy/project/RPI-tests/ai
#   git clone https://github.com/ifzhang/ByteTrack.git ByteTrack_src
#   pip install loguru scipy filterpy motmetrics depthai numpy opencv-python hailo-platform
#
# Then run:
#   ./wasp_tracker.py --hef ~/models/yolov8n.hef --conf 0.3

# -*- coding: utf-8 -*-
"""
WaspTracker
-----------
Oak-D Lite → Hailo-8 YOLOv8n → ByteTrack → (x,y,z) mm 좌표 스트림
"""
import os
import sys
import signal
import argparse
from pathlib import Path

# ByteTrack 소스 경로 설정
script_dir = Path(__file__).resolve().parent
byte_src = script_dir / 'ByteTrack_src'
# ByteTrack_src/yolox 폴더를 모듈 검색 경로에 추가
# ByteTrack_src(내 tracker 패키지 포함) 폴더를 모듈 검색 경로에 추가
# ByteTrack_src 디렉토리를 모듈 검색 경로에 추가
sys.path.insert(0, str(byte_src))

import depthai as dai
import numpy as np
import cv2
from yolox.tracker.byte_tracker import BYTETracker
from hailo_platform.pyhailort import Device, Hef

# ---------------- Hailo 초기화 ---------------- #
class HailoYolo:
    def __init__(self, hef_path):
        # Hailo-8 디바이스 초기화
        self.device = Device()
        with Hef(hef_path) as hef:
            networks = self.device.configure(hef)
        self.net = networks[0]
        # 입출력 스트림 생성
        self.input_stream, = self.net.create_input_vstreams()
        self.output_streams, = self.net.create_output_vstreams()

    def infer(self, frame):
        # 프레임 전처리: resize, normalize, CHW
        img = cv2.resize(frame, (640, 640)).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1).copy()
        self.input_stream.send(img)
        return self.output_streams[0].recv()

# ---------------- Oak-D 파이프라인 구성 ---------------- #
def create_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(640, 640)
    cam.setInterleaved(False)
    cam.setFps(60)

    mono_left = pipeline.createMonoCamera()
    mono_right = pipeline.createMonoCamera()
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    stereo = pipeline.createStereoDepth()
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    stereo.setSubpixel(True)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    xout_rgb = pipeline.createXLinkOut()
    xout_rgb.setStreamName("rgb")
    cam.preview.link(xout_rgb.input)

    xout_depth = pipeline.createXLinkOut()
    xout_depth.setStreamName("depth")
    stereo.disparity.link(xout_depth.input)

    return pipeline

# -------------- 메인 추적 클래스 -------------- #
class WaspTracker:
    def __init__(self, hef_path, conf_th=0.25):
        self.detector = HailoYolo(hef_path)
        self.tracker = BYTETracker(track_thresh=conf_th)
        self.target_id = None
        self.conf_th = conf_th

        self.pipeline = create_pipeline()
        self.device = dai.Device(self.pipeline)
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.q_depth = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    def _select_track(self, tracks):
        if self.target_id is None and tracks:
            self.target_id = tracks[0].track_id

    def run(self):
        while True:
            in_rgb = self.q_rgb.get()
            in_depth = self.q_depth.get()
            frame = in_rgb.getCvFrame()
            depth_frame = in_depth.getFrame()

            detections = self.detector.infer(frame)
            dets = []
            for d in detections:
                x1, y1, x2, y2, conf, cls, _ = d
                if conf < self.conf_th or int(cls) != 0:
                    continue
                dets.append([x1, y1, x2, y2, conf])

            tracks = self.tracker.update(np.array(dets), frame.shape)
            self._select_track(tracks)

            if self.target_id is not None:
                match = next((t for t in tracks if t.track_id == self.target_id), None)
                if match:
                    x1, y1, x2, y2 = map(int, match.tlbr)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    z_mm = int(np.median(depth_frame[y1:y2, x1:x2]))
                    print(f"ID {self.target_id}: X={cx}, Y={cy}, Z={z_mm}mm")
                else:
                    self.target_id = None
                    print("ID lost, selecting new...")
            else:
                print("No target yet...")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="~/models/yolov8n.hef", help="Path to HEF file")
    ap.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    args = ap.parse_args()
    hef_path = os.path.expanduser(args.hef)

    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    wt = WaspTracker(hef_path, conf_th=args.conf)
    wt.run()
