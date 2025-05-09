#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WaspTracker
-----------
Oak-D Lite → Hailo-8 YOLOv8n → ByteTrack → (x,y,z) mm 좌표 스트림
"""
import os
import depthai as dai
import numpy as np
import cv2
from yolox.tracker.byte_tracker import BYTETracker
from hailo_platform.pyhailort import _pyhailort as core

# ---------------- Hailo 초기화 ---------------- #
class HailoYolo:
    def __init__(self, hef_path):
        device = HailoDevice()
        with core.Hef(hef_path) as hef:
            ngs = device.configure(hef)
        self.net = ngs[0]
        self.input_vstream, = self.net.create_input_vstreams()
        self.output_vstreams, = self.net.create_output_vstreams()

    def infer(self, frame):
        img = cv2.resize(frame, (640,640)).astype(np.float32) / 255.0
        img = img.transpose(2,0,1).copy()
        self.input_vstream.send(img)
        return self.output_vstreams[0].recv()

# ---------------- Oak-D 파이프 ---------------- #
def create_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(640, 640)
    cam.setInterleaved(False)
    cam.setFps(60)

    stereo = pipeline.createStereoDepth()
    mono_left  = pipeline.createMonoCamera()
    mono_right = pipeline.createMonoCamera()
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
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
    stereo.depth.link(xout_depth.input)
    return pipeline

# -------------- 메인 추적 클래스 -------------- #
class WaspTracker:
    def __init__(self, hef_path, conf_th=0.25):
        self.detector = HailoYolo(hef_path)
        self.tracker  = BYTETracker()
        self.target_id = None
        self.conf_th = conf_th

        self.pipeline = create_pipeline()
        self.device = dai.Device(self.pipeline)
        self.q_rgb   = self.device.getOutputQueue("rgb",   maxSize=4, blocking=False)
        self.q_depth = self.device.getOutputQueue("depth", maxSize=4, blocking=False)

    def _select_track(self, tracks):
        """첫 프레임에서 임의(0번째) 트랙을 고정."""
        if self.target_id is None and tracks:
            self.target_id = tracks[0].track_id

    def run(self):
        while True:
            in_rgb = self.q_rgb.get()
            in_depth = self.q_depth.get()
            frame = in_rgb.getCvFrame()
            depth_frame = in_depth.getFrame()  # uint16 disparity 깊이(mm)

            detections = self.detector.infer(frame)
            # YOLOv8 NMS 출력: (n,7)[x1,y1,x2,y2,score,class,id] id = -1 (NMS)
            dets = []
            for d in detections:
                x1,y1,x2,y2,conf,cls,_ = d
                if conf < self.conf_th or int(cls)!=0:  # 0번 클래스를 'Wasp'로 가정
                    continue
                dets.append([x1,y1,x2,y2,conf])
            tracks = self.tracker.update(np.array(dets), frame.shape)

            self._select_track(tracks)

            if self.target_id is not None:
                # 타깃 트랙 찾기
                match = next((t for t in tracks if t.track_id==self.target_id), None)
                if match:
                    x1,y1,x2,y2 = map(int, match.tlbr)
                    cx,cy = (x1+x2)//2, (y1+y2)//2
                    z_mm  = int(np.median(depth_frame[y1:y2, x1:x2]))
                    print(f"{self.target_id}: ({cx},{cy},{z_mm})")
                else:        # 프레임에서 사라짐
                    self.target_id = None
                    print("null")
            else:
                print("null")

if __name__ == "__main__":
    import argparse, signal, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="~/models/yolov8n.hef")
    args = ap.parse_args()
    wt = WaspTracker(os.path.expanduser(args.hef))
    signal.signal(signal.SIGINT, lambda s,f: sys.exit(0))
    wt.run()
