#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple YOLOv8n Inference on Oak-D Lite (USB) with Hailo-8
Usage:
  ./yolov8_infer.py --hef /path/to/yolov8n.hef --labels labels.json --conf 0.3
"""
import os
import sys
import signal
import argparse
import json
from pathlib import Path

import depthai as dai
import numpy as np
import cv2

from hailo_platform.pyhailort.pyhailort import (
    VDevice as HailoDevice, HEF,
    InputVStreamParams, OutputVStreamParams,
    InputVStreams, OutputVStreams,
)

script_dir = Path(__file__).resolve().parent

class HailoYolo:
    def __init__(self, hef_path):
        # open virtual device
        params = HailoDevice.create_params()
        self.device = HailoDevice(params)

        # load & configure HEF
        hef = HEF(hef_path)
        networks = self.device.configure(hef)
        self.net = networks[0]

        # activate network group
        self._act_cm = self.net.activate()
        self._act_cm.__enter__()

        # prepare I/O vstream params
        in_params  = InputVStreamParams.make_from_network_group(self.net)
        out_params = OutputVStreamParams.make_from_network_group(self.net)

        # open input/output vstreams
        self._ivs_cm = InputVStreams(self.net, in_params)
        self.ivs = self._ivs_cm.__enter__()
        self._ovs_cm = OutputVStreams(self.net, out_params)
        self.ovs = self._ovs_cm.__enter__()

        # expect exactly one in & one out
        ins = list(self.ivs._vstreams.keys())
        outs = list(self.ovs._vstreams.keys())
        if len(ins) != 1 or len(outs) != 1:
            raise RuntimeError(f"Expected 1→1 vstreams, got in={ins}, out={outs}")
        self.in_name, self.out_name = ins[0], outs[0]

    def infer(self, frame):
        # preprocess: ensure 3-channel BGR, resize to 640×640, normalize, CHW
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        img = cv2.resize(frame, (640, 640)).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1).copy()

        # send & recv
        vin = self.ivs.get(name=self.in_name)
        vin.send(img)
        vin.flush()
        vout = self.ovs.get(name=self.out_name)
        return vout.recv()

    def close(self):
        # teardown
        self._ovs_cm.__exit__(None, None, None)
        self._ivs_cm.__exit__(None, None, None)
        self._act_cm.__exit__(None, None, None)


def create_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.createColorCamera()  # on Oak-D Lite this yields a mono preview
    cam.setPreviewSize(640, 640)
    cam.setInterleaved(False)
    cam.setFps(30)

    xout = pipeline.createXLinkOut()
    xout.setStreamName("rgb")
    cam.preview.link(xout.input)
    return pipeline


def main(opts):
    model = HailoYolo(opts.hef)

    with open(opts.labels, 'r') as f:
        labels = json.load(f).get("labels", [])

    dai_dev = dai.Device(create_pipeline())
    q_rgb = dai_dev.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    print("Starting inference. Press Ctrl+C to exit.")

    try:
        while True:
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()

            dets = model.infer(frame)
            # flatten list-of-arrays if needed
            if isinstance(dets, list):
                dets = np.vstack(dets) if dets else np.zeros((0,6), dtype=np.float32)

            # draw
            for *box, score, cls in dets:
                if score < opts.conf:
                    continue
                y1, x1, y2, x2 = map(int, box)
                lbl = labels[int(cls)] if int(cls) < len(labels) else str(int(cls))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{lbl}:{score:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            cv2.imshow("Hailo YOLOv8n", frame)
            if cv2.waitKey(1) == ord('q'):
                break

    finally:
        print("Cleaning up…")
        model.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--hef',    default=str(script_dir / 'models' / 'yolov8n2.hef'))
    ap.add_argument('--labels', default=str(script_dir / 'resources/wasp_bee/wasp_bee_labels.json'))
    ap.add_argument('--conf',   type=float, default=0.3)
    opts = ap.parse_args()
    opts.hef    = os.path.expanduser(opts.hef)
    opts.labels = os.path.expanduser(opts.labels)
    main(opts)
