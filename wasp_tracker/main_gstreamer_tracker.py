import gi
import os
import cv2
import numpy as np
import hailo
import depthai as dai

gi.require_version('Gst', '1.0')
from gi.repository import Gst

from hailo_apps_infra.hailo_rpi_common import get_caps_from_pad, get_numpy_from_buffer, app_callback_class
from hailo_apps_infra.detection_pipeline import GStreamerDetectionApp
from tracker import WaspTracker
from utils import overlay_detections
from class_names import CLASS_NAMES

# Oak-D Lite 기반 깊이 정보 추출용
class OakCamera:
    def __init__(self, resolution=(640, 480)):
        self.pipeline = dai.Pipeline()
        mono_left = self.pipeline.create(dai.node.MonoCamera)
        mono_right = self.pipeline.create(dai.node.MonoCamera)
        stereo = self.pipeline.create(dai.node.StereoDepth)

        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        xout = self.pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("depth")
        stereo.depth.link(xout.input)

        self.device = dai.Device(self.pipeline)
        self.q = self.device.getOutputQueue("depth", maxSize=1, blocking=False)
        calib = self.device.readCalibration()
        M = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, *resolution))
        self.fx, self.fy, self.cx, self.cy = M[0, 0], M[1, 1], M[0, 2], M[1, 2]

    def get_depth(self):
        try:
            return self.q.get().getFrame()
        except:
            return None

    def get_intrinsics(self):
        return self.fx, self.fy, self.cx, self.cy


class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.tracker = WaspTracker()
        self.oak = OakCamera()

    def on_new_buffer(self, buffer, frame, width, height):
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        wasp_dets = []
        for det in detections:
            if det.get_label() != "Wasp":
                continue
            bbox = det.get_bbox()
            x1 = int(bbox.xmin() * width)
            y1 = int(bbox.ymin() * height)
            x2 = int(bbox.xmax() * width)
            y2 = int(bbox.ymax() * height)
            score = det.get_confidence()
            wasp_dets.append((x1, y1, x2, y2, score, 1))

        tracked = self.tracker.update(wasp_dets)
        depth = self.oak.get_depth()
        intrinsics = self.oak.get_intrinsics()
        annotated = overlay_detections(frame.copy(), tracked, depth, intrinsics, CLASS_NAMES)

        save_path = "/home/ssafy/wasp_output.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"[INFO] Saved: {save_path} | Wasp count: {len(wasp_dets)}")


def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()

    fmt, w, h = get_caps_from_pad(pad)
    frame = None
    if fmt and w and h:
        frame = get_numpy_from_buffer(buffer, fmt, w, h)

    if frame is not None:
        user_data.on_new_buffer(buffer, frame, w, h)

    return Gst.PadProbeReturn.OK


if __name__ == "__main__":
    Gst.init(None)
    user_data = user_app_callback_class()
    app = GStreamerDetectionApp(app_callback, user_data)
    app.run()
