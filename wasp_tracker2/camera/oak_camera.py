import depthai as dai
import numpy as np

class OakCamera:
    def __init__(self, resolution=(640, 480), fps=30):
        self.pipeline = dai.Pipeline()

        # === RGB 카메라 설정 ===
        cam = self.pipeline.create(dai.node.ColorCamera)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setFps(fps)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        # RGB 영상 출력 (video: 1920x1080 원본 영상 사용)
        xout_color = self.pipeline.create(dai.node.XLinkOut)
        xout_color.setStreamName("color")
        cam.video.link(xout_color.input)

        # === Mono 카메라 및 스테레오 깊이 설정 ===
        mono_left = self.pipeline.create(dai.node.MonoCamera)
        mono_right = self.pipeline.create(dai.node.MonoCamera)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo = self.pipeline.create(dai.node.StereoDepth)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # 깊이 영상 출력
        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        # === 장치 초기화 ===
        self.device = dai.Device(self.pipeline)
        self.q_color = self.device.getOutputQueue("color", maxSize=1, blocking=False)
        self.q_depth = self.device.getOutputQueue("depth", maxSize=1, blocking=False)

        # === 내부 카메라 Intrinsics 계산 ===
        calib = self.device.readCalibration()
        M = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, *resolution))
        self.fx, self.fy = M[0, 0], M[1, 1]
        self.cx, self.cy = M[0, 2], M[1, 2]

    def get_frames(self):
        try:
            color = self.q_color.get().getCvFrame()
            depth = self.q_depth.get().getFrame()
            return color, depth
        except:
            return None, None

    def get_intrinsics(self):
        return self.fx, self.fy, self.cx, self.cy
