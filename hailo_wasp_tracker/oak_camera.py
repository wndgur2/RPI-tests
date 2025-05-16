import depthai as dai
import numpy as np

class OakCamera:
    def __init__(self, resolution=(640, 480), fps=30):
        self.pipeline = dai.Pipeline()
        width, height = resolution

        # RGB 카메라 설정
        cam = self.pipeline.create(dai.node.ColorCamera)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setVideoSize(width, height)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(fps)

        # 모노 카메라 및 스테레오 깊이 설정
        mono_left = self.pipeline.create(dai.node.MonoCamera)
        mono_right = self.pipeline.create(dai.node.MonoCamera)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo = self.pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(width, height)
        stereo.setConfidenceThreshold(200)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(False)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        # 컬러 출력: ✅ preview 대신 isp 또는 video 사용
        xout_color = self.pipeline.create(dai.node.XLinkOut)
        xout_color.setStreamName("color")
        cam.isp.link(xout_color.input)  # 또는 cam.video.link(...)

        # 깊이 출력
        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        # 장치 연결 및 큐 생성
        self.device = dai.Device(self.pipeline)
        self.q_color = self.device.getOutputQueue("color", maxSize=4, blocking=False)
        self.q_depth = self.device.getOutputQueue("depth", maxSize=4, blocking=False)

        # 내부 파라미터
        calib = self.device.readCalibration()
        M_rgb = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, width, height))
        self.f_x = M_rgb[0, 0]
        self.f_y = M_rgb[1, 1]
        self.c_x = M_rgb[0, 2]
        self.c_y = M_rgb[1, 2]

    def get_frames(self):
        try:
            in_color = self.q_color.get()
            in_depth = self.q_depth.get()
            return in_color.getCvFrame(), in_depth.getFrame()
        except Exception as e:
            print(f"[ERROR] get_frames failed: {e}", flush=True)
            return None, None

    def close(self):
        self.device.close()
