import depthai as dai
import numpy as np

class OakCamera:
    def __init__(self, resolution=(640, 480), fps=30):
        # 파이프라인 생성
        self.pipeline = dai.Pipeline()
        width, height = resolution

        # 컬러 카메라 설정
        cam = self.pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(width, height)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)  # 1080p 센서 모드
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(fps)

        # 깊이를 위한 모노 카메라(좌/우)
        mono_left = self.pipeline.create(dai.node.MonoCamera)
        mono_right = self.pipeline.create(dai.node.MonoCamera)
        mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        # 스테레오 깊이 노드
        stereo = self.pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)           # 컬러 프레임에 깊이 맵 정렬
        stereo.setOutputSize(width, height)                       # 컬러 프레임 크기와 동일
        stereo.setConfidenceThreshold(200)                        # 깊이 신뢰도 임계값(0–255)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(False)

        # 모노 카메라를 깊이 노드에 연결
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        # 컬러 출력
        xout_color = self.pipeline.create(dai.node.XLinkOut)
        xout_color.setStreamName("color")
        cam.preview.link(xout_color.input)
        # 깊이 출력
        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        # 장치 시작
        self.device = dai.Device(self.pipeline)
        self.q_color = self.device.getOutputQueue("color", maxSize=4, blocking=False)
        self.q_depth = self.device.getOutputQueue("depth", maxSize=4, blocking=False)

        # 카메라 내부 파라미터 가져오기 (3x3 행렬)
        calib = self.device.readCalibration()
        M_rgb = calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, width, height)
        M_rgb = np.array(M_rgb)
        self.f_x = M_rgb[0, 0]   # 초점 거리 x
        self.f_y = M_rgb[1, 1]   # 초점 거리 y
        self.c_x = M_rgb[0, 2]   # 주점 x
        self.c_y = M_rgb[1, 2]   # 주점 y

    def get_frames(self):
        """컬러 프레임과 정렬된(depth-aligned) 깊이 맵을 반환."""
        in_color = self.q_color.get()   # 최신 컬러 프레임
        in_depth = self.q_depth.get()   # 최신 깊이 맵
        color_frame = in_color.getCvFrame()    # OpenCV BGR 이미지
        depth_frame = in_depth.getFrame()      # UINT16, 밀리미터 단위 깊이 맵
        return color_frame, depth_frame

    def close(self):
        """DepthAI 장치 해제."""
        try:
            self.device.close()
        except Exception:
            pass
