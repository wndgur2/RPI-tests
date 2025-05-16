import depthai as dai
import numpy as np

class OakCamera:
    def __init__(self, resolution=(640, 480), fps=30):
        self.pipeline = dai.Pipeline()
        width, height = resolution

        cam = self.pipeline.create(dai.node.ColorCamera)
        cam.setPreviewSize(width, height)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setFps(fps)
        cam.setInterleaved(False)

        mono_l = self.pipeline.create(dai.node.MonoCamera)
        mono_r = self.pipeline.create(dai.node.MonoCamera)
        mono_l.setBoardSocket(dai.CameraBoardSocket.LEFT)
        mono_r.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        mono_l.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        mono_r.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

        stereo = self.pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        stereo.setOutputSize(width, height)
        stereo.setConfidenceThreshold(200)

        mono_l.out.link(stereo.left)
        mono_r.out.link(stereo.right)

        xout_color = self.pipeline.create(dai.node.XLinkOut)
        xout_color.setStreamName("color")
        cam.preview.link(xout_color.input)

        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        self.device = dai.Device(self.pipeline)
        self.q_color = self.device.getOutputQueue("color", maxSize=4, blocking=False)
        self.q_depth = self.device.getOutputQueue("depth", maxSize=4, blocking=False)

        calib = self.device.readCalibration()
        M_rgb = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB, width, height))
        self.f_x, self.f_y = M_rgb[0, 0], M_rgb[1, 1]
        self.c_x, self.c_y = M_rgb[0, 2], M_rgb[1, 2]

    def get_frames(self):
        frame = self.q_color.get().getCvFrame()
        depth = self.q_depth.get().getFrame()
        return frame, depth

    def close(self):
        try:
            self.device.close()
        except:
            pass
