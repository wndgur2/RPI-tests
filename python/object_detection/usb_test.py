import depthai as dai
import cv2


with dai.Device() as device:
    print("USB speed:", device.getUsbSpeed())

    pipeline = dai.Pipeline()
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(300, 300)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)

    xout = pipeline.createXLinkOut()
    xout.setStreamName("preview")
    cam.preview.link(xout.input)

    device.startPipeline(pipeline)
    q = device.getOutputQueue(name="preview", maxSize=4, blocking=False)

    while True:
        frame = q.get().getCvFrame()

        print(f"[INFO] Frame received: shape={frame.shape}")
        if cv2.waitKey(1) == ord('q'):
            break
