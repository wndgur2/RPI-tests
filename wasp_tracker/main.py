from oak_camera import OakCamera
from hailo_infer import HailoInfer
from tracker import WaspTracker
from utils import overlay_detections
from class_names import CLASS_NAMES
import cv2
import time

def main():
    camera = OakCamera()
    infer = HailoInfer("models/yolov8n2.hef")
    tracker = WaspTracker()

    while True:
        frame, depth = camera.get_frames()
        ############################임시############################
        frame = cv2.imread("sample2.jpg")
        # frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ###########################################################
        if frame is None or depth is None:
            continue

        detections = infer.run_inference(frame)
        print(f"[MAIN] Detections: {len(detections)}")
        tracked = tracker.update(detections)

        annotated = overlay_detections(frame.copy(), tracked, depth, camera.get_intrinsics(), CLASS_NAMES)
        cv2.imwrite("output.jpg", annotated)
        time.sleep(0.03)  # 30 FPS 제한

if __name__ == "__main__":
    main()
